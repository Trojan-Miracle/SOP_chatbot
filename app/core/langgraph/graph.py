"""This file contains the Agentic RAG LangGraph workflow and interactions with the LLM."""

from typing import (
    AsyncGenerator,
    Optional,
    cast,
)
from urllib.parse import quote_plus

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    convert_to_openai_messages,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import (
    END,
    StateGraph,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import (
    CompiledStateGraph,
)
from langgraph.types import StateSnapshot
from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool

from app.core.config import (
    Environment,
    settings,
)
from app.core.langgraph.nodes import (
    generate_node,
    grade_node,
    retrieve_node,
    rewrite_node,
)
from app.core.logging import logger
from app.core.observability import langfuse_callback_handler
from app.schemas import (
    GraphState,
    Message,
    RetrievedChunk,
)
from app.services.memory import memory_service
from app.utils import dump_messages, extract_text_content, spawn_background_task

PostgresConnPool = AsyncConnectionPool[AsyncConnection[DictRow]]


class LangGraphAgent:
    """Manages the Agentic RAG LangGraph workflow and interactions with the LLM.

    The workflow is a retrieve -> grade -> (generate | rewrite -> retrieve)
    loop: chunks are pulled from the SOP vector store, an LLM judges whether
    they're sufficient to answer the question, and either generates a cited
    answer or reformulates the query and tries again (bounded by
    ``settings.RAG_MAX_REWRITES``).
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        self._connection_pool: Optional[PostgresConnPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    async def _get_connection_pool(self) -> Optional[PostgresConnPool]:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool or None when the pool fails to initialise in
            production (the app keeps running in a degraded mode).
        """
        if self._connection_pool is None:
            try:
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                        "row_factory": dict_row,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        """Create and configure the Agentic RAG LangGraph workflow.

        Returns:
            Optional[CompiledStateGraph]: The configured LangGraph instance or None if init fails
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                graph_builder.add_node("retrieve", retrieve_node, destinations=("grade",))
                graph_builder.add_node("grade", grade_node, destinations=("generate", "rewrite"))
                graph_builder.add_node("rewrite", rewrite_node, destinations=("retrieve",))
                graph_builder.add_node("generate", generate_node, destinations=(END,))
                graph_builder.set_entry_point("retrieve")
                graph_builder.set_finish_point("generate")

                connection_pool = await self._get_connection_pool()
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                else:
                    checkpointer = None
                    if settings.ENVIRONMENT != Environment.PRODUCTION:
                        raise Exception("Connection pool initialization failed")

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_graph")
                    return None
                raise e

        return self._graph

    async def _get_graph(self) -> CompiledStateGraph:
        """Return the compiled graph, creating it on first access.

        Raises:
            RuntimeError: When ``create_graph()`` swallowed an init failure
                (production-only path) and returned ``None``. Callers can
                rely on the return being non-``None``.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        if self._graph is None:
            raise RuntimeError("graph initialization failed")
        return self._graph

    @staticmethod
    def _build_config(session_id: str, user_id: Optional[str], username: Optional[str]) -> RunnableConfig:
        callbacks: list[BaseCallbackHandler] = [langfuse_callback_handler] if settings.LANGFUSE_TRACING_ENABLED else []
        return {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
    ) -> tuple[list[Message], list[RetrievedChunk]]:
        """Get a response from the Agentic RAG graph.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.
            document_ids (Optional[list[str]]): Restrict retrieval to these document IDs only.

        Returns:
            tuple[list[Message], list[RetrievedChunk]]: The response messages
                and the SOP chunks cited in the latest answer.
        """
        graph = await self._get_graph()
        config = self._build_config(session_id, user_id, username)

        try:
            relevant_memory = await memory_service.search(user_id, messages[-1].content)
            relevant_memory = relevant_memory or "No relevant memory found."

            response = await graph.ainvoke(
                input={
                    "messages": dump_messages(messages),
                    "long_term_memory": relevant_memory,
                    "query": "",
                    "document_ids": document_ids,
                    "retrieved_docs": [],
                    "rewrite_count": 0,
                    "grade": None,
                    "sources": [],
                },
                config=config,
            )

            openai_msgs = cast(list[dict], convert_to_openai_messages(response["messages"]))
            spawn_background_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))

            sources = self.__coerce_sources(response.get("sources", []))
            return self.__process_messages(response["messages"]), sources
        except Exception as e:
            logger.exception("get_response_failed", error=str(e), session_id=session_id)
            raise

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the Agentic RAG graph.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.
            document_ids (Optional[list[str]]): Restrict retrieval to these document IDs only.

        Yields:
            str: Tokens of the LLM response (only the ``generate`` node streams).
        """
        config = self._build_config(session_id, user_id, username)
        graph = await self._get_graph()

        try:
            relevant_memory = await memory_service.search(user_id, messages[-1].content)
            relevant_memory = relevant_memory or "No relevant memory found."
            graph_input = {
                "messages": dump_messages(messages),
                "long_term_memory": relevant_memory,
                "query": "",
                "document_ids": document_ids,
                "retrieved_docs": [],
                "rewrite_count": 0,
                "grade": None,
                "sources": [],
            }

            emitted = False
            async for token, metadata in graph.astream(
                graph_input,
                config,
                stream_mode="messages",
            ):
                if not isinstance(metadata, dict) or metadata.get("langgraph_node") != "generate":
                    continue
                if not isinstance(token, (AIMessage, AIMessageChunk)):
                    continue

                text = extract_text_content(token.content)
                if text:
                    emitted = True
                    yield text

            state = await graph.aget_state(config)
            if not emitted and state.values.get("messages"):
                yield extract_text_content(state.values["messages"][-1].content)
            if state.values and "messages" in state.values:
                openai_msgs = cast(list[dict], convert_to_openai_messages(state.values["messages"]))
                spawn_background_task(memory_service.add(user_id, openai_msgs, config.get("metadata")))
        except GraphInterrupt:
            raise
        except Exception as stream_error:
            logger.exception("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        graph = await self._get_graph()

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        return self.__process_messages(state.values["messages"]) if state.values else []

    async def get_last_sources(self, session_id: str) -> list[RetrievedChunk]:
        """Get the SOP chunks cited in the most recent turn for a session.

        The streaming endpoint only yields text tokens (no room for a
        structured payload mid-stream), so the API layer calls this
        separately once streaming finishes to attach sources to the final
        SSE event.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[RetrievedChunk]: Sources cited in the latest answer, or empty if none.
        """
        graph = await self._get_graph()
        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        return self.__coerce_sources(state.values.get("sources", [])) if state.values else []

    @staticmethod
    def __coerce_sources(raw_sources: list) -> list[RetrievedChunk]:
        """Coerce checkpoint-deserialized source entries back into ``RetrievedChunk``.

        The Postgres checkpointer round-trips state through JSON, so entries
        may come back as plain dicts rather than the original pydantic model.
        """
        return [s if isinstance(s, RetrievedChunk) else RetrievedChunk(**s) for s in raw_sources]

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            conn_pool = await self._get_connection_pool()
            if conn_pool is None:
                raise RuntimeError("connection pool unavailable; cannot clear chat history")

            async with conn_pool.connection() as conn:
                async with conn.pipeline():
                    for table in settings.CHECKPOINT_TABLES:
                        await conn.execute(
                            sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                            (session_id,),
                        )
                logger.info(
                    "checkpoint_tables_cleared_for_session",
                    tables=settings.CHECKPOINT_TABLES,
                    session_id=session_id,
                )

        except Exception as e:
            logger.error(
                "clear_chat_history_operation_failed",
                session_id=session_id,
                error=str(e),
            )
            raise
