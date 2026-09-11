"""Generation node: produce the final answer, grounded in retrieved SOP chunks."""

from langgraph.graph import END
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import Command

from app.core.prompts import SYSTEM_PROMPT, build_context_block
from app.core.rag.context import format_context
from app.schemas.graph import GraphState
from app.services.llm import llm_service
from app.utils import dump_messages, prepare_messages, process_llm_response


async def generate_node(state: GraphState, config: RunnableConfig) -> Command:
    """Generate the final answer from the retrieved chunks and update ``sources``."""
    if state.grade is None or not state.grade.sufficient or not state.retrieved_docs:
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="现有 SOP 资料不足以可靠回答这个问题。请补充适用文档、设备型号或具体场景，或转交负责人确认。"
                    )
                ],
                "sources": [],
            },
            goto=END,
        )
    username = config.get("metadata", {}).get("username")
    context = format_context(state.retrieved_docs)

    messages = prepare_messages(state.messages, SYSTEM_PROMPT)

    # Per-turn dynamic content (date, retrieved excerpts, long-term memory) is
    # appended to the *current* human message rather than folded into the
    # system prompt, which stays fixed across every call — see
    # app/core/prompts/__init__.py for why (LLM provider prefix caching).
    context_block = build_context_block(
        username=username, long_term_memory=state.long_term_memory, sop_context=context
    )
    messages[-1].content = f"{messages[-1].content}\n\n{context_block}"

    response_message = await llm_service.call(dump_messages(messages), config=config)
    response_message = process_llm_response(response_message)

    return Command(update={"messages": [response_message], "sources": state.retrieved_docs}, goto=END)
