"""Harness API endpoints — tool-calling ReAct agent, parallel to the SOP RAG chatbot."""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from langchain.agents.middleware.types import InputAgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
    convert_to_openai_messages,
)

from app.api.v1.auth import get_current_session
from app.core.config import settings
from app.core.harness.agent import get_harness_agent
from app.core.limiter import limiter
from app.core.logging import logger
from app.models.session import Session
from app.schemas.chat import Message
from app.schemas.harness import (
    HarnessChatRequest,
    HarnessChatResponse,
    ToolCallTrace,
)
from app.utils import dump_messages

router = APIRouter()


@router.post("/chat", response_model=HarnessChatResponse)
@limiter.limit(settings.RATE_LIMIT_ENDPOINTS["chat"][0])
async def harness_chat(
    request: Request,
    chat_request: HarnessChatRequest,
    session: Session = Depends(get_current_session),
):
    """Ask the tool-calling harness a question.

    Args:
        request: The FastAPI request object for rate limiting.
        chat_request: The chat request containing messages.
        session: The current session from the auth token.

    Returns:
        HarnessChatResponse: The response messages and the tools invoked while answering.

    Raises:
        HTTPException: If there's an error processing the request.
    """
    try:
        agent = get_harness_agent()
        config: RunnableConfig = {"configurable": {"thread_id": session.id}}

        # MemorySaver accumulates the full conversation across turns — capture
        # the message count beforehand so tool_calls only reports what this
        # turn actually did, not every tool call made since the session began.
        prior_state = await agent.aget_state(config)
        prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

        agent_input: InputAgentState = {"messages": [message for message in dump_messages(chat_request.messages)]}
        result = await agent.ainvoke(agent_input, config=config)

        raw_messages = result["messages"]
        new_messages = raw_messages[prior_count:]
        tool_results_by_id = {m.tool_call_id: m.content for m in new_messages if isinstance(m, ToolMessage)}

        tool_calls: list[ToolCallTrace] = []
        for m in new_messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    tool_calls.append(
                        ToolCallTrace(
                            tool=tc["name"],
                            args=tc["args"],
                            result=str(tool_results_by_id.get(tc.get("id") or "", "")),
                        )
                    )

        openai_msgs = convert_to_openai_messages(raw_messages)
        messages = [
            Message(role=m["role"], content=str(m["content"]))
            for m in openai_msgs
            if m["role"] in ("assistant", "user") and m["content"]
        ]

        logger.info("harness_chat_processed", session_id=session.id, tool_call_count=len(tool_calls))
        return HarnessChatResponse(messages=messages, tool_calls=tool_calls)
    except Exception as e:
        logger.exception("harness_chat_failed", session_id=session.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
