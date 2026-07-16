"""Harness middleware: context management and tool-result validation.

Without these, the harness had two real gaps versus the RAG graph:
- No context management — MemorySaver just accumulates every message forever,
  unlike the RAG graph's ``prepare_messages``/``trim_messages`` token-bounded
  history.
- No feedback loop — the RAG graph's ``grade`` node judges whether retrieved
  chunks are sufficient and, if not, reformulates the query (``rewrite``).
  The harness had no equivalent: a knowledge-base search could come back
  empty or off-topic and the model would just answer from nothing.
"""

from langchain_core.messages import ToolMessage
from langchain.agents.middleware import SummarizationMiddleware, wrap_tool_call

from app.core.config import settings
from app.core.logging import logger
from app.schemas.graph import GradeResult
from app.services.llm import llm_service
from app.services.llm.registry import LLMRegistry

# Context management: once the conversation crosses 20 messages, summarize
# everything except the most recent 10 into a single context-preserving
# message. Uses the cheap/fast model (deepseek-v4-flash) — summarization is a
# utility task, not the final answer.
context_management_middleware = SummarizationMiddleware(
    model=LLMRegistry.get(settings.GRADE_LLM_MODEL),
    trigger=("messages", 20),
    keep=("messages", 10),
)

_VALIDATE_PROMPT = """You are grading whether a knowledge-base search result is sufficient \
to answer the query it was run for.

Query: {query}

Result:
{result}

Judge whether this result contains enough relevant information. If it's empty, off-topic, \
or only partially relevant, mark as insufficient."""


@wrap_tool_call
async def validate_kb_search_results(request, handler):
    """Grade ``search_knowledge_base`` results and flag insufficient ones.

    Mirrors the RAG graph's ``grade`` node, but expressed as feedback on the
    tool result rather than a graph-level rewrite/retry: an insufficient
    result gets an appended note telling the model to try a different query,
    and the ReAct loop's own tool-calling lets the model act on that itself
    (no separate rewrite step needed — that's already what the harness
    paradigm is for).
    """
    response = await handler(request)

    if request.tool_call["name"] != "search_knowledge_base":
        return response

    result_message = response if isinstance(response, ToolMessage) else None
    if result_message is None:
        return response

    query = request.tool_call["args"].get("query", "")
    prompt = _VALIDATE_PROMPT.format(query=query, result=result_message.content)

    try:
        grade: GradeResult = await llm_service.call(
            [{"role": "user", "content": prompt}],
            model_name=settings.GRADE_LLM_MODEL,
            response_format=GradeResult,
        )
    except Exception as e:
        logger.warning("kb_result_validation_failed", error=str(e))
        return response

    logger.info("kb_search_result_graded", query=query, sufficient=grade.sufficient, reasoning=grade.reasoning)

    if not grade.sufficient:
        result_message.content += (
            f"\n\n(NOTE: this result may be insufficient — {grade.reasoning} "
            "Consider calling search_knowledge_base again with a more specific or differently worded query.)"
        )

    return result_message
