"""Query-rewriting node: reformulate the retrieval query when it fell short."""

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import Command

from app.core.config import settings
from app.core.logging import logger
from app.schemas.graph import GraphState, RewrittenQuery
from app.services.llm import llm_service

REWRITE_PROMPT = """The following SOP retrieval query did not return enough information to \
answer the user's question.

Query used: {query}
Why it was insufficient: {reasoning}

Rewrite the query to be more specific and likely to match the relevant SOP section — for \
example by using more precise terminology, expanding abbreviations, or narrowing the scope. \
Keep it a short search query, not a full sentence."""


async def rewrite_node(state: GraphState, config: RunnableConfig) -> Command:
    """Rewrite the retrieval query based on the grading feedback, then loop back to retrieve."""
    reasoning = state.grade.reasoning if state.grade else ""
    prompt = REWRITE_PROMPT.format(query=state.query, reasoning=reasoning)

    result = await llm_service.call(
        [{"role": "user", "content": prompt}],
        model_name=settings.GRADE_LLM_MODEL,
        response_format=RewrittenQuery,
        config=config,
    )

    logger.info("query_rewritten", original=state.query, rewritten=result.query, attempt=state.rewrite_count + 1)
    return Command(update={"query": result.query, "rewrite_count": state.rewrite_count + 1}, goto="retrieve")
