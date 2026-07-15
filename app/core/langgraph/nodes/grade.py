"""Grading node: judge whether retrieved SOP chunks are sufficient to answer."""

from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import Command

from app.core.config import settings
from app.core.logging import logger
from app.core.rag.context import format_context
from app.schemas.graph import GradeResult, GraphState
from app.services.llm import llm_service

GRADE_PROMPT = """You are grading whether retrieved SOP (Standard Operating Procedure) excerpts \
are sufficient to answer a user's question.

Question: {question}

Retrieved excerpts:
{context}

Judge whether these excerpts contain enough information to answer the question accurately \
and specifically. If the excerpts are empty, off-topic, or only partially relevant, mark as \
insufficient."""


async def grade_node(state: GraphState, config: RunnableConfig) -> Command:
    """Grade the last retrieval and route to ``generate`` or ``rewrite``."""
    prompt = GRADE_PROMPT.format(question=state.query, context=format_context(state.retrieved_docs))

    grade = await llm_service.call(
        [{"role": "user", "content": prompt}],
        model_name=settings.GRADE_LLM_MODEL,
        response_format=GradeResult,
        config=config,
    )

    logger.info(
        "retrieval_graded",
        sufficient=grade.sufficient,
        rewrite_count=state.rewrite_count,
        reasoning=grade.reasoning,
    )

    if grade.sufficient or state.rewrite_count >= settings.RAG_MAX_REWRITES:
        return Command(update={"grade": grade}, goto="generate")
    return Command(update={"grade": grade}, goto="rewrite")
