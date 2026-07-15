"""This file contains the graph schema for the application."""

from typing import Annotated

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
)


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store."""

    content: str = Field(description="The chunk's text content")
    filename: str = Field(description="Source PDF filename")
    page: int = Field(description="1-indexed source page number")
    score: float = Field(
        default=0.0,
        description=(
            "Ranking score. Pure vector mode: Chroma distance (lower is more similar). "
            "Hybrid mode: reciprocal-rank-fusion score (higher is better) — the two are "
            "not comparable to each other."
        ),
    )


class GradeResult(BaseModel):
    """Structured output for the retrieval-grading node."""

    sufficient: bool = Field(description="Whether the retrieved chunks are sufficient to answer the question")
    reasoning: str = Field(description="Brief reasoning for the sufficiency judgement")


class RewrittenQuery(BaseModel):
    """Structured output for the query-rewriting node."""

    query: str = Field(description="The rewritten, more specific retrieval query")


class GraphState(BaseModel):
    """State definition for the Agentic RAG LangGraph workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    long_term_memory: str = Field(default="", description="The long term memory of the conversation")

    query: str = Field(default="", description="Current retrieval query (may be a rewrite of the original question)")
    document_ids: list[str] | None = Field(
        default=None, description="Optional filter — restrict retrieval to these document IDs only"
    )
    retrieved_docs: list[RetrievedChunk] = Field(default_factory=list, description="Chunks from the last retrieval")
    rewrite_count: int = Field(default=0, description="Number of query rewrites performed so far")
    grade: GradeResult | None = Field(default=None, description="Result of the last sufficiency grading")
    sources: list[RetrievedChunk] = Field(default_factory=list, description="Chunks cited in the final answer")
