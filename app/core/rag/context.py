"""Shared formatting helpers for retrieved SOP chunks."""

from app.schemas.graph import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a citation-friendly context block for prompts."""
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(f"[来源: {c.filename}, 第{c.page}页]\n{c.content}" for c in chunks)
