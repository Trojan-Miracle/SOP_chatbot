"""Knowledge-base tool using the same hybrid retrieval service as the other workflows.

This is the deliberate contrast with the ``app/core/langgraph`` graph: there,
retrieval is a fixed node every query always passes through. Here, the model
decides for itself whether a question needs the knowledge base at all, and
can call this tool zero, one, or several times with different queries in a
single turn.
"""

from langchain_core.tools import tool

from app.core.rag.context import format_context
from app.core.rag.retrieval import search_sop


@tool
async def search_knowledge_base(query: str) -> str:
    """Search the internal SOP knowledge base for information relevant to a question.

    Use this whenever the user asks about company procedures, SOPs, or any
    topic that might be documented internally rather than general knowledge.

    Args:
        query: A focused search query — specific keywords work better than a
            full question (e.g. "reactor cleaning validation" rather than
            "what does the SOP say about cleaning reactors").

    Returns:
        The top matching excerpts with their source filename and page, or a
        message saying nothing was found.
    """
    hits = await search_sop(query)
    if not hits:
        return "No matching SOP content found for this query."
    return format_context(hits)
