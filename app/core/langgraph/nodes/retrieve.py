"""Graph adapter around the shared SOP retrieval service."""

from langchain_core.messages import HumanMessage
from langgraph.graph.state import Command

from app.core.rag.retrieval import search_sop
from app.schemas.graph import GraphState


def _latest_question(messages: list) -> str:
    """Extract the text of the most recent human message in the conversation."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


async def retrieve_node(state: GraphState) -> Command:
    """Retrieve the current or rewritten query, then route to grading."""
    query = state.query or _latest_question(state.messages)
    retrieved = await search_sop(query, document_ids=state.document_ids)
    return Command(update={"query": query, "retrieved_docs": retrieved}, goto="grade")
