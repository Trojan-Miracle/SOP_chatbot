"""Lightweight ReAct harness — DeepSeek + ``create_agent`` + a small toolset.

Deliberate contrast with ``app/core/langgraph`` (the fixed
retrieve->grade->rewrite->generate graph): here the model itself decides
which tools to call, how many times, and in what order, instead of being
forced through a predetermined pipeline. See the README for the paradigm
this is meant to illustrate.

Uses ``langchain.agents.create_agent`` rather than the older
``langgraph.prebuilt.create_react_agent`` — the latter is deprecated in
favor of this one as of the installed langgraph/langchain versions.
"""

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.core.harness.tools import tools
from app.services.llm.registry import LLMRegistry

SYSTEM_PROMPT = """You are an internal assistant with access to tools. Use them when they help:
- search_knowledge_base: look up internal SOP/documentation content
- get_current_time: get the current date/time in a given timezone
- read_project_file / grep_project_files: explore this project's own codebase (for code-related questions)

Only call a tool when the question actually requires it — for general knowledge questions, just answer directly.
Cite sources when you use search_knowledge_base results."""

# MemorySaver, not the Postgres checkpointer the main RAG graph uses — this is
# a secondary demo path and doesn't need cross-restart persistence; trades
# that away for zero extra coupling to the main agent's connection pool.
_checkpointer = MemorySaver()

_agent = create_agent(
    model=LLMRegistry.get(settings.DEFAULT_LLM_MODEL),
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=_checkpointer,
)


def get_harness_agent():
    """Return the compiled ReAct harness agent."""
    return _agent
