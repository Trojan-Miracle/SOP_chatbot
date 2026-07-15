"""Tool registry for the lightweight ReAct harness."""

from app.core.harness.tools.code_tools import grep_project_files, read_project_file
from app.core.harness.tools.kb_search import search_knowledge_base
from app.core.harness.tools.time_tool import get_current_time

tools = [search_knowledge_base, get_current_time, read_project_file, grep_project_files]

__all__ = ["tools", "search_knowledge_base", "get_current_time", "read_project_file", "grep_project_files"]
