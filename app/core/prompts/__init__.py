"""This file contains the prompts for the agent."""

import os
from datetime import datetime
from typing import Optional

from app.core.config import settings

_PROMPTS_DIR = os.path.dirname(__file__)

# Read templates once at module load — no file I/O per request
with open(os.path.join(_PROMPTS_DIR, "system.md"), "r", encoding="utf-8") as _f:
    _SYSTEM_PROMPT_TEMPLATE = _f.read()

with open(os.path.join(_PROMPTS_DIR, "session_title.md"), "r", encoding="utf-8") as _f:
    SESSION_TITLE_PROMPT = _f.read()

# Formatted once at import time (agent_name never changes at runtime), so this
# is a fixed string across every request — DeepSeek (and most providers) cache
# the KV-state of a repeated prompt *prefix*. Folding per-turn content (date,
# retrieved SOP excerpts — different on every call) into the system prompt
# would break that prefix match on every single request; see
# build_context_block() below for where that content goes instead.
SYSTEM_PROMPT = _SYSTEM_PROMPT_TEMPLATE.format(agent_name=settings.PROJECT_NAME + " Agent")


def build_context_block(username: Optional[str] = None, **sections: str) -> str:
    """Build the per-turn dynamic context block.

    Appended to the *current* human message's content instead of folded into
    the system prompt — keeps the system prompt byte-identical across calls
    (see ``SYSTEM_PROMPT`` above for why that matters).

    Args:
        username: Display name of the user, if known.
        **sections: Extra named sections to include verbatim (e.g.
            ``long_term_memory="..."``, ``sop_context="..."``), each rendered
            as its own "# Title Case" section and skipped if falsy.

    Returns:
        A ``<context>...</context>``-wrapped block.
    """
    lines = [f"# Current date and time\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    if username:
        lines.append(f"# User\nYou are talking to {username}.")
    for key, value in sections.items():
        if value:
            lines.append(f"# {key.replace('_', ' ').title()}\n{value}")
    return "<context>\n" + "\n\n".join(lines) + "\n</context>"
