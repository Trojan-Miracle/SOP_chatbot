"""Codebase exploration tools — read and grep this project's own source.

Security boundary: both tools are sandboxed to ``PROJECT_ROOT`` via
``_resolve_safe_path``, which resolves symlinks/``..`` and rejects anything
that escapes the root. These are exposed over a network-facing API — without
this check, ``read_project_file("../../../../etc/passwd")`` would be a real
arbitrary-file-read vulnerability, not a hypothetical one.
"""

import re
from pathlib import Path

from langchain_core.tools import tool

# app/core/harness/tools/code_tools.py -> parents[4] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]

_ALLOWED_SUFFIXES = {".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".json", ".cfg", ".ini"}
_MAX_FILE_BYTES = 200_000
_MAX_GREP_MATCHES = 50


def _resolve_safe_path(relative_path: str) -> Path:
    """Resolve ``relative_path`` under ``PROJECT_ROOT``, raising on any escape attempt."""
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if candidate != PROJECT_ROOT and PROJECT_ROOT not in candidate.parents:
        raise ValueError(f"path '{relative_path}' escapes the project root — not allowed")
    return candidate


@tool
def read_project_file(path: str) -> str:
    """Read a text file from this project's own codebase (read-only, sandboxed).

    Args:
        path: Path relative to the project root, e.g. "app/main.py".

    Returns:
        The file's contents (truncated if very large), or an error message.
    """
    try:
        resolved = _resolve_safe_path(path)
    except ValueError as e:
        return f"Error: {e}"

    if not resolved.is_file():
        return f"Error: '{path}' is not a file"
    if resolved.suffix not in _ALLOWED_SUFFIXES:
        return f"Error: reading '{resolved.suffix}' files is not allowed (allowed: {sorted(_ALLOWED_SUFFIXES)})"

    content = resolved.read_text(encoding="utf-8", errors="replace")
    if len(content) > _MAX_FILE_BYTES:
        content = content[:_MAX_FILE_BYTES] + "\n... (truncated)"
    return content


@tool
def grep_project_files(pattern: str, path_glob: str = "**/*.py") -> str:
    """Search this project's own codebase for a regex pattern (read-only, sandboxed).

    Args:
        pattern: A Python regular expression to search for.
        path_glob: Glob (relative to the project root) restricting which
            files to search, e.g. "app/**/*.py". Defaults to all Python files.

    Returns:
        Up to 50 matching lines as "path:line_number: text", or an error message.
    """
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex — {e}"

    matches: list[str] = []
    for file_path in PROJECT_ROOT.glob(path_glob):
        try:
            resolved = _resolve_safe_path(str(file_path.relative_to(PROJECT_ROOT)))
        except ValueError:
            continue
        if not resolved.is_file() or resolved.suffix not in _ALLOWED_SUFFIXES:
            continue

        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines, start=1):
            if regex.search(line):
                rel = resolved.relative_to(PROJECT_ROOT)
                matches.append(f"{rel}:{i}: {line.strip()}")
                if len(matches) >= _MAX_GREP_MATCHES:
                    return "\n".join(matches) + "\n... (truncated at 50 matches)"

    return "\n".join(matches) if matches else "No matches found."
