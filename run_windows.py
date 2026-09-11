"""Windows-native dev entrypoint.

`uv run uvicorn app.main:app` fails on Windows: psycopg's async driver
(used by LangGraph's Postgres checkpointer) requires a SelectorEventLoop, but
modern uvicorn (>=0.36) hardcodes ProactorEventLoop on win32 via an explicit
`loop_factory` passed to `asyncio.Runner` — this bypasses `asyncio.set_event_loop_policy()`
entirely, so the usual policy-based workaround doesn't work. Passing a custom
loop factory via uvicorn's `loop=` import-string mechanism does.

Note: reload=False. Uvicorn's --reload spawns worker subprocesses that need
ProactorEventLoop, which conflicts with the SelectorEventLoop this script
forces — auto-reload isn't compatible with this workaround. Restart manually
after code changes, or develop inside Docker/WSL2 (Linux has neither loop).

Usage: uv run python run_windows.py
"""

import asyncio

import uvicorn


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Create the selector loop required by the async PostgreSQL driver."""
    return asyncio.SelectorEventLoop()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, loop="run_windows:selector_loop_factory")
