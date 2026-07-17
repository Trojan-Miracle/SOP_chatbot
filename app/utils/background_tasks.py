"""Fire-and-forget background task tracking.

``asyncio.create_task()`` returns a ``Task`` that the event loop only holds a
weak reference to while pending — the asyncio docs explicitly warn that
without a strong reference held somewhere else, a task can be garbage
collected mid-execution. This module keeps a single shared set of strong
references, discarding each task once it completes.
"""

import asyncio
from typing import Coroutine

_background_tasks: set[asyncio.Task] = set()


def spawn_background_task(coro: Coroutine) -> asyncio.Task:
    """Schedule a coroutine as a tracked background task (fire-and-forget, GC-safe)."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
