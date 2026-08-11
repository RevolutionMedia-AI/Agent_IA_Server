"""Bounded executor for sync helpers used by the audio path.

Replaces ad-hoc ``asyncio.to_thread`` calls so a slow call (e.g. a Twilio
HTTP fetch) on one path doesn't queue behind other tasks in Python's
default executor.
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

_pool = ThreadPoolExecutor(
    max_workers=min(32, max(4, (os.cpu_count() or 1) * 4)),
    thread_name_prefix="audio-thread",
)


async def to_thread(func, *args, **kwargs):
    """Like :func:`asyncio.to_thread` but dispatches to our bounded pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_pool, lambda: func(*args, **kwargs))


def shutdown(wait: bool = True) -> None:
    """Shutdown the pool. Call from FastAPI lifespan shutdown."""
    _pool.shutdown(wait=wait)


def stats() -> dict:
    """Return pool stats for diagnostics. ``active`` is -1 (sentinel)."""
    return {
        "max_workers": _pool._max_workers,
        "active": -1,
    }
