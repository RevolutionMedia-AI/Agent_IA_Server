"""Run the sync data layer off the event loop.

The ``db_*.py`` modules are sync because psycopg2 is sync. Calling them
straight from an async handler blocks the event loop — and that same
loop is what receives the audio frames of every live call, so one slow
query delays every other call's media frames, not just its own.

This module hands those calls to a dedicated thread pool. It is
deliberately NOT the shared pool in ``services/thread_pool.py``: that
one also runs VAD and other CPU work, so a blocked query would sit
behind a backlog of frame-sized jobs (head-of-line blocking). DB work
gets its own threads and its own budget.

Usage:

    from STT_server.services.db_async import db_call

    row = await db_call(get_agent, agent_id, user_id)

The function stays sync; only the *call site* becomes async.
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

log = logging.getLogger("stt_server.db")

# ponytail: sized to match the psycopg2 pool (ThreadedConnectionPool
# maxconn=10 in STT_server/db.py). More threads than connections would
# just queue on pool.getconn() anyway; fewer would under-use the pool.
_DB_MAX_WORKERS = max(2, int(os.getenv("DB_EXECUTOR_MAX_WORKERS", "10")))

_pool = ThreadPoolExecutor(
    max_workers=_DB_MAX_WORKERS,
    thread_name_prefix="db-thread",
)


async def db_call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a sync DB function on the DB thread pool and await its result.

    Exceptions propagate unchanged, so callers keep their existing
    ``try/except`` around the DB call.
    """
    loop = asyncio.get_running_loop()
    if kwargs:
        call = lambda: fn(*args, **kwargs)  # noqa: E731
    else:
        call = lambda: fn(*args)  # noqa: E731
    return await loop.run_in_executor(_pool, call)
