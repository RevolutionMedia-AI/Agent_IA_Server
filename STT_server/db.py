"""Postgres connection helpers for the BE.

The BE has two storage backends:
  - Postgres (preferred in production): the BE has a DATABASE_URL env
    var pointing to a Postgres instance (Railway provides one
    automatically when both services live in the same project).
  - JSON files (fallback for local dev): STT_server/data/*.json.

This module is the SHIM that picks one. Callers should not import
psycopg2 directly; they should call get_conn() and run queries
through the helper functions in db_users.py, db_agents.py, etc.

Migrations are NOT auto-applied here. The repo has db/migrations/*.sql
and the user runs them manually (the project policy is "no migraciones
en codigo"). The schema in 001_schema.sql is the source of truth.

Connection pool: psycopg2.pool.ThreadedConnectionPool. One pool per
process; lazy init on first use so importing this module is cheap
and JSON-mode deployments never touch psycopg2.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("stt_server.db")

# ponytail: don't import psycopg2 at module import time. The JSON-only
# deployments (local dev) shouldn't pay the cost. Lazy import inside
# the pool init.
_pool = None
_pool_lock = threading.Lock()


def database_url() -> str | None:
    """Read DATABASE_URL from the environment.

    Railway injects this when both the BE service and the Postgres
    service live in the same project. Returns None when not set; callers
    should fall back to the JSON storage backend in that case.
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    # Railway also exposes the URL split across PG* vars. Reassemble.
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER")
    pwd = os.environ.get("PGPASSWORD")
    db = os.environ.get("PGDATABASE")
    if all([host, user, pwd, db]):
        return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    return None


def is_postgres() -> bool:
    return bool(database_url())


def _init_pool():
    """Lazy init the connection pool. Called on first get_conn().

    ponytail: previous version released the lock BEFORE creating the
    pool. Two threads hitting _init_pool simultaneously both saw
    ``_pool is None``, both entered the with-block, but only the
    first to win the lock-check left. The other thread exited the
    ``with _pool_lock`` block and started creating a SECOND pool —
    meaning two ThreadedConnectionPool instances leaked, each
    holding 1..10 idle Postgres connections against the same DB.
    Under load the connection count went 1 + 2 + 3 ... and we
    exhausted the DB's max_connections. Move ALL of the construction
    inside the lock; the cost is a one-shot ~200ms stall on the
    first concurrent call to get_conn().
    """
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        url = database_url()
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. Cannot use Postgres backend. "
                "Either set DATABASE_URL in the environment, or use the JSON "
                "backend (don't call db.get_conn() when DATABASE_URL is unset)."
            )
        # Lazy import — keeps the JSON-only deployments free of psycopg2.
        import psycopg2
        from psycopg2 import pool as pg_pool
        from psycopg2.extras import RealDictCursor
        log.warning("[db] connecting to Postgres: host=%s db=%s", url.split("@")[-1], url.split("/")[-1])
        # ponytail: RealDictCursor so fetchall() returns dicts and we can do
        # row["id"] instead of row[0]. The plain cursor returned tuples,
        # which made _row_to_user crash with "tuple indices must be integers".
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=1, maxconn=10, dsn=url, cursor_factory=RealDictCursor,
        )
        log.warning("[db] pool ready (min=1 max=10)")
        return _pool


@contextmanager
def get_conn(max_attempts: int = 10):
    """Yield a psycopg2 connection from the pool.

    The connection is auto-committed on success and rolled back on
    exception. Caller should not call .commit()/.rollback() manually.

    ponytail: 2026-10-02 — bounded immediate retry on pool exhaustion.
    The pool is 10 slots and the server is bursty (live calls + API +
    background tasks contend), so a single getconn() attempt 500s on
    transient spikes that clear in milliseconds. Production saw exactly
    that: [DB_POOL] exhausted, then the very next call seconds later
    proceeded normally.

    The retry is NON-BLOCKING (time.sleep(0) only yields the GIL) because
    get_conn() is called from async handlers and a real sleep would stall
    the event loop past the 20 ms voice frame budget. This handles the
    race where a slot is about to be returned; if the pool is genuinely
    saturated through all attempts, it still fails fast with PoolError so
    callers keep their existing fallback behaviour.
    """
    pool = _init_pool()
    import psycopg2
    conn = None
    for attempt in range(max_attempts):
        try:
            conn = pool.getconn()
            break
        except psycopg2.pool.PoolError:
            if attempt + 1 >= max_attempts:
                # ponytail: never log the DSN — only safe counters.
                log.warning(
                    "[DB_POOL] exhausted max=10 after %d attempts; "
                    "failing fast instead of queuing",
                    max_attempts,
                )
                raise
            # Yield the GIL so the thread returning a slot can run.
            # No real sleep: this is called from async handlers and even
            # 50 ms would blow the voice frame budget.
            time.sleep(0)
    assert conn is not None  # for type checkers; loop breaks or raises
    try:
        yield conn
        conn.commit()
    except BaseException:
        # ponytail: BaseException (not just Exception) so CancelledError
        # / KeyboardInterrupt also roll back. Otherwise the slot goes
        # back to the pool with an open transaction and the next
        # borrower inherits idle-in-transaction (plus any advisory
        # xact lock taken on it).
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def close_pool():
    """Close the pool. Called on FastAPI shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
