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
    """Lazy init the connection pool. Called on first get_conn()."""
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
    return _pool


@contextmanager
def get_conn():
    """Yield a psycopg2 connection from the pool.

    The connection is auto-committed on success and rolled back on
    exception. Caller should not call .commit()/.rollback() manually.
    """
    pool = _init_pool()
    import psycopg2
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
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
