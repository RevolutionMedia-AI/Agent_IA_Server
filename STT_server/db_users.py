"""Postgres-backed implementations of the auth file functions.

The original auth.py uses module-level JSON files:
  - load_users()    reads STT_server/data/users.json
  - save_users(list) writes the same file
  - load_sessions()  reads STT_server/data/sessions.json
  - save_sessions()  writes the same file

This module replaces those with Postgres queries. The function
signatures match exactly, so auth.py can swap the import without
touching any other line.

Schema (mirrors db/migrations/001_schema.sql):
  users(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,             -- SHA-256 hex
    role TEXT NOT NULL DEFAULT 'admin',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
  )

  auth_sessions(
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
  )
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from STT_server.db import get_conn, is_postgres

log = logging.getLogger("stt_server.db_users")

# JSON file paths (kept for the JSON backend fallback).
# db_users.py lives in STT_server/, so .parent = STT_server/.
DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_FILE = DATA_DIR / "users.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"


# ── JSON backend (preserved for local dev) ────────────────────────────

def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _write_json(path, data):
    _ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_users_json() -> list:
    return _read_json(USERS_FILE, [])


def save_users_json(users: list) -> None:
    _write_json(USERS_FILE, users)


def load_sessions_json() -> dict:
    return _read_json(SESSIONS_FILE, {})


def save_sessions_json(sessions: dict) -> None:
    _write_json(SESSIONS_FILE, sessions)


# ── Postgres backend (preferred) ────────────────────────────────────

def _row_to_user(row) -> dict:
    """psycopg2 RealDictRow (or tuple) -> user dict matching the
    JSON-file shape, so auth.py treats both backends identically.
    """
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "password": row["password"],
        "role": row["role"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def _row_to_session(row) -> dict:
    return {
        "user_id": row["user_id"],
        "email": row["email"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
    }


def load_users_db() -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, email, password, role, created_at, updated_at FROM users ORDER BY created_at")
            return [_row_to_user(r) for r in cur.fetchall()]


def save_users_db(users: list) -> None:
    """Replace the users table with the given list.

    We use a single transaction: DELETE all then INSERT the new set.
    Simpler than upserting each row, and the user set is small (single
    digits), so perf is fine. The auth flow only ever adds a new user
    or updates an existing one — never a partial set.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users")
            for u in users:
                cur.execute(
                    """
                    INSERT INTO users (id, name, email, password, role, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s,
                            COALESCE(%s, NOW()),
                            COALESCE(%s, NOW()))
                    """,
                    (
                        u.get("id"),
                        u.get("name"),
                        u.get("email"),
                        u.get("password"),
                        u.get("role", "admin"),
                        # Parse created_at if it came in as a string; NULL otherwise
                        _parse_iso(u.get("created_at")),
                        _parse_iso(u.get("updated_at")),
                    ),
                )


def load_sessions_db() -> dict:
    """Returns a dict {token: {user_id, email, created_at, expires_at}}
    matching the JSON-file shape, so callers can iterate the same way.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT token, user_id, email, created_at, expires_at FROM auth_sessions"
            )
            out = {}
            for r in cur.fetchall():
                out[r["token"]] = _row_to_session(r)
            return out


def save_sessions_db(sessions: dict) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM auth_sessions")
            for token, s in sessions.items():
                cur.execute(
                    """
                    INSERT INTO auth_sessions (token, user_id, email, created_at, expires_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        token,
                        s.get("user_id"),
                        s.get("email"),
                        _parse_iso(s.get("created_at")) or datetime.now(timezone.utc),
                        _parse_iso(s.get("expires_at")) or datetime.now(timezone.utc),
                    ),
                )


def _parse_iso(s):
    """Parse an ISO-8601 string into a datetime, or return None on garbage."""
    if not s:
        return None
    try:
        # ponytail: tolerate both '...Z' and '...+00:00' representations
        # because the JSON files were written with mixed formats historically.
        s2 = s.replace("Z", "+00:00") if isinstance(s, str) else s
        return datetime.fromisoformat(s2)
    except (ValueError, TypeError):
        return None


def update_user_name(user_id: str, name: str) -> None:
    """Mirror a display-name change into the users table.

    Used by routes/api.py's settings PUT when running on Postgres.
    On the JSON backend the settings handler writes users.json directly
    so this helper is unused there.
    """
    if not is_postgres():
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET name = %s, updated_at = NOW() WHERE id = %s",
                (name, user_id),
            )


# ── Public API: pick the right backend at import time ────────────────

# The two globals below are read by routes/auth.py. They are
# reassigned once at import so the rest of the code can stay
# backend-agnostic.

if is_postgres():
    log.warning("[db] using Postgres backend for auth (users + sessions)")
    load_users = load_users_db
    save_users = save_users_db
    load_sessions = load_sessions_db
    save_sessions = save_sessions_db
else:
    log.warning("[db] DATABASE_URL not set - using JSON file backend for auth")
    load_users = load_users_json
    save_users = save_users_json
    load_sessions = load_sessions_json
    save_sessions = save_sessions_json
