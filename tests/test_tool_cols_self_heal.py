"""Tests for the self-healing schema check in db_tools.

The 014_service_credentials.sql migration adds a `credentials`
JSONB column to agent_tools. The operator hit a 500 on Railway
because the migration didn't run on startup (start.sh splitter,
image cache, or some other edge case). The BE now self-heals:
the first call to _tool_cols() runs an idempotent ALTER TABLE if
the column is missing, then caches the result for the rest of
the process lifetime.

These tests cover the JSON path (no schema to check; always
returns the full column list) and the Postgres self-heal path
(mocked so we don't need a live Postgres).
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Any

import pytest


# ── JSON path: always full cols, no ALTER ─────────────────────────────────


def test_json_path_returns_full_tool_cols():
    """When is_postgres() is False (JSON fallback), _tool_cols()
    returns the full column list including credentials. The resolver
    ignores it; the JSON storage doesn't care about schema."""
    from STT_server import db_tools

    # Force the JSON path and reset the cache.
    db_tools._TOOL_COLS_EXTRA = []
    db_tools._columns_check_done = False
    with patch.object(db_tools, "is_postgres", return_value=False):
        cols = db_tools._tool_cols()

    assert "credentials" in cols, f"JSON path should include credentials, got: {cols}"
    assert "id" in cols and "updated_at" in cols
    # The cache was populated.
    assert db_tools._columns_check_done is True
    assert "credentials" in db_tools._TOOL_COLS_EXTRA


def test_tool_cols_returns_cached_value_on_second_call():
    """The second call must NOT hit information_schema. We assert this
    by leaving is_postgres() to raise: if the cached value is
    returned, no DB call is made."""
    from STT_server import db_tools

    db_tools._TOOL_COLS_EXTRA = ["credentials"]
    db_tools._columns_check_done = True
    with patch.object(db_tools, "is_postgres", side_effect=AssertionError("must not be called")):
        cols1 = db_tools._tool_cols()
        cols2 = db_tools._tool_cols()
    assert cols1 == cols2
    assert "credentials" in cols1


# ── Postgres self-heal path (mocked) ──────────────────────────────────────


@dataclass
class _ScriptedCursor:
    """A fake cursor that returns a scriptable sequence of (rows, side_effects).

    ponytail: the self-heal runs TWO information_schema SELECTs
    (initial check + post-ALTER confirm) on separate connections from
    the pool. The mock returns the same result for both so the test
    can simulate "column already there" without re-populating the
    queue.
    """
    fetchone_results: list[Any] = field(default_factory=list)
    fetchall_results: list[Any] = field(default_factory=list)
    # ponytail: if set, fetchone() returns this value forever (the
    # cursor is recreated on every get_conn() call, so the queue
    # would otherwise drain between the initial check and the
    # post-ALTER confirm). For the "column present" case both
    # checks return the same row, so the loop never needs to drain.
    fetchone_static: Any = None
    executed: list = field(default_factory=list)

    def __enter__(self): return self
    def __exit__(self, *a): return False

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchone(self):
        if self.fetchone_static is not None:
            return self.fetchone_static
        return self.fetchone_results.pop(0) if self.fetchone_results else None

    def fetchall(self):
        return self.fetchall_results.pop(0) if self.fetchall_results else []


def _make_stub_get_conn(cursor: _ScriptedCursor):
    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return cursor
        def commit(self): pass
        def rollback(self): pass
    return lambda: _Conn()


def test_postgres_path_self_heals_when_credentials_column_missing():
    """Simulate: information_schema says no credentials column. The
    self-heal runs ALTER TABLE, then re-checks, then caches.
    Subsequent calls do NOT hit information_schema again."""
    from STT_server import db_tools

    cursor = _ScriptedCursor(
        fetchone_results=[
            None,                  # first info_schema → no creds col
            ("credentials",),      # post-ALTER confirm → col present
        ],
    )

    db_tools._TOOL_COLS_EXTRA = []
    db_tools._columns_check_done = False

    with patch.object(db_tools, "is_postgres", return_value=True), \
         patch.object(db_tools, "get_conn", _make_stub_get_conn(cursor)):
        cols = db_tools._tool_cols()

    assert "credentials" in cols
    assert db_tools._columns_check_done is True
    assert "credentials" in db_tools._TOOL_COLS_EXTRA

    # Three SQL statements ran: info_schema SELECT, ALTER TABLE,
    # info_schema SELECT (verify).
    alter_calls = [q for q, _ in cursor.executed if "ALTER TABLE" in q.upper()]
    assert len(alter_calls) == 1, f"expected 1 ALTER, got {len(alter_calls)}: {cursor.executed}"
    assert "credentials JSONB" in alter_calls[0]


def test_postgres_path_no_alter_when_credentials_column_present():
    """If the column is already there, the self-heal is a no-op. No
    ALTER issued; the confirm SELECT still runs (cheap, defensive
    against a half-applied ALTER on a previous deploy)."""
    from STT_server import db_tools

    cursor = _ScriptedCursor(
        fetchone_static=("credentials",),  # both checks return same row
    )

    db_tools._TOOL_COLS_EXTRA = []
    db_tools._columns_check_done = False

    with patch.object(db_tools, "is_postgres", return_value=True), \
         patch.object(db_tools, "get_conn", _make_stub_get_conn(cursor)):
        cols = db_tools._tool_cols()

    assert "credentials" in cols
    alter_calls = [q for q, _ in cursor.executed if "ALTER TABLE" in q.upper()]
    assert len(alter_calls) == 0, f"expected 0 ALTER, got {alter_calls}"
    # Two SELECTs ran: initial check + post-ALTER confirm (defensive).
    assert len(cursor.executed) == 2, (
        f"expected 2 info_schema SELECTs, got {len(cursor.executed)}: {cursor.executed}"
    )


def test_postgres_path_second_call_does_not_re_check():
    """The cache means the second call to _tool_cols() doesn't open
    any connection. Verify by setting is_postgres to raise on the
    second call."""
    from STT_server import db_tools

    cursor = _ScriptedCursor(
        fetchone_static=("credentials",),  # first call: col present
    )

    db_tools._TOOL_COLS_EXTRA = []
    db_tools._columns_check_done = False

    # First call: real (mocked) is_postgres + connection.
    # Second call: is_postgres raises (would explode if called).
    call_count = {"n": 0}

    def is_postgres_counting():
        call_count["n"] += 1
        if call_count["n"] > 1:
            raise AssertionError("is_postgres called on cached path; cache is broken")
        return True

    with patch.object(db_tools, "is_postgres", side_effect=is_postgres_counting), \
         patch.object(db_tools, "get_conn", _make_stub_get_conn(cursor)):
        cols1 = db_tools._tool_cols()
        cols2 = db_tools._tool_cols()
        cols3 = db_tools._tool_cols()

    assert cols1 == cols2 == cols3
    assert "credentials" in cols1
    # Only 1 is_postgres call across the 3 invocations.
    assert call_count["n"] == 1


# ── Smoke test: full insert/select roundtrip on JSON ─────────────────────


def test_json_create_and_get_tool_roundtrip():
    """Sanity: even with the new dynamic column resolver, the JSON
    create / get path still works end-to-end. The tool_cols() returns
    the full list so the JSON path never sees a missing-column error
    (which was the operator's symptom on Postgres)."""
    import sys
    sys.path.insert(0, '.')
    sys.path.insert(0, 'STT_server')
    os.environ['CREDENTIAL_ENCRYPTION_KEY'] = '3caLHixTmxCJ1OAQEK11TEn4k5soMyJhybJIyAFVMfk='
    os.environ['DATABASE_URL'] = ''

    from STT_server import db_tools
    from STT_server.routes import api as api_mod
    from STT_server.services import session_runtime as rt_mod

    with tempfile.TemporaryDirectory() as td:
        fp = pathlib.Path(td) / 'agent_tools.json'
        fp.write_text('[]')
        db_tools._AGENT_TOOLS_FILE = fp
        api_mod.TOOLS_FILE = str(fp)
        rt_mod._TOOLS_FILE = str(fp)

        # Test the BE route's contract end-to-end. We don't go through
        # FastAPI's dependency injection (which would wrap auth in a
        # Depends object); instead call the db helpers directly with
        # the auth dict as a kwarg.
        body = api_mod.ApiKeyUpdate(
            credentials={"api_key": "sk-test1234567890abcdefABCDEF"}
        )
        from STT_server.security.credentials import encrypt_credentials
        encrypted = encrypt_credentials({"api_key": "sk-test1234567890abcdefABCDEF"})
        payload = {
            "agent_id": "__shared__",
            "name": "OpenAI",
            "description": "",
            "webhook_url": "",
            "filler_phrase": "Let me check the system...",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "kind": "webhook",
            "destination": None,
            "assignments": [],
            "function_name": "openai",
            "test_data_model": "gpt-4o-mini",
            "credentials": encrypted,
        }
        # Ponytail: go through the actual storage layer (db_create_tool)
        # to exercise the new dynamic _tool_cols() function. The route
        # layer would do the same after validate_credentials + encrypt.
        result = db_tools.create_tool("u1", payload, tool_id="openai")
        assert result["id"] == "openai"
        assert result["agent_id"] == "__shared__"

        # Read it back via the resolver-shaped helper.
        row = db_tools.get_tool("openai", "u1")
        assert row is not None
        assert row["id"] == "openai"
        # The credentials dict is preserved (Fernet ciphertext).
        assert isinstance(row.get("credentials"), dict)
        assert "api_key" in row["credentials"]
