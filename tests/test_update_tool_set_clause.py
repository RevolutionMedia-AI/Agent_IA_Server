"""Regression tests for the agent_tools UPDATE SET-clause builder.

The dynamic SET in db_tools.update_tool (Postgres branch) used to
include `updated_at = %s` from the AgentTool.to_dict() payload AND
`updated_at = NOW()` appended at the end of the loop. Postgres
rejects the resulting SQL with `multiple assignments to same column
"updated_at"`, surfacing as a 500 on PUT /tools/{id} from the FE.

These tests stub the Postgres connection so update_tool exercises
the dynamic-SET branch in-process. We capture the query and assert
it has exactly one assignment per column.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

import pytest


VALID_PAYLOAD = {
    "name": "google_cal",
    "description": "Test desc",
    "webhook_url": "https://revomedia.app.n8n.cloud/webhook/agendar-cita-dinamica",
    "filler_phrase": "Let me check the system...",
    "parameters": {
        "type": "object",
        "properties": {"x": {"type": "string", "description": "x"}},
        "required": ["x"],
    },
}


# --- Postgres stubs ----------------------------------------------------------


@dataclass
class Captured:
    queries: list = field(default_factory=list)


def _make_stub_conn(captured: Captured):
    """Return a context-manager-compatible class that records
    execute() calls and returns a row that matches _TOOL_COLS."""

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query: str, params: tuple) -> None:
            captured.queries.append((query, params))

        def fetchone(self):
            return {
                "id": "tool-existing-1",
                "user_id": "user-test-001",
                "agent_id": "__shared__",
                "name": "google_cal",
                "description": "Test desc",
                "webhook_url": "https://revomedia.app.n8n.cloud/webhook/agendar-cita-dinamica",
                "filler_phrase": "Let me check the system...",
                "parameters": json.dumps(VALID_PAYLOAD["parameters"]),
                "kind": "webhook",
                "destination": None,
                "assignments": json.dumps([]),
                "function_name": "google_cal",
                "test_data_model": "gpt-4o-mini",
                "last_tested_at": None,
                "last_test_result": None,
                "last_test_error": None,
                "last_test_error_at": None,
                "last_invoked_at": None,
                "last_invocation_status": None,
                "last_invocation_error": None,
                "last_invocation_error_at": None,
                "invocation_count": 0,
                "created_at": None,
                "updated_at": None,
            }

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self):
            return _Cursor()

    return _Conn


def _force_postgres_branch(monkeypatch: pytest.MonkeyPatch, captured: Captured):
    """Patch db_tools so update_tool runs the Postgres SET branch
    without needing a real connection or DATABASE_URL."""
    from STT_server import db_tools

    monkeypatch.setattr(db_tools, "is_postgres", lambda: True)
    monkeypatch.setattr(db_tools, "get_conn", lambda: _make_stub_conn(captured)())


def _build_typical_payload():
    from STT_server.routes import api as api_mod

    body = type("Body", (), {
        "name": "google_cal",
        "description": "Test desc (edited)",
        "webhook_url": "https://revomedia.app.n8n.cloud/webhook/agendar-cita-dinamica",
        "filler_phrase": "Let me check the system...",
        "parameters": VALID_PAYLOAD["parameters"],
        "kind": "webhook",
        "destination": None,
        "test_data_model": "gpt-4o-mini",
    })()
    return api_mod._build_tool_payload("__shared__", body)


def _seed_tool_row(tmp_path: pathlib.Path) -> pathlib.Path:
    from STT_server import db_tools

    fp = tmp_path / "agent_tools.json"
    seed = [{
        "id": "tool-existing-1",
        "user_id": "user-test-001",
        "agent_id": "__shared__",
        "name": "google_cal",
        "description": "Test desc",
        "webhook_url": "https://revomedia.app.n8n.cloud/webhook/agendar-cita-dinamica",
        "filler_phrase": "Let me check the system...",
        "parameters": VALID_PAYLOAD["parameters"],
        "kind": "webhook",
        "destination": None,
        "assignments": [],
        "function_name": "google_cal",
        "test_data_model": "gpt-4o-mini",
        "last_tested_at": None,
        "last_test_result": None,
        "last_invoked_at": None,
        "last_invocation_status": None,
        "invocation_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }]
    fp.write_text(json.dumps(seed), encoding="utf-8")
    db_tools._AGENT_TOOLS_FILE = fp
    return fp


def _set_clause_from(query: str) -> tuple[list[str], list[str]]:
    """Return (assignments, cols) parsed from the SET clause between
    the `SET` keyword and the `WHERE` keyword of an UPDATE."""
    assert query.startswith("UPDATE agent_tools SET"), query
    set_part = query.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
    assignments = [a.strip() for a in set_part.split(",")]
    cols = [a.split(" = ", 1)[0].strip() for a in assignments]
    return assignments, cols


# --- Tests -------------------------------------------------------------------


def test_update_set_clause_has_no_duplicate_updated_at(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full PUT body from the FE must yield a SET clause with
    exactly ONE `updated_at` assignment (the trailing NOW()).

    Regression for the operator's 500 after my push:
    psycopg2.errors.SyntaxError: multiple assignments to same
    column "updated_at".
    """
    from STT_server.db_tools import update_tool

    captured = Captured()
    _seed_tool_row(tmp_path)
    _force_postgres_branch(monkeypatch, captured)

    payload = _build_typical_payload()
    update_tool("tool-existing-1", "user-test-001", payload)

    assert captured.queries, "expected update_tool to execute at least one query"
    query = captured.queries[-1][0]
    assignments, cols = _set_clause_from(query)

    # Each column must appear exactly once.
    counts: dict[str, int] = {}
    for c in cols:
        counts[c] = counts.get(c, 0) + 1
    dups = [c for c, n in counts.items() if n > 1]
    assert not dups, (
        f"duplicate assignments in SET: {dups}\nfull clause: {set_part}\nquery: {query}"
        for set_part in [", ".join(assignments)]
    ) if False else not dups

    # updated_at must be present once (the trailing NOW()).
    assert cols.count("updated_at") == 1, (
        f"updated_at must appear exactly once; cols={cols}"
    )

    # id, user_id, created_at are write-once on INSERT and must
    # never reach the SET clause — otherwise we'd rename the PK
    # or clobber the original creation timestamp.
    for forbidden in ("id", "user_id", "created_at"):
        assert forbidden not in cols, (
            f"{forbidden} should not be in SET (DB-managed); got cols={cols}"
        )

    # test_data_model must be present and scalar.
    test_assignments = [a for a in assignments if a.startswith("test_data_model")]
    assert test_assignments, f"test_data_model missing; cols={cols}"
    assert "::jsonb" not in test_assignments[0], (
        f"test_data_model is TEXT, not JSONB; got {test_assignments[0]}"
    )


def test_update_set_clause_skips_none_observability(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """last_* observability fields are None on the freshly-built
    AgentTool row and must be skipped (continue), not serialised
    as `last_* = NULL`."""
    from STT_server.db_tools import update_tool

    captured = Captured()
    _seed_tool_row(tmp_path)
    _force_postgres_branch(monkeypatch, captured)

    payload = _build_typical_payload()
    update_tool("tool-existing-1", "user-test-001", payload)

    assert captured.queries
    _, cols = _set_clause_from(captured.queries[-1][0])
    for col in (
        "last_tested_at", "last_test_result",
        "last_invoked_at", "last_invocation_status",
    ):
        assert col not in cols, f"{col} should be skipped (None); got cols={cols}"


def test_update_set_clause_handles_empty_payload(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty payload is not even worth a SET — update_tool
    must short-circuit and re-read instead of producing
    `UPDATE ... SET updated_at = NOW() WHERE ...` with no
    preceding payloads (which Postgres accepts but is wasted I/O).
    """
    from STT_server.db_tools import update_tool

    captured = Captured()
    _seed_tool_row(tmp_path)
    _force_postgres_branch(monkeypatch, captured)

    out = update_tool("tool-existing-1", "user-test-001", {})
    assert out is not None, "expected update_tool to short-circuit and re-read"
    # Only the SELECT from get_tool ran, not an UPDATE.
    assert not any(q.startswith("UPDATE") for q, _ in captured.queries), (
        f"unexpected UPDATE on empty payload: {captured.queries}"
    )


def test_json_path_filter_skips_db_managed_fields(
    tmp_path: pathlib.Path
) -> None:
    """The JSON-file fallback in update_tool should also skip
    the DB-managed fields. Otherwise `created_at` would get
    rewritten by the freshly-built AgentTool.to_dict() value
    and we'd lose the original creation timestamp."""
    from STT_server.db_tools import update_tool

    fp = _seed_tool_row(tmp_path)

    payload = _build_typical_payload()
    update_tool("tool-existing-1", "user-test-001", payload)

    after = json.loads(fp.read_text(encoding="utf-8"))[0]
    # created_at must stay as the seed value.
    assert after["created_at"] == "2026-01-01T00:00:00Z", (
        f"created_at was rewritten; got {after['created_at']!r}"
    )
    # id stays put (no PK rename on the JSON path either).
    assert after["id"] == "tool-existing-1"
