"""Route-level validation for stt_latency_mode (B8 / B9 / B10 end to end).

The unit tests cover openai_stt_models.validate; these go through the
actual HTTP handlers, because the thing that must never happen is an
invalid pair reaching the agent row and then failing OpenAI's
session.update at call time.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from STT_server.routes import api


@pytest.fixture
def pg_backend(monkeypatch):
    """Force the Postgres branch of update_agent.

    The route short-circuits into a JSON-file loop when is_postgres() is
    False, which is the case in CI (no DATABASE_URL).     Both the JSON loop
    and db_update_agent need covering, so the tests below patch
    db_update_agent directly for the SQL path and patch the JSON loader for
    the file path, rather than depending on a live database.

    Yields the monkeypatch fixture so each test can keep patching with the
    same object.
    """
    monkeypatch.setattr(api, "is_postgres", lambda: True)
    return monkeypatch


@pytest.fixture
def json_backend(monkeypatch):
    """Force the JSON-file branch (CI default)."""
    monkeypatch.setattr(api, "is_postgres", lambda: False)
    return monkeypatch


def _create(**over):
    base = {
        "name": "Latency Dial Agent",
        "stt_provider": "openai",
        "stt_model": "gpt-live-transcribe",
        "stt_latency_mode": "low",
    }
    base.update(over)
    return api.AgentCreate(**base)


AUTH = {"user_id": "user-admin-001"}


def test_valid_pair_persists(monkeypatch):
    """The happy path: the normalized value reaches the payload."""
    seen = {}

    def fake_create(user_id, payload):
        seen.update(payload)
        return {"id": "agent-x", **payload}

    monkeypatch.setattr(api, "db_create_agent", fake_create)
    out = api.create_agent(_create(stt_latency_mode="xhigh"), auth=AUTH)
    assert out["stt_latency_mode"] == "xhigh"


def test_absent_mode_persists_null(monkeypatch):
    """B10 — an agent saved without the field stores NULL, and the adapter
    substitutes the platform default at call time. Nothing is invented at
    write time, so a legacy row and a deliberately-default row are the
    same thing on disk."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    api.create_agent(
        api.AgentCreate(name="A", stt_provider="openai",
                        stt_model="gpt-live-transcribe", stt_latency_mode=None),
        auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_invalid_mode_is_400(monkeypatch):
    """B8 — a bogus level is rejected with a message naming the legal ones."""
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda *a, **k: pytest.fail("must not reach the store"),
    )
    with pytest.raises(HTTPException) as exc:
        api.create_agent(_create(stt_latency_mode="turbo"), auth=AUTH)
    assert exc.value.status_code == 400
    assert "turbo" in str(exc.value.detail)
    assert "minimal" in str(exc.value.detail)


def test_mode_illegal_for_that_specific_model_is_400(monkeypatch):
    """The interesting case: a level that is legal for the OTHER model but
    not this one. A global enum check would let this through."""
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda *a, **k: pytest.fail("must not reach the store"),
    )
    # gpt-live-transcribe accepts all five; a future model with fewer
    # levels must still 400. Assert the mechanism with the real models:
    # gpt-transcribe accepts none at all.
    with pytest.raises(HTTPException) as exc:
        api.create_agent(
            _create(stt_model="gpt-transcribe", stt_latency_mode="low"),
            auth=AUTH,
        )
    assert exc.value.status_code == 400
    assert "no latency" in str(exc.value.detail)


def test_committed_turn_model_rejects_an_explicit_mode(monkeypatch):
    """gpt-transcribe stores NULL, and an explicitly-sent mode is a 400
    rather than a silent drop — otherwise a client that always posts the
    field looks like it saved when it did not."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    with pytest.raises(HTTPException) as exc:
        api.create_agent(
            _create(stt_model="gpt-transcribe", stt_latency_mode="high"),
            auth=AUTH,
        )
    assert exc.value.status_code == 400
    assert "committed-turn" in str(exc.value.detail)
    assert "stt_latency_mode" not in seen

    # ...and omitting the field is a clean save with NULL.
    api.create_agent(
        api.AgentCreate(name="A", stt_provider="openai",
                        stt_model="gpt-transcribe", stt_latency_mode=None),
        auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_non_openai_provider_never_stores_a_mode(monkeypatch):
    """The dial is an OpenAI concept. Deepgram and Inworld rows must not
    carry one, whatever the client sends."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    api.create_agent(
        api.AgentCreate(name="A", stt_provider="deepgram", stt_model="nova-3",
                        stt_latency_mode="low"),
        auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_unknown_stt_model_stores_null_rather_than_400(monkeypatch):
    """An stt_model outside the OpenAI lineup (a legacy row, or a model
    from another provider) must not be blocked on the dial."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    api.create_agent(
        _create(stt_model="whisper-1", stt_latency_mode="low"), auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_update_switching_to_dial_less_model_clears_the_stored_mode(pg_backend, monkeypatch):
    """The debt this fixes: db_update_agent treats None as "don't touch", so
    without an explicit clear the old mode sat on the row forever. Harmless
    to the adapter (it re-derives per call) but a stale column, and it
    resurrects the moment the operator switches back to a model with a dial.
    """
    pg_backend.setattr(api, "_agent_stt_provider", lambda a, u: "openai")
    monkeypatch.setattr(api, "_agent_stt_model", lambda a, u: "gpt-live-transcribe")
    captured = {}

    def fake_update(agent_id, user_id, payload, clear_fields=None):
        captured["payload"] = dict(payload)
        captured["clear"] = set(clear_fields or ())
        return {"id": agent_id, **payload}

    monkeypatch.setattr(api, "db_update_agent", fake_update)
    api.update_agent(
        "agent-1",
        api.AgentUpdate(stt_provider="openai", stt_model="gpt-transcribe"),
        auth=AUTH,
    )
    assert "stt_latency_mode" not in captured["payload"], (
        "a dial-less model must not carry a latency value in the SET clause"
    )
    assert "stt_latency_mode" in captured["clear"]


def test_update_switching_away_from_openai_clears_the_mode(pg_backend):
    """Leaving OpenAI entirely also has to clear it."""
    captured = {}
    pg_backend.setattr(
        api, "db_update_agent",
        lambda a, u, p, clear_fields=None: (
            captured.update(payload=dict(p), clear=set(clear_fields or ())),
            {"id": a},
        )[1],
    )
    api.update_agent(
        "agent-1",
        api.AgentUpdate(stt_provider="deepgram", stt_model="nova-3"),
        auth=AUTH,
    )
    assert "stt_latency_mode" in captured["clear"]


def test_update_with_a_legal_mode_persists_it_and_clears_nothing(pg_backend, monkeypatch):
    """The ordinary case: a value is SET and nothing is cleared."""
    pg_backend.setattr(api, "_agent_stt_provider", lambda a, u: "openai")
    monkeypatch.setattr(api, "_agent_stt_model", lambda a, u: "gpt-live-transcribe")
    captured = {}
    monkeypatch.setattr(
        api, "db_update_agent",
        lambda a, u, p, clear_fields=None: (
            captured.update(payload=dict(p), clear=set(clear_fields or ())),
            {"id": a},
        )[1],
    )
    api.update_agent(
        "agent-1",
        api.AgentUpdate(stt_provider="openai", stt_model="gpt-live-transcribe",
                        stt_latency_mode="high"),
        auth=AUTH,
    )
    assert captured["payload"]["stt_latency_mode"] == "high"
    assert captured["clear"] == set()


def test_update_untouched_by_latency_does_not_clear(pg_backend):
    """A PUT that never mentions STT must leave the column alone — this is
    the regression a too-eager clear would introduce."""
    captured = {}
    pg_backend.setattr(
        api, "db_update_agent",
        lambda a, u, p, clear_fields=None: (
            captured.update(payload=dict(p), clear=set(clear_fields or ())),
            {"id": a},
        )[1],
    )
    api.update_agent("agent-1", api.AgentUpdate(name="Renamed"), auth=AUTH)
    assert "stt_latency_mode" not in captured["clear"]
    assert "stt_latency_mode" not in captured["payload"]


def test_db_update_agent_supports_explicit_null(monkeypatch):
    """The db layer itself: clear_fields emits `col = NULL`, and a column
    in both payload and clear_fields ends up NULL (clear wins)."""
    import STT_server.db_agents as dba

    sql = {}
    monkeypatch.setattr(dba, "is_postgres", lambda: True)

    class _Cur:
        def __init__(self): self.sql = ""
        def execute(self, s, p): sql["stmt"] = s; sql["params"] = p
        def fetchone(self): return None
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Conn:
        def cursor(self): return _Cur()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(dba, "get_conn", lambda: _Conn())
    dba.update_agent(
        "a1", "u1", {"stt_latency_mode": "low"}, clear_fields={"stt_latency_mode"},
    )
    assert "stt_latency_mode = NULL" in sql["stmt"]
    # A clear must not ALSO bind a parameter for the same column, or the
    # column gets assigned twice and Postgres rejects the statement.
    # Compare on split clauses so the substring "stt_latency_mode" matching
    # inside a longer clause cannot produce a false pass either way.
    # Only the SET list matters; the trailing RETURNING column list also
    # contains the bare name, so cut the statement at RETURNING first.
    set_part = sql["stmt"].split(" SET ", 1)[1].split(" RETURNING ", 1)[0]
    hits = [c.strip() for c in set_part.split(",") if "stt_latency_mode" in c]
    assert hits == ["stt_latency_mode = NULL"], hits


def test_db_update_agent_ignores_unknown_clear_columns(monkeypatch):
    """clear_fields is a hint from the route, not trusted blindly: a typo
    must not reach the SQL."""
    import STT_server.db_agents as dba

    sql = {}
    monkeypatch.setattr(dba, "is_postgres", lambda: True)

    class _Cur:
        def execute(self, s, p): sql["stmt"] = s
        def fetchone(self): return None
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _Conn:
        def cursor(self): return _Cur()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(dba, "get_conn", lambda: _Conn())
    dba.update_agent("a1", "u1", {"name": "X"}, clear_fields={"not_a_column"})
    assert "not_a_column" not in sql["stmt"]


def test_json_backend_also_clears(monkeypatch):
    """The JSON-file branch writes nulls directly instead of going through
    db_update_agent, so it needs its own clear assertion — otherwise the
    CI-default backend would silently keep the stale value."""
    monkeypatch.setattr(api, "is_postgres", lambda: False)
    row = {"id": "agent-1", "user_id": AUTH["user_id"], "stt_latency_mode": "xhigh"}
    monkeypatch.setattr(api, "_load", lambda *a, **k: [dict(row)])
    saved = {}
    monkeypatch.setattr(api, "_save", lambda *a, **k: saved.update(agents=a[1]))

    api.update_agent(
        "agent-1",
        api.AgentUpdate(stt_provider="openai", stt_model="gpt-transcribe"),
        auth=AUTH,
    )
    assert saved["agents"][0]["stt_latency_mode"] is None


def test_update_validates_against_the_stored_model(pg_backend, monkeypatch):
    """A partial update that only carries the mode is checked against the
    model the agent actually has, not against a missing field."""
    pg_backend.setattr(api, "_agent_stt_provider", lambda a, u: "openai")
    monkeypatch.setattr(api, "_agent_stt_model", lambda a, u: "gpt-live-transcribe")
    monkeypatch.setattr(
        api, "db_update_agent",
        lambda *a, **k: pytest.fail("must not reach the store"),
    )
    with pytest.raises(HTTPException) as exc:
        api.update_agent(
            "agent-1",
            api.AgentUpdate(stt_latency_mode="nope"),
            auth=AUTH,
        )
    assert exc.value.status_code == 400
    assert "nope" in str(exc.value.detail)
