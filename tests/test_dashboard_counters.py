"""Regression tests for the dashboard / agents counters.

The 2026-09-03 release rewrote /dashboard/stats and /agents so the
front-end reads live counters instead of placeholder strings. The
tests below pin the contract:

  * /dashboard/stats returns live numbers from aggregate_usage() +
    db_call_sessions.count_open_sessions().
  * /agents stamps tools_count / integrations_count / minutes_usage /
    calls on every row.
  * avg_qa_score is gone (replaced by usage).
  * list_open_sessions() still flips rows to closed (recovery path
    is unchanged); count_open_sessions() is read-only.
"""
from __future__ import annotations

import importlib

import pytest


def _import_api():
    api = importlib.import_module("STT_server.routes.api")
    return importlib.reload(api)


def test_dashboard_stats_drops_avg_qa_and_adds_usage(monkeypatch):
    api = _import_api()

    fake_usage = {
        "totals": {
            "calls": 12,
            "duration_seconds": 1860.0,
            "platform_duration_seconds": 600.0,
            "own_duration_seconds": 1260.0,
            "cost_usd": 1.23,
        },
        "per_agent": [
            {"agent_id": "agent-a", "calls": 4, "duration_seconds": 600.0},
        ],
    }

    class FakeAggregate:
        def __call__(self, user_id):
            assert user_id == "user-1"
            return fake_usage

    class FakeTools:
        @staticmethod
        def list_tools(user_id):
            return [
                {"id": "t1", "agent_id": "agent-a", "webhook_url": "https://x"},
                {"id": "t2", "agent_id": "__shared__", "webhook_url": "https://y"},
                {"id": "t3", "agent_id": "agent-a", "kind": "credentials"},  # filtered
            ]

    class FakeIntegrations:
        @staticmethod
        def list_integrations(user_id, agent_id=None):
            return [
                {"id": "i1", "connection_status": "connected"},
                {"id": "i2", "connection_status": "failed"},
                {"id": "i3", "connection_status": "connected"},
            ]

    class FakeCallSessions:
        @staticmethod
        def count_open_sessions(*, user_id=None):
            return 3

        @staticmethod
        def list_active_for_user(*, user_id=None, limit=50):
            return [
                {"session_key": "s1", "call_sid": "CA1", "agent_id": "agent-a", "started_at": "2026-09-03T22:00:00Z"},
                {"session_key": "s2", "call_sid": "CA2", "agent_id": "agent-b", "started_at": "2026-09-03T22:00:01Z"},
            ]

    monkeypatch.setattr(api._load, "__defaults__", (), raising=False)
    monkeypatch.setattr(api, "_load", lambda path, default: (
        [{"id": "agent-a", "user_id": "user-1", "status": "Active", "name": "Eduardo"}]
        if "agents" in path
        else [{"id": "num-1", "user_id": "user-1"}]
    ))
    monkeypatch.setattr("STT_server.services.usage_store.aggregate_usage", FakeAggregate())
    monkeypatch.setattr("STT_server.db_tools.list_tools", FakeTools.list_tools)
    monkeypatch.setattr("STT_server.db_integrations.list_integrations", FakeIntegrations.list_integrations)
    monkeypatch.setattr("STT_server.db_call_sessions.count_open_sessions", FakeCallSessions.count_open_sessions)
    monkeypatch.setattr("STT_server.db_call_sessions.list_active_for_user", FakeCallSessions.list_active_for_user)

    out = api.dashboard_stats(auth={"user_id": "user-1"})

    assert out["active_agents"] == 1
    assert out["calls_today"] == 12
    assert out["live_calls"] == 3
    assert out["live_calls_detail"][0]["agent_id"] == "agent-a"
    assert out["tools_count"] == 2  # provider-credential row filtered out
    assert out["integrations_count"] == 2  # only 'connected'
    assert "avg_qa_score" not in out
    usage = out["usage"]
    assert usage["calls"] == 12
    assert usage["total_minutes"] == 31.0  # 1860 / 60
    assert usage["total_cost_usd"] == 1.23
    assert usage["platform_minutes"] == 10.0
    assert usage["own_minutes"] == 21.0
    assert usage["usage_label"] == "31.0 min"
    assert out["numbers_count"] == 1
    assert out["minutes_by_agent"]["agent-a"] == 10.0  # 600 / 60


def test_list_agents_stamps_counters(monkeypatch):
    api = _import_api()

    class FakeListAgents:
        @staticmethod
        def __call__(user_id):
            return [
                {"id": "agent-a", "user_id": user_id, "name": "Eduardo"},
                {"id": "agent-b", "user_id": user_id, "name": "Mateo"},
            ]

    class FakeTools:
        @staticmethod
        def list_tools(user_id, agent_id=None):
            if agent_id == "agent-a":
                return [
                    {"id": "t1", "webhook_url": "https://x"},
                    {"id": "t2", "webhook_url": "https://y"},
                ]
            return []

    class FakeIntegrations:
        @staticmethod
        def list_integrations(user_id, agent_id=None):
            if agent_id == "agent-b":
                return [{"id": "i1", "connection_status": "connected"}]
            return []

    class FakeCallSessions:
        @staticmethod
        def list_active_for_user(*, user_id=None, limit=200):
            return [
                {"session_key": "s1", "call_sid": "CA1", "agent_id": "agent-a", "started_at": "2026-09-03T22:00:00Z"},
                {"session_key": "s2", "call_sid": "CA2", "agent_id": "agent-a", "started_at": "2026-09-03T22:00:01Z"},
                {"session_key": "s3", "call_sid": "CA3", "agent_id": "agent-b", "started_at": "2026-09-03T22:00:02Z"},
            ]

    usage = {
        "totals": {"calls": 7, "duration_seconds": 0, "cost_usd": 0},
        "per_agent": [
            {"agent_id": "agent-a", "calls": 5, "duration_seconds": 600.0},
            {"agent_id": "agent-b", "calls": 2, "duration_seconds": 60.0},
        ],
    }

    monkeypatch.setattr(api, "db_list_agents", FakeListAgents.__call__)
    monkeypatch.setattr("STT_server.services.usage_store.aggregate_usage", lambda user_id: usage)
    monkeypatch.setattr("STT_server.db_tools.list_tools", FakeTools.list_tools)
    monkeypatch.setattr("STT_server.db_integrations.list_integrations", FakeIntegrations.list_integrations)
    monkeypatch.setattr("STT_server.db_call_sessions.list_active_for_user", FakeCallSessions.list_active_for_user)

    rows = api.list_agents(auth={"user_id": "user-1"})

    assert rows[0]["calls"] == 5
    assert rows[0]["minutes_usage"] == 10.0
    assert rows[0]["tools_count"] == 2
    assert rows[0]["integrations_count"] == 0
    assert rows[0]["active_calls"] == 2
    assert rows[1]["calls"] == 2
    assert rows[1]["minutes_usage"] == 1.0
    assert rows[1]["tools_count"] == 0
    assert rows[1]["integrations_count"] == 1
    assert rows[1]["active_calls"] == 1


def test_count_open_sessions_does_not_mutate(monkeypatch):
    """list_open_sessions flips rows to closed; count_open_sessions is
    read-only so polling dashboards don't accidentally tear down the
    call_sessions ledger.
    """
    db = importlib.import_module("STT_server.db_call_sessions")
    importlib.reload(db)

    writes: list[list[dict]] = []
    sample_rows = [
        {"session_key": "a", "user_id": "u1", "closed": False},
        {"session_key": "b", "user_id": "u1", "closed": True},
        {"session_key": "c", "user_id": "u2", "closed": False},
    ]

    monkeypatch.setattr(db, "_read_json", lambda: [dict(r) for r in sample_rows])
    monkeypatch.setattr(db, "_write_json", lambda rows: writes.append(rows))

    # monkeypatch the JSON-backend shortcut by forcing is_postgres() False
    monkeypatch.setattr(db, "is_postgres", lambda: False)

    total = db.count_open_sessions()
    own = db.count_open_sessions(user_id="u1")
    other = db.count_open_sessions(user_id="u2")
    detail = db.list_active_for_user(user_id="u1")

    assert total == 2
    assert own == 1
    assert other == 1
    assert writes == [], "count + list must never mutate the ledger"
    assert [r.get("session_key") for r in detail] == ["a"]


def test_agents_api_no_longer_round_trips_tool_counts_per_card():
    """Sanity guard: the FE import surface for Agents.jsx must not
    import agentToolsApi anymore. We don't want the legacy
    per-card /agents/{id}/tools round-trip to come back.
    """
    import pathlib
    here = pathlib.Path(__file__).resolve().parent.parent
    candidates = [
        here.parent / "AgentsAi_Frontend" / "src" / "pages" / "Agents.jsx",
        here / "Agents.jsx",
    ]
    src_path = next((p for p in candidates if p.exists()), None)
    if src_path is None:
        pytest.skip("Agents.jsx not co-located with this test run")
    src = src_path.read_text(encoding="utf-8")
    assert "agentToolsApi" not in src, (
        "Agents.jsx should read tools_count straight from /agents"
    )
    assert "minutes_usage" in src
    assert "Minutes Usage" in src
    assert "agent.perf" not in src  # no more agent.perf references
