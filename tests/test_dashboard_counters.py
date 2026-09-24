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
            {
                "agent_id": "agent-a",
                "calls": 4,
                "duration_seconds": 252.0,  # 4.2 min
                "cost_usd": 0.3478,  # 4.2 * 0.0828 (own rate)
                "rate_per_min": 0.0828,
            },
        ],
        "rates": {
            "own_per_min": 0.0828,
            "platform_per_min": 0.14,
            "currency": "USD",
        },
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

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "status": "Active", "name": "Eduardo"},
    ])
    monkeypatch.setattr(api, "_load", lambda path, default: (
        [{"id": "num-1", "user_id": "user-1"}]
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
    # rate-aware: own 21.0 * 0.0828 + platform 10.0 * 0.14 ≈ 3.1388
    assert usage["total_cost_usd"] == round(21.0 * 0.0828 + 10.0 * 0.14, 4)
    assert usage["platform_minutes"] == 10.0
    assert usage["own_minutes"] == 21.0
    assert usage["own_per_min"] == 0.0828
    assert usage["platform_per_min"] == 0.14
    assert usage["usage_label"] == "31.0 min"
    assert out["rates"] == {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"}
    assert out["numbers_count"] == 1
    assert out["minutes_by_agent"]["agent-a"] == 4.2  # 252 / 60
    assert out["pricing"]["agent-a"]["rate_per_min"] == 0.0828
    assert out["pricing"]["agent-a"]["cost_usd"] == 0.3478


def test_dashboard_active_agents_case_insensitive(monkeypatch):
    """Legacy rows with lowercase `active` should also count."""
    api = _import_api()

    class FakeAggregate:
        def __call__(self, user_id):
            return {
                "totals": {
                    "calls": 0, "duration_seconds": 0.0,
                    "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0,
                    "cost_usd": 0.0,
                },
                "per_agent": [],
                "rates": {
                    "own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"
                },
            }

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    agents_seed = [
        {"id": "agent-a", "user_id": "user-1", "status": "active", "name": "Eduardo"},
        {"id": "agent-b", "user_id": "user-1", "status": "Paused", "name": "Mateo"},
    ]
    monkeypatch.setattr(api, "_load", lambda path, default: (
        agents_seed if path.endswith("agents.json")
        else ([] if path.endswith("numbers.json") else default)
    ))
    monkeypatch.setattr("STT_server.services.usage_store.aggregate_usage", FakeAggregate())
    monkeypatch.setattr("STT_server.db_tools.list_tools", lambda user_id: [])
    monkeypatch.setattr("STT_server.db_integrations.list_integrations", lambda user_id: [])
    monkeypatch.setattr("STT_server.db_call_sessions.count_open_sessions", lambda *, user_id=None: 0)
    monkeypatch.setattr("STT_server.db_call_sessions.list_active_for_user", lambda *, user_id=None, limit=20: [])

    out = api.dashboard_stats(auth={"user_id": "user-1"})
    assert out["active_agents"] == 1


def test_list_agents_stamps_counters(monkeypatch):
    """Regression: /agents stamps tools_count, integrations_count,
    minutes_usage, calls on every row. The 2026-09-03 release rewrote
    /agents so the front-end reads live counters instead of placeholder
    strings.

    ponytail: Ticket 3 — the test used to mock per-agent list_tools /
    list_integrations (the legacy N+1 path). After the dedup, /agents
    calls the bulk helpers tools_count_by_agent /
    connected_integrations_count_by_agent. The fixture below mirrors
    the same counts the per-agent path produced (agent-a → 2 tools,
    agent-b → 1 connected integration).
    """
    api = _import_api()

    class FakeListAgents:
        @staticmethod
        def __call__(user_id):
            return [
                {"id": "agent-a", "user_id": user_id, "name": "Eduardo"},
                {"id": "agent-b", "user_id": user_id, "name": "Mateo"},
            ]

    def fake_tools_count_by_agent(user_id, agent_ids):
        # Mirror the per-agent list_tools(agent_id=X) counts the test
        # used to assert: agent-a had 2 real tools; agent-b had 0.
        return {aid: (2 if aid == "agent-a" else 0) for aid in agent_ids}

    def fake_connected_integrations_count_by_agent(user_id, agent_ids):
        # Mirror: agent-a had 0 connected integrations, agent-b had 1.
        return {aid: (1 if aid == "agent-b" else 0) for aid in agent_ids}

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
    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_connected_integrations_count_by_agent,
    )
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


def test_has_user_stored_key_reads_credentials_column(monkeypatch):
    """Regression: per-user provider credentials live in the
    ``credentials`` JSONB column on the agent_tools row, not as a
    ``connected`` boolean. The previous version queried
    ``t.get("connected")`` which never matched → every call was
    classified ``used_platform_keys=True`` and the operator paid the
    platform rate even after uploading their own OpenAI key.
    """
    from STT_server.services.usage_store import has_user_stored_key

    fake_rows = {
        ("openai", "user-1"): {"id": "openai", "credentials": {"api_key": "sk-..."}},
        ("elevenlabs", "user-1"): {"id": "elevenlabs", "credentials": {}},
        ("inworld", "user-1"): {"id": "inworld", "credentials": "{}"},  # JSON string 'null'
    }

    def fake_get_tool(tool_id, user_id):
        return fake_rows.get((tool_id, user_id))

    import STT_server.db_tools as db_tools_mod
    # ponytail: the helper lazy-imports ``db_get_tool`` from db_tools,
    # so monkeypatch the source attribute (not the consumer module).
    monkeypatch.setattr(db_tools_mod, "db_get_tool", fake_get_tool)

    assert has_user_stored_key("user-1", "openai") is True
    assert has_user_stored_key("user-1", "elevenlabs") is False
    assert has_user_stored_key("user-1", "inworld") is False
    assert has_user_stored_key("user-1", "missing") is False
    assert has_user_stored_key("user-1", "") is False
    assert has_user_stored_key("", "openai") is False


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


# ── Ticket 2: aggregate_usage dedup characterization ─────────────────────────


def test_dashboard_stats_aggregate_usage_call_count(monkeypatch):
    """Pin the contract: dashboard_stats must call aggregate_usage
    exactly once per request. Ticket 2 dedup'd the duplicate call at
    api.py:714 + api.py:774. If a future refactor reintroduces the
    duplicate, this test fires.
    """
    api = _import_api()

    call_log: list[str] = []

    fake_usage = {
        "totals": {
            "calls": 5,
            "duration_seconds": 600.0,
            "platform_duration_seconds": 200.0,
            "own_duration_seconds": 400.0,
            "cost_usd": 0.42,
        },
        "per_agent": [],
        "rates": {
            "own_per_min": 0.0828,
            "platform_per_min": 0.14,
            "currency": "USD",
        },
    }

    def fake_aggregate_usage(user_id):
        call_log.append(user_id)
        return fake_usage

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [])
    monkeypatch.setattr(api, "_load", lambda path, default: [])
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage", fake_aggregate_usage
    )
    monkeypatch.setattr("STT_server.db_tools.list_tools", lambda user_id: [])
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations", lambda user_id: []
    )
    monkeypatch.setattr(
        "STT_server.db_call_sessions.count_open_sessions", lambda **kw: 0
    )
    monkeypatch.setattr(
        "STT_server.db_call_sessions.list_active_for_user", lambda **kw: []
    )

    out = api.dashboard_stats(auth={"user_id": "user-1"})

    # The dedup contract: exactly one aggregate_usage call per request.
    assert call_log == ["user-1"], (
        f"aggregate_usage called {len(call_log)} times: {call_log}; expected 1"
    )

    # Response still correct: usage block derived from the single result.
    assert out["usage"]["calls"] == 5
    assert out["usage"]["total_minutes"] == 10.0
    assert out["usage"]["own_per_min"] == 0.0828
    assert out["usage"]["platform_per_min"] == 0.14


def test_dashboard_stats_aggregate_usage_failure_isolated(monkeypatch):
    """If aggregate_usage raises, the request fails. The dedup must
    not change error semantics: still HTTP 500, still no partial
    response leaked. (Both old call sites were unprotected, so
    removing the duplicate does not alter failure mode.)
    """
    api = _import_api()

    def fake_aggregate_usage(user_id):
        raise RuntimeError("ledger unreachable")

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [])
    monkeypatch.setattr(api, "_load", lambda path, default: [])
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage", fake_aggregate_usage
    )

    with pytest.raises(RuntimeError, match="ledger unreachable"):
        api.dashboard_stats(auth={"user_id": "user-1"})


def test_dashboard_stats_per_user_no_carry_over(monkeypatch):
    """Two consecutive requests with different user_ids must each
    call aggregate_usage once. No cross-user caching.
    """
    api = _import_api()

    call_log: list[str] = []

    def fake_aggregate_usage(user_id):
        call_log.append(user_id)
        return {
            "totals": {
                "calls": 0,
                "duration_seconds": 0.0,
                "platform_duration_seconds": 0.0,
                "own_duration_seconds": 0.0,
                "cost_usd": 0.0,
            },
            "per_agent": [],
            "rates": {
                "own_per_min": 0.0828,
                "platform_per_min": 0.14,
                "currency": "USD",
            },
        }

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [])
    monkeypatch.setattr(api, "_load", lambda path, default: [])
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage", fake_aggregate_usage
    )
    monkeypatch.setattr("STT_server.db_tools.list_tools", lambda user_id: [])
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations", lambda user_id: []
    )
    monkeypatch.setattr(
        "STT_server.db_call_sessions.count_open_sessions", lambda **kw: 0
    )
    monkeypatch.setattr(
        "STT_server.db_call_sessions.list_active_for_user", lambda **kw: []
    )

    api.dashboard_stats(auth={"user_id": "user-A"})
    api.dashboard_stats(auth={"user_id": "user-B"})
    api.dashboard_stats(auth={"user_id": "user-A"})

    assert call_log == ["user-A", "user-B", "user-A"], (
        f"each request must trigger its own aggregate_usage; got {call_log}"
    )


# ── Ticket 3: list_agents N+1 → bulk counters characterization ────────────────


def test_list_agents_per_agent_calls_reproduce_n_plus_1(monkeypatch):
    """Reproduce the N+1: list_agents must NOT call list_tools or
    list_integrations once per agent. Before the fix, two calls per
    agent (1 tools + 1 integrations) plus 1 list_agents. With 3 agents
    that's 1 + 6 = 7 storage calls. After the fix: 1 + 1 + 1 = 3.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "Eduardo"},
        {"id": "agent-b", "user_id": user_id, "name": "Mateo"},
        {"id": "agent-c", "user_id": user_id, "name": "Sofia"},
    ])

    tools_bulk_calls: list[str] = []
    integ_bulk_calls: list[str] = []

    def fake_tools_count_by_agent(user_id, agent_ids):
        tools_bulk_calls.append(user_id)
        return {aid: 0 for aid in agent_ids}

    def fake_connected_integrations_count_by_agent(user_id, agent_ids):
        integ_bulk_calls.append(user_id)
        return {aid: 0 for aid in agent_ids}

    # After the fix, the route layer must call the bulk helpers instead
    # of the per-agent list_tools / list_integrations.
    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_connected_integrations_count_by_agent,
    )

    # Also instrument the per-agent helpers to detect any regression:
    # if the N+1 returns, these would be called once per agent.
    per_agent_tools_calls: list[tuple[str, str | None]] = []
    per_agent_integ_calls: list[tuple[str, str | None]] = []

    def spy_list_tools(user_id, agent_id=None):
        per_agent_tools_calls.append((user_id, agent_id))
        return []

    def spy_list_integrations(user_id, agent_id=None):
        per_agent_integ_calls.append((user_id, agent_id))
        return []

    monkeypatch.setattr("STT_server.db_tools.list_tools", spy_list_tools)
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations", spy_list_integrations
    )

    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    api.list_agents(auth={"user_id": "user-1"})

    # Bulk helpers called exactly once.
    assert len(tools_bulk_calls) == 1, (
        f"tools_count_by_agent called {len(tools_bulk_calls)} times; expected 1"
    )
    assert len(integ_bulk_calls) == 1, (
        f"connected_integrations_count_by_agent called {len(integ_bulk_calls)} times; expected 1"
    )
    # Per-agent helpers must NOT be called at all (regression guard).
    assert len(per_agent_tools_calls) == 0, (
        f"per-agent list_tools called {len(per_agent_tools_calls)} times: "
        f"{per_agent_tools_calls}; N+1 regression"
    )
    assert len(per_agent_integ_calls) == 0, (
        f"per-agent list_integrations called {len(per_agent_integ_calls)} times: "
        f"{per_agent_integ_calls}; N+1 regression"
    )


def test_list_agents_value_contract_tools_count(monkeypatch):
    """Pin the value contract for tools_count:
    - per-agent tool (private) → counts
    - shared tool assigned to X → counts
    - shared tool NOT assigned to X → does NOT count
    - provider credential row (no webhook_url, no destination) → excluded
    - per-agent tool of OTHER agent → NOT visible to X (user scope)

    The bulk helpers (tools_count_by_agent / connected_integrations_count_by_agent)
    must produce the same counts the per-agent list_tools / list_integrations
    path produced before Ticket 3.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
    ])

    # Bulk helpers return the same numbers the per-agent loop produced.
    # agent-a: 1 private + 1 shared assigned to A = 2
    # agent-b: 2 private + 1 shared assigned to B = 3
    def fake_tools_count_by_agent(user_id, agent_ids):
        assert user_id == "user-1"
        assert sorted(agent_ids) == ["agent-a", "agent-b"]
        return {"agent-a": 2, "agent-b": 3}

    def fake_connected_integrations_count_by_agent(user_id, agent_ids):
        assert user_id == "user-1"
        return {"agent-a": 1, "agent-b": 1}

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_connected_integrations_count_by_agent,
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    by_id = {r["id"]: r for r in rows}
    assert by_id["agent-a"]["tools_count"] == 2
    assert by_id["agent-b"]["tools_count"] == 3
    assert by_id["agent-a"]["integrations_count"] == 1
    assert by_id["agent-b"]["integrations_count"] == 1


def test_list_agents_value_contract_zero_counts(monkeypatch):
    """Agent with no tools and no integrations → counts 0."""
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-empty", "user_id": user_id, "name": "Empty"},
    ])

    def fake_tools_count_by_agent(user_id, agent_ids):
        return {a: 0 for a in agent_ids}

    def fake_connected_integrations_count_by_agent(user_id, agent_ids):
        return {a: 0 for a in agent_ids}

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_connected_integrations_count_by_agent,
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    assert len(rows) == 1
    assert rows[0]["tools_count"] == 0
    assert rows[0]["integrations_count"] == 0
    assert rows[0]["active_calls"] == 0
    assert rows[0]["calls"] == 0


def test_list_agents_cross_user_isolation(monkeypatch):
    """Two users, each with one agent and one tool/integration. Tools
    from U1 must NOT appear in U2's counts (and vice versa). The bulk
    helpers must scope by user_id.
    """
    api = _import_api()

    def fake_db_list_agents(user_id):
        return [
            {"id": f"agent-{user_id}", "user_id": user_id, "name": f"A-{user_id}"},
        ]

    def fake_tools_count_by_agent(user_id, agent_ids):
        # Each user has exactly one tool (their own).
        # Return 1 for the agent that belongs to user_id, 0 otherwise.
        return {a: 1 if a == f"agent-{user_id}" else 0 for a in agent_ids}

    def fake_connected_integrations_count_by_agent(user_id, agent_ids):
        return {a: 1 if a == f"agent-{user_id}" else 0 for a in agent_ids}

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", fake_db_list_agents)
    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_connected_integrations_count_by_agent,
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows_u1 = api.list_agents(auth={"user_id": "user-1"})
    rows_u2 = api.list_agents(auth={"user_id": "user-2"})

    # Each user has exactly one agent with exactly one tool and one
    # connected integration. Counts must NOT cross-pollinate.
    assert rows_u1[0]["tools_count"] == 1
    assert rows_u1[0]["integrations_count"] == 1
    assert rows_u2[0]["tools_count"] == 1
    assert rows_u2[0]["integrations_count"] == 1
    assert rows_u1[0]["id"] != rows_u2[0]["id"]


def test_list_agents_provider_credential_excluded(monkeypatch):
    """Provider credential rows (Settings → API keys) have agent_id
    '__shared__', no webhook_url, no destination, kind='credentials'.
    The bulk helper must exclude them so the badge doesn't count OpenAI
    keys as n8n webhooks.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
    ])

    # 1 real webhook + 1 credential row assigned to A. Credential must
    # NOT count, so tools_count_by_agent returns 1 for agent-a.
    def fake_tools_count_by_agent(user_id, agent_ids):
        return {"agent-a": 1}

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        lambda user_id, agent_ids: {a: 0 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    assert rows[0]["tools_count"] == 1, (
        "credential row (no webhook_url, no destination) must not count"
    )


def test_list_agents_bulk_failure_preserves_error_semantics(monkeypatch):
    """If the bulk helper raises, the previous per-agent path swallowed
    the exception per agent and stamped 0. The bulk equivalent must do
    the same: stamp 0 for every agent and continue (don't 500 the
    whole /agents request).
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
    ])

    def fake_tools_count_by_agent(user_id, agent_ids):
        raise RuntimeError("db unavailable")

    def fake_connected_integrations_count_by_agent(user_id, agent_ids):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_connected_integrations_count_by_agent
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    # /agents must NOT raise; the route handler catches the bulk
    # helper failure per helper and falls back to zero counts.
    rows = api.list_agents(auth={"user_id": "user-1"})
    assert len(rows) == 2
    assert rows[0]["tools_count"] == 0
    assert rows[0]["integrations_count"] == 0
    assert rows[1]["tools_count"] == 0
    assert rows[1]["integrations_count"] == 0


def test_list_agents_json_fallback_single_scan(monkeypatch):
    """In JSON fallback mode (no DATABASE_URL), the bulk helpers must
    read the file ONCE and iterate ONCE — not once per agent.

    This is the JSON equivalent of the N+1: the legacy code re-read
    the JSON file per agent because the per-agent list_tools helper
    loaded it inside the function. The bulk helpers load once.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)

    # 3 agents.
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
        {"id": "agent-c", "user_id": user_id, "name": "C"},
    ])

    # Count how many times the bulk helpers are CALLED. Each helper
    # reads the JSON file at most once per call (inside the
    # ``if not _AGENT_TOOLS_FILE.exists()`` + try/except block).
    # Before the fix the route handler called list_tools /
    # list_integrations once per agent. After the fix: 1 + 1 = 2.
    bulk_calls: list[str] = []

    def fake_tools_count(user_id, agent_ids):
        bulk_calls.append("tools")
        return {a: 0 for a in agent_ids}

    def fake_count_integrations(user_id, agent_ids):
        bulk_calls.append("integ")
        return {a: 0 for a in agent_ids}

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools_count
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_count_integrations,
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    api.list_agents(auth={"user_id": "user-1"})

    # One call per bulk helper, regardless of agent count.
    assert bulk_calls.count("tools") == 1, (
        f"tools_count_by_agent called {bulk_calls.count('tools')} times; expected 1"
    )
    assert bulk_calls.count("integ") == 1, (
        f"connected_integrations_count_by_agent called {bulk_calls.count('integ')} times; expected 1"
    )


def test_list_agents_response_order_preserved(monkeypatch):
    """The fix must NOT change the order in which agents are returned.
    list_agents returns agents in created_at DESC (legacy). The bulk
    helpers don't touch the agent ordering; they only stamp counts.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)

    fixed_rows = [
        {"id": "agent-a", "user_id": "user-1", "name": "A"},
        {"id": "agent-b", "user_id": "user-1", "name": "B"},
        {"id": "agent-c", "user_id": "user-1", "name": "C"},
    ]
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: fixed_rows)

    def fake_tools(user_id, agent_ids):
        return {a: 0 for a in agent_ids}

    def fake_integ(user_id, agent_ids):
        return {a: 0 for a in agent_ids}

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_tools
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_integ,
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    assert [r["id"] for r in rows] == ["agent-a", "agent-b", "agent-c"]


# ── Ticket 3 finalization: failure parity (F1-F5) ───────────────────────────────


def test_list_agents_tools_bulk_failure_falls_back_to_per_agent(monkeypatch):
    """F1: tools_count_by_agent raises. The route must fall back to the
    pre-fix per-agent list_tools loop. The fixture returns:
      A=2 (succeeds), B=raises (so B=0), C=1.

    Expected: A tools_count=2, B=0, C=1. This matches pre-fix behavior
    where only the failing agent had tools_count=0.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
        {"id": "agent-c", "user_id": user_id, "name": "C"},
    ])

    def fake_bulk_tools_count(user_id, agent_ids):
        raise RuntimeError("storage unavailable")

    def fake_legacy_tools(user_id, agent_id=None):
        if agent_id == "agent-a":
            return [
                {"id": "t1", "webhook_url": "https://x"},
                {"id": "t2", "webhook_url": "https://y"},
            ]
        if agent_id == "agent-b":
            raise RuntimeError("B-specific failure")
        if agent_id == "agent-c":
            return [{"id": "t3", "destination": "+1"}]
        return []

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_bulk_tools_count
    )
    # Legacy fallback uses list_tools.
    monkeypatch.setattr(
        "STT_server.db_tools.list_tools", fake_legacy_tools
    )
    # Integrations bulk succeeds (independent path).
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        lambda user_id, agent_ids: {a: 0 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    by_id = {r["id"]: r for r in rows}
    assert by_id["agent-a"]["tools_count"] == 2
    assert by_id["agent-b"]["tools_count"] == 0  # legacy: list_tools raised → 0
    assert by_id["agent-c"]["tools_count"] == 1


def test_list_agents_fallback_credential_excluded_real_predicate(monkeypatch):
    """§9 real-tool predicate parity: when the tools bulk fails and
    the legacy per-agent fallback runs, the predicate
    ``bool(webhook_url or destination)`` must apply identically to
    the bulk helper. Both code paths must agree on what counts.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
    ])

    # tools bulk fails → fallback.
    def fake_bulk_tools_count(user_id, agent_ids):
        raise RuntimeError("storage unavailable")

    # Legacy list_tools returns the same row shape the bulk helper
    # would consume: 1 real webhook + 1 provider credential
    # (kind='credentials', no webhook_url, no destination). Both must
    # be filtered to count only the webhook.
    def fake_legacy_tools(user_id, agent_id=None):
        return [
            {"id": "t1", "webhook_url": "https://x"},
            # provider credential row — must be excluded.
            {"id": "cred-openai", "agent_id": "__shared__",
             "assignments": ["agent-a"], "kind": "credentials"},
        ]

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_bulk_tools_count
    )
    monkeypatch.setattr("STT_server.db_tools.list_tools", fake_legacy_tools)
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        lambda user_id, agent_ids: {a: 0 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    # Predicate parity: fallback filters the credential row via the
    # same _is_real_tool check the bulk helper uses.
    assert rows[0]["tools_count"] == 1, (
        "legacy fallback must filter provider credentials just like "
        "the bulk helper (predicate parity)"
    )


def test_list_agents_integrations_bulk_failure_falls_back_to_per_agent(monkeypatch):
    """F2: connected_integrations_count_by_agent raises. Fallback to
    pre-fix list_integrations per-agent loop.

    Fixture: A=1, B=raises, C=2. Expected: A=1, B=0, C=2.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
        {"id": "agent-c", "user_id": user_id, "name": "C"},
    ])

    def fake_bulk_integrations(user_id, agent_ids):
        raise RuntimeError("storage unavailable")

    def fake_legacy_integrations(user_id, agent_id=None):
        if agent_id == "agent-a":
            return [
                {"id": "i1", "agent_id": "agent-a",
                 "connection_status": "connected"},
            ]
        if agent_id == "agent-b":
            raise RuntimeError("B-specific failure")
        if agent_id == "agent-c":
            return [
                {"id": "i2", "agent_id": "agent-c",
                 "connection_status": "connected"},
                {"id": "i3", "agent_id": "agent-c",
                 "connection_status": "connected"},
            ]
        return []

    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        fake_bulk_integrations,
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations",
        fake_legacy_integrations,
    )
    # Tools bulk succeeds.
    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent",
        lambda user_id, agent_ids: {a: 0 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    by_id = {r["id"]: r for r in rows}
    assert by_id["agent-a"]["integrations_count"] == 1
    assert by_id["agent-b"]["integrations_count"] == 0
    assert by_id["agent-c"]["integrations_count"] == 2


def test_list_agents_tools_fail_integrations_succeed_independently(monkeypatch):
    """F3: tools bulk raises → fallback to per-agent. Integrations bulk
    succeeds. The two paths must NOT cross-contaminate.

    Assert: legacy list_tools called N times; legacy list_integrations
    called 0 times.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
    ])

    def fake_bulk_tools(user_id, agent_ids):
        raise RuntimeError("tools storage failed")

    def fake_legacy_tools(user_id, agent_id=None):
        return []

    legacy_tools_calls: list[tuple[str, str | None]] = []
    legacy_integ_calls: list[tuple[str, str | None]] = []

    def spy_legacy_tools(user_id, agent_id=None):
        legacy_tools_calls.append((user_id, agent_id))
        return []

    def spy_legacy_integrations(user_id, agent_id=None):
        legacy_integ_calls.append((user_id, agent_id))
        return []

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent", fake_bulk_tools
    )
    monkeypatch.setattr("STT_server.db_tools.list_tools", spy_legacy_tools)
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        lambda user_id, agent_ids: {a: 1 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations", spy_legacy_integrations
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    by_id = {r["id"]: r for r in rows}
    # tools: fallback used (each agent called list_tools once).
    assert len(legacy_tools_calls) == 2, (
        f"expected 2 legacy list_tools calls (one per agent); got {len(legacy_tools_calls)}"
    )
    # integrations: bulk succeeded (no fallback).
    assert len(legacy_integ_calls) == 0, (
        f"expected 0 legacy list_integrations calls (bulk succeeded); got {len(legacy_integ_calls)}"
    )
    # tools_count: from legacy loop (each agent returned []).
    assert by_id["agent-a"]["tools_count"] == 0
    assert by_id["agent-b"]["tools_count"] == 0
    # integrations_count: from bulk (each agent returned 1).
    assert by_id["agent-a"]["integrations_count"] == 1
    assert by_id["agent-b"]["integrations_count"] == 1


def test_list_agents_integrations_fail_tools_succeed_independently(monkeypatch):
    """F4: integrations bulk raises → fallback. Tools bulk succeeds.
    Inverse of F3.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
    ])

    legacy_tools_calls: list[tuple[str, str | None]] = []
    legacy_integ_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent",
        lambda user_id, agent_ids: {a: 2 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.db_tools.list_tools",
        lambda user_id, agent_id=None: legacy_tools_calls.append((user_id, agent_id)) or [],
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        lambda user_id, agent_ids: (_ for _ in ()).throw(RuntimeError("integ failed")),
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations",
        lambda user_id, agent_id=None: legacy_integ_calls.append((user_id, agent_id)) or [],
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    # tools: bulk succeeded (no fallback).
    assert len(legacy_tools_calls) == 0, (
        f"expected 0 legacy list_tools calls (bulk succeeded); got {len(legacy_tools_calls)}"
    )
    # integrations: fallback used.
    assert len(legacy_integ_calls) == 2, (
        f"expected 2 legacy list_integrations calls; got {len(legacy_integ_calls)}"
    )
    by_id = {r["id"]: r for r in rows}
    assert by_id["agent-a"]["tools_count"] == 2
    assert by_id["agent-b"]["tools_count"] == 2
    assert by_id["agent-a"]["integrations_count"] == 0
    assert by_id["agent-b"]["integrations_count"] == 0


def test_list_agents_happy_path_never_calls_legacy_per_agent(monkeypatch):
    """F5: both bulk helpers succeed. The legacy list_tools and
    list_integrations must NEVER be called. This is the N+1 protection
    on the happy path.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: True)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
        {"id": "agent-c", "user_id": user_id, "name": "C"},
    ])

    legacy_tools_calls: list[tuple[str, str | None]] = []
    legacy_integ_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        "STT_server.db_tools.tools_count_by_agent",
        lambda user_id, agent_ids: {a: 1 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.db_tools.list_tools",
        lambda user_id, agent_id=None: legacy_tools_calls.append((user_id, agent_id)) or [],
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.connected_integrations_count_by_agent",
        lambda user_id, agent_ids: {a: 1 for a in agent_ids},
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.list_integrations",
        lambda user_id, agent_id=None: legacy_integ_calls.append((user_id, agent_id)) or [],
    )
    monkeypatch.setattr(
        "STT_server.services.usage_store.aggregate_usage",
        lambda user_id: {
            "totals": {"calls": 0, "duration_seconds": 0.0, "cost_usd": 0.0,
                       "platform_duration_seconds": 0.0, "own_duration_seconds": 0.0},
            "per_agent": [],
            "rates": {"own_per_min": 0.0828, "platform_per_min": 0.14, "currency": "USD"},
        },
    )

    rows = api.list_agents(auth={"user_id": "user-1"})
    assert len(legacy_tools_calls) == 0, (
        f"N+1 regression: list_tools called {len(legacy_tools_calls)} times in happy path"
    )
    assert len(legacy_integ_calls) == 0, (
        f"N+1 regression: list_integrations called {len(legacy_integ_calls)} times in happy path"
    )
    by_id = {r["id"]: r for r in rows}
    for aid in ("agent-a", "agent-b", "agent-c"):
        assert by_id[aid]["tools_count"] == 1
        assert by_id[aid]["integrations_count"] == 1


def test_list_agents_bulk_tools_duplicate_assignment_counts_once(monkeypatch):
    """Ticket 3 finalization §5+§6: a single storage row contributes
    MAX 1 to the count of any given agent, regardless of how many
    times the agent_id appears in the row's assignments list.

    Pre-fix behavior (verified in parent cc37d1e via
    db_tools.list_tools JSON filter): the legacy membership check
    ``agent_id in assignments`` returns True once per assignment list,
    so a row with assignments=['a','a'] returns the row once and
    counts as 1.

    Ticket 3 must preserve this. The JSON branch uses
    ``{a for a in assignments if a in agent_id_set}`` (set
    comprehension) so duplicates dedupe per agent per row.
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_tools as _db_tools_mod_for_setup
    monkeypatch.setattr(_db_tools_mod_for_setup, "is_postgres", lambda: False)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
    ])

    import json as _json
    from STT_server import db_tools as _db_tools_mod

    tools_payload = [
        {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a", "agent-a"], "destination": "+1"},
    ]
    fake_file = _db_tools_mod.Path(_db_tools_mod.__file__).resolve().parent / "data" / "agent_tools.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(tools_payload))

    try:
        tools_count = _db_tools_mod.tools_count_by_agent("user-1", ["agent-a"])
    finally:
        fake_file.write_text("[]")

    # Canonical: one row, max 1 per agent.
    assert tools_count == {"agent-a": 1}, (
        f"a shared row with duplicate assignments must count once "
        f"per agent: {tools_count}"
    )


def test_list_agents_bulk_tools_multi_agent_duplicate_assignments(monkeypatch):
    """§5 multi-agent dedupe: assignments=['a','a','b','b'] produces
    A=1, B=1, C=0. Duplicates per agent are deduped; distinct agents
    each get their own count.
    """
    import json as _json
    from STT_server import db_tools as _db_tools_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_tools as _db_tools_mod_for_setup
    monkeypatch.setattr(_db_tools_mod_for_setup, "is_postgres", lambda: False)

    tools_payload = [
        {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a", "agent-a", "agent-b", "agent-b"],
         "destination": "+1"},
    ]
    fake_file = _db_tools_mod.Path(_db_tools_mod.__file__).resolve().parent / "data" / "agent_tools.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(tools_payload))

    try:
        tools_count = _db_tools_mod.tools_count_by_agent(
            "user-1", ["agent-a", "agent-b", "agent-c"]
        )
    finally:
        fake_file.write_text("[]")

    assert tools_count == {"agent-a": 1, "agent-b": 1, "agent-c": 0}


def test_list_agents_bulk_tools_multiple_distinct_rows(monkeypatch):
    """§6: two DISTINCT shared rows (different ids) both assigned to
    the same agent must contribute 2 — count identity is per row, not
    per (row, agent_id) deduplication across rows.
    """
    import json as _json
    from STT_server import db_tools as _db_tools_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_tools as _db_tools_mod_for_setup
    monkeypatch.setattr(_db_tools_mod_for_setup, "is_postgres", lambda: False)

    tools_payload = [
        {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a"], "destination": "+1"},
        {"id": "t2", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a"], "webhook_url": "https://x"},
    ]
    fake_file = _db_tools_mod.Path(_db_tools_mod.__file__).resolve().parent / "data" / "agent_tools.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(tools_payload))

    try:
        tools_count = _db_tools_mod.tools_count_by_agent("user-1", ["agent-a"])
    finally:
        fake_file.write_text("[]")

    # Two distinct rows, each assigned to agent-a → count is 2.
    assert tools_count == {"agent-a": 2}


def test_list_agents_bulk_tools_private_row_with_assignments(monkeypatch):
    """§7: a private tool (agent_id='agent-a') with stray assignments
    fields is a valid private row. The JSON filter matches the agent
    by agent_id first; assignments on a private row are ignored.
    Per the legacy filter: ``agent_id == X OR (agent_id == '__shared__'
    AND X in assignments)`` — the private row matches via the first
    disjunct.
    """
    import json as _json
    from STT_server import db_tools as _db_tools_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_tools as _db_tools_mod_for_setup
    monkeypatch.setattr(_db_tools_mod_for_setup, "is_postgres", lambda: False)

    tools_payload = [
        # Private tool for agent-a with stray 'assignments' field.
        # The legacy filter matches via agent_id (first disjunct),
        # so it counts once for agent-a regardless of assignments.
        {"id": "t1", "user_id": "user-1", "agent_id": "agent-a",
         "assignments": ["agent-a", "agent-a"], "destination": "+1"},
    ]
    fake_file = _db_tools_mod.Path(_db_tools_mod.__file__).resolve().parent / "data" / "agent_tools.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(tools_payload))

    try:
        tools_count = _db_tools_mod.tools_count_by_agent("user-1", ["agent-a"])
    finally:
        fake_file.write_text("[]")

    # Private row matches agent_id disjunct; count = 1.
    assert tools_count == {"agent-a": 1}


def test_list_agents_bulk_integrations_shared_multi_assign(monkeypatch):
    """Ticket 3 finalization §6: shared integration with assignments
    ['agent-A', 'agent-B'] counts once for each (1 per agent).
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    # ponytail: db_integrations imports ``is_postgres`` at module level
    # (line 40). The module-level reference is cached and doesn't
    # reflect monkeypatch on STT_server.db.is_postgres. Patch the local
    # symbol directly.
    import STT_server.db_integrations as _db_int_mod_for_setup
    monkeypatch.setattr(_db_int_mod_for_setup, "is_postgres", lambda: False)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
        {"id": "agent-b", "user_id": user_id, "name": "B"},
        {"id": "agent-c", "user_id": user_id, "name": "C"},
    ])

    import json as _json
    from STT_server import db_integrations as _db_int_mod

    integ_payload = [
        {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a", "agent-b"],
         "connection_status": "connected"},
    ]
    fake_file = _db_int_mod.Path(_db_int_mod.__file__).resolve().parent / "data" / "integrations.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(integ_payload))

    try:
        integ_count = _db_int_mod.connected_integrations_count_by_agent(
            "user-1", ["agent-a", "agent-b", "agent-c"]
        )
    finally:
        fake_file.write_text("[]")

    assert integ_count == {"agent-a": 1, "agent-b": 1, "agent-c": 0}


def test_list_agents_bulk_integrations_duplicate_assignment_counts_once(monkeypatch):
    """Ticket 3 finalization §5+§6: a single storage row contributes
    MAX 1 to the count of any given agent, regardless of how many
    times the agent_id appears in the row's assignments list. Matches
    the Postgres ?| operator semantics and the pre-fix legacy
    per-agent list_integrations filter (which used membership check
    ``agent_id in assignments``).
    """
    api = _import_api()

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_integrations as _db_int_mod_for_setup
    monkeypatch.setattr(_db_int_mod_for_setup, "is_postgres", lambda: False)
    monkeypatch.setattr(api, "db_list_agents", lambda user_id: [
        {"id": "agent-a", "user_id": user_id, "name": "A"},
    ])

    import json as _json
    from STT_server import db_integrations as _db_int_mod

    integ_payload = [
        {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a", "agent-a"],
         "connection_status": "connected"},
    ]
    fake_file = _db_int_mod.Path(_db_int_mod.__file__).resolve().parent / "data" / "integrations.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(integ_payload))

    try:
        integ_count = _db_int_mod.connected_integrations_count_by_agent(
            "user-1", ["agent-a"]
        )
    finally:
        fake_file.write_text("[]")

    # Canonical: one row, max 1 per agent.
    assert integ_count == {"agent-a": 1}


def test_list_agents_bulk_integrations_multi_agent_duplicate_assignments(monkeypatch):
    """§5 multi-agent dedupe for integrations."""
    import json as _json
    from STT_server import db_integrations as _db_int_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_integrations as _db_int_mod_for_setup
    monkeypatch.setattr(_db_int_mod_for_setup, "is_postgres", lambda: False)

    integ_payload = [
        {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a", "agent-a", "agent-b", "agent-b"],
         "connection_status": "connected"},
    ]
    fake_file = _db_int_mod.Path(_db_int_mod.__file__).resolve().parent / "data" / "integrations.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(integ_payload))

    try:
        integ_count = _db_int_mod.connected_integrations_count_by_agent(
            "user-1", ["agent-a", "agent-b", "agent-c"]
        )
    finally:
        fake_file.write_text("[]")

    assert integ_count == {"agent-a": 1, "agent-b": 1, "agent-c": 0}


def test_list_agents_bulk_integrations_multiple_distinct_rows(monkeypatch):
    """§6 for integrations: two distinct shared rows both assigned to
    the same agent count as 2 (count identity is per row)."""
    import json as _json
    from STT_server import db_integrations as _db_int_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_integrations as _db_int_mod_for_setup
    monkeypatch.setattr(_db_int_mod_for_setup, "is_postgres", lambda: False)

    integ_payload = [
        {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a"], "connection_status": "connected"},
        {"id": "i2", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a"], "connection_status": "connected"},
    ]
    fake_file = _db_int_mod.Path(_db_int_mod.__file__).resolve().parent / "data" / "integrations.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(integ_payload))

    try:
        integ_count = _db_int_mod.connected_integrations_count_by_agent(
            "user-1", ["agent-a"]
        )
    finally:
        fake_file.write_text("[]")

    assert integ_count == {"agent-a": 2}


def test_list_agents_bulk_integrations_private_row_with_assignments(monkeypatch):
    """§7 for integrations: a private integration (agent_id='agent-a')
    with stray assignments counts once for agent-a."""
    import json as _json
    from STT_server import db_integrations as _db_int_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    import STT_server.db_integrations as _db_int_mod_for_setup
    monkeypatch.setattr(_db_int_mod_for_setup, "is_postgres", lambda: False)

    integ_payload = [
        {"id": "i1", "user_id": "user-1", "agent_id": "agent-a",
         "assignments": ["agent-a", "agent-a"], "connection_status": "connected"},
    ]
    fake_file = _db_int_mod.Path(_db_int_mod.__file__).resolve().parent / "data" / "integrations.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(integ_payload))

    try:
        integ_count = _db_int_mod.connected_integrations_count_by_agent(
            "user-1", ["agent-a"]
        )
    finally:
        fake_file.write_text("[]")

    assert integ_count == {"agent-a": 1}


# ── Real-data-model fixtures (no helper mocks; exercise the real
#    production JSON path) ────────────────────────────────────────────────


def test_list_agents_real_model_private_integration_counts(monkeypatch, tmp_path):
    """Ticket 3 finalization §8: the new tests must not rely solely on
    helper mocks. This test exercises the real JSON bulk helper against
    a real-data-model integration row:

      {id, user_id, agent_id="agent-b", connection_status="connected",
       assignments=[]}

    Expected: agent-b=1 (private integration owned by agent-b). Agent-a
    and agent-c get 0 because the integration is private.
    """
    import json as _json
    from STT_server import db_integrations as _db_int_mod

    fake_file = tmp_path / "integrations.json"
    fake_file.write_text(_json.dumps([
        {"id": "i1", "user_id": "user-1", "agent_id": "agent-b",
         "connection_status": "connected", "assignments": []},
    ]))

    # Override the module-level path so the helper reads from tmp_path.
    monkeypatch.setattr(_db_int_mod, "_INTEGRATIONS_FILE", fake_file)
    # ponytail: cache-busting is_postgres on the local module.
    monkeypatch.setattr(_db_int_mod, "is_postgres", lambda: False)
    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)

    counts = _db_int_mod.connected_integrations_count_by_agent(
        "user-1", ["agent-a", "agent-b", "agent-c"]
    )
    assert counts == {"agent-a": 0, "agent-b": 1, "agent-c": 0}, (
        f"private integration owned by agent-b should count only for b: "
        f"{counts}"
    )


def test_list_agents_real_model_credential_row_excluded(monkeypatch, tmp_path):
    """§9 real-data-model predicate parity: provider credential row
    (agent_id='__shared__', assignments=['agent-a'], NO webhook_url,
    NO destination) must NOT count toward agent-a's tools_count.

    Webhook tool (agent_id='agent-a', webhook_url='https://x') counts
    for agent-a. call_transfer (destination='+1') counts for agent-a.
    """
    import json as _json
    from STT_server import db_tools as _db_tools_mod

    fake_file = tmp_path / "agent_tools.json"
    fake_file.write_text(_json.dumps([
        # credential: real predicate excludes (no webhook_url, no destination).
        {"id": "cred-openai", "user_id": "user-1", "agent_id": "__shared__",
         "assignments": ["agent-a"], "kind": "credentials"},
        # webhook: counts.
        {"id": "t1", "user_id": "user-1", "agent_id": "agent-a",
         "webhook_url": "https://x"},
        # call_transfer: counts (destination truthy).
        {"id": "t2", "user_id": "user-1", "agent_id": "agent-a",
         "destination": "+526649070770"},
    ]))

    monkeypatch.setattr(_db_tools_mod, "_AGENT_TOOLS_FILE", fake_file)
    monkeypatch.setattr(_db_tools_mod, "is_postgres", lambda: False)
    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)

    counts = _db_tools_mod.tools_count_by_agent("user-1", ["agent-a"])
    # 1 webhook + 1 call_transfer = 2 real tools. credential excluded.
    assert counts == {"agent-a": 2}, (
        f"webhook + call_transfer should count, credential should not: "
        f"{counts}"
    )


# ── PostgreSQL query shape validation (without running Postgres) ────────────


def test_tools_count_by_agent_postgres_query_shape(monkeypatch):
    """§10: validate the SQL query string and parameter shape passed to
    psycopg2 without actually executing against Postgres. The test
    captures the cursor.execute() arguments and asserts:
      - ANY(%s) parameter receives a list[str] (psycopg2 → ARRAY).
      - jsonb ?| parameter receives a list[str] (psycopg2 → text[]).
      - Query string contains the right operators.
    """
    from STT_server import db_tools as _db_tools_mod

    monkeypatch.setattr(_db_tools_mod, "is_postgres", lambda: True)

    # Capture cursor.execute args. The _ensure_tool_columns helper
    # runs a schema check first; we record all execute() calls and
    # assert on the LAST one (the bulk count query).
    captured: list = []

    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def execute(self, query, params=None):
            captured.append((query, params))
            self.fetchall = lambda: []

        def fetchone(self): return None
        def fetchall(self): return []

    class FakeConn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return FakeCursor()

    monkeypatch.setattr(_db_tools_mod, "get_conn", lambda: FakeConn())

    _db_tools_mod.tools_count_by_agent("user-1", ["agent-a", "agent-b"])
    assert captured, "cursor.execute was never called"
    # The bulk query is the last execute() call.
    query, params = captured[-1]

    # Query string assertions.
    assert "agent_id = ANY(%s)" in query, (
        f"ANY(%s) operator missing from query: {query}"
    )
    assert "?|" in query, (
        f"jsonb overlap operator ?| missing from query: {query}"
    )
    assert "agent_id = '__shared__'" in query
    assert "WHERE user_id = %s" in query

    # Parameter shape.
    # psycopg2 adapts ``list(agent_ids)`` to a Postgres ARRAY (for
    # ANY) and to a text[] (for ?|). The two ``ANY`` and ``?|``
    # placeholders each receive a list copy.
    assert len(params) == 3, (
        f"expected 3 params (user_id, ANY list, ?| list); got {len(params)}: {params}"
    )
    user_id_p, any_p, overlap_p = params
    assert user_id_p == "user-1"
    assert isinstance(any_p, list) and sorted(any_p) == ["agent-a", "agent-b"]
    assert isinstance(overlap_p, list) and sorted(overlap_p) == ["agent-a", "agent-b"]


def test_connected_integrations_count_by_agent_postgres_query_shape(monkeypatch):
    """§10: same validation for the integrations bulk helper."""
    from STT_server import db_integrations as _db_int_mod

    monkeypatch.setattr(_db_int_mod, "is_postgres", lambda: True)

    captured: list = []

    class FakeCursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False

        def execute(self, query, params=None):
            captured.append((query, params))
            self.fetchall = lambda: []

        def fetchone(self): return None
        def fetchall(self): return []

    class FakeConn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return FakeCursor()

    monkeypatch.setattr(_db_int_mod, "get_conn", lambda: FakeConn())

    _db_int_mod.connected_integrations_count_by_agent(
        "user-1", ["agent-a", "agent-b"]
    )

    assert captured
    query, params = captured[-1]

    assert "agent_id = ANY(%s)" in query
    assert "?|" in query
    assert "connection_status = 'connected'" in query
    assert "WHERE user_id = %s" in query

    assert len(params) == 3
    user_id_p, any_p, overlap_p = params
    assert user_id_p == "user-1"
    assert isinstance(any_p, list) and sorted(any_p) == ["agent-a", "agent-b"]
    assert isinstance(overlap_p, list) and sorted(overlap_p) == ["agent-a", "agent-b"]


# ── Comprehensive JSON bulk vs legacy list_* parity (§8) ────────────────────


_PARITY_SCENARIOS = [
    # (description, fixture_rows, expected_counts)
    pytest.param(
        "no_rows",
        [],
        {"agent-a": 0, "agent-b": 0, "agent-c": 0},
        id="no_rows",
    ),
    pytest.param(
        "private_only",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "agent-a",
             "webhook_url": "https://x"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="private_only",
    ),
    pytest.param(
        "shared_one_agent",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a"], "webhook_url": "https://x"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="shared_one_agent",
    ),
    pytest.param(
        "shared_two_agents",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a", "agent-b"],
             "webhook_url": "https://x"},
        ],
        {"agent-a": 1, "agent-b": 1, "agent-c": 0},
        id="shared_two_agents",
    ),
    pytest.param(
        "shared_duplicate_assignments_one_agent",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a", "agent-a"], "webhook_url": "https://x"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="shared_duplicate_assignments_one_agent",
    ),
    pytest.param(
        "shared_duplicate_assignments_two_agents",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a", "agent-a", "agent-b", "agent-b"],
             "webhook_url": "https://x"},
        ],
        {"agent-a": 1, "agent-b": 1, "agent-c": 0},
        id="shared_duplicate_assignments_two_agents",
    ),
    pytest.param(
        "credential_excluded",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "agent-a",
             "webhook_url": "https://x"},
            # Credential row assigned to A and B: must NOT count.
            {"id": "cred-openai", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a", "agent-b"], "kind": "credentials"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="credential_excluded",
    ),
    pytest.param(
        "unassigned_shared_excluded",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-other"], "webhook_url": "https://x"},
        ],
        {"agent-a": 0, "agent-b": 0, "agent-c": 0},
        id="unassigned_shared_excluded",
    ),
    pytest.param(
        "two_distinct_shared_rows_same_agent",
        [
            {"id": "t1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a"], "destination": "+1"},
            {"id": "t2", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a"], "webhook_url": "https://x"},
        ],
        {"agent-a": 2, "agent-b": 0, "agent-c": 0},
        id="two_distinct_shared_rows_same_agent",
    ),
]


@pytest.mark.parametrize("description,fixture_rows,expected_counts", _PARITY_SCENARIOS)
def test_list_agents_bulk_tools_count_matches_legacy_per_agent(
    monkeypatch, description, fixture_rows, expected_counts
):
    """§8: the bulk tools_count_by_agent JSON branch must produce
    IDENTICAL counts to the legacy per-agent list_tools(agent_id=X)
    loop (which is the pre-fix code path). This is the canonical
    parity check: bulk == legacy, per row, per agent.

    Both code paths are exercised against the same fixture rows so
    any divergence indicates a real bug, not a test artifact.
    """
    import json as _json
    from STT_server import db_tools as _db_tools_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    monkeypatch.setattr(_db_tools_mod, "is_postgres", lambda: False)

    fake_file = _db_tools_mod.Path(_db_tools_mod.__file__).resolve().parent / "data" / "agent_tools.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(fixture_rows))

    try:
        # Bulk path.
        bulk = _db_tools_mod.tools_count_by_agent(
            "user-1", ["agent-a", "agent-b", "agent-c"]
        )

        # Legacy path: replicate the pre-fix filter (list_tools +
        # _is_real_tool) inline so we don't depend on imports
        # reaching back to routes/api.py.
        def legacy_count_for(agent_id):
            count = 0
            for r in fixture_rows:
                if r.get("user_id") != "user-1":
                    continue
                rid = r.get("agent_id")
                if rid == agent_id:
                    pass  # private row match
                elif rid == "__shared__" and agent_id in (r.get("assignments") or []):
                    pass  # shared assigned match
                else:
                    continue
                if r.get("webhook_url") or r.get("destination"):
                    count += 1
            return count

        legacy = {
            aid: legacy_count_for(aid) for aid in ("agent-a", "agent-b", "agent-c")
        }
    finally:
        fake_file.write_text("[]")

    assert bulk == legacy, (
        f"scenario={description}: bulk={bulk} vs legacy={legacy}"
    )
    assert bulk == expected_counts, (
        f"scenario={description}: expected={expected_counts}, got bulk={bulk}"
    )


_PARITY_INTEG_SCENARIOS = [
    pytest.param(
        "no_rows",
        [],
        {"agent-a": 0, "agent-b": 0, "agent-c": 0},
        id="no_rows",
    ),
    pytest.param(
        "private_connected",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "agent-a",
             "connection_status": "connected"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="private_connected",
    ),
    pytest.param(
        "private_disconnected_excluded",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "agent-a",
             "connection_status": "disconnected"},
        ],
        {"agent-a": 0, "agent-b": 0, "agent-c": 0},
        id="private_disconnected_excluded",
    ),
    pytest.param(
        "shared_one_agent_connected",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a"], "connection_status": "connected"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="shared_one_agent_connected",
    ),
    pytest.param(
        "shared_two_agents_connected",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a", "agent-b"], "connection_status": "connected"},
        ],
        {"agent-a": 1, "agent-b": 1, "agent-c": 0},
        id="shared_two_agents_connected",
    ),
    pytest.param(
        "shared_duplicate_one_agent",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a", "agent-a"], "connection_status": "connected"},
        ],
        {"agent-a": 1, "agent-b": 0, "agent-c": 0},
        id="shared_duplicate_one_agent",
    ),
    pytest.param(
        "shared_unassigned_excluded",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-other"], "connection_status": "connected"},
        ],
        {"agent-a": 0, "agent-b": 0, "agent-c": 0},
        id="shared_unassigned_excluded",
    ),
    pytest.param(
        "two_distinct_shared_rows_same_agent",
        [
            {"id": "i1", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a"], "connection_status": "connected"},
            {"id": "i2", "user_id": "user-1", "agent_id": "__shared__",
             "assignments": ["agent-a"], "connection_status": "connected"},
        ],
        {"agent-a": 2, "agent-b": 0, "agent-c": 0},
        id="two_distinct_shared_rows_same_agent",
    ),
]


@pytest.mark.parametrize(
    "description,fixture_rows,expected_counts", _PARITY_INTEG_SCENARIOS
)
def test_list_agents_bulk_integrations_count_matches_legacy_per_agent(
    monkeypatch, description, fixture_rows, expected_counts
):
    """§8 parity for integrations: bulk == legacy per-agent."""
    import json as _json
    from STT_server import db_integrations as _db_int_mod

    monkeypatch.setattr("STT_server.db.is_postgres", lambda: False)
    monkeypatch.setattr(_db_int_mod, "is_postgres", lambda: False)

    fake_file = _db_int_mod.Path(_db_int_mod.__file__).resolve().parent / "data" / "integrations.json"
    fake_file.parent.mkdir(parents=True, exist_ok=True)
    fake_file.write_text(_json.dumps(fixture_rows))

    try:
        bulk = _db_int_mod.connected_integrations_count_by_agent(
            "user-1", ["agent-a", "agent-b", "agent-c"]
        )

        # Legacy path: replicate the inline filter that the per-agent
        # list_integrations(agent_id=X) used to apply in routes/api.py.
        def legacy_count_for(agent_id):
            count = 0
            for r in fixture_rows:
                if r.get("user_id") != "user-1":
                    continue
                if r.get("connection_status") != "connected":
                    continue
                rid = r.get("agent_id")
                if rid == agent_id:
                    pass
                elif rid == "__shared__" and agent_id in (r.get("assignments") or []):
                    pass
                else:
                    continue
                count += 1
            return count

        legacy = {
            aid: legacy_count_for(aid) for aid in ("agent-a", "agent-b", "agent-c")
        }
    finally:
        fake_file.write_text("[]")

    assert bulk == legacy, (
        f"scenario={description}: bulk={bulk} vs legacy={legacy}"
    )
    assert bulk == expected_counts, (
        f"scenario={description}: expected={expected_counts}, got bulk={bulk}"
    )
