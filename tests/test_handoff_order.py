"""BE tests for the unified handoff order (canonical config).

Canonical order: one node list (the single source the UI edits):
  {"type": "ai"}                                             (exactly once)
  {"type": "transfer_tool", "tool_id": "<id>"}               (0..N, unique)
  {"type": "phone_destination", "destination": "+...",
   "timeout_sec": 20}                                        (raw legacy)

Persistence (no new DB column): split_handoff_order writes the pair
(transfer_cascade with tool_id, transfer_chain) atomically;
derive_handoff_order reads it back losslessly. Runtime readers are
untouched (/voice reads cascade destinations, turn_manager reads
chain ids).

Identity rule under test: a transfer tool keeps its tool_id on BOTH
sides of the AI. Unassign removes by exact tool_id — sibling tools
sharing one destination are never confused.
"""
from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient

from STT_server.services.transfer_cascade import (
    build_transfer_chain,
    derive_handoff_order,
    parse_cascade_with_ids,
    resolve_cascade_steps,
    split_handoff_order,
    validate_handoff_order,
)

A = "+15550001111"
B = "+15550002222"
SAME = "+15550003333"


def _tools(*defs):
    """defs: (id, dest). All call_transfer."""
    return {
        tid: {"name": tid, "kind": "call_transfer", "destination": dest,
              "ring_timeout_sec": 20}
        for tid, dest in defs
    }


def _ai():
    return {"type": "ai"}


def _tool(tid):
    # ponytail: transfer_tool nodes carry identity only — timeout lives
    # on the tool row, so there is no per-node timeout to pass here.
    return {"type": "transfer_tool", "tool_id": tid}


def _raw(dest, timeout=20):
    return {"type": "phone_destination", "destination": dest,
            "timeout_sec": timeout}


# ── B1: legacy normalization ──────────────────────────────────────────

def test_B1_legacy_cascade_chain_normalizes_to_order():
    """cascade=[raw A] chain=[B] -> [raw:A, AI, tool:B]. Raw entries with
    no tool_id stay raw (never silently adopted as tools)."""
    order = derive_handoff_order(
        [{"destination": A, "timeout_sec": 20}], ["tool-b"])
    assert order == [_raw(A), _ai(), _tool("tool-b")]


# ── B5–B9: split derivation ───────────────────────────────────────────

def test_B5_reorder_A_AI_B_splits():
    tools = _tools(("a", A), ("b", B))
    (halves, err) = split_handoff_order([_tool("a"), _ai(), _tool("b")], tools)
    assert err is None, err
    cascade, chain = halves
    assert cascade == [{"destination": A, "timeout_sec": 20, "tool_id": "a"}]
    assert chain == ["b"]


def test_B7_derived_runtime_matches_A_AI_B():
    tools = _tools(("a", A), ("b", B))
    (halves, err) = split_handoff_order([_tool("a"), _ai(), _tool("b")], tools)
    assert err is None, err
    cascade, chain = halves
    assert [c["destination"] for c in cascade] == [A]   # before AI
    assert chain == ["b"]                              # after AI


def test_B8_AI_A_B_derives_empty_cascade():
    tools = _tools(("a", A), ("b", B))
    (halves, err) = split_handoff_order([_ai(), _tool("a"), _tool("b")], tools)
    assert err is None, err
    cascade, chain = halves
    assert cascade == []
    assert chain == ["a", "b"]


def test_B9_A_B_AI_derives_empty_chain():
    tools = _tools(("a", A), ("b", B))
    (halves, err) = split_handoff_order([_tool("a"), _tool("b"), _ai()], tools)
    assert err is None, err
    cascade, chain = halves
    assert [c["tool_id"] for c in cascade] == ["a", "b"]
    assert chain == []


def test_B6_reorder_variants_accepted():
    tools = _tools(("a", A), ("b", B))
    for order in (
        [_ai(), _tool("a"), _tool("b")],
        [_tool("b"), _tool("a"), _ai()],
        [_tool("b"), _ai(), _tool("a")],
        [_tool("a"), _tool("b"), _ai()],
    ):
        (halves, err) = split_handoff_order(order, tools)
        assert err is None, f"{order}: {err}"
        # round-trip: derive(split(x)) == x (timeout defaults materialize)
        back = derive_handoff_order(*halves)
        assert [n.get("tool_id", n.get("type")) for n in back] == [
            n.get("tool_id", n.get("type")) for n in order
        ]


def test_raw_after_AI_rejected():
    tools = _tools(("a", A))
    (halves, err) = split_handoff_order([_ai(), _raw(A)], tools)
    assert halves is None and "before the AI" in (err or "")


def test_B10_duplicate_tool_rejected():
    nodes, err = validate_handoff_order([_tool("a"), _ai(), _tool("a")])
    assert nodes is None and "duplicate" in (err or "").lower()


def test_B12_duplicate_and_missing_AI_rejected():
    nodes, err = validate_handoff_order([_ai(), _tool("a"), _ai()])
    assert nodes is None and "exactly once" in (err or "")
    nodes, err = validate_handoff_order([_tool("a")])
    assert nodes is None and "exactly once" in (err or "")


def test_B11_unassigned_tool_rejected_in_split():
    tools = _tools(("a", A))
    (halves, err) = split_handoff_order([_ai(), _tool("ghost")], tools)
    assert halves is None and "ghost" in (err or "")


def test_non_transfer_kind_rejected_in_split():
    tools = {"w": {"name": "w", "kind": "webhook", "destination": "",
                   "ring_timeout_sec": 20}}
    (halves, err) = split_handoff_order([_ai(), _tool("w")], tools)
    assert halves is None and "call_transfer" in (err or "")


# ── API-level tests (assign / unassign / reorder / idempotency) ───────

CALL_TRANSFER_PAYLOAD = {
    "name": "reception",
    "description": "Front desk transfer",
    "kind": "call_transfer",
    "destination": A,
    "ring_timeout_sec": 20,
}


def _fresh(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _make_tool(client, auth_token, dest=A, name="reception"):
    payload = {**CALL_TRANSFER_PAYLOAD, "destination": dest, "name": name}
    r = await client.post("/tools", json=payload,
                          headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


async def _make_agent(agent_id):
    from STT_server.db_agents import create_agent as db_create_agent
    db_create_agent("user-test-001", {"id": agent_id, "name": "Test"})


async def _halves(agent_id):
    from STT_server.db_agents import get_agent as _get_agent
    row = _get_agent(agent_id, "user-test-001")
    assert row is not None, f"agent {agent_id} not found"
    return row.get("transfer_cascade") or [], row.get("transfer_chain") or []


async def _put_order(client, auth_token, agent_id, order):
    r = await client.put(f"/agents/{agent_id}", json={"handoff_order": order},
                         headers={"Authorization": f"Bearer {auth_token}"})
    return r


async def test_B2_assign_appends_transfer_tool_exactly_once(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-b2")
    tool_id = await _make_tool(client, auth_token)
    await _make_agent(agent_id)
    r = await client.post(f"/agents/{agent_id}/tools/{tool_id}/assign",
                          headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    _, chain = await _halves(agent_id)
    assert chain == [tool_id]
    # canonical view: [AI, tool]
    cascade, chain = await _halves(agent_id)
    assert derive_handoff_order(cascade, chain) == [_ai(), _tool(tool_id)]


async def test_B13_assign_idempotent(client: AsyncClient, auth_token: str):
    agent_id = _fresh("agent-b13")
    tool_id = await _make_tool(client, auth_token)
    await _make_agent(agent_id)
    for _ in range(2):
        r = await client.post(
            f"/agents/{agent_id}/tools/{tool_id}/assign",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200, r.text
    _, chain = await _halves(agent_id)
    assert chain == [tool_id]


async def test_B3_unassign_removes_from_chain(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-b3")
    tool_id = await _make_tool(client, auth_token)
    await _make_agent(agent_id)
    await client.post(f"/agents/{agent_id}/tools/{tool_id}/assign",
                      headers={"Authorization": f"Bearer {auth_token}"})
    r = await client.delete(
        f"/agents/{agent_id}/tools/{tool_id}/assign",
        headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert chain == [] and cascade == []


async def test_B14_unassign_idempotent(client: AsyncClient, auth_token: str):
    agent_id = _fresh("agent-b14")
    tool_id = await _make_tool(client, auth_token)
    await _make_agent(agent_id)
    await client.post(f"/agents/{agent_id}/tools/{tool_id}/assign",
                      headers={"Authorization": f"Bearer {auth_token}"})
    for _ in range(2):
        r = await client.delete(
            f"/agents/{agent_id}/tools/{tool_id}/assign",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert chain == [] and cascade == []


async def test_reorder_crosses_AI_both_directions(
    client: AsyncClient, auth_token: str,
):
    """A,AI,B -> AI,A,B -> B,A,AI, each persisting across refetch."""
    agent_id = _fresh("agent-xcross")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    b = await _make_tool(client, auth_token, dest=B, name="tb")
    await _make_agent(agent_id)
    for tid in (a, b):
        await client.post(f"/agents/{agent_id}/tools/{tid}/assign",
                          headers={"Authorization": f"Bearer {auth_token}"})
    for order in (
        [_tool(a), _ai(), _tool(b)],
        [_ai(), _tool(a), _tool(b)],
        [_tool(b), _tool(a), _ai()],
        [_ai(), _tool(b), _tool(a)],
    ):
        r = await _put_order(client, auth_token, agent_id, order)
        assert r.status_code == 200, r.text
        cascade, chain = await _halves(agent_id)
        back = derive_handoff_order(cascade, chain)
        assert [n.get("tool_id", "ai") for n in back] == [
            n.get("tool_id", "ai") for n in order
        ], f"round-trip failed for {order}: got {back}"


async def test_B4_same_destination_unassign_keeps_sibling(
    client: AsyncClient, auth_token: str,
):
    """CRITICAL: A and B share +15550003333. Order A,AI,B; unassign A
    must remove exactly A's node; B survives with identity intact."""
    agent_id = _fresh("agent-b4")
    a = await _make_tool(client, auth_token, dest=SAME, name="sa")
    b = await _make_tool(client, auth_token, dest=SAME, name="sb")
    await _make_agent(agent_id)
    for tid in (a, b):
        r = await client.post(
            f"/agents/{agent_id}/tools/{tid}/assign",
            headers={"Authorization": f"Bearer {auth_token}"})
        assert r.status_code == 200, r.text
    r = await _put_order(client, auth_token, agent_id,
                         [_tool(a), _ai(), _tool(b)])
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert [c.get("tool_id") for c in cascade] == [a]
    assert chain == [b]
    # Unassign A.
    r = await client.delete(
        f"/agents/{agent_id}/tools/{a}/assign",
        headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [], f"A's cascade node must be gone, got {cascade}"
    assert chain == [b], f"sibling B must survive, got {chain}"
    assert derive_handoff_order(cascade, chain) == [_ai(), _tool(b)]


async def test_B10_API_duplicate_tool_rejected(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-b10")
    tool_id = await _make_tool(client, auth_token)
    await _make_agent(agent_id)
    await client.post(f"/agents/{agent_id}/tools/{tool_id}/assign",
                      headers={"Authorization": f"Bearer {auth_token}"})
    r = await _put_order(client, auth_token, agent_id,
                         [_tool(tool_id), _ai(), _tool(tool_id)])
    assert r.status_code == 400, r.text


async def test_B11_API_unassigned_tool_rejected(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-b11")
    tool_id = await _make_tool(client, auth_token)  # never assigned
    await _make_agent(agent_id)
    r = await _put_order(client, auth_token, agent_id,
                         [_ai(), _tool(tool_id)])
    assert r.status_code == 400, r.text


async def test_B12_API_AI_shape_rejected(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-b12")
    await _make_agent(agent_id)
    r = await _put_order(client, auth_token, agent_id, [_ai(), _ai()])
    assert r.status_code == 400, r.text
    r = await _put_order(client, auth_token, agent_id,
                         [{"type": "transfer_tool", "tool_id": "x"}])
    assert r.status_code == 400, r.text


async def test_raw_after_AI_rejected_over_API(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-rawafter")
    await _make_agent(agent_id)
    r = await _put_order(client, auth_token, agent_id,
                         [_ai(), _raw(A)])
    assert r.status_code == 400, r.text


async def test_create_with_tool_order_rejected(
    client: AsyncClient, auth_token: str,
):
    """Nothing can be assigned before the agent exists."""
    tool_id = await _make_tool(client, auth_token)
    r = await client.post(
        "/agents",
        json={"name": "x",
              "handoff_order": [_ai(), _tool(tool_id)]},
        headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 400, r.text


async def test_raw_entries_survive_unassign(
    client: AsyncClient, auth_token: str,
):
    """Unrelated raw rows belong to no tool: unassigning a transfer tool
    never removes them. (A same-destination raw would have been promoted
    to the tool at assign time — only genuinely unrelated raws remain.)"""
    agent_id = _fresh("agent-rawsurv")
    tool_id = await _make_tool(client, auth_token, dest=A)
    await _make_agent(agent_id)
    await client.post(f"/agents/{agent_id}/tools/{tool_id}/assign",
                      headers={"Authorization": f"Bearer {auth_token}"})
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [
            {"destination": B, "timeout_sec": 15},
        ], "transfer_chain": [tool_id]},
        headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    r = await client.delete(
        f"/agents/{agent_id}/tools/{tool_id}/assign",
        headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": B, "timeout_sec": 15}], cascade
    assert chain == [], chain


async def test_webhook_unassign_untouched(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-webhook")
    payload = {
        "name": "lookup",
        "description": "lookup",
        "kind": "webhook",
        "webhook_url": "https://n8n.example.com/wh",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    r = await client.post("/tools", json=payload,
                          headers={"Authorization": f"Bearer {auth_token}"})
    tool_id = r.json()["id"]
    await _make_agent(agent_id)
    await client.post(f"/agents/{agent_id}/tools/{tool_id}/assign",
                      headers={"Authorization": f"Bearer {auth_token}"})
    await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": B, "timeout_sec": 30}]},
        headers={"Authorization": f"Bearer {auth_token}"})
    await client.delete(
        f"/agents/{agent_id}/tools/{tool_id}/assign",
        headers={"Authorization": f"Bearer {auth_token}"})
    cascade, _ = await _halves(agent_id)
    assert cascade == [{"destination": B, "timeout_sec": 30}], cascade


async def _order_ids(agent_id):
    """Canonical order as [tool ids + 'ai'] after a fresh BE refetch."""
    cascade, chain = await _halves(agent_id)
    return [n.get("tool_id", "ai") for n in derive_handoff_order(cascade, chain)]


async def _tool_timeout(client, auth_token, agent_id, tool_id):
    r = await client.get(f"/agents/{agent_id}/tools",
                         headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    return next(t for t in r.json() if t["id"] == tool_id)["ring_timeout_sec"]


async def test_T1_single_tool_timeout_put_aligns_tool_and_cascade(
    client: AsyncClient, auth_token: str,
):
    """TIMEOUT ATOMICITY: one PUT /agents/{id}/tools/{tid} with
    ring_timeout_sec=37 must align BOTH the tool row and every
    materialized cascade snapshot — no second request needed."""
    agent_id = _fresh("agent-t1atomic")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    r = await _put_order(client, auth_token, agent_id, [_tool(a), _ai()])
    assert r.status_code == 200, r.text
    await _set_tool_timeout(client, auth_token, agent_id, a, 37)
    assert await _tool_timeout(client, auth_token, agent_id, a) == 37
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": A, "timeout_sec": 37,
                        "tool_id": a}], cascade
    assert chain == []


async def test_T1_reports_sync_status(client: AsyncClient, auth_token: str):
    agent_id = _fresh("agent-t1rep")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _put_order(client, auth_token, agent_id, [_tool(a), _ai()])
    r = await client.get(f"/agents/{agent_id}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body["ring_timeout_sec"] = 37
    r = await client.put(f"/agents/{agent_id}/tools/{a}", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["cascade_sync"] == {"updated": [agent_id], "failed": []}


async def test_T2_after_AI_timeout_single_put(
    client: AsyncClient, auth_token: str,
):
    """CASE 1: AI,A — no cascade involved; tool=37, nothing else moves."""
    agent_id = _fresh("agent-t2atomic")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _set_tool_timeout(client, auth_token, agent_id, a, 37)
    assert await _tool_timeout(client, auth_token, agent_id, a) == 37
    cascade, chain = await _halves(agent_id)
    assert cascade == [] and chain == [a]


async def test_T3_changing_A_never_changes_B(
    client: AsyncClient, auth_token: str,
):
    """CASE 3: A=11,B=43 in A,AI,B — A→31 leaves B=43 everywhere."""
    agent_id = _fresh("agent-t3atomic")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    b = await _make_tool(client, auth_token, dest=B, name="tb")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for tid in (a, b):
        await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
    await _set_tool_timeout(client, auth_token, agent_id, a, 11)
    await _set_tool_timeout(client, auth_token, agent_id, b, 43)
    r = await _put_order(client, auth_token, agent_id, [_tool(a), _ai(), _tool(b)])
    assert r.status_code == 200, r.text
    await _set_tool_timeout(client, auth_token, agent_id, a, 31)
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": A, "timeout_sec": 31, "tool_id": a}]
    assert chain == [b]
    assert await _tool_timeout(client, auth_token, agent_id, a) == 31
    assert await _tool_timeout(client, auth_token, agent_id, b) == 43


async def test_T4_sync_failure_keeps_tool_and_reports(
    client: AsyncClient, auth_token: str, monkeypatch,
):
    """Failure injection: cascade writes fail AFTER the tool row commits.
    The tool value stands (source of truth), the failure is reported
    (not silent), and the next halves-write heals the copy."""
    import STT_server.routes.api as api_mod
    agent_id = _fresh("agent-t4atomic")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _put_order(client, auth_token, agent_id, [_tool(a), _ai()])

    real_update = api_mod.db_update_agent

    def flaky(agent_id_, user_id_, payload):
        if isinstance(payload, dict) and "transfer_cascade" in payload:
            raise RuntimeError("injected cascade failure")
        return real_update(agent_id_, user_id_, payload)

    monkeypatch.setattr(api_mod, "db_update_agent", flaky)
    r = await client.get(f"/agents/{agent_id}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body["ring_timeout_sec"] = 37
    r = await client.put(f"/agents/{agent_id}/tools/{a}", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["cascade_sync"] == {"updated": [], "failed": [agent_id]}
    assert "partial" in " ".join(r.json()["change_log"])
    # source of truth committed; derived copy stale — reported, not hidden
    assert await _tool_timeout(client, auth_token, agent_id, a) == 37
    cascade, _ = await _halves(agent_id)
    assert cascade[0]["timeout_sec"] == 20

    # heal: next halves-write normalizes the copy to the tool row
    # (restore only our patch — undo() would also revert conftest's
    # auth/data_dir fixtures and every later call would 401).
    monkeypatch.setattr(api_mod, "db_update_agent", real_update)
    r = await client.put(f"/agents/{agent_id}",
                         json={"transfer_cascade": cascade}, headers=H)
    assert r.status_code == 200, r.text
    cascade, _ = await _halves(agent_id)
    assert cascade == [{"destination": A, "timeout_sec": 37, "tool_id": a}]


async def test_T5_shared_tool_syncs_all_agents(
    client: AsyncClient, auth_token: str,
):
    """CASE 4: ring_timeout_sec is GLOBAL on the shared row. One PUT
    aligns every assigned agent's snapshot — before-AI or after-AI."""
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    g1, g2 = _fresh("agent-t5a"), _fresh("agent-t5b")
    await _make_agent(g1)
    await _make_agent(g2)
    H = {"Authorization": f"Bearer {auth_token}"}
    for g in (g1, g2):
        await client.post(f"/agents/{g}/tools/{a}/assign", headers=H)
    await _put_order(client, auth_token, g1, [_tool(a), _ai()])
    await _put_order(client, auth_token, g2, [_ai(), _tool(a)])
    await _set_tool_timeout(client, auth_token, g1, a, 37)
    c1, ch1 = await _halves(g1)
    c2, ch2 = await _halves(g2)
    assert c1 == [{"destination": A, "timeout_sec": 37, "tool_id": a}]
    assert ch1 == []
    assert c2 == [] and ch2 == [a]
    # shared endpoint also syncs and keeps its bare-tool shape
    r = await client.get(f"/agents/{g1}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body["ring_timeout_sec"] = 41
    r = await client.put(f"/tools/{a}", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["ring_timeout_sec"] == 41  # bare shape kept
    c1, _ = await _halves(g1)
    assert c1 == [{"destination": A, "timeout_sec": 41, "tool_id": a}]


async def test_kind_flip_away_strips_handoff_links(    client: AsyncClient, auth_token: str,
):
    """Flipping call_transfer→webhook removes tool links from both halves
    instead of orphaning entries that would 400 future writes."""
    agent_id = _fresh("agent-flip")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _put_order(client, auth_token, agent_id, [_tool(a), _ai()])
    r = await client.get(f"/agents/{agent_id}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body.update({"kind": "webhook",
                 "webhook_url": "https://n8n.example.com/wh",
                 "destination": None})
    r = await client.put(f"/agents/{agent_id}/tools/{a}", json=body, headers=H)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [] and chain == [], (cascade, chain)


async def _set_tool_timeout(client, auth_token, agent_id, tool_id, timeout):
    """Full-body tool PUT (the endpoint validates a complete ToolCreate)."""
    r = await client.get(f"/agents/{agent_id}/tools",
                         headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    cur = next(t for t in r.json() if t["id"] == tool_id)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body["ring_timeout_sec"] = timeout
    r = await client.put(f"/agents/{agent_id}/tools/{tool_id}", json=body,
                         headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text


async def test_B1_assign_over_unique_legacy_raw_promotes_in_place(
    client: AsyncClient, auth_token: str,
):
    """B1: legacy cascade [RAW A] + assign tool A(dest A) promotes the raw
    IN PLACE to a tool-linked entry — one node, same position, identity
    becomes tool_id. The destination rings exactly once."""
    agent_id = _fresh("agent-b1")
    await _make_agent(agent_id)
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": SAME, "timeout_sec": 20}]},
        headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    a = await _make_tool(client, auth_token, dest=SAME, name="ta")
    r = await client.post(f"/agents/{agent_id}/tools/{a}/assign",
                          headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": SAME, "timeout_sec": 20,
                        "tool_id": a}], cascade
    assert chain == [], chain
    order = derive_handoff_order(cascade, chain)
    assert order == [{"type": "transfer_tool", "tool_id": a},
                     {"type": "ai"}], order
    # runtime materializes the destination exactly once
    resolved = resolve_cascade_steps(
        parse_cascade_with_ids(cascade),
        await _tools_by_id(client, auth_token, agent_id))
    assert [s["destination"] for s in resolved] == [SAME]


async def test_B2_contaminated_duplicate_normalizes_on_save(
    client: AsyncClient, auth_token: str,
):
    """B2: pre-existing contamination [RAW A, linked A] heals on the next
    handoff save — one tool-linked representation survives."""
    agent_id = _fresh("agent-b2")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    H = {"Authorization": f"Bearer {auth_token}"}
    # contaminate directly at the DB layer (legacy shape the old UI
    # produced): validated writes would already heal this, so seeding
    # must bypass them — exactly like the production rows did.
    from STT_server.db_agents import create_agent as db_create_agent
    db_create_agent("user-test-001", {
        "id": agent_id, "name": "Test",
        "transfer_cascade": [
            {"destination": A, "timeout_sec": 20},
            {"destination": A, "timeout_sec": 20, "tool_id": a},
        ],
        "transfer_chain": [],
    })
    # tool assignment row (as production has it)
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    cascade, _ = await _halves(agent_id)
    assert len(cascade) == 2  # contaminated as production shows
    # next canonical save consolidates (derived order re-sent, as FE does)
    nodes = ([{"type": "phone_destination", "destination": A, "timeout_sec": 20},
              {"type": "transfer_tool", "tool_id": a}, {"type": "ai"}])
    r = await _put_order(client, auth_token, agent_id, nodes)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": A, "timeout_sec": 20, "tool_id": a}]
    assert chain == []


async def test_B3_unassign_after_promotion_leaves_AI_only(
    client: AsyncClient, auth_token: str,
):
    """B3: promoted [Tool A, AI] + unassign A -> [AI]. The legacy raw
    was REPLACED at promotion, so nothing resurrects."""
    agent_id = _fresh("agent-b3promo")
    await _make_agent(agent_id)
    await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": SAME, "timeout_sec": 20}]},
        headers={"Authorization": f"Bearer {auth_token}"})
    a = await _make_tool(client, auth_token, dest=SAME, name="ta")
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    cascade, _ = await _halves(agent_id)
    assert any(c.get("tool_id") == a for c in cascade)
    r = await client.delete(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [], cascade
    assert chain == [], chain
    assert derive_handoff_order(cascade, chain) == [{"type": "ai"}]


async def test_T3_timeout_survives_AI_crossings(
    client: AsyncClient, auth_token: str,
):
    """Timeout lives on the tool row: A,AI -> AI,A -> A,AI keeps 37."""
    agent_id = _fresh("agent-t3")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _set_tool_timeout(client, auth_token, agent_id, a, 37)
    for order in ([_tool(a), _ai()], [_ai(), _tool(a)], [_tool(a), _ai()]):
        r = await _put_order(client, auth_token, agent_id, order)
        assert r.status_code == 200, r.text
        cascade, chain = await _halves(agent_id)
        if order[0].get("type") == "transfer_tool":
            assert cascade == [{"destination": A, "timeout_sec": 37,
                                "tool_id": a}], cascade
            assert chain == []
        else:
            assert cascade == []
            assert chain == [a]
    # tool row untouched by all the crossings
    r = await client.get(f"/agents/{agent_id}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    assert cur["ring_timeout_sec"] == 37


async def test_T4_distinct_timeouts_never_swap(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-t4")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    b = await _make_tool(client, auth_token, dest=B, name="tb")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for tid in (a, b):
        await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
    await _set_tool_timeout(client, auth_token, agent_id, a, 11)
    await _set_tool_timeout(client, auth_token, agent_id, b, 43)
    orders = [
        [_tool(a), _ai(), _tool(b)],
        [_ai(), _tool(b), _tool(a)],
        [_tool(b), _tool(a), _ai()],
        [_ai(), _tool(a), _tool(b)],
    ]
    for order in orders:
        r = await _put_order(client, auth_token, agent_id, order)
        assert r.status_code == 200, r.text
        cascade, chain = await _halves(agent_id)
        by_tool = {c["tool_id"]: c["timeout_sec"] for c in cascade if c.get("tool_id")}
        ai = next(i for i, n in enumerate(order) if n["type"] == "ai")
        for n in order[:ai]:
            assert by_tool[n["tool_id"]] == (11 if n["tool_id"] == a else 43)
        assert chain == [n["tool_id"] for n in order[ai + 1:]]


def test_T5_round_trip_preserves_metadata():
    """derive(split(order)) == order for every supported shape, full
    metadata (type / tool_id / destination / timeout / position)."""
    tools = _tools(("a", A), ("b", B))
    for t in tools.values():
        t["kind"] = "call_transfer"
    tools["a"]["ring_timeout_sec"] = 11
    tools["b"]["ring_timeout_sec"] = 43
    cases = [
        [_ai()],
        [_ai(), _tool("a")],
        [_tool("a"), _ai()],
        [_ai(), _tool("a"), _tool("b")],
        [_tool("a"), _ai(), _tool("b")],
        [_tool("a"), _tool("b"), _ai()],
        [_raw(A), _ai()],
        [_raw(A), _tool("a"), _ai()],
        [_raw(A), _ai(), _tool("a")],
        [_raw(A), _tool("a"), _ai(), _tool("b")],
    ]
    for order in cases:
        (halves, err) = split_handoff_order(order, tools)
        assert err is None, f"{order}: {err}"
        back = derive_handoff_order(*halves)
        assert back == order, f"round-trip failed: {order} -> {halves} -> {back}"
        # timeouts resolve from the tool rows, never drift
        cascade, chain = halves
        for c in cascade:
            if c.get("tool_id"):
                assert c["timeout_sec"] == tools[c["tool_id"]]["ring_timeout_sec"]
    # RAW after AI stays invalid (only raw restriction left)
    (halves, err) = split_handoff_order([_raw(A), _ai(), _tool("a"), _raw(B)], tools)
    assert halves is None and err is not None


async def test_T7_same_tool_both_halves_rejected(
    client: AsyncClient, auth_token: str,
):
    """A tool in cascade.tool_id AND chain is ambiguous — 400, no drift."""
    agent_id = _fresh("agent-t7")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": A, "timeout_sec": 20,
                                    "tool_id": a}],
              "transfer_chain": [a]},
        headers=H)
    assert r.status_code == 400, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [] and chain == [a]  # untouched by rejected write


async def test_T9_assign_preserves_existing_order(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-t9")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    b = await _make_tool(client, auth_token, dest=B, name="tb")
    c = await _make_tool(client, auth_token, dest=SAME, name="tc")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for tid in (a, b):
        await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
    r = await _put_order(client, auth_token, agent_id, [_tool(a), _ai(), _tool(b)])
    assert r.status_code == 200, r.text
    r = await client.post(f"/agents/{agent_id}/tools/{c}/assign", headers=H)
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == [a, "ai", b, c]


async def test_T10_unassign_preserves_relative_order(
    client: AsyncClient, auth_token: str,
):
    agent_id = _fresh("agent-t10")
    tools = {}
    for tid_name, dest in (("a", A), ("b", B), ("c", SAME)):
        tools[tid_name] = await _make_tool(client, auth_token, dest=dest,
                                           name=f"t{tid_name}")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for tid in tools.values():
        await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": "+19998887777",
                                    "timeout_sec": 20}]},
        headers=H)
    assert r.status_code == 200, r.text
    a, b, c = tools["a"], tools["b"], tools["c"]
    r = await _put_order(client, auth_token, agent_id,
                         [{"type": "phone_destination",
                           "destination": "+19998887777", "timeout_sec": 20},
                          _tool(a), _ai(), _tool(b), _tool(c)])
    assert r.status_code == 200, r.text
    r = await client.delete(f"/agents/{agent_id}/tools/{b}/assign", headers=H)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    back = derive_handoff_order(cascade, chain)
    assert [n.get("tool_id", n.get("destination", "ai")) for n in back] == [
        "+19998887777", a, "ai", c]
    r = await client.delete(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    back = derive_handoff_order(cascade, chain)
    assert [n.get("tool_id", n.get("destination", "ai")) for n in back] == [
        "+19998887777", "ai", c]


async def test_integration_scenario_full_product_flow(
    client: AsyncClient, auth_token: str,
):
    """REQUIRED integration scenario: assign A+B, free reorder across
    AI with reload after every step, unassign, re-assign. No
    duplicates, no orphans, no snapback."""
    agent_id = _fresh("agent-e2e")
    a = await _make_tool(client, auth_token, dest=A, name="ea")
    b = await _make_tool(client, auth_token, dest=B, name="eb")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}

    for tid in (a, b):
        r = await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
        assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == ["ai", a, b]  # AI,A,B

    r = await _put_order(client, auth_token, agent_id, [_tool(a), _ai(), _tool(b)])
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == [a, "ai", b]  # A,AI,B reload

    r = await _put_order(client, auth_token, agent_id, [_tool(b), _tool(a), _ai()])
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == [b, a, "ai"]  # B,A,AI reload

    r = await _put_order(client, auth_token, agent_id, [_ai(), _tool(b), _tool(a)])
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == ["ai", b, a]  # AI,B,A reload

    r = await client.delete(f"/agents/{agent_id}/tools/{b}/assign", headers=H)
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == ["ai", a]  # AI,A reload

    r = await client.post(f"/agents/{agent_id}/tools/{b}/assign", headers=H)
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == ["ai", a, b]  # AI,A,B, no dupes

    # runtime halves match the visible order: nothing before AI, both after
    cascade, chain = await _halves(agent_id)
    assert cascade == []
    assert chain == [a, b]


async def _tools_by_id(client, auth_token, agent_id):
    r = await client.get(f"/agents/{agent_id}/tools",
                         headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200, r.text
    return {t["id"]: t for t in r.json() if isinstance(t, dict) and t.get("id")}


def test_resolver_pure_contract():
    """resolve_cascade_steps: linked steps resolve from tool rows, raw
    steps pass through, unresolvable links fall back to snapshots."""
    tools = {"a": {"kind": "call_transfer", "destination": A,
                   "ring_timeout_sec": 37}}
    steps = [{"destination": A, "timeout_sec": 20, "tool_id": "a"},
             {"destination": B, "timeout_sec": 15}]
    assert resolve_cascade_steps(steps, tools) == [
        {"destination": A, "timeout_sec": 37, "tool_id": "a"},
        {"destination": B, "timeout_sec": 15},
    ]
    # missing tool / wrong kind / bad tool destination -> snapshot kept
    assert resolve_cascade_steps(
        [{"destination": A, "timeout_sec": 20, "tool_id": "ghost"}], tools
    ) == [{"destination": A, "timeout_sec": 20, "tool_id": "ghost"}]
    assert resolve_cascade_steps(
        [{"destination": A, "timeout_sec": 20, "tool_id": "w"}],
        {"w": {"kind": "webhook", "destination": "", "ring_timeout_sec": 5}},
    ) == [{"destination": A, "timeout_sec": 20, "tool_id": "w"}]
    assert resolve_cascade_steps(
        [{"destination": A, "timeout_sec": 20, "tool_id": "a"}],
        {"a": {"kind": "call_transfer", "destination": "bad",
               "ring_timeout_sec": 99}},
    ) == [{"destination": A, "timeout_sec": 20, "tool_id": "a"}]
    # total on garbage
    assert resolve_cascade_steps(None, None) == []
    assert resolve_cascade_steps("nope", "nope") == []


def test_resolver_distinguishes_missing_tool_from_failed_lookup():
    """Fallback semantics: a completed lookup with an absent id is a
    LEGITIMATELY missing tool (snapshot is the compatible fallback).
    A lookup that never completed (None) returns snapshots VERBATIM —
    the caller logs it as an explicit error, never as missing-tool."""
    steps = [{"destination": A, "timeout_sec": 20, "tool_id": "a"}]
    # completed lookup, tool gone -> per-step snapshot fallback
    assert resolve_cascade_steps(steps, {}) == steps
    # failed lookup -> verbatim degraded output (caller logs ERROR)
    out = resolve_cascade_steps(steps, None)
    assert out == steps
    assert out is not steps and out[0] is not steps[0]  # copies, no aliasing


async def test_T2_failure_runtime_still_canonical(
    client: AsyncClient, auth_token: str, monkeypatch,
):
    """T2: injected cascade-sync failure leaves tool=37/cascade=20
    persisted — but a NEW call resolves 37 (runtime never serves stale
    for tool-linked entries)."""
    import STT_server.routes.api as api_mod
    agent_id = _fresh("agent-t2rt")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _put_order(client, auth_token, agent_id, [_tool(a), _ai()])

    real_update = api_mod.db_update_agent

    def flaky(agent_id_, user_id_, payload):
        if isinstance(payload, dict) and "transfer_cascade" in payload:
            raise RuntimeError("injected cascade failure")
        return real_update(agent_id_, user_id_, payload)

    monkeypatch.setattr(api_mod, "db_update_agent", flaky)
    await _set_tool_timeout(client, auth_token, agent_id, a, 37)
    monkeypatch.setattr(api_mod, "db_update_agent", real_update)

    cascade, _ = await _halves(agent_id)
    assert cascade[0]["timeout_sec"] == 20  # stale snapshot persisted
    # what /voice would ring: resolve like STT_Server does
    resolved = resolve_cascade_steps(
        parse_cascade_with_ids(cascade),
        await _tools_by_id(client, auth_token, agent_id))
    assert resolved[0]["timeout_sec"] == 37
    assert resolved[0]["destination"] == A


async def test_T4_multi_agent_halfway_failure_no_partial_runtime(
    client: AsyncClient, auth_token: str, monkeypatch,
):
    """T4: shared tool on 3 agents, cascade write fails on the 2nd.
    Persisted copies diverge — runtime resolves 37 for ALL of them."""
    import STT_server.routes.api as api_mod
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    agents = [_fresh(f"agent-t4m{i}") for i in range(3)]
    for g in agents:
        await _make_agent(g)
    H = {"Authorization": f"Bearer {auth_token}"}
    for g in agents:
        await client.post(f"/agents/{g}/tools/{a}/assign", headers=H)
        await _put_order(client, auth_token, g, [_tool(a), _ai()])

    real_update = api_mod.db_update_agent

    def flaky(agent_id_, user_id_, payload):
        if (isinstance(payload, dict) and "transfer_cascade" in payload
                and agent_id_ == agents[1]):
            raise RuntimeError("injected failure on 2nd agent")
        return real_update(agent_id_, user_id_, payload)

    monkeypatch.setattr(api_mod, "db_update_agent", flaky)
    r = await client.get(f"/agents/{agents[0]}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body["ring_timeout_sec"] = 37
    r = await client.put(f"/agents/{agents[0]}/tools/{a}", json=body, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["cascade_sync"]["failed"] == [agents[1]]
    monkeypatch.setattr(api_mod, "db_update_agent", real_update)

    from STT_server.db_agents import get_agent as _get_agent
    rows = [_get_agent(g, "user-test-001") for g in agents]
    assert rows[0]["transfer_cascade"][0]["timeout_sec"] == 37
    assert rows[1]["transfer_cascade"][0]["timeout_sec"] == 20  # stale copy
    assert rows[2]["transfer_cascade"][0]["timeout_sec"] == 37
    # runtime: no partial state — every agent's new call rings 37
    for g in agents:
        tools = await _tools_by_id(client, auth_token, g)
        row = _get_agent(g, "user-test-001")
        resolved = resolve_cascade_steps(
            parse_cascade_with_ids(row["transfer_cascade"]), tools)
        assert resolved[0]["timeout_sec"] == 37, g


async def test_T5_raw_timeout_independent_of_tool_update(
    client: AsyncClient, auth_token: str,
):
    """T5: UNRELATED raw rows are their own source of truth — a tool
    timeout PUT never rewrites them."""
    agent_id = _fresh("agent-t5raw")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": B, "timeout_sec": 15}],
              "transfer_chain": [a]}, headers=H)
    await _set_tool_timeout(client, auth_token, agent_id, a, 37)
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": B, "timeout_sec": 15}], cascade
    assert chain == [a]
    resolved = resolve_cascade_steps(
        parse_cascade_with_ids(cascade),
        await _tools_by_id(client, auth_token, agent_id))
    assert resolved == [{"destination": B, "timeout_sec": 15}]


async def test_B4_runtime_rings_tool_once(
    client: AsyncClient, auth_token: str,
):
    """B4: normalized [Reception, AI] materializes the destination
    exactly once pre-AI (no raw+tool double ring)."""
    agent_id = _fresh("agent-b4rt")
    await _make_agent(agent_id)
    await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": SAME, "timeout_sec": 20}]},
        headers={"Authorization": f"Bearer {auth_token}"})
    a = await _make_tool(client, auth_token, dest=SAME, name="ta")
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    cascade, chain = await _halves(agent_id)
    resolved = resolve_cascade_steps(
        parse_cascade_with_ids(cascade),
        await _tools_by_id(client, auth_token, agent_id))
    assert [s["destination"] for s in resolved] == [SAME]
    assert chain == []


async def test_B5_reassign_never_duplicates(
    client: AsyncClient, auth_token: str,
):
    """B5: AI + assign A -> AI,A; unassign -> AI; assign -> AI,A."""
    agent_id = _fresh("agent-b5")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for _ in range(2):
        r = await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
        assert r.status_code == 200, r.text
        assert await _order_ids(agent_id) == ["ai", a]
        r = await client.delete(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
        assert r.status_code == 200, r.text
        assert await _order_ids(agent_id) == ["ai"]


async def test_B6_two_tools_same_destination_stay_distinct(
    client: AsyncClient, auth_token: str,
):
    """B6: A(X)+B(X), no raw — both assigned tools coexist as two tool
    nodes. No destination dedupe, ever."""
    agent_id = _fresh("agent-b6")
    a = await _make_tool(client, auth_token, dest=SAME, name="sa")
    b = await _make_tool(client, auth_token, dest=SAME, name="sb")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for tid in (a, b):
        r = await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
        assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == ["ai", a, b]
    r = await _put_order(client, auth_token, agent_id, [_tool(b), _ai(), _tool(a)])
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == [b, "ai", a]


async def test_B7_ambiguous_raw_never_promoted_nor_dropped(
    client: AsyncClient, auth_token: str,
):
    """B7: RAW X + assigned A(X) + assigned B(X) — destination cannot
    identify the owner, so the raw is preserved AND both tools persist.
    Documented coexistence (ambiguous case only)."""
    agent_id = _fresh("agent-b7")
    a = await _make_tool(client, auth_token, dest=SAME, name="sa")
    b = await _make_tool(client, auth_token, dest=SAME, name="sb")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    for tid in (a, b):
        await client.post(f"/agents/{agent_id}/tools/{tid}/assign", headers=H)
    # legacy halves write carrying all three: accepted as-is, no
    # destructive reconciliation
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": SAME, "timeout_sec": 20}],
              "transfer_chain": [a, b]}, headers=H)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": SAME, "timeout_sec": 20}], cascade
    assert chain == [a, b], chain
    # and assigning a THIRD same-dest tool appends (no promotion either)
    c = await _make_tool(client, auth_token, dest=SAME, name="sc")
    r = await client.post(f"/agents/{agent_id}/tools/{c}/assign", headers=H)
    assert r.status_code == 200, r.text
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": SAME, "timeout_sec": 20}], cascade
    assert chain == [a, b, c], chain


async def test_B8_unrelated_raw_accepted_and_kept(
    client: AsyncClient, auth_token: str,
):
    """B8: RAW Y (no tool claims it) + tool A(X) persist side by side."""
    agent_id = _fresh("agent-b8")
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": B, "timeout_sec": 15}]},
        headers=H)
    assert r.status_code == 200, r.text
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": B, "timeout_sec": 15}], cascade
    assert chain == [a], chain
    order = derive_handoff_order(cascade, chain)
    assert order[0] == {"type": "phone_destination", "destination": B,
                        "timeout_sec": 15}
    assert order[-1] == {"type": "transfer_tool", "tool_id": a}


async def test_B9_assign_preserves_raw_position(
    client: AsyncClient, auth_token: str,
):
    """B9: [RAW A, AI, B] + assign A -> [A, AI, B]. In-place promotion,
    never append-to-end."""
    agent_id = _fresh("agent-b9")
    b = await _make_tool(client, auth_token, dest=B, name="tb")
    await _make_agent(agent_id)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{b}/assign", headers=H)
    r = await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": A, "timeout_sec": 20}]},
        headers=H)
    assert r.status_code == 200, r.text
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    r = await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    assert r.status_code == 200, r.text
    assert await _order_ids(agent_id) == [a, "ai", b]


async def test_B10_promotion_keeps_canonical_timeout(
    client: AsyncClient, auth_token: str,
):
    """B10: raw timeout=20, tool ring_timeout_sec=37 -> promoted snapshot
    and runtime use 37. Never copied raw->tool."""
    agent_id = _fresh("agent-b10")
    await _make_agent(agent_id)
    await client.put(
        f"/agents/{agent_id}",
        json={"transfer_cascade": [{"destination": A, "timeout_sec": 20}]},
        headers={"Authorization": f"Bearer {auth_token}"})
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{agent_id}/tools/{a}/assign", headers=H)
    await _set_tool_timeout(client, auth_token, agent_id, a, 37)
    # move after AI and back to prove nothing was lost/reset
    await _put_order(client, auth_token, agent_id, [_ai(), _tool(a)])
    await _put_order(client, auth_token, agent_id, [_tool(a), _ai()])
    cascade, chain = await _halves(agent_id)
    assert cascade == [{"destination": A, "timeout_sec": 37, "tool_id": a}]
    assert chain == []
    assert await _tool_timeout(client, auth_token, agent_id, a) == 37
    resolved = resolve_cascade_steps(
        parse_cascade_with_ids(cascade),
        await _tools_by_id(client, auth_token, agent_id))
    assert resolved[0]["timeout_sec"] == 37


def test_T6_after_AI_chain_uses_tool_row():
    """T6: the after-AI path never had a snapshot — build_transfer_chain
    always materialized from tool rows."""
    tools = {"a": {"name": "A", "destination": A, "ring_timeout_sec": 37,
                   "kind": "call_transfer"}}
    got = build_transfer_chain("a", ["a"], tools)
    assert got[0]["timeout_sec"] == 37
    assert got[0]["destination"] == A


async def test_shared_row_survives_agent_route_update(
    client: AsyncClient, auth_token: str,
):
    """Timeout PUT via the agent route must NOT rewrite ownership or
    membership: agent_id stays __shared__, assignments intact."""
    import STT_server.db_tools as db_tools_mod
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    g1, g2 = _fresh("agent-own1"), _fresh("agent-own2")
    await _make_agent(g1)
    await _make_agent(g2)
    H = {"Authorization": f"Bearer {auth_token}"}
    for g in (g1, g2):
        await client.post(f"/agents/{g}/tools/{a}/assign", headers=H)
    await _set_tool_timeout(client, auth_token, g1, a, 37)
    row = db_tools_mod.get_tool(a, "user-test-001")
    assert row["agent_id"] == "__shared__", row["agent_id"]
    assert sorted(row["assignments"]) == sorted([g1, g2]), row["assignments"]


async def test_shared_route_update_preserves_ownership(
    client: AsyncClient, auth_token: str,
):
    """Same invariant via PUT /tools/{id}: edits never touch agent_id
    or assignments (assign flows own membership)."""
    import STT_server.db_tools as db_tools_mod
    a = await _make_tool(client, auth_token, dest=A, name="ta")
    g1 = _fresh("agent-owns1")
    await _make_agent(g1)
    H = {"Authorization": f"Bearer {auth_token}"}
    await client.post(f"/agents/{g1}/tools/{a}/assign", headers=H)
    r = await client.get(f"/agents/{g1}/tools", headers=H)
    cur = next(t for t in r.json() if t["id"] == a)
    body = {k: cur.get(k) for k in (
        "name", "description", "kind", "webhook_url", "destination",
        "parameters", "filler_phrase", "test_data_model",
        "integration_id", "action")}
    body["ring_timeout_sec"] = 41
    r = await client.put(f"/tools/{a}", json=body, headers=H)
    assert r.status_code == 200, r.text
    row = db_tools_mod.get_tool(a, "user-test-001")
    assert row["agent_id"] == "__shared__", row["agent_id"]
    assert row["assignments"] == [g1], row["assignments"]
    assert row["ring_timeout_sec"] == 41
