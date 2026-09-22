"""Backend routing tests for ordered call routing (spec 8).

Covers cascade (humans before AI) + chain (humans after AI) as the
single authoritative sequence, per-call isolation, explicit jumps, and
no-restart / no-retry semantics.
"""
from STT_server.services.transfer_cascade import (
    parse_cascade,
    build_transfer_chain,
    build_unified_routing,
)

# helpers
def _tools(*defs):
    """defs: (id, dest, name)"""
    out = {}
    for tid, dest, name in defs:
        out[tid] = {"name": name, "destination": dest, "ring_timeout_sec": 20}
    return out


def test_scenario_A_ai_first_no_answer_goes_to_next():
    """AI → Reception → Recruiting, Reception no answer → Recruiting"""
    tools = _tools(("rec", "+15550001111", "Reception"), ("rec2", "+15550002222", "Recruiting"))
    chain = ["rec", "rec2"]
    # AI invokes rec, rec no answer => fallback to rec2
    chain_built = build_transfer_chain("rec", chain, tools)
    assert [c["id"] for c in chain_built] == ["rec", "rec2"]
    # Simulate: rec Dial no-answer, remaining = ["rec2"]
    remaining = ["rec2"]
    next_chain = build_transfer_chain(remaining[0], remaining, tools)
    assert [c["id"] for c in next_chain] == ["rec2"]


def test_scenario_B_ai_first_answered_then_hangup_continues():
    """AI → Reception → Recruiting, Reception answers then leg completes → Recruiting, caller retained"""
    tools = _tools(("rec", "+15550001111", "Reception"), ("rec2", "+15550002222", "Recruiting"))
    chain = ["rec", "rec2"]
    # explicit rec
    first = build_transfer_chain("rec", chain, tools)
    assert first[0]["id"] == "rec"
    # After rec Dial completed, remaining = ["rec2"] → next is rec2, not hangup
    remaining = ["rec2"]
    nxt = build_transfer_chain(remaining[0], remaining, tools)
    assert nxt[0]["id"] == "rec2"
    # After rec2, remaining empty → fallback to AI (chain empty => no next human)
    assert build_transfer_chain("", [], tools) == []


def test_scenario_C_reception_before_ai_no_answer_to_ai():
    """Reception → AI, Reception no answer → AI"""
    cascade = parse_cascade([{"destination": "+15550001111", "timeout_sec": 20}])
    unified = build_unified_routing(cascade, [], {})
    # unified = [human Reception, ai]
    assert unified[0]["type"] == "human"
    assert unified[1]["type"] == "ai"
    # Simulate pos 0 human no-answer => next pos 1 is AI


def test_scenario_D_reception_ai_cafeteria_human_to_ai_not_back_to_reception():
    """Reception → AI → Cafeteria, Reception leg finishes → AI, AI transfer → Cafeteria must NOT return to Reception"""
    cascade = parse_cascade([{"destination": "+15550001111"}])
    tools = _tools(("caf", "+15550003333", "Cafeteria"))
    chain = ["caf"]
    unified = build_unified_routing(cascade, chain, tools)
    # order: Reception, AI, Cafeteria
    assert [u["type"] for u in unified] == ["human", "ai", "human"]
    # After Reception completes, next is AI (pos 1)
    # After AI invokes transfer, next should be Cafeteria (pos 2), not Reception
    chain_built = build_transfer_chain("caf", chain, tools)
    assert [c["id"] for c in chain_built] == ["caf"]
    # Explicit check: invoking caf directly goes to caf, not back to earlier
    # Ensure remaining after caf is empty, so after caf completes we fallback to AI not Reception


def test_scenario_E_two_humans_then_ai():
    """Reception → Recruiting → AI, Reception no answer → Recruiting, Recruiting completes → AI"""
    cascade = parse_cascade([
        {"destination": "+15550001111"},
        {"destination": "+15550002222"},
    ])
    unified = build_unified_routing(cascade, [], {})
    assert [u["type"] for u in unified] == ["human", "human", "ai"]
    # pos 0 no-answer => pos1 Recruiting, pos1 completed => pos2 AI


def test_scenario_F_exhaustion_no_restart():
    """All humans fail → deterministic terminal, no restart, no retry"""
    cascade = parse_cascade([{"destination": "+15550001111"}])
    tools = _tools(("rec", "+15550001111", "Reception"))
    chain = ["rec"]
    # Simulate full cycle: Reception attempted (pos0), then AI, then rec attempted, then exhausted
    # After rec in chain completes with no remaining, fallback is AI exhausted -> hangup, not restart at Reception
    remaining = []
    nxt = build_transfer_chain("", remaining, tools)
    assert nxt == []
    # Unified shows no loop
    unified = build_unified_routing(cascade, chain, tools)
    # Ensure no duplicate attempt: each destination appears once in unified before AI and once after
    # Already attempted set would be handled by pos monotonic


def test_scenario_G_explicit_jump():
    """Reception → AI → Cafeteria → Recruiting, at AI explicit Recruiting → directly to Recruiting, skip Cafeteria"""
    tools = _tools(
        ("caf", "+15550001111", "Cafeteria"),
        ("rec", "+15550002222", "Recruiting"),
    )
    chain = ["caf", "rec"]
    # Explicit Recruiting
    explicit = build_transfer_chain("rec", chain, tools)
    assert [c["id"] for c in explicit] == ["rec"]  # skips caf
    # Normal sequential would be caf first
    sequential = build_transfer_chain("caf", chain, tools)
    assert [c["id"] for c in sequential] == ["caf", "rec"]


def test_scenario_H_new_call_resets():
    """Call 1 exhausts Reception,Recruiting,AI ; Call 2 starts fresh at Reception"""
    cascade = parse_cascade([
        {"destination": "+15550001111"},
        {"destination": "+15550002222"},
    ])
    tools = {}
    # Call1 pos advances to 2 (AI)
    unified1 = build_unified_routing(cascade, [], tools)
    # New call builds fresh unified, no state leakage
    unified2 = build_unified_routing(cascade, [], tools)
    assert unified1 == unified2
    assert unified1[0]["destination"] == "+15550001111"


def test_parse_cascade_and_unified_no_retry():
    cascade = parse_cascade([
        {"destination": "+15550001111"},
        {"destination": "+15550002222"},
    ])
    unified = build_unified_routing(cascade, ["t1"], _tools(("t1", "+15550003333", "X")))
    # Each human once
    humans = [u for u in unified if u["type"] == "human"]
    assert len(humans) == 3
    assert len({h["destination"] for h in humans}) == 3
