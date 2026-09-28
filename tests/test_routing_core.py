"""Exhaustive matrix for the pure routing core.

The decision "ring the next human, or start the AI, or resume the AI, or
end the call" is the only part of routing whose bugs reach the caller as
SILENCE instead of an error, so it lives in transfer_cascade.decide_routing
— free of FastAPI, Twilio, DB and provider SDKs — and is pinned here
without booting the app (STT_Server pulls openai/inworld/assemblyai/
webrtcvad at import, which is exactly why this suite exists).

Every one of these assertions pins CURRENT behaviour, including the
parts that look like bugs (duplicate destinations are NOT deduped, all
five DialCallStatus values advance identically). Changing any of them is
a behaviour change and belongs in its own ticket.
"""
import pytest

from STT_server.services.transfer_cascade import (
    ACTION_DIAL,
    ACTION_END_CALL,
    ACTION_RESUME_AI,
    ACTION_START_AI,
    DIAL_STATUS_MISSING,
    DIAL_STATUS_UNKNOWN,
    PHASE_POST_AI,
    PHASE_PRE_AI,
    RoutingStep,
    build_transfer_chain,
    decide_routing,
    normalize_dial_status,
    parse_cascade_with_ids,
    resolve_cascade_steps,
)

H1 = "+15550001111"
H2 = "+15550002222"
H3 = "+15550003333"

ALL_STATUSES = [
    "no-answer", "busy", "failed", "canceled", "completed",
    None, "", "   ", "weird-new-status", "COMPLETED", "No-Answer",
]
OUTCOMES = [H1, H2, H3]


def _steps(*dests, timeout=20):
    return [RoutingStep(destination=d, timeout_sec=timeout) for d in dests]


# ── status normalization ───────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("no-answer", "no-answer"),
    ("busy", "busy"),
    ("failed", "failed"),
    ("canceled", "canceled"),
    ("completed", "completed"),
    ("No-Answer", "no-answer"),
    ("  completed  ", "completed"),
    (None, DIAL_STATUS_MISSING),
    ("", DIAL_STATUS_MISSING),
    ("   ", DIAL_STATUS_MISSING),
    ("weird-new-status", DIAL_STATUS_UNKNOWN),
    ("in-progress", DIAL_STATUS_UNKNOWN),
    (123, DIAL_STATUS_UNKNOWN),
])
def test_normalize_dial_status(raw, expected):
    assert normalize_dial_status(raw) == expected


# ── the full status x phase x aliveness matrix ─────────────────────────────

@pytest.mark.parametrize("status", ALL_STATUSES)
@pytest.mark.parametrize("phase", [PHASE_PRE_AI, PHASE_POST_AI])
def test_every_dial_status_advances_identically(status, phase):
    """CURRENT BEHAVIOUR, PINNED: DialCallStatus carries no weight in the
    decision. no-answer, busy, failed, canceled and completed all advance
    to the next human. Whether `completed` *should* advance is a product
    question, not a code one — but it is not decided by this string."""
    d = decide_routing(phase, status, None, _steps(H1, H2))
    assert d.action == ACTION_DIAL
    assert d.destination == H1
    assert [s.destination for s in d.rest] == [H2]
    assert d.chosen_index == 0


@pytest.mark.parametrize("status", ALL_STATUSES)
def test_every_dial_status_exhausts_to_the_right_ai_flavour(status):
    """pre_ai START (caller never heard the AI); post_ai RESUME (they
    did, and the conversation comes back with them)."""
    assert decide_routing(PHASE_PRE_AI, status, None, []).action == ACTION_START_AI
    assert decide_routing(PHASE_POST_AI, status, None, []).action == ACTION_RESUME_AI


@pytest.mark.parametrize("status", ALL_STATUSES)
def test_caller_gone_never_dials_and_never_starts_ai(status):
    """THE ESSENTIAL RULE: once the caller is gone there is nobody to
    bridge to. No status, and no amount of pending humans, may resurrect
    the sequence — and critically this is decided by caller_alive, NOT by
    reading DialCallStatus."""
    for phase in (PHASE_PRE_AI, PHASE_POST_AI):
        d = decide_routing(phase, status, False, _steps(H1, H2, H3))
        assert d.action == ACTION_END_CALL
        assert d.reason == "caller_gone"
        assert d.destination is None
        assert d.rest == ()


def test_completed_with_caller_alive_vs_gone_are_independent():
    """`completed` alone must decide nothing. The same DialCallStatus
    yields Dial when the caller is alive and EndCall when they are not —
    which is the only way to tell A (leg ended, caller alive) apart from
    B (caller gone) without guessing."""
    alive = decide_routing(PHASE_POST_AI, "completed", True, _steps(H1))
    gone = decide_routing(PHASE_POST_AI, "completed", False, _steps(H1))
    assert alive.action == ACTION_DIAL
    assert gone.action == ACTION_END_CALL
    assert alive.dial_status == gone.dial_status == "completed"


def test_caller_alive_true_does_not_short_circuit():
    """True means "still connected", not "call finished"."""
    d = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1, H2))
    assert d.action == ACTION_DIAL
    assert d.destination == H1
    assert decide_routing(PHASE_POST_AI, "no-answer", True, []).action == ACTION_RESUME_AI


# ── exhaustion walk, the two product shapes ────────────────────────────────

def test_pre_ai_cascade_walks_to_exhaustion_then_starts_ai():
    """H1 fail -> H2 fail -> exhausted -> StartAI. Monotonic, terminates."""
    pending = _steps(H1, H2, H3)
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, pending)
    assert d.action == ACTION_DIAL and d.destination == H1
    pending = list(d.rest)

    for expected in (H2, H3):
        d = decide_routing(PHASE_PRE_AI, "no-answer", True, pending)
        assert d.action == ACTION_DIAL and d.destination == expected
        pending = list(d.rest)

    d = decide_routing(PHASE_PRE_AI, "no-answer", True, pending)
    assert d.action == ACTION_START_AI
    assert d.reason == "sequence_exhausted"


def test_post_ai_chain_walks_to_exhaustion_then_resumes_ai():
    """AI requests human -> suffix -> exhausted -> ResumeAI."""
    pending = _steps(H1, H2)
    d = decide_routing(PHASE_POST_AI, "no-answer", True, pending)
    assert d.destination == H1
    d = decide_routing(PHASE_POST_AI, "no-answer", True, list(d.rest))
    assert d.destination == H2
    d = decide_routing(PHASE_POST_AI, "no-answer", True, list(d.rest))
    assert d.action == ACTION_RESUME_AI


@pytest.mark.parametrize("cascade,chain,expect", [
    ([], [], "ai-first"),                      # no pre-AI humans
    ([H1], [H2], "ai-middle"),                 # humans on both sides
    ([H1], [], "ai-last"),                     # humans, then AI, no chain
])
def test_ai_position_variants(cascade, chain, expect):
    """AI-first / middle / last, derived from the two stored halves."""
    steps = parse_cascade_with_ids([{"destination": d} for d in cascade])
    chain_ids = [f"t{i}" for i in range(len(chain))]
    tools = {
        f"t{i}": {"id": f"t{i}", "kind": "call_transfer",
                  "destination": d, "ring_timeout_sec": 20}
        for i, d in enumerate(chain)
    }
    built = build_transfer_chain(chain_ids[0] if chain_ids else "", chain_ids, tools)

    if steps:
        first = decide_routing(PHASE_PRE_AI, "no-answer", True, steps)
        assert first.action == ACTION_DIAL
    else:
        assert decide_routing(PHASE_PRE_AI, "no-answer", True, steps).action == ACTION_START_AI

    if built:
        d = decide_routing(PHASE_POST_AI, "no-answer", True, built)
        assert d.action == ACTION_DIAL
    else:
        assert decide_routing(PHASE_POST_AI, "no-answer", True, built).action == ACTION_RESUME_AI
    assert expect in ("ai-first", "ai-middle", "ai-last")


def test_second_transfer_after_ai_resume_starts_a_fresh_pass():
    """After ResumeAI the AI may invoke another transfer. The core is
    stateless per call, so a second round is just another pass — it does
    NOT continue the old one and does not loop by itself."""
    round1 = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1, H2))
    assert round1.destination == H1
    round2 = decide_routing(PHASE_POST_AI, "no-answer", True, [round1])
    assert round2.action == ACTION_RESUME_AI


# ── malformed / hostile input: total, never raises ─────────────────────────

@pytest.mark.parametrize("pending", [
    None,
    [],
    (),
    [None],
    [None, None],
    [""],
    ["not-a-number"],
    [{}],
    [{"destination": ""}],
    [{"destination": None}],
    [{"destination": "5551234567"}],          # no +
    [{"destination": "+0155500011111"}],      # leading zero after +
    [{"destination": "+123"}],                # too short
    [{"destination": "+" + "1" * 20}],        # too long
    ["   "],
    [123],
    [object()],
    [{"destination": "not e164"}, RoutingStep(destination=H1)],
])
def test_malformed_pending_never_raises(pending):
    for phase in (PHASE_PRE_AI, PHASE_POST_AI):
        for status in ALL_STATUSES:
            for alive in (None, True, False):
                d = decide_routing(phase, status, alive, pending)
                assert d.action in (
                    ACTION_DIAL, ACTION_START_AI, ACTION_RESUME_AI, ACTION_END_CALL,
                )


def test_invalid_destination_is_skipped_not_dialed():
    """A junk entry must never be dialed, and must not abort the rest."""
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       ["garbage", {"destination": "+015550001111"},
                        RoutingStep(destination=H1)])
    assert d.action == ACTION_DIAL
    assert d.destination == H1
    assert d.chosen_index == 2, "chosen_index must index the INPUT list"
    assert d.rest == ()


def test_all_invalid_pending_falls_through_to_ai():
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, ["x", "", {}])
    assert d.action == ACTION_START_AI
    assert decide_routing(PHASE_POST_AI, "no-answer", True, ["x"]).action == ACTION_RESUME_AI


def test_malformed_status_and_phase_are_total():
    d = decide_routing(PHASE_PRE_AI, {"weird": "dict"}, None, _steps(H1))
    assert d.dial_status == DIAL_STATUS_UNKNOWN
    assert d.action == ACTION_DIAL
    # an unrecognised phase is treated as pre-AI (the safe default:
    # Start rather than Resume)
    assert decide_routing("nonsense", "no-answer", None, []).action == ACTION_START_AI
    assert decide_routing(None, None, None, []).action == ACTION_START_AI


# ── duplicate destinations: CURRENT behaviour, pinned ─────────────────────

def test_duplicate_destination_is_NOT_deduped():
    """CURRENT BEHAVIOUR, PINNED: nothing dedupes by destination. The
    same number can ring twice. `validate_transfer_chain` dedupes tool
    IDS on save, not numbers at runtime, and parse_cascade keeps every
    well-formed step. Changing this is a behaviour change."""
    pending = _steps(H1, H1, H2)
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, pending)
    assert d.destination == H1
    nxt = decide_routing(PHASE_PRE_AI, "no-answer", True, list(d.rest))
    assert nxt.destination == H1, "the duplicate still rings — pinned, not endorsed"


def test_same_phone_in_two_tools_rings_twice():
    """Two distinct tool ids, same E.164: the chain has two links."""
    tools = {
        "a": {"id": "a", "kind": "call_transfer", "destination": H1, "ring_timeout_sec": 20},
        "b": {"id": "b", "kind": "call_transfer", "destination": H1, "ring_timeout_sec": 30},
    }
    chain = build_transfer_chain("a", ["a", "b"], tools)
    assert [c["destination"] for c in chain] == [H1, H1]
    assert [c["id"] for c in chain] == ["a", "b"]
    d = decide_routing(PHASE_POST_AI, "no-answer", True, chain)
    assert d.destination == H1
    assert [s["id"] for s in d.rest] == ["b"]


# ── snapshot policy: cascade keeps, chain skips (INCONSISTENT, pinned) ─────

def test_snapshot_policy_pre_ai_deleted_tool_still_rings():
    """POLICY B (live resolution) with a snapshot fallback: a cascade
    step whose tool was deleted KEEPS the persisted destination, so the
    number still rings and the caller is not stranded."""
    steps = parse_cascade_with_ids([
        {"destination": H1, "timeout_sec": 20, "tool_id": "gone"},
        {"destination": H2, "timeout_sec": 25, "tool_id": "alive"},
    ])
    tools = {"alive": {"id": "alive", "kind": "call_transfer",
                       "destination": H3, "ring_timeout_sec": 45}}
    resolved = resolve_cascade_steps(steps, tools)
    assert [s["destination"] for s in resolved] == [H1, H3]
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, resolved)
    assert d.destination == H1, "deleted tool's snapshot survives"


def test_snapshot_policy_post_ai_deleted_tool_is_skipped():
    """The OPPOSITE policy: a chain link whose tool is gone simply does
    not exist, so that number stops ringing. Same operator action, same
    moment in the call, different outcome from the cascade. This is the
    documented inconsistency — pinned, NOT fixed here."""
    chain = build_transfer_chain("a", ["a", "deleted", "c"], {
        "a": {"id": "a", "kind": "call_transfer", "destination": H1, "ring_timeout_sec": 20},
        "c": {"id": "c", "kind": "call_transfer", "destination": H3, "ring_timeout_sec": 20},
    })
    assert [c["id"] for c in chain] == ["a", "c"]
    d = decide_routing(PHASE_POST_AI, "no-answer", True, chain)
    assert d.destination == H1
    nxt = decide_routing(PHASE_POST_AI, "no-answer", True, list(d.rest))
    assert nxt.destination == H3


def test_db_lookup_failure_returns_snapshots_verbatim():
    """tools_by_id=None means the lookup itself failed (not a missing
    tool). Snapshots come back untouched, so the call still routes."""
    steps = parse_cascade_with_ids([{"destination": H1, "timeout_sec": 33, "tool_id": "t"}])
    resolved = resolve_cascade_steps(steps, None)
    assert resolved == [{"destination": H1, "timeout_sec": 33, "tool_id": "t"}]
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, resolved)
    assert d.destination == H1 and d.timeout_sec == 33


def test_stale_snapshot_timeout_is_clamped():
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       [RoutingStep(destination=H1, timeout_sec=9999)])
    assert d.timeout_sec == 60
    # 0 is falsy, so it falls back to the default rather than to the
    # floor — same `int(x or DEFAULT)` shape the cascade/chain builders
    # use everywhere else.
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       [RoutingStep(destination=H1, timeout_sec=0)])
    assert d.timeout_sec == 20
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       [RoutingStep(destination=H1, timeout_sec=-3)])
    assert d.timeout_sec == 5
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       [RoutingStep(destination=H1, timeout_sec="nonsense")])
    assert d.timeout_sec == 20
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       [RoutingStep(destination=H1, timeout_sec=None)])
    assert d.timeout_sec == 20


# ── loop guards: THERE ARE NONE. Pinned so a future guard is visible. ──────

def test_no_loop_guard_exists_today():
    """A per-call round counter does NOT exist. Nothing in the core stops
    AI -> humans -> AI -> humans -> AI; the only brakes are the prompt
    note ("Do NOT immediately re-invoke a transfer tool") and
    MAX_CASCADE_STEPS / MAX_CHAIN_TOOLS bounding a SINGLE pass. The core
    is stateless and will happily return DIAL forever if fed the same
    pending list. Pinned so that adding a real guard is a visible,
    deliberate diff rather than an accidental one."""
    pending = _steps(H1, H2)
    actions = set()
    for _ in range(25):
        d = decide_routing(PHASE_POST_AI, "no-answer", True, pending)
        actions.add(d.action)
    assert actions == {ACTION_DIAL}, "no round limit is enforced by the core"


def test_scenarios_from_the_product_spec():
    """The two shapes the operator named, end to end through the core."""
    # H1 does not answer -> H2 does not answer -> AI takes the call.
    pending = _steps(H1, H2)
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, pending)
    assert (d.destination, d.action) == (H1, ACTION_DIAL)
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, list(d.rest))
    assert (d.destination, d.action) == (H2, ACTION_DIAL)
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, list(d.rest))
    assert d.action == ACTION_START_AI

    # AI answers first, caller asks for a human, nobody answers -> AI resumes.
    chain = build_transfer_chain("r", ["r", "s"], {
        "r": {"id": "r", "kind": "call_transfer", "destination": H1, "ring_timeout_sec": 20},
        "s": {"id": "s", "kind": "call_transfer", "destination": H2, "ring_timeout_sec": 20},
    })
    d = decide_routing(PHASE_POST_AI, "no-answer", True, chain)
    assert d.destination == H1
    d = decide_routing(PHASE_POST_AI, "no-answer", True, list(d.rest))
    assert d.destination == H2
    d = decide_routing(PHASE_POST_AI, "no-answer", True, list(d.rest))
    assert d.action == ACTION_RESUME_AI

    # A human answers (H1), the leg ends, the caller is still there:
    # advance to the next one rather than hanging up. H1's dial already
    # happened, so it is no longer pending.
    d = decide_routing(PHASE_PRE_AI, "completed", True, _steps(H2))
    assert d.action == ACTION_DIAL and d.destination == H2

    # The caller hung up mid-chain: stop, do not ring H2.
    d = decide_routing(PHASE_POST_AI, "completed", False, _steps(H1, H2))
    assert d.action == ACTION_END_CALL


# ── equivalence with the inline logic the core replaced ───────────────────
# The two callbacks used to decide inline. This ticket was extraction
# only, so the core MUST agree with that old code on every input it can
# actually receive. The references below are verbatim transcriptions of
# the pre-extraction branches, and the inputs are always parser/builder
# output — which is the only thing those routes ever passed in, and is
# why the core's extra E164 guard changes nothing.

def _legacy_cascade(steps, step):
    """Pre-extraction /voice/cascade branch."""
    try:
        idx = int(step)
    except (TypeError, ValueError):
        idx = len(steps)
    if 0 <= idx < len(steps):
        return ACTION_DIAL, steps[idx]["destination"], steps[idx]["timeout_sec"]
    return ACTION_START_AI, None, None


def _legacy_chain(chain):
    """Pre-extraction /voice/transfer-fallback branch."""
    if chain:
        return ACTION_DIAL, chain[0]["destination"], chain[0]["timeout_sec"]
    return ACTION_RESUME_AI, None, None


@pytest.mark.parametrize("status", ALL_STATUSES)
@pytest.mark.parametrize("raw_cascade", [
    [],
    [{"destination": H1}],
    [{"destination": H1}, {"destination": H2}],
    [{"destination": H1, "timeout_sec": 45}, {"destination": H2, "timeout_sec": 10},
     {"destination": H3, "timeout_sec": 60}],
    # malformed entries are dropped by the parser before the route sees them
    [{"destination": "junk"}, {"destination": H1}, "not-a-dict"],
    [{"destination": H1, "timeout_sec": 9999}],
])
@pytest.mark.parametrize("step", [0, 1, 2, 3, -1, 999, "2", "abc", None, ""])
def test_core_matches_legacy_cascade_branch(status, raw_cascade, step):
    steps = parse_cascade_with_ids(raw_cascade)
    pending = steps[int(step):] if _legacy_idx(steps, step) is not None else []
    d = decide_routing(PHASE_PRE_AI, status, None, pending)
    legacy = _legacy_cascade(steps, step)
    assert (d.action, d.destination, d.timeout_sec) == legacy


def _legacy_idx(steps, step):
    try:
        idx = int(step)
    except (TypeError, ValueError):
        idx = len(steps)
    return idx if 0 <= idx < len(steps) else None


@pytest.mark.parametrize("status", ALL_STATUSES)
@pytest.mark.parametrize("raw_chain,invoked", [
    ([], ""),
    (["a"], "a"),
    (["a", "b"], "a"),
    (["a", "b", "c"], "b"),
    (["missing", "b"], "missing"),
])
def test_core_matches_legacy_chain_branch(status, raw_chain, invoked):
    tools = {
        tid: {"id": tid, "kind": "call_transfer", "destination": d,
              "ring_timeout_sec": 20}
        for tid, d in zip(raw_chain, [H1, H2, H3, H1, H2])
    }
    chain = build_transfer_chain(invoked, raw_chain, tools)
    d = decide_routing(PHASE_POST_AI, status, None, chain)
    assert (d.action, d.destination, d.timeout_sec) == _legacy_chain(chain)


def test_core_is_a_superset_only_where_intended():
    """The ONE intended behavioural difference: the core can end the call
    when the caller is gone. Every route currently passes None, so this
    is dormant — it becomes real only when a route starts reading
    CallStatus. Asserted explicitly so the difference is documented
    rather than discovered in production."""
    steps = parse_cascade_with_ids([{"destination": H1}, {"destination": H2}])
    assert decide_routing(PHASE_PRE_AI, "completed", None, steps[1:]).action == ACTION_DIAL
    assert decide_routing(PHASE_PRE_AI, "completed", False, steps[1:]).action == ACTION_END_CALL
