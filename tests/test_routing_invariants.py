"""Runtime invariants closed in this ticket.

Three things the routing core could not decide for itself:

  1. PARENT CALLER LIVENESS — Twilio sends DialCallStatus (the dialed
     leg) and CallStatus (the parent call) as independent fields. Only
     the second one says whether there is still a caller to hand the
     call to. caller_alive=False must never produce a Dial, a StartAI or
     a ResumeAI.

  2. POLICY A (immutable in-flight plan) — once a call's destinations are
     frozen, a tool delete, a tool edit, an assignment change or a DB
     outage must not move THAT call. The next call reads new config.

  3. LOOP GUARDS — a pre-AI cascade is not a handoff round (no AI was
     involved) but it does burn dial attempts; both caps are sticky and
     a departed caller outranks both.

Plus the sealed-plan codec (encrypt -> URL -> decrypt) round-trip.
"""
import pytest

from STT_server.services.call_plan import open_call_plan, seal_call_plan
from STT_server.services.transfer_cascade import (
    ACTION_DIAL,
    ACTION_END_CALL,
    ACTION_RESUME_AI,
    ACTION_START_AI,
    MAX_HANDOFF_ROUNDS_PER_CALL,
    MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL,
    PHASE_POST_AI,
    PHASE_PRE_AI,
    RoutingLimits,
    RoutingStep,
    advance_plan,
    caller_alive_from_call_status,
    decide_routing,
    plan_from_steps,
)

H1 = "+15550001111"
H2 = "+15550002222"
H3 = "+15550003333"
LIMITS = RoutingLimits(max_handoff_rounds=3, max_human_dial_attempts=20)


def _steps(*dests, timeout=20):
    return [RoutingStep(destination=d, timeout_sec=timeout, label="x") for d in dests]


# ── 1. parent caller liveness ─────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("initiated", True), ("ringing", True), ("in-progress", True),
    ("In-Progress", True), ("  ringing  ", True),
    ("completed", False), ("busy", False), ("failed", False),
    ("no-answer", False), ("canceled", False), ("COMPLETED", False),
    # missing / unknown MUST stay None, never collapse to False
    (None, None), ("", None), ("   ", None),
    ("some-new-twilio-status", None), ("queued", None), (123, None),
])
def test_caller_alive_from_call_status(raw, expected):
    assert caller_alive_from_call_status(raw) is expected


@pytest.mark.parametrize("status", ["no-answer", "busy", "failed", "canceled", "completed"])
def test_absolute_invariant_caller_gone_never_advances(status):
    """caller_alive=False can ONLY produce END_CALL — never dial, never
    start_ai, never resume_ai, with or without pending humans, with or
    without a limit already spent."""
    for phase in (PHASE_PRE_AI, PHASE_POST_AI):
        for pending in ([], _steps(H1), _steps(H1, H2, H3)):
            for rounds, attempts in ((0, 0), (99, 99)):
                d = decide_routing(phase, status, False, pending,
                                   rounds_used=rounds, dial_attempts=attempts,
                                   limits=LIMITS)
                assert d.action == ACTION_END_CALL
                assert d.destination is None
                assert d.rest == ()
                assert d.handoff_disabled is True


def test_completed_plus_parent_active_advances():
    """A human answered, the leg ended, the caller is still there: the
    call advances. H1 is no longer pending, so pending starts at H2."""
    d = decide_routing(PHASE_PRE_AI, "completed", True, _steps(H2), limits=LIMITS)
    assert d.action == ACTION_DIAL
    assert d.destination == H2


def test_completed_plus_parent_terminal_ends():
    d = decide_routing(PHASE_PRE_AI, "completed", False, _steps(H1, H2), limits=LIMITS)
    assert d.action == ACTION_END_CALL


def test_no_answer_plus_parent_terminal_ends():
    d = decide_routing(PHASE_POST_AI, "no-answer", False, _steps(H1, H2), limits=LIMITS)
    assert d.action == ACTION_END_CALL


def test_missing_call_status_keeps_previous_behaviour():
    """None aliveness => the historical advance-on-everything. An
    unrecognised Twilio status must not start ending live calls."""
    for phase in (PHASE_PRE_AI, PHASE_POST_AI):
        for raw in (None, "", "brand-new-status"):
            alive = caller_alive_from_call_status(raw)
            assert alive is None
            d = decide_routing(phase, "no-answer", alive, _steps(H1, H2), limits=LIMITS)
            assert d.action == ACTION_DIAL


def test_dial_status_alone_never_decides_liveness():
    """The two fields are orthogonal: the same DialCallStatus yields
    different outcomes purely on the parent status."""
    for dial_status in ("completed", "no-answer", "busy", ""):
        alive = decide_routing(PHASE_POST_AI, dial_status, True, _steps(H1), limits=LIMITS)
        gone = decide_routing(PHASE_POST_AI, dial_status, False, _steps(H1), limits=LIMITS)
        unknown = decide_routing(PHASE_POST_AI, dial_status, None, _steps(H1), limits=LIMITS)
        assert alive.action == ACTION_DIAL
        assert gone.action == ACTION_END_CALL
        assert unknown.action == ACTION_DIAL
        assert alive.dial_status == gone.dial_status == unknown.dial_status


# ── 2. Policy A: the in-flight plan is immutable ─────────────────────────

def test_plan_survives_tool_deletion():
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2))
    # tool rows are gone entirely; the plan does not consult them again
    assert [s.destination for s in plan.steps] == [H1, H2]
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, plan.steps, limits=LIMITS)
    assert d.destination == H1


def test_plan_survives_tool_edit():
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2))
    # an operator edits the tool to point somewhere else; the plan is
    # already frozen so the call keeps its original targets
    edited = [RoutingStep(destination=H3, timeout_sec=45)]
    assert [s.destination for s in plan.steps] == [H1, H2]
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, plan.steps, limits=LIMITS)
    assert d.destination == H1
    # only the NEXT call sees the edit
    fresh = plan_from_steps(PHASE_PRE_AI, edited)
    assert fresh.steps[0].destination == H3


def test_first_call_old_config_next_call_new_config():
    """The in-flight call rings the old plan; the next call rings the new
    config. That is the whole point of Policy A."""
    old_config = _steps(H1)
    new_config = _steps(H3)
    in_flight = plan_from_steps(PHASE_PRE_AI, old_config)
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, in_flight.steps, limits=LIMITS)
    assert d.destination == H1
    next_call = plan_from_steps(PHASE_PRE_AI, new_config)
    d2 = decide_routing(PHASE_PRE_AI, "no-answer", True, next_call.steps, limits=LIMITS)
    assert d2.destination == H3


def test_plan_survives_db_disappearing():
    """No DB read happens on the plan path at all: the steps came off the
    sealed token. A dead DB is a non-event for an in-flight call."""
    plan = plan_from_steps(PHASE_POST_AI, _steps(H1, H2))
    for hypothetical_db_dead in (True, False):   # nothing reads it
        d = decide_routing(PHASE_POST_AI, "no-answer", True, plan.steps, limits=LIMITS)
        assert d.destination == H1


def test_plan_preserves_full_step_metadata():
    steps = [RoutingStep(destination=H1, timeout_sec=33, tool_id="t-a", label="Recepcion")]
    plan = plan_from_steps(PHASE_POST_AI, steps)
    s = plan.steps[0]
    assert (s.destination, s.timeout_sec, s.tool_id, s.label) == (H1, 33, "t-a", "Recepcion")


def test_plan_drops_invalid_destinations_at_freeze_time():
    plan = plan_from_steps(PHASE_PRE_AI, ["junk", None, {"destination": H1}, "also junk"])
    assert [s.destination for s in plan.steps] == [H1]


# ── 3. loop guards ────────────────────────────────────────────────────────

@pytest.mark.parametrize("rounds_used", [0, 1, 2])
def test_rounds_below_limit_still_dial(rounds_used):
    d = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1, H2),
                       rounds_used=rounds_used, dial_attempts=0, limits=LIMITS)
    assert d.action == ACTION_DIAL


def test_round_limit_plus_one_is_blocked():
    d = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1),
                       rounds_used=LIMITS.max_handoff_rounds, dial_attempts=0,
                       limits=LIMITS)
    assert d.action == ACTION_RESUME_AI, "AI already existed -> resume it"
    assert d.reason == "limit_reached"
    assert d.handoff_disabled is True


def test_pre_ai_is_governed_by_the_dial_cap_not_the_round_cap():
    """A pre-AI cascade contains no AI, so it consumes no handoff rounds
    and the round cap cannot apply to it — rounds_used is 0 there by
    construction. What stops a runaway cascade is the DIAL cap, and when
    that fires the AI has not spoken yet, hence StartAI."""
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, _steps(H1),
                       rounds_used=0, dial_attempts=0,
                       limits=RoutingLimits(max_handoff_rounds=0,
                                            max_human_dial_attempts=20))
    assert d.action == ACTION_DIAL, "round cap must not strand a pre-AI cascade"
    d2 = decide_routing(PHASE_PRE_AI, "no-answer", True, _steps(H1),
                        rounds_used=0, dial_attempts=20,
                        limits=RoutingLimits(max_handoff_rounds=0,
                                             max_human_dial_attempts=20))
    assert d2.action == ACTION_START_AI
    assert d2.handoff_disabled is True


@pytest.mark.parametrize("attempts", [0, 1, 10, 19])
def test_dial_attempts_below_limit_still_dial(attempts):
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, _steps(H1, H2),
                       rounds_used=0, dial_attempts=attempts, limits=LIMITS)
    assert d.action == ACTION_DIAL


def test_dial_attempt_limit_plus_one_is_blocked():
    d = decide_routing(PHASE_PRE_AI, "no-answer", True, _steps(H1),
                       rounds_used=0,
                       dial_attempts=LIMITS.max_human_dial_attempts, limits=LIMITS)
    assert d.action == ACTION_START_AI
    assert d.handoff_disabled is True
    d2 = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1),
                        rounds_used=0,
                        dial_attempts=LIMITS.max_human_dial_attempts, limits=LIMITS)
    assert d2.action == ACTION_RESUME_AI


def test_limits_combine_pre_ai_and_post_ai_attempts():
    """A pre-AI cascade burns dial attempts; the chain that follows sees
    the running total, so the combined spend is capped."""
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2, H3))
    assert plan.dial_attempts == 0 and plan.rounds_used == 0

    d = decide_routing(PHASE_PRE_AI, "no-answer", True, plan.steps, limits=LIMITS)
    plan = advance_plan(plan, d)
    assert plan.dial_attempts == 1, "pre-AI dials count as attempts"
    assert plan.rounds_used == 0, "but NOT as handoff rounds (no AI involved)"

    # hand the running total to the post-AI phase
    chain = plan_from_steps(PHASE_POST_AI, _steps(H1), "CA1")
    carried = chain.__class__(**{**chain.__dict__,
                                 "dial_attempts": plan.dial_attempts})
    d = decide_routing(PHASE_POST_AI, "no-answer", True, carried.steps,
                       dial_attempts=carried.dial_attempts, limits=LIMITS)
    assert d.action == ACTION_DIAL
    after = advance_plan(carried, d)
    assert after.dial_attempts == 2
    assert after.rounds_used == 1, "the post-AI dial DOES consume a round"


def test_handoff_disabled_is_sticky():
    """Once set it stays set for the rest of the call, even if a later
    hop somehow still had pending humans and fresh counters."""
    d = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1, H2),
                       rounds_used=0, dial_attempts=0,
                       handoff_disabled=True, limits=LIMITS)
    assert d.action == ACTION_RESUME_AI
    assert d.handoff_disabled is True
    assert d.destination is None


def test_limit_reached_beats_pending_destinations():
    """Guard precedence: exhausted budget wins over "there is still a
    human to ring". We never dial past the cap."""
    d = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1, H2, H3),
                       rounds_used=99, dial_attempts=0, limits=LIMITS)
    assert d.action == ACTION_RESUME_AI
    assert d.reason == "limit_reached"
    assert d.destination is None


def test_caller_hangup_outranks_every_other_decision():
    """Belt and braces: gone beats the guard, beats exhaustion, beats
    pending humans."""
    d = decide_routing(PHASE_POST_AI, "completed", False, _steps(H1, H2, H3),
                       rounds_used=99, dial_attempts=99,
                       handoff_disabled=True, limits=LIMITS)
    assert d.action == ACTION_END_CALL
    assert d.reason == "caller_gone"


def test_limits_do_not_apply_to_pre_ai_rounds():
    """A pre-AI cascade has rounds_used=0 by construction, so the round
    cap can never strand a fresh cascade."""
    for limit in (0, 1, 3, 99):
        d = decide_routing(PHASE_PRE_AI, "no-answer", True, _steps(H1),
                           rounds_used=0, dial_attempts=0,
                           limits=RoutingLimits(max_handoff_rounds=limit,
                                                max_human_dial_attempts=20))
        assert d.action == ACTION_DIAL


def test_defaults_match_the_documented_caps():
    assert MAX_HANDOFF_ROUNDS_PER_CALL == 3
    assert MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL == 20
    assert RoutingLimits().max_handoff_rounds == 3
    assert RoutingLimits().max_human_dial_attempts == 20


def test_garbage_counters_are_total_not_crashing():
    for bad in ("x", None, object(), [1]):
        d = decide_routing(PHASE_POST_AI, "no-answer", True, _steps(H1),
                           rounds_used=bad, dial_attempts=bad, limits=LIMITS)
        assert d.action == ACTION_DIAL


# ── sealed plan round-trip (Policy A over a real hop) ─────────────────────

def test_sealed_plan_round_trip_preserves_everything():
    plan = plan_from_steps(
        PHASE_POST_AI,
        [RoutingStep(destination=H1, timeout_sec=25, tool_id="t-a", label="Recepcion"),
         RoutingStep(destination=H2, timeout_sec=30, tool_id="t-b", label="Cafeteria")],
        "CA-round-trip",
    )
    plan = plan.__class__(**{**plan.__dict__, "rounds_used": 2, "dial_attempts": 5,
                             "handoff_disabled": True})
    token = seal_call_plan(plan)
    assert token, "sealing must produce a token"
    back = open_call_plan(token)
    assert back is not None
    assert back.phase == plan.phase
    assert back.call_sid == "CA-round-trip"
    assert back.rounds_used == 2
    assert back.dial_attempts == 5
    assert back.handoff_disabled is True
    assert [s.destination for s in back.steps] == [H1, H2]
    assert [s.timeout_sec for s in back.steps] == [25, 30]
    assert [s.tool_id for s in back.steps] == ["t-a", "t-b"]
    assert [s.label for s in back.steps] == ["Recepcion", "Cafeteria"]


def test_sealed_plan_hides_destinations_in_the_token():
    """The whole point of sealing: no E.164 readable in the URL, so it
    cannot leak into an access log, a trace or Twilio's request log."""
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2, H3), "CA-x")
    token = seal_call_plan(plan)
    assert H1 not in token and H2 not in token and H3 not in token
    assert "1555000" not in token


def test_sealed_plan_drives_a_real_hop_without_touching_config():
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2, H3), "CA-hop")
    d1 = decide_routing(PHASE_PRE_AI, "no-answer", True, plan.steps, limits=LIMITS)
    assert d1.destination == H1
    sealed = seal_call_plan(advance_plan(plan, d1))
    # next hop: config is entirely out of the picture
    plan2 = open_call_plan(sealed)
    d2 = decide_routing(PHASE_PRE_AI, "busy", True, plan2.steps, limits=LIMITS)
    assert d2.destination == H2
    plan3 = advance_plan(plan2, d2)
    assert plan3.dial_attempts == 2
    sealed2 = seal_call_plan(plan3)
    plan4 = open_call_plan(sealed2)
    d3 = decide_routing(PHASE_PRE_AI, "no-answer", True, plan4.steps, limits=LIMITS)
    assert d3.destination == H3


def test_exhausted_chain_over_a_real_sealed_hop_reaches_ai():
    plan = plan_from_steps(PHASE_POST_AI, _steps(H1), "CA-exh")
    d = decide_routing(PHASE_POST_AI, "no-answer", True, plan.steps, limits=LIMITS)
    final = advance_plan(plan, d)
    assert final.steps == (), "human phase is over"
    sealed = seal_call_plan(final)
    back = open_call_plan(sealed)
    assert back.dial_attempts == 1 and back.rounds_used == 1
    d2 = decide_routing(PHASE_POST_AI, "no-answer", True, back.steps,
                        rounds_used=back.rounds_used,
                        dial_attempts=back.dial_attempts, limits=LIMITS)
    assert d2.action == ACTION_RESUME_AI


@pytest.mark.parametrize("token", [None, "", "not-a-token", "!!!!", 42, b"bytes"])
def test_open_call_plan_degrades_to_none(token):
    """Any junk -> None -> the route falls back to live config, which is
    the behaviour in production today. Never an exception on a live
    call path."""
    assert open_call_plan(token) is None


def test_seal_empty_plan_returns_empty_token():
    assert seal_call_plan(plan_from_steps(PHASE_PRE_AI, [], "CA")) == ""
    assert seal_call_plan(None) == ""


def test_undecryptable_token_degrades_to_legacy_path():
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1))
    token = seal_call_plan(plan)
    # flip a few characters: Fernet authentication must reject it
    tampered = ("A" if token[10] != "A" else "B") + token[11:]
    assert open_call_plan(tampered) is None


# ── the sealed state as a hostile surface ─────────────────────────────────
# Fernet proves the blob is INTACT and CONFIDENTIAL. It does NOT prove
# the blob belongs to THIS call, so the plan is bound to the CallSid
# explicitly. A valid token from call A must not route call B to A's
# numbers.

def test_valid_sealed_plan_is_accepted_for_its_own_call():
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2), "CA-OWN")
    assert open_call_plan(seal_call_plan(plan), "CA-OWN") is not None


def test_token_from_a_different_call_is_rejected():
    plan = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2), "CA-CALL-A")
    token = seal_call_plan(plan)
    # same token, different call -> refused, route falls back to config
    assert open_call_plan(token, "CA-CALL-B") is None
    # and the real owner still gets it
    assert open_call_plan(token, "CA-CALL-A") is not None


def test_binding_tolerates_unbound_plan_and_unknown_caller():
    """A plan sealed before binding existed, or a webhook with no
    CallSid, must still work: the check only rejects a CONTRADICTION."""
    unbound = plan_from_steps(PHASE_PRE_AI, _steps(H1), "")
    assert open_call_plan(seal_call_plan(unbound), "CA-ANY") is not None
    bound = plan_from_steps(PHASE_PRE_AI, _steps(H1), "CA-X")
    assert open_call_plan(seal_call_plan(bound), "") is not None
    assert open_call_plan(seal_call_plan(bound), None) is not None


def test_rejected_token_never_yields_another_calls_destinations():
    """The failure mode that matters: a rejected token must not leave
    this caller routed to the other call's numbers."""
    plan_a = plan_from_steps(PHASE_PRE_AI, _steps(H1, H2), "CA-A")
    rejected = open_call_plan(seal_call_plan(plan_a), "CA-B")
    assert rejected is None
    # so the route has no plan and resolves from its own live config
    live_b = [RoutingStep(destination=H3, timeout_sec=20)]
    d = decide_routing(PHASE_PRE_AI, "no-answer", True,
                       live_b if rejected is None else rejected.steps,
                       limits=LIMITS)
    assert d.destination == H3, "must ring call B's own config, not call A's"


def test_empty_but_counted_plan_preserves_counters_through_a_hop():
    """The exhausted chain: no steps left, but the budget has to survive
    or the AI resets it and the loop is unbounded again."""
    plan = plan_from_steps(PHASE_POST_AI, _steps(H1), "CA-EMPTY")
    d = decide_routing(PHASE_POST_AI, "no-answer", True, plan.steps, limits=LIMITS)
    final = advance_plan(plan, d)
    assert final.steps == ()
    token = seal_call_plan(final)
    assert token, "an exhausted-but-counted plan MUST still seal"
    back = open_call_plan(token, "CA-EMPTY")
    assert back is not None
    assert back.dial_attempts == 1 and back.rounds_used == 1
    d2 = decide_routing(PHASE_POST_AI, "no-answer", True, back.steps,
                        rounds_used=back.rounds_used,
                        dial_attempts=back.dial_attempts, limits=LIMITS)
    assert d2.action == ACTION_RESUME_AI


def test_max_configuration_stays_inside_a_sane_url():
    """Worst case: the largest cascade (5) and the largest chain (10) with
    realistic ids and labels, i.e. what an operator could actually
    configure. Sealing inflates the payload, and this rides in an action
    URL that crosses Twilio, proxies and access logs — so measure rather
    than assume."""
    from STT_server.services.transfer_cascade import (
        MAX_CASCADE_STEPS, MAX_CHAIN_TOOLS, cascade_action_url,
        transfer_fallback_url,
    )

    def _big(n):
        return [RoutingStep(
            destination="+1555000" + f"{i:04d}",
            timeout_sec=20 + i,
            tool_id="tool-abcdef0123456789" + str(i),
            label="Recepcion principal sede norte " + str(i),
        ) for i in range(n)]

    # Sanity: these really are the configured maxima
    assert MAX_CASCADE_STEPS == 5 and len(_big(MAX_CASCADE_STEPS)) == 5
    assert MAX_CHAIN_TOOLS == 10 and len(_big(MAX_CHAIN_TOOLS)) == 10

    cascade_url = cascade_action_url(
        "https://example.onrailway.app", "agent-0123456789abcdef", 1,
        tenant_id="tenant-0123456789abcdef",
        plan=seal_call_plan(plan_from_steps(
            PHASE_PRE_AI, _big(MAX_CASCADE_STEPS), "CA-CASCADE")),
    )
    chain_url = transfer_fallback_url(
        "https://example.onrailway.app", "agent-0123456789abcdef",
        [f"tool-abcdef0123456789{i}" for i in range(MAX_CHAIN_TOOLS)],
        tenant_id="tenant-0123456789abcdef",
        plan=seal_call_plan(plan_from_steps(
            PHASE_POST_AI, _big(MAX_CHAIN_TOOLS), "CA-CHAIN")),
    )
    for label, url in (("cascade", cascade_url), ("chain", chain_url)):
        assert len(url) < 4096, f"worst-case {label} action URL is {len(url)} chars"
    assert len(chain_url) > len(cascade_url), (
        "the 10-step chain should be the larger URL"
    )
    # Fernet overhead is the thing to watch as caps grow: bounded, not
    # doubling per step.
    one = seal_call_plan(plan_from_steps(PHASE_POST_AI, _big(1), "CA"))
    ten = seal_call_plan(plan_from_steps(PHASE_POST_AI, _big(10), "CA"))
    assert len(ten) < len(one) * 6, "sealed growth is not tracking step count"


def test_sealed_token_cannot_break_the_twiml_action_attribute():
    """A raw '&' inside the TwiML action attribute is malformed XML and
    Twilio may silently DROP the callback — which looks exactly like
    "the cascade dials but never falls through". The token rides inside
    that attribute, so it must be XML-safe by construction."""
    import re as _re
    from STT_server.services.transfer_cascade import (
        MAX_CHAIN_TOOLS, dial_twiml, transfer_fallback_url, PHASE_POST_AI,
    )

    steps = [RoutingStep(
        destination="+1555000%04d" % i, timeout_sec=20,
        tool_id="t<>&\"'%d" % i,          # deliberately XML-hostile ids
        label="a&b<c>d\"e'f %d" % i,
    ) for i in range(MAX_CHAIN_TOOLS)]
    token = seal_call_plan(plan_from_steps(PHASE_POST_AI, steps, "CA-XML"))
    assert token
    # Fernet output is base64url: nothing that breaks an XML attribute.
    assert not _re.search(r"[&<>\"']", token), "token contains an XML-hostile char"

    url = transfer_fallback_url("https://x.test", "ag", [], plan=token)
    xml = dial_twiml("+15550001111", 20, url)
    # exactly one attribute delimiter, and no unescaped ampersand
    action = xml.split('action="', 1)[1].split('"', 1)[0]
    assert "&amp;" in action
    assert not _re.search(r"&(?!amp;)", action), "unescaped & in the action URL"
    # the escaping survives a round trip back to the real URL
    assert action.replace("&amp;", "&") == url


def test_dial_attempt_cap_is_enforced_exactly_over_many_sealed_hops():
    """The dial cap must be real, not advisory, and the count must
    survive the seal/open hop on every leg. The round cap is lifted here
    so the DIAL cap is the binding constraint."""
    generous = RoutingLimits(max_handoff_rounds=999, max_human_dial_attempts=20)
    plan = plan_from_steps(PHASE_POST_AI, _steps(*([H1] * 40)), "CA-many")
    dials = 0
    for _ in range(80):
        d = decide_routing(PHASE_POST_AI, "no-answer", True, plan.steps,
                           rounds_used=plan.rounds_used,
                           dial_attempts=plan.dial_attempts, limits=generous)
        if d.action != ACTION_DIAL:
            break
        dials += 1
        assert dials <= generous.max_human_dial_attempts, "dialed past the cap"
        plan = open_call_plan(seal_call_plan(advance_plan(plan, d)))
        assert plan is not None, "sealed plan must always round-trip"
    else:
        pytest.fail("the dial-attempt cap was never reached")

    assert dials == generous.max_human_dial_attempts
    assert plan.dial_attempts == dials
    assert d.action == ACTION_RESUME_AI
    assert d.reason == "limit_reached"
    assert d.handoff_disabled is True


def test_round_cap_is_the_binding_constraint_for_repeated_chains():
    """With the shipped defaults the round cap (3) bites before the dial
    cap (20): three AI->human sequences, then the AI keeps the caller."""
    plan = plan_from_steps(PHASE_POST_AI, _steps(*([H1] * 40)), "CA-rounds")
    dials = 0
    for _ in range(80):
        d = decide_routing(PHASE_POST_AI, "no-answer", True, plan.steps,
                           rounds_used=plan.rounds_used,
                           dial_attempts=plan.dial_attempts, limits=LIMITS)
        if d.action != ACTION_DIAL:
            break
        dials += 1
        plan = open_call_plan(seal_call_plan(advance_plan(plan, d)))
    assert dials == LIMITS.max_handoff_rounds == 3
    assert plan.dial_attempts == 3, "each chain dial also burned an attempt"
    assert d.action == ACTION_RESUME_AI
    assert d.reason == "limit_reached"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("test_routing_invariants: all green")
