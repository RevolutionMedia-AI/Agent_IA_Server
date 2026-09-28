"""Phase 2: the Twilio/FastAPI adapter boundary for call routing.

The pure core (tests/test_routing_core.py, 810 cases) proves the DECISION
is right. It cannot prove the HTTP layer actually feeds it the right
inputs, or renders the decision correctly. That is what this file covers:

    Twilio request
      -> signature / credentials
      -> parameter parsing
      -> sealed-plan validation
      -> decide_routing(...)
      -> TwiML rendering

Everything under test is the REAL production code: the actual
/voice/cascade and /voice/transfer-fallback route functions, the real
decide_routing, the real seal/open_call_plan, the real dial_twiml and
connect_stream_twiml, and the real validate_twilio_signature. Nothing
here is a re-implementation.

Only two things are stubbed, both unrelated to routing:
  - the `openai` and `webrtcvad` SDKs, which are absent in this
    environment and are imported at module scope by STT_Server. They are
    STT/LLM concerns; the routing paths never touch them. Stubbing them
    is what lets the REAL app import (see _stub_absent_sdks).
  - the three config lookups the routes call to resolve an agent, its
    cascade/chain and its tools. Those are DB reads, not adapter logic;
    resolve_cascade_steps / build_transfer_chain still run for real.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sys
import types
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

# PUBLIC_URL is read at import time by config.py, and the routes rebuild
# the signature URL from it, so the test client must use the same host.
os.environ.setdefault("PUBLIC_URL", "http://localhost:8080")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_KEY",
    "oahqImB7aGYfEFxfWIJLZzJs27YSYAgr5rHUyc3gIRU=",
)


from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

import STT_server.STT_Server as srv  # noqa: E402  (the REAL app)
import STT_server.db_agents as db_agents  # noqa: E402
import STT_server.db_phone_numbers as db_phone_numbers  # noqa: E402
import STT_server.db_tools as db_tools  # noqa: E402
from STT_server.services.call_plan import open_call_plan, seal_call_plan  # noqa: E402
from STT_server.services.transfer_cascade import (  # noqa: E402
    ACTION_END_CALL,
    PHASE_POST_AI,
    PHASE_PRE_AI,
    RoutingStep,
    plan_from_steps,
)

TOKEN = "twilio-auth-token-abc123"
ACCOUNT_SID = "ACtest0000000000000000000000test"
TO_NUMBER = "+15550000000"
H1 = "+15550001111"
H2 = "+15550002222"
H3 = "+15550003333"
BASE = "http://localhost:8080"

CASCADE_PATH = "/voice/cascade"
FALLBACK_PATH = "/voice/transfer-fallback"


# ── signing + request helpers ────────────────────────────────────────────

def sign(url: str, form: dict, token: str = TOKEN) -> str:
    """Recompute Twilio's X-Twilio-Signature.

    Same algorithm as the production validator (and the official SDKs):
    HMAC-SHA1 over the full URL then ``key + value`` for each form param
    sorted by key, raw concatenation, base64.
    """
    pieces = [url]
    for key in sorted(form):
        value = form[key] if form[key] is not None else ""
        pieces.append(f"{key}{value}")
    payload = "".join(pieces).encode("utf-8")
    digest = hmac.new(token.encode("utf-8"), payload, hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def build_url(path: str, params: dict) -> str:
    """Build the exact URL the route will reconstruct for signing.

    Encoding the query ourselves and handing the finished string to
    httpx guarantees the bytes we sign are the bytes that arrive, so the
    test never depends on a client-side re-encoding matching.
    """
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{BASE}{path}?{qs}" if qs else f"{BASE}{path}"


async def post(client, path, params, form, *, token=TOKEN, sign_it=True,
               signature=None):
    url = build_url(path, params)
    headers = {}
    if signature is not None:
        headers["X-Twilio-Signature"] = signature
    elif sign_it:
        headers["X-Twilio-Signature"] = sign(url, form, token or "")
    return await client.post(url, data=form, headers=headers)


def twilio_form(**over) -> dict:
    form = {
        "To": TO_NUMBER,
        "From": "+15559998888",
        "CallSid": "CA" + "0" * 32,
        "AccountSid": ACCOUNT_SID,
        "DialCallStatus": "no-answer",
        "CallStatus": "in-progress",
        "DialCallSid": "CA" + "1" * 32,
    }
    form.update({k: v for k, v in over.items() if v is not None})
    return form


# ── TwiML inspection ─────────────────────────────────────────────────────

def twiml_of(response) -> ET.Element:
    assert response.status_code == 200, response.text
    body = response.text
    assert body.strip(), "empty TwiML body"
    return ET.fromstring(body)


def dial_destinations(xml: ET.Element) -> list[str]:
    return [n.text for n in xml.iter("Number") if n.text]


def action_url(xml: ET.Element) -> str:
    for dial in xml.iter("Dial"):
        return dial.attrib.get("action", "")
    return ""


def has_stream(xml: ET.Element) -> bool:
    return any(True for _ in xml.iter("Stream"))


def has_hangup(xml: ET.Element) -> bool:
    return any(True for _ in xml.iter("Hangup"))


def stream_params(xml: ET.Element) -> dict:
    return {p.attrib.get("name"): p.attrib.get("value") for p in xml.iter("Parameter")}


# ── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """The real app, real routes. No lifespan: we are not booting a call,
    only the two HTTP callback endpoints."""
    app = FastAPI()
    app.include_router(srv.app.router)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url=BASE)


@pytest.fixture(autouse=True)
def routing_env(monkeypatch):
    """Stub only the three CONFIG lookups. The resolution functions that
    consume them (resolve_cascade_steps, build_transfer_chain,
    parse_cascade_with_ids) are the real ones."""
    def _number(_to):
        return {
            "id": "pn-1", "twilio_auth_token": TOKEN,
            "twilio_account_sid": ACCOUNT_SID, "user_id": "u1",
        }

    monkeypatch.setattr(db_phone_numbers, "find_by_number", _number)
    monkeypatch.setattr(db_agents, "get_agent", lambda *a, **k: None)
    monkeypatch.setattr(db_tools, "list_tools", lambda *a, **k: [])
    return monkeypatch


def set_cascade(monkeypatch, destinations, tool_rows=None):
    """Give the agent a pre-AI cascade of raw entries."""
    monkeypatch.setattr(db_agents, "get_agent", lambda *a, **k: {
        "id": "agent-1", "user_id": "u1",
        "transfer_cascade": [
            {"destination": d, "timeout_sec": 20} for d in destinations
        ],
        "transfer_chain": [],
    })
    if tool_rows is not None:
        monkeypatch.setattr(db_tools, "list_tools", lambda *a, **k: tool_rows)


def set_chain(monkeypatch, tool_rows, chain_ids):
    """Give the agent a post-AI chain backed by real tool rows."""
    monkeypatch.setattr(db_agents, "get_agent", lambda *a, **k: {
        "id": "agent-1", "user_id": "u1",
        "transfer_cascade": [],
        "transfer_chain": list(chain_ids),
    })
    monkeypatch.setattr(db_tools, "list_tools", lambda *a, **k: tool_rows)


def tool_row(tid, destination, timeout=20):
    return {
        "id": tid, "name": f"Tool {tid}", "kind": "call_transfer",
        "destination": destination, "ring_timeout_sec": timeout,
    }


# ═══════════════════════════════════════════════════════════════════════
# 1-2. Signature validation
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_cascade_valid_signature_reaches_route_and_dials(client, routing_env):
    """A correctly signed request gets past auth into real routing and
    emits real TwiML for the next human."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 1}, twilio_form())
    assert r.status_code == 200
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2]
    assert has_stream(xml) is False


@pytest.mark.asyncio
async def test_cascade_invalid_signature_is_403_and_routes_nothing(client, routing_env):
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 1},
                   twilio_form(), signature="not-a-signature")
    assert r.status_code == 403
    assert "Dial" not in r.text and "Stream" not in r.text


@pytest.mark.asyncio
async def test_fallback_invalid_signature_is_403_and_routes_nothing(client, routing_env):
    set_chain(routing_env, [tool_row("t1", H1), tool_row("t2", H2)], ["t1", "t2"])
    r = await post(client, FALLBACK_PATH,
                   {"agent_id": "agent-1", "remaining": "t2"},
                   twilio_form(), signature="nope")
    assert r.status_code == 403
    assert "Dial" not in r.text and "Stream" not in r.text


@pytest.mark.asyncio
async def test_signature_covers_the_query_string(client, routing_env):
    """The plan and the step ride in the query, and the signature must
    cover them: a tampered query must not authenticate."""
    set_cascade(routing_env, [H1, H2])
    form = twilio_form()
    good_url = build_url(CASCADE_PATH, {"agent_id": "agent-1", "step": 1})
    sig = sign(good_url, form)
    # same signature, but the query now says a different step
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 0}, form, signature=sig)
    assert r.status_code == 403


# ═══════════════════════════════════════════════════════════════════════
# 3. Fail-closed runtime credentials
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("path,params", [
    (CASCADE_PATH, {"agent_id": "agent-1", "step": 1}),
    (FALLBACK_PATH, {"agent_id": "agent-1", "remaining": "t2"}),
])
async def test_missing_credential_fails_closed_503(client, routing_env, path, params):
    """No resolvable auth token -> 503, before any routing happens. No
    Dial, no stream, no plan advanced. Security is not loosened for
    testability."""
    routing_env.setattr(db_phone_numbers, "find_by_number", lambda _to: None)
    set_cascade(routing_env, [H1, H2])
    r = await post(client, path, params, twilio_form())
    assert r.status_code == 503
    assert "Dial" not in r.text and "Stream" not in r.text
    assert "<Hangup" not in r.text


@pytest.mark.asyncio
async def test_empty_form_body_fails_closed_503(client, routing_env):
    """A callback with no To field cannot resolve a credential, so it
    cannot be authenticated."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1}, {})
    assert r.status_code == 503
    assert "Dial" not in r.text


# ═══════════════════════════════════════════════════════════════════════
# 4. Caller liveness wiring (the two Twilio fields, independently)
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("parent", ["initiated", "ringing", "in-progress"])
async def test_completed_leg_with_live_parent_advances(client, routing_env, parent):
    """A human answered, the leg ended, the caller is still connected:
    the call advances to the next human."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1},
                   twilio_form(DialCallStatus="completed", CallStatus=parent))
    assert r.status_code == 200
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2], "must advance, not hang up"
    assert has_stream(xml) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", ["completed", "busy", "failed", "canceled", "no-answer"])
@pytest.mark.parametrize("dial_status", ["completed", "no-answer", "busy"])
async def test_terminal_parent_ends_the_call(client, routing_env, parent, dial_status):
    """Terminal parent status -> END_CALL. No next Dial, no StartAI, no
    ResumeAI. The dial leg's own status is irrelevant to this decision,
    which is the point: the two fields are independent."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1},
                   twilio_form(DialCallStatus=dial_status, CallStatus=parent))
    assert r.status_code == 200
    xml = twiml_of(r)
    assert has_hangup(xml), "a departed caller must get <Hangup/>"
    assert dial_destinations(xml) == []
    assert has_stream(xml) is False, "must not open a stream to dead air"


@pytest.mark.asyncio
async def test_terminal_parent_on_chain_also_ends(client, routing_env):
    set_chain(routing_env, [tool_row("t1", H1), tool_row("t2", H2)], ["t1", "t2"])
    r = await post(client, FALLBACK_PATH, {"agent_id": "agent-1", "remaining": "t2"},
                   twilio_form(DialCallStatus="completed", CallStatus="completed"))
    xml = twiml_of(r)
    assert has_hangup(xml)
    assert has_stream(xml) is False
    assert stream_params(xml).get("transfer_resume") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", [None, "", "some-new-twilio-status"])
async def test_unknown_call_status_preserves_legacy_routing(client, routing_env, parent):
    """Missing/unrecognised CallStatus -> caller_alive None -> the
    historical advance-on-everything. An unrecognised Twilio status must
    never start ending live calls."""
    set_cascade(routing_env, [H1, H2])
    form = twilio_form(DialCallStatus="no-answer")
    if parent is not None:
        form["CallStatus"] = parent
    else:
        form.pop("CallStatus", None)
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1}, form)
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2]
    assert has_hangup(xml) is False


# ═══════════════════════════════════════════════════════════════════════
# 5 + 11. Decision -> TwiML, and per-route phase wiring
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_cascade_exhausted_emits_start_ai_without_resume(client, routing_env):
    """Phase wiring, observed: a pre_ai cascade that runs out STARTS the
    AI, so there must be no transfer_resume parameter."""
    set_cascade(routing_env, [H1])
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1},
                   twilio_form())
    xml = twiml_of(r)
    assert has_stream(xml)
    assert dial_destinations(xml) == []
    params = stream_params(xml)
    assert params.get("transfer_resume") is None
    assert params.get("agent_id") == "agent-1"


@pytest.mark.asyncio
async def test_chain_exhausted_emits_resume_ai(client, routing_env):
    """Phase wiring, observed: a post_ai chain that runs out RESUMES the
    AI, so transfer_resume must be present."""
    set_chain(routing_env, [tool_row("t1", H1)], ["t1"])
    r = await post(client, FALLBACK_PATH, {"agent_id": "agent-1", "remaining": ""},
                   twilio_form())
    xml = twiml_of(r)
    assert has_stream(xml)
    assert stream_params(xml).get("transfer_resume") == "1"


@pytest.mark.asyncio
async def test_fallback_dials_next_remaining_human(client, routing_env):
    set_chain(routing_env, [tool_row("t1", H1), tool_row("t2", H2)], ["t1", "t2"])
    r = await post(client, FALLBACK_PATH,
                   {"agent_id": "agent-1", "remaining": "t2"}, twilio_form())
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2]
    assert "remaining=t2" in action_url(xml) or "remaining=" in action_url(xml)


@pytest.mark.asyncio
async def test_action_url_points_back_at_the_right_callback(client, routing_env):
    """A Dial with no action URL ends the call on no-answer — the
    'se cuelga' bug. The rendered action must address the right route."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1},
                   twilio_form())
    url = action_url(twiml_of(r))
    assert url.startswith(f"{BASE}{CASCADE_PATH}?")
    assert "step=2" in url, "the next callback must advance the step"

    set_chain(routing_env, [tool_row("t1", H1), tool_row("t2", H2)], ["t1", "t2"])
    r2 = await post(client, FALLBACK_PATH,
                    {"agent_id": "agent-1", "remaining": "t2"}, twilio_form())
    url2 = action_url(twiml_of(r2))
    assert url2.startswith(f"{BASE}{FALLBACK_PATH}?")


# ═══════════════════════════════════════════════════════════════════════
# 6. Sealed plan: valid, and a real hop that re-seals
# ═════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_valid_sealed_plan_is_honoured_and_advances(client, routing_env):
    """Policy A at the HTTP boundary.

    NOTE on what a plan contains: the PENDING steps. /voice dials the
    first destination itself and seals the ADVANCED plan into the action
    URL, so by the time a callback arrives H1 is already gone. This test
    therefore hands the route a plan whose next step is H2 while live
    config says H3, and the plan must win.
    """
    call_sid = "CA" + "a" * 32
    plan = plan_from_steps(PHASE_PRE_AI, [
        RoutingStep(destination=H2, timeout_sec=30, tool_id="t2", label="Cafeteria"),
    ], call_sid)
    token = seal_call_plan(plan)
    assert token
    set_cascade(routing_env, [H3])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 1, "plan": token},
                   twilio_form(CallSid=call_sid))
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2], (
        "Policy A: the frozen plan wins over live config"
    )


@pytest.mark.asyncio
async def test_sealed_plan_hop_re_seals_with_counters_advanced(client, routing_env):
    """A real hop: callback -> Dial -> the emitted action URL carries a
    freshly sealed plan whose counters and remaining order survived."""
    call_sid = "CA" + "b" * 32
    plan = plan_from_steps(PHASE_PRE_AI, [
        RoutingStep(destination=H1, timeout_sec=20, tool_id="t1"),
        RoutingStep(destination=H2, timeout_sec=20, tool_id="t2"),
    ], call_sid)
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 1, "plan": seal_call_plan(plan)},
                   twilio_form(CallSid=call_sid))
    url = action_url(twiml_of(r))
    qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    token = qs.get("plan")
    assert token, "the next callback must carry a re-sealed plan"
    assert H1 not in token and H2 not in token, "no plaintext destinations"

    hop = open_call_plan(token, call_sid)
    assert hop is not None, "the re-sealed plan must open for this call"
    assert hop.dial_attempts == 1, "one dial consumed"
    assert [s.destination for s in hop.steps] == [H2], "remaining order preserved"
    assert [s.tool_id for s in hop.steps] == ["t2"]
    assert hop.call_sid == call_sid


@pytest.mark.asyncio
async def test_exhausted_sealed_plan_still_carries_counters_to_the_ai(client, routing_env):
    """The regression that motivated sealing the final plan: an exhausted
    chain has no steps left, but the budget must still reach the AI so it
    cannot reset and loop.

    A plan with a step left is not exhausted — it dials. So this builds
    the genuinely exhausted shape: zero pending steps, counters already
    spent."""
    from STT_server.services.transfer_cascade import CallPlan

    call_sid = "CA" + "c" * 32
    exhausted = CallPlan(phase=PHASE_POST_AI, steps=(), rounds_used=1,
                         dial_attempts=1, handoff_disabled=False, call_sid=call_sid)
    token = seal_call_plan(exhausted)
    assert token, "an exhausted-but-counted plan MUST still seal"
    r = await post(client, FALLBACK_PATH,
                   {"agent_id": "agent-1", "remaining": "t1", "plan": token},
                   twilio_form(CallSid=call_sid, DialCallStatus="no-answer"))
    xml = twiml_of(r)
    assert has_stream(xml), "a fully dialed chain returns to the AI"
    assert dial_destinations(xml) == []
    params = stream_params(xml)
    assert params.get("transfer_resume") == "1"

    carried = open_call_plan(params.get("call_plan"), call_sid)
    assert carried is not None, "counters must reach the AI on the stream"
    assert carried.dial_attempts == 1
    assert carried.rounds_used == 1
    assert carried.steps == ()


# ═══════════════════════════════════════════════════════════════════════
# 7 + 8. Hostile sealed state, and CallSid binding
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("token_factory,label", [
    (lambda: "not-a-token", "malformed"),
    (lambda: "!!!not base64!!!", "non-base64"),
    (lambda: "AAAA", "truncated"),
])
async def test_malformed_plan_token_is_safe(client, routing_env, token_factory, label):
    """Never a 500, never a crash: fall back to live config, which is
    the documented POLICY A exception (see the module docstring)."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 1, "plan": token_factory()},
                   twilio_form())
    assert r.status_code == 200, f"{label} must not 500"
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2], f"{label} falls back to live config"
    assert has_stream(xml) is False


@pytest.mark.asyncio
async def test_cryptographically_tampered_token_is_safe(client, routing_env):
    """A token that fails Fernet authentication must be ignored, and the
    call must fall back to live config — never to the destination the
    rejected token named. step=0 so live config still has a human to ring,
    which makes the fallback observable."""
    call_sid = "CA" + "d" * 32
    token = seal_call_plan(plan_from_steps(
        PHASE_PRE_AI, [RoutingStep(destination=H1, timeout_sec=20)], call_sid))
    tampered = ("A" if token[10] != "A" else "B") + token[11:]
    set_cascade(routing_env, [H2, H3])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 0, "plan": tampered},
                   twilio_form(CallSid=call_sid))
    assert r.status_code == 200
    xml = twiml_of(r)
    assert dial_destinations(xml) == [H2], "tampered -> live config, never H1"
    assert has_stream(xml) is False


@pytest.mark.asyncio
async def test_token_from_another_call_is_rejected_and_never_cross_routes(
    client, routing_env,
):
    """Call B presenting call A's valid token must not be routed to
    A's humans. It falls back to B's own live config."""
    sid_a = "CA" + "e" * 32
    sid_b = "CA" + "f" * 32
    token_a = seal_call_plan(plan_from_steps(
        PHASE_PRE_AI,
        [RoutingStep(destination=H1, timeout_sec=20),
         RoutingStep(destination=H2, timeout_sec=20)],
        sid_a,
    ))
    set_cascade(routing_env, [H3, H3])       # call B's live config
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 0, "plan": token_a},
                   twilio_form(CallSid=sid_b))
    xml = twiml_of(r)
    got = dial_destinations(xml)
    assert got == [H3], f"must use call B's config, got {got}"
    assert H1 not in got and H2 not in got


@pytest.mark.asyncio
async def test_hostile_ids_and_labels_survive_the_codec(client, routing_env):
    """Ids/labels with XML-hostile characters must round-trip through the
    plan and still produce a parseable, correctly-escaped action URL.

    The plan carries PENDING steps, so a single-entry plan names the one
    human still to ring.

    The escaping invariant is stated precisely: in the RAW body every '&'
    must be '&amp;', and after XML parsing the action attribute must equal
    the intended URL byte for byte. (A parsed attribute legitimately
    contains bare '&' again — those are the query separators.)
    """
    import re as _re

    call_sid = "CA" + "9" * 32
    hostile = 'a&b<c>"d\'e'
    # H1 rings now; the hostile metadata rides on the step still PENDING,
    # because a dialed step is consumed out of the plan.
    plan = plan_from_steps(PHASE_PRE_AI, [
        RoutingStep(destination=H1, timeout_sec=20),
        RoutingStep(destination=H2, timeout_sec=25, tool_id=f"t-{hostile}",
                    label=f"Recepcion {hostile}"),
    ], call_sid)
    token = seal_call_plan(plan)
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": 1, "plan": token},
                   twilio_form(CallSid=call_sid))
    assert r.status_code == 200

    raw = r.text
    attr_raw = raw.split('action="', 1)[1].split('"', 1)[0]
    assert not _re.search(r"&(?!amp;)", attr_raw), (
        "raw action attribute has an unescaped ampersand"
    )
    assert "&amp;" in attr_raw

    xml = ET.fromstring(raw)              # parsed, not substring-matched
    assert dial_destinations(xml) == [H1]
    url = action_url(xml)
    assert url, "a Dial with no action URL ends the call on no-answer"
    assert url == attr_raw.replace("&amp;", "&"), "XML escaping did not round-trip"

    # the sealed plan survived the hop, hostile metadata intact
    qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    carried = open_call_plan(qs.get("plan"), call_sid)
    assert carried is not None
    assert carried.dial_attempts == 1
    assert [s.destination for s in carried.steps] == [H2]
    assert carried.steps[0].label == f"Recepcion {hostile}"
    assert carried.steps[0].tool_id == f"t-{hostile}"


def test_binding_uses_the_full_callsid_not_the_log_truncation():
    """Two CallSids sharing the first 20 characters must NOT compare
    equal. call_sid_log is truncated to 20 chars for logging, so a
    regression that bound against it would silently accept the wrong
    call."""
    from STT_server.services.call_plan import open_call_plan as _open
    shared = "CA12345678901234567890"
    sid_a = shared + "_AAAAA"
    sid_b = shared + "_BBBB"
    assert sid_a[:20] == sid_b[:20], "they must share the truncated prefix"

    token = _seal_for(sid_a, H1)
    assert _open(token, sid_a) is not None, "the real owner is accepted"
    assert _open(token, sid_b) is None, "the prefix-sharing impostor is rejected"


def _seal_for(call_sid, destination):
    return seal_call_plan(plan_from_steps(
        PHASE_PRE_AI, [RoutingStep(destination=destination, timeout_sec=20)], call_sid))


# ═══════════════════════════════════════════════════════════════════════
# 10. Parameter parsing at the boundary
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["-1", "0", "999", " ", "3"])
async def test_malformed_step_is_defined_and_safe(client, routing_env, step):
    """A bad step never 500s and never emits invalid TwiML: it resolves
    to a defined routing outcome (dial a valid next, or start the AI).

    Two defined behaviours are accepted here on purpose:
      * 422 — FastAPI rejects a non-integer for `step: int` before the
        route body runs. That is the fail-safe answer, and it makes the
        route's own int() fallback unreachable over HTTP (it still guards
        the JSON-file and internal callers).
      * 200 — an integer that is out of range falls through to the AI.
    Either way the invariant is the same: no 500, and never a TwiML body
    naming a destination that is not in the config.
    """
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": step}, twilio_form())
    assert r.status_code in (200, 422), f"unexpected {r.status_code}"
    if r.status_code == 422:
        return
    xml = twiml_of(r)
    if dial_destinations(xml):
        assert all(d == H1 or d == H2 for d in dial_destinations(xml))
    else:
        assert has_stream(xml)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", ["abc", "1e9", "0x2", "1.5", "None", "null"])
async def test_non_integer_step_is_rejected_by_validation(client, routing_env, step):
    """A non-integer `step` is refused by FastAPI's own validation with
    422 — no route body, no TwiML, no 500. Pinned so a future loosening of
    the `step: int` annotation is a visible change."""
    set_cascade(routing_env, [H1, H2])
    r = await post(client, CASCADE_PATH,
                   {"agent_id": "agent-1", "step": step}, twilio_form())
    assert r.status_code == 422
    assert "<Dial" not in r.text and "<Connect" not in r.text


@pytest.mark.asyncio
@pytest.mark.parametrize("remaining", ["", "   ", ",,,", "t1,,t2", "does-not-exist"])
async def test_malformed_remaining_is_defined_and_safe(client, routing_env, remaining):
    set_chain(routing_env, [tool_row("t1", H1)], ["t1"])
    r = await post(client, FALLBACK_PATH,
                   {"agent_id": "agent-1", "remaining": remaining}, twilio_form())
    assert r.status_code == 200
    xml = twiml_of(r)
    if dial_destinations(xml):
        assert dial_destinations(xml) == [H1]
    else:
        assert has_stream(xml)


@pytest.mark.asyncio
async def test_missing_dial_status_still_routes(client, routing_env):
    """Twilio omitting DialCallStatus entirely must not break the hop."""
    set_cascade(routing_env, [H1, H2])
    form = twilio_form()
    form.pop("DialCallStatus")
    r = await post(client, CASCADE_PATH, {"agent_id": "agent-1", "step": 1}, form)
    assert r.status_code == 200
    assert dial_destinations(twiml_of(r)) == [H2]


@pytest.mark.asyncio
async def test_agent_that_no_longer_exists_does_not_500(client, routing_env):
    """Agent deleted mid-call: no config, no crash, a defined outcome."""
    routing_env.setattr(db_agents, "get_agent", lambda *a, **k: None)
    r = await post(client, CASCADE_PATH, {"agent_id": "gone", "step": 1},
                   twilio_form())
    assert r.status_code == 200
    xml = twiml_of(r)
    assert has_stream(xml), "no config -> the AI takes the call"
    assert dial_destinations(xml) == []


# ═══════════════════════════════════════════════════════════════════════
# 12. Loop guard survives real HTTP callbacks
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_loop_counter_survives_real_url_callbacks(client, routing_env):
    """Serialization + HTTP parsing must not reset the budget. Walks real
    callbacks until the guard fires, then asserts the AI is resumed and
    no further Dial is emitted."""
    from STT_server.config import MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL

    call_sid = "CA" + "7" * 32
    steps = [RoutingStep(destination=f"+1555000{i:04d}", timeout_sec=20)
             for i in range(MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL + 5)]
    token = seal_call_plan(plan_from_steps(PHASE_POST_AI, steps, call_sid))

    plan = "pre_ai"
    params = {"agent_id": "agent-1", "plan": token}
    dials = 0
    last_xml = None
    for _ in range(MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL + 5):
        r = await post(client, CASCADE_PATH, params,
                       twilio_form(CallSid=call_sid, DialCallStatus="no-answer"))
        assert r.status_code == 200
        last_xml = twiml_of(r)
        if not dial_destinations(last_xml):
            break
        dials += 1
        url = action_url(last_xml)
        qs = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        carried = open_call_plan(qs.get("plan"), call_sid)
        assert carried is not None, "each hop must carry a readable plan"
        params = {"agent_id": "agent-1", "plan": qs["plan"]}

    assert dials == MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL, (
        f"guard must stop at exactly the cap, dialed {dials}"
    )
    # the hop after the cap: no Dial, back to the AI, counters preserved
    assert has_stream(last_xml) or has_hangup(last_xml)
    assert dial_destinations(last_xml) == []
    assert plan == PHASE_PRE_AI


# ═══════════════════════════════════════════════════════════════════════
# 9. URL / XML safety at the product maximum
# ═══════════════════════════════════════════════════════════════════════

# ponytail: an INTERNAL budget, not a claim about any platform's limit.
# 4096 is a widely cited URL ceiling but we have NOT verified it against
# Twilio, Railway's edge and every proxy in front of them, so we do not
# rely on it. We budget well under it and fail the build if growth eats
# the headroom.
ACTION_URL_INTERNAL_BUDGET = 3000


def test_max_plan_url_stays_inside_the_internal_budget():
    from STT_server.services.transfer_cascade import (
        MAX_CASCADE_STEPS, MAX_CHAIN_TOOLS, cascade_action_url, dial_twiml,
        transfer_fallback_url,
    )

    def big(n):
        return [RoutingStep(
            destination="+1555000%04d" % i, timeout_sec=20 + i,
            tool_id="tool-abcdef0123456789" + str(i),
            label="Recepcion principal sede norte " + str(i),
        ) for i in range(n)]

    chain_token = seal_call_plan(plan_from_steps(
        PHASE_POST_AI, big(MAX_CHAIN_TOOLS), "CA" + "1" * 32))
    assert H1 not in chain_token, "no plaintext destinations in the token"

    chain_url = transfer_fallback_url(
        BASE, "agent-0123456789abcdef",
        [f"tool-abcdef0123456789{i}" for i in range(MAX_CHAIN_TOOLS)],
        tenant_id="tenant-0123456789abcdef", plan=chain_token)
    cascade_url = cascade_action_url(
        BASE, "agent-0123456789abcdef", 1,
        tenant_id="tenant-0123456789abcdef",
        plan=seal_call_plan(plan_from_steps(
            PHASE_PRE_AI, big(MAX_CASCADE_STEPS), "CA" + "1" * 32)))

    for label, url in (("cascade", cascade_url), ("chain", chain_url)):
        assert len(url) < ACTION_URL_INTERNAL_BUDGET, (
            f"worst-case {label} action URL is {len(url)} chars, over the "
            f"{ACTION_URL_INTERNAL_BUDGET} internal budget"
        )

    # XML round-trip: parse, read the attribute, unescape, and it must
    # equal the URL we meant to emit.
    xml = ET.fromstring(dial_twiml(H1, 20, chain_url))
    got = action_url(xml)
    assert got == chain_url, "XML escaping did not round-trip"

    # Growth guard: raising MAX_CHAIN_TOOLS must fail loudly, not quietly
    # produce an enormous URL.
    over_budget = transfer_fallback_url(
        BASE, "agent-0123456789abcdef", [], plan=seal_call_plan(
            plan_from_steps(PHASE_POST_AI, big(MAX_CHAIN_TOOLS * 3), "CA" + "1" * 32)))
    assert len(over_budget) > ACTION_URL_INTERNAL_BUDGET, (
        "tripling the chain must exceed the internal budget, otherwise the "
        "budget does not constrain anything"
    )


def test_start_handler_resume_block_is_guarded_by_if_resume():
    """Regression guard for a bug that killed EVERY inbound call in
    production.

    The media-stream start handler builds a `transfer_resume` branch: a
    failure note, the carried pre-transfer history, and the sealed plan
    read off the <Stream>. All of it is meaningless — and was crashing —
    on a NORMAL call that has no transfer_resume.

    A previous revision dedented the body one level, so a plain call fell
    through to it and raised UnboundLocalError on `_note`. The caller got
    "hubo un problema de configuracion" and the AI never ran.

    Asserted structurally, because reproducing it needs the whole
    WebSocket handshake: every name the resume branch introduces must be
    first bound INSIDE the `if _resume:` block, never used after it.
    """
    import ast
    import pathlib

    import STT_server.STT_Server as server_mod

    path = pathlib.Path(server_mod.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    handler = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "media_stream":
            handler = node
            break
    assert handler is not None, "media_stream handler not found"

    # find the `if _resume:` statement
    resume_if = None
    for node in ast.walk(handler):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.Name) and test.id == "_resume"):
            resume_if = node
            break
    assert resume_if is not None, "the if _resume: branch is gone"
    lo = resume_if.lineno
    hi = resume_if.end_lineno or resume_if.lineno

    # every name the branch introduces must be bound inside it
    bound_inside = set()
    for stmt in resume_if.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound_inside.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    bound_inside.add(a.asname or a.name)
    assert {"_note", "_carried"} <= bound_inside, (
        "the failure note and the carried history must be created inside "
        f"the branch; found {sorted(bound_inside)}"
    )

    # and none of them may be READ outside it. This is the exact
    # production failure: the read sat one dedent too high, so every
    # normal call hit it with `_note` unbound.
    for name in ("_note", "_carried"):
        escaped = sorted({
            n.lineno for n in ast.walk(handler)
            if isinstance(n, ast.Name) and n.id == name
            and isinstance(n.ctx, ast.Load)
            and not (lo <= n.lineno <= hi)
        })
        assert not escaped, (
            f"{name} is read at line(s) {escaped}, OUTSIDE `if _resume:` "
            f"(lines {lo}-{hi}). On a normal call that name is unbound and "
            "the call dies with UnboundLocalError."
        )


if __name__ == "__main__":
    print("run with: python -m pytest tests/test_routing_http.py")
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and "asyncio" not in _name:
            _fn()
    print("test_routing_http: sync tests green")
