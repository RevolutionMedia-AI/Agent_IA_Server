"""Sealed call plans (Policy A + loop guards).

Why this module exists
----------------------
A call's routing plan and its loop counters must survive three things:

  1. a Twilio callback landing on a DIFFERENT process/instance than the
     one that started the call (Railway replicas; today the repo runs a
     single uvicorn process, but replica count is a dashboard setting,
     not a commit),
  2. a Postgres outage mid-call,
  3. a deploy between two hops of the same call.

They also must not leak phone numbers into access logs, traces, or
Twilio's own request log.

So the plan rides INSIDE the action URL, sealed with the same Fernet key
already used for provider credentials: no shared storage, no DB read on
the callback path, identical behaviour on every replica, and no E.164
in the clear. Twilio's own X-Twilio-Signature already authenticates the
whole query string, so a tampered blob fails twice over: bad signature
-> 403, and even past that, decryption fails closed.

Pure decision/serialization logic lives in transfer_cascade.py; this
module is only the crypto seam.
"""
from __future__ import annotations

import logging

from STT_server.security.credentials import decrypt_value, encrypt_value
from STT_server.services.transfer_cascade import (
    CallPlan,
    RoutingLimits,
    RoutingStep,
    decode_plan_payload,
    encode_plan_payload,
)

log = logging.getLogger("stt_server.call_plan")

# Query parameter carrying the sealed plan on /voice/cascade,
# /voice/transfer-fallback and the AI <Stream> element.
PLAN_PARAM = "plan"


def routing_limits() -> RoutingLimits:
    """The env-configured loop caps, read fresh on each decision.

    Reading per call (not at import) means changing the env var and
    restarting is all it takes, and tests can monkeypatch config without
    reimporting the pure core.
    """
    from STT_server.config import (
        MAX_HANDOFF_ROUNDS_PER_CALL,
        MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL,
    )
    return RoutingLimits(
        max_handoff_rounds=MAX_HANDOFF_ROUNDS_PER_CALL,
        max_human_dial_attempts=MAX_HUMAN_DIAL_ATTEMPTS_PER_CALL,
    )


def _plan_is_worth_sealing(plan: CallPlan) -> bool:
    """A plan carries state only if it has destinations left OR running
    counters / a sticky flag.

    The exhausted-but-counted case matters most: when a chain runs out we
    seal the FINAL plan (no steps, but the running rounds/attempts) so
    the loop guards survive the hand-off to the AI session. Sealing on
    `plan.steps` alone would silently drop them and the caps would reset
    on every return to the AI — the exact unbounded loop this guards.
    """
    return bool(
        plan.steps
        or plan.dial_attempts
        or plan.rounds_used
        or plan.handoff_disabled
    )


def seal_call_plan(plan: CallPlan) -> str:
    """Encode + Fernet-encrypt a plan into a URL-safe token.

    Returns "" when the plan has no state or encryption is unavailable,
    so the caller falls back to the legacy live-config path rather than
    emitting a broken URL.
    """
    if not isinstance(plan, CallPlan) or not _plan_is_worth_sealing(plan):
        return ""
    try:
        token = encrypt_value(encode_plan_payload(plan))
    except Exception as exc:
        # No CREDENTIAL_ENCRYPTION_KEY, or Fernet unavailable. Degrade to
        # the pre-Policy-A behaviour instead of breaking the call.
        log.warning("[call_plan] could not seal plan for %s: %s", plan.call_sid or "?", exc)
        return ""
    return token or ""


def open_call_plan(token, expected_call_sid: str = "") -> CallPlan | None:
    """Decrypt + decode a sealed plan. None means "use the legacy path".

    ``expected_call_sid`` binds the token to one call. Fernet proves the
    blob is intact and confidential; it does NOT prove the blob belongs
    to THIS call, so a valid token from call A would otherwise be
    accepted verbatim on call B and route that caller to A's numbers.
    When the ids disagree we return None and the caller falls back to
    live config — a mismatch is far more likely to be our own assembly
    bug than an attack, and degrading keeps the caller in a working call
    instead of killing it.

    None is also returned for a missing token (a call that started before
    this shipped, i.e. mid-deploy), an undecryptable token, or a
    malformed payload. Every one of those degrades to resolving
    destinations from live config — the behaviour in production today.
    """
    if not token or not isinstance(token, str):
        return None
    try:
        plaintext = decrypt_value(token)
    except Exception as exc:
        # decrypt_value already fails closed and logs. We add routing
        # context and degrade.
        log.warning("[call_plan] undecryptable plan token, falling back to live config: %s", exc)
        return None
    if not plaintext:
        return None
    plan = decode_plan_payload(plaintext)
    if plan is None:
        log.warning("[call_plan] malformed plan payload, falling back to live config")
        return None
    want = str(expected_call_sid or "").strip()
    if want and plan.call_sid and plan.call_sid != want:
        log.warning(
            "[call_plan] plan belongs to call %s but arrived on call %s — "
            "refusing to use it, falling back to live config",
            plan.call_sid, want,
        )
        return None
    return plan


def plan_steps(plan: CallPlan | None) -> list[RoutingStep]:
    """The plan's remaining steps, or [] when there is no plan."""
    if not isinstance(plan, CallPlan):
        return []
    return list(plan.steps)
