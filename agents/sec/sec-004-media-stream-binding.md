# sec-004-media-stream-binding

**Severity:** HIGH — Auth gap / Input validation gap (unauthenticated WS).

## Scope (files)

- `STT_server/STT_Server.py` — `media_stream` handler around 512-679.
- New module: `STT_server/security/stream_nonce.py` — mint/verify/bind.

## Approach (NEEDS_DESIGN_FIRST)

1. Generate a short-lived, single-use stream nonce in the signed `/voice`
   handler after signature validation passes.
2. Embed the nonce in the TwiML stream URL parameter; bind it to expected
   `phone`, `tenant_id`, `agent_id`, and `callSid`.
3. In `media_stream`, reject `start` events whose nonce is missing, expired,
   already consumed, or whose `callSid` does not match.
4. Add origin/host, frame-size, rate, and event-order validation at the WS
   boundary.
5. Atomic consume via Redis-or-similar with TTL = call expected lifetime + 30s.

## Sub-agents

- `sec-004a-nonce-store` — interface + TTL store (in-memory acceptable for
  unit tests, Redis in prod).
- `sec-004b-twilml-binding` — extend TwiML generator to include nonce.
- `sec-004c-ws-origin-check` — origin/host allowlist + frame limits.

## Dependencies

- `sec-003-twilio-voice-signature` — nonce only minted on a validated request.

## Verification

```python
# tests/test_media_stream_binding.py
def test_ws_rejects_missing_nonce():
    ...
def test_ws_rejects_expired_nonce():
    ...
def test_ws_rejects_replayed_nonce():
    ...
def test_ws_rejects_callSid_mismatch():
    ...
def test_ws_accepts_signed_bound_nonce():
    ...
```

## Acceptance

- Direct WS connection without a valid bound nonce is rejected before any
  tenant/agent lookup.
- TwiML produced by `/voice` carries the nonce; consumers can extract it.
- Replay/expire/cross-call attempts all fail.
