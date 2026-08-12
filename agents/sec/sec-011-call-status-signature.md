# sec-011-call-status-signature

**Severity:** MEDIUM — Webhook validation gap / Sensitive logging.

> Status: baseline fix landed in commit `382c102` (signature required +
> sanitized upstream exceptions). Sub-agent exists to keep regression coverage.

## Scope (files)

- `STT_server/routes/api.py` — `/call-status` handler around 198-216.
- Tests: `tests/test_call_status_signature.py`.

## Approach (SAFE_CHANGE)

1. Validate `X-Twilio-Signature` using the resolved per-number/account token
   and canonical callback URL.
2. Bind the callback to an existing outbound call (lookup by `CallSid`);
   reject unknown calls before parsing / logging fields.
3. Remove sensitive fields from the log line; keep only `call_id`, `status`,
   `duration`, `outcome=accepted|rejected`.

## Sub-agents

- (none — narrow, focused)

## Dependencies

- `sec-003-twilio-voice-signature` — shares the HMAC helper.

## Verification

```python
def test_call_status_rejects_missing_signature():
    r = client.post("/call-status", data={"CallSid": "CA123"})
    assert r.status_code in (401, 403)

def test_call_status_rejects_unknown_call():
    r = post_signed("/call-status", {"CallSid": "CAunknown"}, TOKEN)
    assert r.status_code == 404

def test_call_status_accepts_known_signed_callback():
    seed_outbound(call_sid="CA123", token=TOKEN)
    r = post_signed("/call-status", {"CallSid": "CA123", "CallStatus": "completed"}, TOKEN)
    assert r.status_code == 200
```

## Acceptance

- Replay/invalid-signature/missing-signature paths reject before logging.
- Unknown CallSid returns 404; the call is not in memory.
- Log line never includes token, payload, or arbitrary caller fields.
