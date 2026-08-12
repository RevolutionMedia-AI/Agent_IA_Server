# sec-003-twilio-voice-signature

**Severity:** HIGH — Webhook validation gap (fail-open on parse failure).

## Scope (files)

- `STT_server/STT_Server.py` — `POST /voice` handler around 243-340.

## Approach (NEEDS_TESTS_FIRST)

1. Reject parse failures and empty bodies immediately; do not fall through to
   TwiML generation.
2. Require `To` + resolved per-number token before any TwiML generation.
3. Require a valid `X-Twilio-Signature` (HMAC) on every production request.
4. Centralize the verification call through a single helper that returns a
   typed result (no `if form_dict:` gate around it).
5. Restrict the local-development bypass behind an explicit env flag.

## Sub-agents

- (none — single-file fix)

## Dependencies

- Existing HMAC helper in `adapters/twilio_api.py:23-69`.
- Per-number token lookup helper (likely needs to be added or reused from
  `db_phone_numbers.py`).

## Verification

```python
# tests/test_twilio_voice_signature.py
@pytest.mark.parametrize("body,headers", [
    (b"", {"X-Twilio-Signature": "x"}),                  # empty body
    (b"%ZZ", {"X-Twilio-Signature": "x"}),               # malformed form
    (b"To=%2B1...", {}),                                 # missing signature
    (b"To=%2B1...", {"X-Twilio-Signature": "wrong"}),    # invalid signature
    (b"To=%2B15550000000", {"X-Twilio-Signature": "x"}), # unknown number
])
def test_voice_rejects_unsigned_or_unparsable(body, headers):
    r = client.post("/voice", content=body, headers=headers)
    assert r.status_code in (401, 403, 400)

def test_voice_accepts_valid_signature():
    body = form_encode({"To": "+15550000000", "From": "+15559999999"})
    sig = compute_twilio_signature(PUBLIC_URL + "/voice", body, TOKEN)
    r = client.post("/voice", content=body, headers={"X-Twilio-Signature": sig})
    assert r.status_code == 200 and "<Response>" in r.text
```

## Acceptance

- Empty / malformed / unsigned / invalid-signature / unknown-number paths all
  reject before TwiML generation.
- A single signed happy-path test passes.
- No `if form_dict:` gate around signature validation remains.
