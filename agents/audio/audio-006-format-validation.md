# audio-006-format-validation

**Severity:** MEDIUM — Format validation gap.

## Scope (files)

- `STT_server/STT_Server.py` — start handler (559-579), media dispatch
  (892-900).
- `STT_server/services/audio_ingest.py` — base64 decode + decode μ-law
  (52-56, 96-104).
- `STT_server/config.py` — `INBOUND_FORMAT` / rate (37-40).

## Approach (SAFE_CHANGE)

1. Parse `start.mediaFormat` once on the start event. Accept only
   `encoding=audio/pcmu|audio/mulaw`, `sampleRate=8000`, `channels=1`.
   Reject otherwise with a clear error event back to Twilio.
2. Store the validated format on the session; refuse to process media
   payloads whose payload length is incompatible with a 20 ms frame at
   8 kHz (160 bytes ± slack).
3. Use `base64.b64decode(..., validate=True)`; cap payload size per event
   to a configurable max (e.g. 4096 bytes raw).
4. Add tests with sanitized real Twilio envelopes (start + media + mark).

## Sub-agents

- `audio-006a-format-check` — strict schema validation, error event emitter.
- `audio-006b-base64-hardening` — `validate=True` + size cap.
- `audio-006c-twilio-fixtures` — sanitized fixtures under
  `tests/fixtures/twilio/`.

## Dependencies

- None.

## Verification

```python
def test_start_rejects_unsupported_sample_rate():
    r = dispatch({"event": "start", "start": {"mediaFormat": {"encoding": "audio/pcmu", "sampleRate": 16000, "channels": 1}}})
    assert r["type"] == "error"

def test_media_decode_rejects_oversized_payload():
    r = dispatch({"event": "media", "media": {"payload": "A" * 10000}})
    assert r.get("type") == "drop"

def test_valid_start_accepted():
    r = dispatch({"event": "start", "start": {"mediaFormat": {"encoding": "audio/pcmu", "sampleRate": 8000, "channels": 1}}})
    assert r["accepted"] is True
```

## Acceptance

- Any `mediaFormat` that is not PCMU/8000/1 causes the session to close
  with a structured error.
- Oversized or malformed base64 is counted + dropped without crashing.
- Sanitized Twilio fixture suite passes.
