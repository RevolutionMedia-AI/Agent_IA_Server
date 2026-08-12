# audio-005-unbounded-speech

**Severity:** HIGH — Unbounded audio accumulation.

## Scope (files)

- `STT_server/domain/session.py` — `speech_frames` field (88-90).
- `STT_server/services/audio_ingest.py` — append + END_SILENCE clear
  (106-118, 180-214).

## Approach (SAFE_CHANGE)

1. Instrument: `speech_frames_bytes`, `speech_frames_duration_ms`,
   `speech_frames_resets_total`.
2. Enforce a hard cap on `speech_frames` bytes/duration; on cap, emit a
   synthetic end-of-utterance to STT and reset the local buffer without
   touching the STT audio queue.
3. Ensure `vad_buffer`, `pre_speech_frames`, and `speech_frames` are all
   cleared on session cleanup.

## Sub-agents

- `audio-005a-byte-counter` — fast counters + tests.
- `audio-005b-hard-cap` — bounded list + reset path.
- `audio-005c-cleanup-audit` — `session.close()` must drain all three.

## Dependencies

- `audio-012-observability` — bytes/duration metric export.

## Verification

```python
def test_continuous_signal_resets_at_cap():
    session = make_session(cap_bytes=32000)  # 4s @ 8 kHz μ-law
    feed_silence_then_continuous_speech(duration_s=10)
    assert metrics["speech_frames_resets_total"] >= 2
    assert session.speech_frames_byte_len() <= 32000

def test_cleanup_clears_vad_buffers():
    session = make_session()
    session.feed(...)
    await session.close()
    assert session.vad_buffer.tell() == 0
    assert len(session.pre_speech_frames) == 0
    assert len(session.speech_frames) == 0
```

## Acceptance

- A signal that would otherwise grow `speech_frames` indefinitely triggers
  a controlled reset before memory grows past the cap.
- All three buffers are empty after `session.close()`.
- Counters are visible in the per-call summary.
