# audio-003-bargein-correctness

**Severity:** HIGH — Barge-in correctness.

## Scope (files)

- `STT_server/services/audio_ingest.py` — barge-in detection (134-160).
- `STT_server/config.py` — barge-in thresholds (148-167).
- `STT_server/services/playback_service.py` — interrupt plumbing (55-111).
- `STT_server/adapters/inworld_tts.py` — generation-bound frames (286-294).
- `STT_server/adapters/elevenlabs_tts.py` — same (158-194).
- `STT_server/adapters/rime_tts.py` — same (342-391).

## Approach (NEEDS_TESTS_FIRST)

1. Build a deterministic barge-in regression corpus: soft speech, loud
   speech, noise, echo, mid-sentence interruption.
2. Verify each adapter tags frames with `generation` captured at request
   start; playback drops any frame whose `generation != current_gen`.
3. On `interrupt_current_turn`: increment generation, drain playback, send
   Twilio `clear`, **and** issue provider-specific cancel for in-flight TTS:
   - Inworld: cancel task / close session.
   - ElevenLabs: cancel WS / close stream.
   - Rime: drop the HTTP response reader.
4. Measure: time from `barge_in_detected` to last stale frame on the wire;
   assert < 250 ms.

## Sub-agents

- `audio-003a-generation-tag` — generation stamping on every frame.
- `audio-003b-cancel-task` — per-provider cancellation.
- `audio-003c-corpus-harness` — deterministic replay corpus.
- `audio-003d-staleness-metric` — emit `bargein_staleness_ms` per call.

## Dependencies

- `audio-008-jitter-pacing` — staleness budget depends on outbound pacing.
- `audio-010-vad-calibration` — barge-in gates.

## Verification

```python
def test_bargein_stops_stale_frames():
    adapter = make_stub_tts(emit_for_ms=2000)
    gen0 = start_generation()
    adapter.start()
    time.sleep(0.4)  # let some frames out
    interrupt_current_turn()
    last = collect_frames_for(0.5)
    assert all(f.generation == gen0 for f in last[:5])
    assert last_frame_after_interrupt_ms < 250

def test_inworld_cancel_releases_session():
    ...
def test_elevenlabs_cancel_closes_stream():
    ...
```

## Acceptance

- Every adapter's frames carry the captured `generation`.
- Provider cancel is invoked on barge-in for all three TTS paths.
- `bargein_staleness_ms` < 250 ms in the deterministic corpus.
- Regression tests added under `tests/audio/bargein/`.
