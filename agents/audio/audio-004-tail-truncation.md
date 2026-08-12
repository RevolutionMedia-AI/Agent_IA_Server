# audio-004-tail-truncation

**Severity:** HIGH — Frame truncation.

## Scope (files)

- `STT_server/services/audio_frame_processor.py` — flush policy (26-59).
- `STT_server/services/playback_service.py` — playback framer (196-204).
- `STT_server/adapters/inworld_tts.py` — tail behavior (212-220, 394-399).
- `STT_server/adapters/elevenlabs_tts.py` — tail behavior (44, 302-303).
- `STT_server/adapters/tts_dispatcher.py` — OpenAI/Rime paths (169-187,
  240-246).

## Approach (NEEDS_TESTS_FIRST)

1. Record `dropped_tail_bytes` per provider/segment/call_id.
2. Golden tests with chunks of randomized sizes and phrases ending in
   transient consonants (`s`, `t`, `k`, `sh`).
3. Carry option: a final flush that emits the remainder as a padded last
   frame with a configurable padding strategy (zero pad / μ-law silence byte
   0xFF) — only enabled when the segment is the last in the turn.
4. Cross-segment continuity: pass `carry_bytes` between segments of the same
   generation so the framer joins consecutive chunks into 160-byte frames
   before dropping anything.

## Sub-agents

- `audio-004a-tail-metrics` — counters + dropped-byte histogram.
- `audio-004b-golden-tests` — `tests/audio/golden/tail_truncation/`.
- `audio-004c-final-flush` — opt-in final flush for last segment of turn.
- `audio-004d-segment-carry` — `carry_bytes` between same-gen segments.

## Dependencies

- `audio-011-codec-regression` — codec fix lands before tail retuning.

## Verification

```python
def test_dropped_tail_bytes_recorded_per_segment():
    before = metrics["dropped_tail_bytes_total"]
    play_segment(b"a" * 173)  # 13 bytes short
    after = metrics["dropped_tail_bytes_total"]
    assert after - before >= 13

def test_final_segment_emits_remainder_as_padded_frame():
    frames = capture_frames(play_segment(b"x" * 173, final=True))
    assert frames[-1] == b"\xff" * 160  # last frame is padded silence

def test_intermediate_segment_still_drops_tail():
    frames = capture_frames(play_segment(b"x" * 173, final=False))
    assert frames[-1] != b"\xff" * 160
```

## Acceptance

- `dropped_tail_bytes` is recorded per provider/segment.
- Golden tests pass with random chunk sizes.
- Final-segment carry/flush path reduces perceptually-significant truncation.
- Intermediate segments unchanged in behaviour.
