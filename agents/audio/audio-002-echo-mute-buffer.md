# audio-002-echo-mute-buffer

**Severity:** HIGH — Echo / input routing.

## Scope (files)

- `STT_server/services/audio_ingest.py` — mute buffer + assistant speaking
  gate (86-94, 120-160).
- `STT_server/services/playback_service.py` — `assistant_speaking` lifecycle
  + mark handling (100-111, 326-356).
- `STT_server/config.py` — `STT_MUTE_BUFFER_MAXLEN`, echo ignore window
  (70, 162-167).

## Approach (NEEDS_DESIGN_FIRST)

1. Record full-duplex fixtures with timestamps + Twilio marks so we can
   replay the conversation deterministically.
2. Replace the boolean `assistant_speaking` with a richer state machine:
   `idle → playing → playing_tail → ending → idle`. Each transition has an
   associated timestamp and playout position estimate.
3. Mute buffer re-injection policy becomes position-aware: only re-inject
   audio whose capture timestamp is **before** the playout position that
   triggered the current `mark`.
4. VAD evidence requirement: an inbound chunk is forwarded only if the VAD
   probability over a small window exceeds the echo gate AND the chunk's
   capture time is past the playout position by at least the round-trip
   estimate.
5. Calibration knob per cohort (handset / speakerphone / headset) without
   touching code.

## Sub-agents

- `audio-002a-state-machine` — `PlaybackState` enum + transitions.
- `audio-002b-fixture-recorder` — sanitize + replay harness.
- `audio-002c-position-aware-reinject` — policy that uses playout position
  + echo window.
- `audio-002d-cohort-calibration` — preset config schema.

## Dependencies

- `audio-008-jitter-pacing` — need jitter data to estimate RT.
- `audio-010-vad-calibration` — VAD evidence threshold.

## Verification

```python
def test_no_reinject_after_mark():
    state = PlaybackStateMachine()
    state.mark_acked(position_ms=1200)
    chunks = capture_during_playback_then_quiet(duration_ms=2000)
    forwarded = state.apply_echo_policy(chunks)
    assert not any(c.capture_ms < state.playout_position_ms for c in forwarded)

def test_real_user_speech_during_playback_forwarded():
    # user speaks at t=3000ms while agent is still playing
    forwarded = state.apply_echo_policy(synth_user_speech_at(3000))
    assert len(forwarded) >= 1
```

## Acceptance

- No re-injection of chunks whose capture timestamp is before playout
  position at mark-ack time.
- Real user speech captured during playback is forwarded without false
  suppression in the corpus.
- Speakerphone echo scenario shows reduced phantom transcripts in the
  replay harness.
