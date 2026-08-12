# audio-008-jitter-pacing

**Severity:** MEDIUM — Jitter / pacing.

## Scope (files)

- `STT_server/adapters/twilio_media.py` — seq/arrival observability
  (16-72).
- `STT_server/services/playback_service.py` — outbound pacing (290-303).
- `STT_server/STT_Server.py` — media dispatch loop (892-900).

## Approach (NEEDS_MEASUREMENT_FIRST)

1. Emit per-call metrics: inbound `arrival_jitter_ms_p95`,
   `sequence_gap_count`, `reorder_count`, `duplicate_count`.
2. Emit outbound: `send_deadline_drift_ms_p95`, `frame_send_jitter_ms_p95`.
3. Use a single monotonic stream clock (`time.monotonic_ns`) for both
   directions.
4. Only after data: add a small jitter buffer (e.g. 40 ms window) for
   inbound — never before evidence shows reorders/dupes/loss.
5. Outbound pacing: switch to deadline-absolute scheduling only if
   `send_deadline_drift_ms_p95` > 30 ms under load.

## Sub-agents

- `audio-008a-inbound-metrics` — seq/arrival counters + jitter estimator.
- `audio-008b-outbound-clock` — monotonic stream clock + drift emit.
- `audio-008c-jitter-buffer` — optional, gated on measurements.
- `audio-008d-deadline-pacing` — optional, gated on measurements.

## Dependencies

- `audio-012-observability` for the metric sink.

## Verification

```bash
# Replay a captured traffic trace with jitter
python agents/tools/replay_jitter.py --trace tests/fixtures/twilio/jitter.json --rate 50
# Metrics must include arrival_jitter_ms_p95 and sequence_gap_count
```

```python
def test_outbound_deadline_drift_under_load():
    s = start_outbound(load=200)
    s.send_for(seconds=10)
    assert metrics["send_deadline_drift_ms_p95"] < 30
```

## Acceptance

- Inbound and outbound jitter/drift/gap metrics are emitted on every call.
- No jitter buffer / deadline pacing added until evidence demands it.
- Replay harness reproduces known jitter cases for regression.
