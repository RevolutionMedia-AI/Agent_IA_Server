# audio-001-queue-backpressure

**Severity:** HIGH — Audio loss / backpressure.

## Scope (files)

- `STT_server/services/common.py` — `enqueue_nowait_with_drop` (12-26).
- `STT_server/config.py` — queue size knobs (68-77).
- `STT_server/domain/session.py` — queue ownership (105-130).
- `STT_server/services/audio_ingest.py` — drop-oldest in `stt_audio_queue`
  and `realtime_audio_queue` (86-94).

## Approach (NEEDS_MEASUREMENT_FIRST)

1. Tag every enqueued item with `kind` ∈ `{audio, control, mark, clear,
   segment_end}` and `generation`.
2. `enqueue_nowait_with_drop` becomes kind-aware: control items never get
   dropped; audio items may be dropped oldest-first; mark/clear/segment_end
   have higher priority than audio frames.
3. Emit metrics: `queue_residence_seconds`, `queue_high_water`, `drops_total{kind}`,
   `drops_total{provider}`, `sequence_gap_count`.
4. When audio must be dropped, record a synthetic "gap" marker into the
   downstream provider payload (model-specific: a flush event for OpenAI
   Realtime, a `Finalize` for Deepgram, etc.) so the STT model can recover.
5. After measurements, tune queue sizes; do not change them blind.

## Sub-agents

- `audio-001a-typed-item` — `AudioQueueItem` dataclass + serializers.
- `audio-001b-policy-dispatch` — kind-aware drop policy.
- `audio-001c-metrics-emit` — counters + gauges wired to
  `agents/tools/audio_metrics_collector.py`.
- `audio-001d-gap-marker` — model-specific gap signal.

## Dependencies

- `audio-012-observability` — needs the metrics collector first.

## Verification

```python
def test_control_items_never_dropped_under_backpressure():
    q = make_queue(maxsize=2)
    enqueue_nowait_with_drop(q, audio(b"a"))
    enqueue_nowait_with_drop(q, audio(b"b"))
    enqueue_nowait_with_drop(q, control(kind="clear"))
    enqueue_nowait_with_drop(q, control(kind="mark"))
    assert q.qsize() == 2
    assert peek_kind(q) in {"clear", "mark"}

def test_gap_marker_emitted_on_audio_drop():
    metrics = collect_during(lambda: drop_audio_n_times(5))
    assert metrics["gap_markers_emitted"] >= 1
```

## Acceptance

- No `clear` / `mark` / `segment_end` item is dropped by the policy.
- Drops are recorded per kind, provider, and generation.
- At least one gap-marker is emitted per audio drop on the supported paths.
- Existing `audio_ingest` tests still pass.
