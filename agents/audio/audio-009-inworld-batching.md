# audio-009-inworld-batching

**Severity:** MEDIUM — Provider buffering latency.

## Scope (files)

- `STT_server/adapters/inworld_stt_realtime.py` — chunk/frame batching
  (46-49, 118-153).
- `STT_server/services/audio_ingest.py` — chunk size assumption (52-94).

## Approach (NEEDS_MEASUREMENT_FIRST)

1. Histogram: bytes and duration of every inbound media item.
2. Histogram: batch wait time per send.
3. Replace the implicit "8 chunks = 8×20 ms" assumption with an explicit
   sample/duration accumulator: flush when accumulated samples ≥ 8000×0.16
   (1280 samples) OR a deadline of 80 ms since first chunk in batch.
4. Emit metric `inworld_batch_wait_ms_p50/p95`,
   `inworld_batch_samples_p50/p95`.

## Sub-agents

- `audio-009a-histogram-emit` — bucketed counts + percentile export.
- `audio-009b-duration-batching` — sample-aware flush.
- `audio-009c-deadline-flush` — fallback flush at 80 ms.

## Dependencies

- `audio-012-observability` for histograms.

## Verification

```python
def test_batch_flushes_on_sample_threshold():
    adapter = InworldBatch(target_samples=1280)
    feed_chunks([b"\x00" * 160] * 7)  # 7 × 20 ms = 1120 samples
    assert adapter.flush_pending_samples() == 0
    feed_chunks([b"\x00" * 160])      # 8 × 20 ms = 1280 samples
    assert adapter.flush_pending_samples() == 1280

def test_deadline_flush_fires():
    adapter = InworldBatch(target_samples=99999, deadline_ms=80)
    feed_chunks([b"\x00" * 160])
    await asyncio.sleep(0.1)
    assert adapter.flush_pending_samples() == 160
```

## Acceptance

- Batches are sized in samples / ms, not chunk count.
- A deadline flush prevents unbounded latency on low-traffic calls.
- Histograms emitted and reviewed before any further tuning.
