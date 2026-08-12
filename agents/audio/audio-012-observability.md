# audio-012-observability

**Severity:** LOW — Audio observability gap.

## Scope (files)

- `STT_server/adapters/twilio_media.py` — seq/arrival counters.
- `STT_server/services/audio_frame_processor.py` — framer stats.
- `STT_server/services/playback_service.py` — outbound + mark RTT.
- `STT_server/STT_Server.py` — mark RTT measurement.

## Approach (SAFE_CHANGE)

1. One per-call summary, emitted at end-of-call, **metadata only** (no
   audio content, no transcripts, no tool payloads):
   - format observed (`pcmu/8000/1`),
   - bytes / frames in + out,
   - seq gaps / dupes / reorders,
   - per-queue high-water + drop count,
   - tail bytes dropped per provider,
   - resample path taken (scipy / fallback),
   - pacing drift ms p50/p95,
   - barge-in count + last staleness ms,
   - mark RTT ms p50/p95,
   - generation transitions.
2. Centralize emission through `agents/tools/audio_metrics_collector.py`.
3. Each summary carries `call_id`, `tenant_id`, `agent_id`, and a
   correlation ID; nothing user-identifying.

## Sub-agents

- `audio-012a-summary-record` — `AudioCallSummary` dataclass.
- `audio-012b-collector` — `agents/tools/audio_metrics_collector.py`.
- `audio-012c-emit-hooks` — wire every counter into the summary.

## Dependencies

- This is the substrate other agents build on; ship first or in parallel
  with `audio-001`.

## Verification

```python
def test_summary_contains_required_keys():
    s = collect_summary(make_fake_call())
    for key in ("format", "bytes_in", "bytes_out", "seq_gaps", "queue_high_water",
                "drop_count", "tail_bytes_dropped", "resample_path",
                "pacing_drift_ms_p95", "bargein_count", "mark_rtt_ms_p95",
                "generation_transitions"):
        assert key in s

def test_summary_contains_no_audio_content():
    s = collect_summary(make_fake_call(payload=b"PCMUDATA"))
    assert b"PCMUDATA" not in str(s).encode()
```

## Acceptance

- One summary emitted per call with all required fields populated.
- No raw audio or transcript content appears in any field.
- Existing `audio_metrics.py` continues to function; new collector is
  additive.
