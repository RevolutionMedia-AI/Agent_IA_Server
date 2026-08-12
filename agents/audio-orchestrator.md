# audio-orchestrator

Coordinates remediation of AUDIO-001 → AUDIO-012 across the Twilio → STT → LLM
→ TTS → playback pipeline. Enforces the audit's "measure first, fix second"
discipline: instrumentation lands before any tuning, and tuning is gated on
visible evidence.

## Responsibilities

1. Read `AUDIO_PIPELINE_AUDIT.md` and `agents/audio/*.md`.
2. Order work by the audit's P0/P1/P2 priorities (see `agents/README.md`).
3. Gate any **tuning** change behind a measurement artifact produced by the
   sub-agent's instrumentation step.
4. Centralize audio observability hooks (`agents/tools/audio_metrics_collector.py`)
   so per-finding metrics share one emitter.

## Shared state it owns

- `.audio-state.json` — finding statuses, fixture set versions, last measured
  values for `queue_drop_rate`, `barge_in_success_pct`, `tail_bytes_per_seg`,
  `pacing_drift_ms_p95`.
- `data/audio_metrics/` — append-only CSV/JSONL per-call summaries (no audio
  payload content — `audio-012` invariant).

## Sub-agents it manages

- `audio-001-queue-backpressure` — typed items, priority classes, control
  separation.
- `audio-002-echo-mute-buffer` — playout-position-aware echo policy.
- `audio-003-bargein-correctness` — generation invalidation + provider cancel.
- `audio-004-tail-truncation` — segment-level carry / final flush.
- `audio-005-unbounded-speech` — bounded `speech_frames` + cleanup.
- `audio-006-format-validation` — `start.mediaFormat` + base64 size gates.
- `audio-007-resampling-quality` — scipy path verified, fallback bounded.
- `audio-008-jitter-pacing` — measured before any buffer/pacing redesign.
- `audio-009-inworld-batching` — sample/duration-aware batching.
- `audio-010-vad-calibration` — corpus-driven threshold versioning.
- `audio-011-codec-regression` — pytest G.711 + framing parity.
- `audio-012-observability` — per-call summary without payload content.

## Cross-cutting dependencies

- `audio-001` and `audio-012` touch every queue/buffer — instrument first,
  restructure second.
- `audio-002` / `audio-003` need `audio-008` jitter data to avoid chasing
  echo symptoms caused by pacing drift.
- `audio-004` and `audio-011` share the framer — fix codec regressions before
  re-tuning tail handling.

## Exit condition

All 12 sub-agents reach `done` with at least one measurement artifact per
"NEEDS_MEASUREMENT_FIRST" finding. Existing `tests/test_audio_codec.py` and
`tests/test_audio_frame_processor.py` continue to pass.

## Non-goals

- Does not capture or store any caller audio.
- Does not raise thresholds without corpus evidence.
- Does not introduce new dependencies unless `audio-007` proves the fallback
  path is unacceptable.
