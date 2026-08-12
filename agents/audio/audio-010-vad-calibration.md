# audio-010-vad-calibration

**Severity:** MEDIUM — VAD calibration.

## Scope (files)

- `STT_server/config.py` — VAD mode, RMS, START/END frames, echo window
  (43-49, 145-167).
- `STT_server/services/audio_ingest.py` — VAD integration (38-49, 116-160,
  197-214).

## Approach (NEEDS_MEASUREMENT_FIRST)

1. Build a labeled corpus:
   - telephony bandwidth (8 kHz PCMU),
   - handset vs speakerphone vs headset,
   - multiple languages,
   - background noise + double-talk.
2. Measure FPR / FNR, endpoint delay, barge-in success rate per cohort
   with the existing config.
3. Version the threshold presets as `vad_presets/{cohort}.json`; runtime
   loads by tenant or call metadata.
4. Rollback path: each preset retains the previous version.

## Sub-agents

- `audio-010a-corpus-builder` — sanitization + label schema.
- `audio-010b-replay-harness` — `agents/tools/replay_vad.py`.
- `audio-010c-preset-loader` — JSON preset + version.
- `audio-010d-metrics` — per-cohort FPR/FNR endpoint delay.

## Dependencies

- Consent / anonymization pipeline before corpus build.

## Verification

```bash
python agents/tools/replay_vad.py \
    --corpus tests/fixtures/vad/corpus.jsonl \
    --config config.py --preset vad_presets/handset.json
# exit 0 only if FPR <= 5% and FNR <= 5%
```

```python
def test_preset_version_fallback():
    loader = VADPresetLoader()
    loader.load("handset", version=2)  # newer missing
    assert loader.active_version == 1
```

## Acceptance

- Labeled corpus committed under `tests/fixtures/vad/` (sanitized).
- Preset loader with version fallback is the single path used at runtime.
- Cohort-level FPR/FNR/barge-in metrics emitted on every call.
