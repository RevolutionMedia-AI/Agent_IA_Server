# audio-007-resampling-quality

**Severity:** MEDIUM — Resampling quality.

## Scope (files)

- `STT_server/adapters/inworld_stt_realtime.py` — 8→16 kHz interp (62-93).
- `STT_server/adapters/rime_tts.py` — N→8 kHz resample (130-199).
- `STT_server/adapters/tts_dispatcher.py` — OpenAI 24→8 kHz (161-189).
- `requirements.txt` / `requirements.docker.txt` — `scipy` declaration.

## Approach (NEEDS_TESTS_FIRST)

1. Spectral + RMS golden tests for tones at 1 kHz, impulses, voice bursts,
   silence. Run through every resample path (Inworld up, Rime down via
   scipy, OpenAI down via scipy, Rime fallback decimation).
2. Assert duration, sample count, peak amplitude, RMS, and spectral content
   within tolerances.
3. Smoke test confirms scipy path is exercised in the deployed image; log
   fallback invocations as a counter.
4. If scipy is unavailable at runtime, abort startup — never silently use
   the decimation fallback.

## Sub-agents

- `audio-007a-golden-resample` — `tests/audio/golden/resample/`.
- `audio-007b-scipy-presence-check` — startup self-test.
- `audio-007c-fallback-counter` — log every fallback to metrics.

## Dependencies

- `scipy.signal.resample_poly` already used; verify availability.

## Verification

```python
def test_resample_8_to_16_preserves_duration():
    x = sine_1kHz(duration_ms=200, rate=8000)
    y = upsample_to_16k(x)
    assert abs(len(y) / 16000 - 0.200) < 1e-3

def test_resample_24_to_8_suppresses_alias():
    x = noise(duration_ms=200, rate=24000)
    y = downsample_to_8k(x, method="scipy")
    assert spectrum_above_4kHz(y).max() < 1e-3

def test_startup_aborts_when_scipy_missing(monkeypatch):
    monkeypatch.setattr(scipy.signal, "resample_poly", None)
    with pytest.raises(RuntimeError):
        init_resampler()
```

## Acceptance

- Golden tests pass on the supported resample paths.
- Decimation fallback is never silently used; counter increments when
  invoked from a degraded branch.
- Inworld 8 kHz direct path is benchmarked for WER before/after.
