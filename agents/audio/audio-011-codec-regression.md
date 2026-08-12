# audio-011-codec-regression

**Severity:** MEDIUM — Codec regression risk.

## Scope (files)

- `STT_server/services/audio_codec.py` — G.711 encode/decode + parity
  asserts (53-117, 122-157).
- `STT_server/services/audio_frame_processor.py` — framing parity
  (97-129).

## Approach (SAFE_CHANGE)

1. Move G.111 vectors and property-based tests into pytest:
   - 256 round-trip encode/decode cases,
   - PCM boundary values (`INT16_MIN`, `INT16_MAX`, 0),
   - odd-byte inputs must raise,
   - cross-check against a third-party vector file (RFC 3551 / ITU-T G.711
     table where licence permits).
2. Drop dependency on `audioop` for parity tests; if `audioop` is used in
   production code, gate it behind a small wrapper so non-CPython runtimes
   can fall back.
3. Add framing tests: tail truncation, padded flush, multi-segment carry.

## Sub-agents

- `audio-011a-pytest-g711` — `tests/test_audio_codec_pytest.py` (replaces
  `__main__` asserts).
- `audio-011b-vector-fixtures` — `tests/fixtures/g711/` licensed vector set.
- `audio-011c-audioop-shim` — fallback wrapper if `audioop` is unavailable.
- `audio-011d-framing-tests` — pytest port of the framer's parity block.

## Dependencies

- Existing `tests/test_audio_codec.py` and
  `tests/test_audio_frame_processor.py` (already untracked).

## Verification

```bash
pytest tests/test_audio_codec_pytest.py tests/test_audio_frame_processor.py -v
pytest -k "g711 or framing"
```

```python
def test_g711_roundtrip_all_codes():
    for code in range(256):
        pcm = ulaw2lin(bytes([code]))  # 2 samples
        back = lin2ulaw(pcm)
        assert back == bytes([code])

def test_g711_odd_bytes_raises():
    with pytest.raises(ValueError):
        ulaw2lin(b"\x00\x01\x02")
```

## Acceptance

- `pytest tests/test_audio_codec_pytest.py` is green.
- Property-based tests pass against a fuzzer for 10 k random inputs.
- `audioop` no longer required at runtime; shim verified.
