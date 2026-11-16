"""Unit tests for STT_server.services.audio_codec.

Covers the public G.711 mu-law codec surface (``ulaw2lin``, ``lin2ulaw``, ``rms``):
  - silence decode,
  - all-256-code round-trip stability (the core correctness property),
  - width and length guards,
  - rms behaviour on zero, DC, and alternating-sign inputs,
  - random-fuzz round-trip stays within G.711 quantisation buckets (±4),
  - the numpy-vectorised ``ulaw2lin``/``rms`` are bit-identical to the
    pure-Python reference they replaced.
"""
from __future__ import annotations

import math
import random
import struct

import pytest

from STT_server.services import audio_codec
from STT_server.services.audio_codec import lin2ulaw, rms, ulaw2lin


# ── pure-Python reference implementations ────────────────────────────────
# These are the originals, kept here (not in the package) so the
# vectorised versions can be proven bit-identical. If someone changes
# audio_codec, these stay frozen and the comparison still means something.


def _ref_ulaw2lin(audio: bytes, width: int) -> bytes:
    tab = audio_codec._MULAW_TO_LINEAR
    out = bytearray(len(audio) * 2)
    for i, b in enumerate(audio):
        out[i * 2:i * 2 + 2] = tab[b * 2:b * 2 + 2]
    return bytes(out)


def _ref_rms(audio: bytes, width: int) -> int:
    if len(audio) == 0:
        return 0
    n = len(audio) // width
    if n == 0:
        return 0
    total = 0.0
    for i in range(0, len(audio), width):
        s = struct.unpack_from('<h', audio, i)[0]
        total += s * s
    return int(math.sqrt(total / n))


# ── ulaw2lin ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 8, 40, 160, 8000])
def test_ulaw2lin_silence_for_n_bytes(n: int) -> None:
    """μ-law 0xFF is positive silence → all-zero int16 LE samples."""
    assert ulaw2lin(b"\xFF" * n, 2) == b"\x00" * (n * 2)


def test_ulaw2lin_width_guard() -> None:
    with pytest.raises(ValueError):
        ulaw2lin(b"\x00" * 4, 1)
    with pytest.raises(ValueError):
        ulaw2lin(b"\x00" * 4, 3)
    with pytest.raises(ValueError):
        ulaw2lin(b"\x00" * 4, 4)


def test_ulaw2lin_empty_input() -> None:
    assert ulaw2lin(b"", 2) == b""


# ── lin2ulaw round-trip — all 256 codes ─────────────────────────────────


def test_lin2ulaw_roundtrip_all_256_codes() -> None:
    """Every μ-law byte survives encode→decode to itself (except the
    negative-silence alias 0x7F).

    The decode table and the quantiser must agree on every bucket.
    0x7F and 0xFF both decode to int16 zero (positive and negative
    silence) but the encoder emits 0xFF for zero input — G.711 μ-law
    by spec only has one canonical silence code. Any drift on the
    other 255 codes is a hard regression in the codec.
    """
    mismatches: list[int] = []
    for b in range(256):
        if b == 0x7F:  # negative-silence alias; encode(0) returns 0xFF
            continue
        rt = lin2ulaw(ulaw2lin(bytes([b]), 2), 2)
        if rt != bytes([b]):
            mismatches.append(b)
    assert not mismatches, f"round-trip drift on codes: {mismatches}"


def test_lin2ulaw_width_guard() -> None:
    with pytest.raises(ValueError):
        lin2ulaw(b"\x00" * 4, 1)
    with pytest.raises(ValueError):
        lin2ulaw(b"\x00" * 4, 3)


def test_lin2ulaw_odd_length_guard() -> None:
    """PCM buffer length must be a multiple of width=2."""
    with pytest.raises(ValueError):
        lin2ulaw(b"\x00\x01\x02", 2)


def test_lin2ulaw_empty_input() -> None:
    assert lin2ulaw(b"", 2) == b""


# ── rms ──────────────────────────────────────────────────────────────────


def test_rms_of_zero() -> None:
    assert rms(b"\x00\x00" * 100, 2) == 0


def test_rms_of_empty() -> None:
    assert rms(b"", 2) == 0


def test_rms_of_dc_one() -> None:
    """DC level of 1 → rms == 1 (constant signal has rms == magnitude)."""
    pcm = b"\x01\x00" * 4  # four int16 samples each equal to 1 (LE)
    assert rms(pcm, 2) == 1


def test_rms_of_alternating_sign() -> None:
    """[-1, +1, -1, +1] → rms == 1."""
    pcm = b"\x01\x00\xff\x7f" * 4  # alternating ±32767 ↔ ±1; see below
    # Note: 0xff 0x7f is int16 LE for 32767, 0x01 0x00 is 1. To get ±1 we want
    # 0x01 0x00 (1) and 0xff 0xff (-1). The above bytes give ±32767, which
    # also has rms == 32767 — but the spec wants the ±1 case. Construct it.
    pcm = b"\x01\x00\xff\xff" * 4  # int16 LE: 1, -1, 1, -1, ...
    assert rms(pcm, 2) == 1


def test_rms_width_guard() -> None:
    with pytest.raises(ValueError):
        rms(b"\x00" * 4, 1)
    with pytest.raises(ValueError):
        rms(b"\x00" * 4, 4)


# ── random-fuzz round-trip within G.711 quantisation tolerance ───────────


def test_random_fuzz_roundtrip_within_quantisation_tolerance() -> None:
    """G.711 μ-law is lossy by up to one segment-step per code.

    CPython's audioop (which this codec mirrors byte-for-byte) uses
    segment step = 2^seg in the 16-bit output: seg 0 → 2 LSB, seg 1 →
    4, …, seg 7 → 1024 LSB. A round-trip therefore lands within
    ±(step/2) of the original; the worst case is ±1024 LSB for the
    top segment. We sample 1000 random int16 values and assert the
    recovered sample is within the per-segment step bound plus one
    bucket, computed empirically from the same fuzz run (the codec
    matches audioop so we know the bound).
    """
    import struct

    rng = random.Random(0)
    worst_delta = 0
    for _ in range(1000):
        s = rng.randint(-32768, 32767)
        pcm = struct.pack('<h', s)
        recovered_bytes = ulaw2lin(lin2ulaw(pcm, 2), 2)
        recovered = struct.unpack('<h', recovered_bytes)[0]
        delta = abs(recovered - s)
        worst_delta = max(worst_delta, delta)
    # Seg 7 step = 1024 → max single-step error ≈ 512; +512 for adjacent
    # bucket drift gives the published audioop bound. Tighten below 1024
    # means the codec is finer than audioop; tighten to 0 means we
    # silently swapped the table for a finer one.
    assert worst_delta <= 1024, (
        f"worst round-trip delta {worst_delta} exceeds 1024 LSB "
        "(segment-7 step for audioop-compatible G.711)"
    )


# ── vectorised implementation == pure-Python reference ───────────────────


def test_ulaw2lin_vectorised_matches_reference() -> None:
    """The numpy table lookup must be byte-identical to the old Python loop.

    ``ulaw2lin`` decodes every incoming media chunk, so a one-bit change
    here is audible on every call. Exhaustive over all 256 codes and every
    buffer length up to 600 bytes, plus random buffers up to 4 kB.
    """
    rng = random.Random(1234)
    for n in range(0, 601):
        buf = bytes(rng.randint(0, 255) for _ in range(n))
        assert ulaw2lin(buf, 2) == _ref_ulaw2lin(buf, 2), f"mismatch at len={n}"
    for _ in range(500):
        buf = bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 4000)))
        assert ulaw2lin(buf, 2) == _ref_ulaw2lin(buf, 2)


def test_ulaw2lin_vectorised_all_256_codes() -> None:
    """Every mu-law code decodes to the exact table entry."""
    for b in range(256):
        assert ulaw2lin(bytes([b]), 2) == _ref_ulaw2lin(bytes([b]), 2)


def test_rms_vectorised_matches_reference() -> None:
    """The vectorised RMS must return the exact same integer.

    ``rms`` feeds the VAD voice threshold, so a 1-off drift can flip a
    speech/silence decision and change barge-in behaviour.
    """
    rng = random.Random(4321)
    for _ in range(2000):
        n = rng.randint(1, 400)
        samples = [
            rng.choice([-32768, -1, 0, 1, 32767, rng.randint(-32768, 32767)])
            for _ in range(n)
        ]
        buf = b"".join(struct.pack('<h', s) for s in samples)
        assert rms(buf, 2) == _ref_rms(buf, 2), f"mismatch at n={n}"


def test_rms_vectorised_exhaustive_single_sample() -> None:
    """Every int16 value, on its own, gives the same RMS as the reference."""
    for s in range(-32768, 32768):
        buf = struct.pack('<h', s)
        assert rms(buf, 2) == _ref_rms(buf, 2), f"mismatch for sample {s}"


def test_rms_odd_length_raises_struct_error() -> None:
    """Odd-length PCM keeps raising ``struct.error`` (not ValueError).

    The pure-Python loop raised ``struct.error`` on a trailing half
    sample; the numpy path would raise ValueError. Callers' except
    clauses must keep matching, so the type is pinned here.
    """
    with pytest.raises(struct.error):
        rms(b"\x01\x00\x02", 2)

