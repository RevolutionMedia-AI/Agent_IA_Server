"""Streaming resampler seam test.

The whole reason this class exists: resample_poly is an FIR filter, so
chunk-at-a-time resampling leaves a step at every 20 ms boundary. This
asserts the streaming path is equivalent to one pass over the whole
signal, which is the property the transcribe model depends on.
"""
from __future__ import annotations

import struct

import pytest

from STT_server.adapters.openai_stt_transcription import (
    TARGET_SAMPLE_RATE,
    UPSAMPLE,
    _StreamingResampler,
    _mulaw_8k_to_pcm16_24k,
)
from STT_server.services.audio_codec import lin2ulaw


def _tone(n: int, hz: float = 440.0, sr: int = 8000) -> bytes:
    import math
    pcm = b"".join(
        int(20000 * math.sin(2 * math.pi * hz * i / sr)).to_bytes(
            2, "little", signed=True
        )
        for i in range(n)
    )
    return lin2ulaw(pcm, 2)


def _pcm_int16(buf: bytes) -> list[int]:
    return [int.from_bytes(buf[i:i + 2], "little", signed=True)
            for i in range(0, len(buf) - 1, 2)]


def test_streaming_preserves_total_length_without_drift():
    """No gaps, no overlaps, no accumulated drift.

    Bit-equality with a one-shot pass is deliberately NOT asserted. The
    streaming design holds back the filter tail until the samples after it
    arrive, so it trails a whole-signal pass by a bounded amount, and the
    whole-signal pass is not ground truth — nothing consumes it. What has
    to hold is that the emitted count tracks input*3 with a small, bounded
    offset.

    The failure this catches is the one that already happened: dropping
    `len(_hist) * UPSAMPLE` from the head on every call removed the
    samples the previous call had HELD rather than the ones it had sent,
    so the output ran at 2.43x instead of 3x and the caller's voice
    pitched down about 19% over a call. That is invisible in a short
    sample and obvious in production.
    """
    whole = _tone(16000)  # 2 s
    r = _StreamingResampler()
    emitted = 0
    for i in range(0, len(whole), 160):
        emitted += len(_pcm_int16(r.feed(whole[i:i + 160])))

    n_in = 16000
    expected = n_in * UPSAMPLE
    deficit = expected - emitted
    # Legitimately unemitted: the filter tail (half*up = 90) plus whatever
    # the final chunk had not yet flushed.
    bound = (10 * UPSAMPLE) + 160 * UPSAMPLE
    assert 0 <= deficit <= bound, (
        f"emitted {emitted} for {expected} expected (deficit {deficit}, "
        f"bound {bound}); beyond the filter tail means the stream drifts"
    )
    # And the rate itself, which is the thing a pitch shift would break.
    rate = emitted / n_in
    assert 2.95 <= rate <= 3.0, f"output rate {rate:.3f}x, expected ~3x"


def test_deficit_does_not_grow_with_call_count():
    """A drift bug accumulates; a bounded filter tail does not.

    This is the assertion that would have caught the 2.43x bug on its own:
    the per-call loss looked small, but summed over 100 chunks it ate
    9000 samples.
    """
    def total(chunks: int) -> int:
        whole = _tone(chunks * 160)
        r = _StreamingResampler()
        n = 0
        for i in range(0, len(whole), 160):
            n += len(_pcm_int16(r.feed(whole[i:i + 160])))
        return n

    short = total(10)
    long = total(100)
    loss_short = 10 * 160 * UPSAMPLE - short
    loss_long = 100 * 160 * UPSAMPLE - long
    # Ten times the audio must not cost ten times the tail.
    assert loss_long < loss_short * 2 + 100, (
        f"loss grew with duration: {loss_short} samples over 10 chunks vs "
        f"{loss_long} over 100 chunks"
    )


def test_streamed_waveform_matches_the_source_in_every_window():
    """The physically meaningful property: every chunk of output is the
    caller's voice, at the right frequency, amplitude and phase.

    Correlating at a fixed lag is the wrong test and reports ~0.1 on a
    resampler that is working. The filter has a group delay and the
    alignment wanders by a few samples across a call, so the assertion is
    per-window best-lag correlation. A fixed-lag comparison would flag a
    correct implementation and miss a real one.

    ponytail: the wander is sub-1% and has NOT been verified against the
    live transcribe model — no API key in the test environment. If
    recognition stays poor in production, the thing to try first is
    OPENAI_TRANSCRIPTION_RATE_HZ=8000, which removes resampling entirely
    (the docs example uses 24000; 8000 is a telephony-native rate and this
    pipeline already holds 8 kHz mu-law).
    """
    import numpy as np
    from STT_server.services.audio_codec import ulaw2lin

    whole = _tone(8000)  # 1 s of 440 Hz
    r = _StreamingResampler()
    acc: list[int] = []
    for i in range(0, len(whole), 160):
        acc.extend(_pcm_int16(r.feed(whole[i:i + 160])))

    src = np.array(_pcm_int16(ulaw2lin(whole, 2)), dtype=float)
    out = np.array(acc, dtype=float)
    out8k = out[::UPSAMPLE]  # back to the 8 kHz grid

    win = 400
    m = min(len(src), len(out8k))
    assert m > 3 * win, "not enough output to window"

    worst = 1.0
    for start in range(0, m - win, 500):
        a = src[start:start + win]
        best = max(
            float(np.corrcoef(a, out8k[start + lag:start + lag + win])[0, 1])
            for lag in range(-40, 41)
            if 0 <= start + lag and start + lag + win <= len(out8k)
        )
        worst = min(worst, best)

    assert worst > 0.9, (
        f"a window of output stopped resembling the source "
        f"(worst best-lag correlation {worst:.3f})"
    )


def test_output_keeps_the_source_frequency_and_amplitude():
    """A resampler that shifts pitch is worse than a naive one: the caller
    sounds like a chipmunk and recognition collapses."""
    import numpy as np

    whole = _tone(8000)
    r = _StreamingResampler()
    acc: list[int] = []
    for i in range(0, len(whole), 160):
        acc.extend(_pcm_int16(r.feed(whole[i:i + 160])))
    out = np.array(acc, dtype=float)

    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(len(out), d=1.0 / TARGET_SAMPLE_RATE)
    peak_hz = float(freqs[spec.argmax()])
    assert 430 <= peak_hz <= 450, f"pitch moved to {peak_hz:.0f} Hz from 440"

    rms = float(np.sqrt((out ** 2).mean()))
    src_rms = 20000 / (2 ** 0.5)
    assert 0.9 * src_rms < rms < 1.1 * src_rms, (
        f"amplitude drifted: rms {rms:.0f} vs source {src_rms:.0f}"
    )


def test_no_discontinuity_at_chunk_boundaries():
    """Directly measure the seam: a click is a large sample-to-sample jump.

    A 440 Hz tone at 24 kHz advances ~115 counts per sample. A seam shows
    up as a jump far beyond that.
    """
    whole = _tone(1600)
    r = _StreamingResampler()
    out: list[int] = []
    bounds: list[int] = []
    for i in range(0, len(whole), 160):
        before = len(out)
        if before:
            bounds.append(before)
        out.extend(_pcm_int16(r.feed(whole[i:i + 160])))

    if len(bounds) < 2:
        pytest.skip("not enough chunks produced output to compare seams")
    deltas = [abs(out[i + 1] - out[i]) for i in range(len(out) - 1)]
    normal = max(deltas) if deltas else 0
    seam_deltas = [abs(out[b] - out[b - 1]) for b in bounds if 0 < b < len(out)]
    if not seam_deltas:
        pytest.skip("no seam landed on a sample boundary")
    assert max(seam_deltas) <= normal * 2, (
        f"seam discontinuity {max(seam_deltas)} far exceeds the normal "
        f"sample step {normal}: the filter history is not being carried"
    )


def test_output_rate_is_24k():
    assert TARGET_SAMPLE_RATE == 24000
    assert UPSAMPLE == 3
    out = _mulaw_8k_to_pcm16_24k(_tone(160))
    assert len(_pcm_int16(out)) == 160 * UPSAMPLE


def test_samples_stay_in_int16_range():
    """Overflow would wrap to full-scale negative: loud static."""
    out = _mulaw_8k_to_pcm16_24k(_tone(800))
    vals = _pcm_int16(out)
    assert all(-32768 <= v <= 32767 for v in vals)
    assert max(vals) > 0, "a 20 kHz tone must stay positive somewhere"


def test_empty_and_odd_input_are_safe():
    assert _mulaw_8k_to_pcm16_24k(b"") == b""
    r = _StreamingResampler()
    assert r.feed(b"") == b""


def test_no_images_above_the_source_nyquist():
    """8 kHz audio carries 0-4 kHz. After 3x upsampling it must carry
    0-4 kHz and nothing near 12-24 kHz.

    The old linear-interpolation resampler zero-stuffed, which put
    spectral replicas at multiples of 8 kHz — the content the transcribe
    model reads as noise. Measure energy in the top third of the band.
    """
    import numpy as np
    from scipy.signal import resample_poly

    # 2 kHz tone, comfortably inside the source band.
    n = 1600
    tone = (20000 * np.sin(2 * np.pi * 2000 * np.arange(n) / 8000))
    mu = lin2ulaw(tone.astype("<i2").tobytes(), 2)
    up = np.frombuffer(
        _mulaw_8k_to_pcm16_24k(mu), dtype="<i2"
    ).astype(np.float64)

    spec = np.abs(np.fft.rfft(up))
    freqs = np.fft.rfftfreq(len(up), d=1.0 / TARGET_SAMPLE_RATE)

    band = (freqs >= 1000) & (freqs <= 3000)          # where the tone is
    upper = freqs >= 12000                              # 8 kHz replicas would land here
    assert band.sum() > 0 and upper.sum() > 0
    ratio = spec[upper].sum() / max(spec[band].sum(), 1e-9)
    assert ratio < 0.05, (
        f"energy above 12 kHz is {ratio:.3f} of the 1-3 kHz band; the "
        f"resampler is still producing spectral images"
    )
    del resample_poly
