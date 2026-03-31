#!/usr/bin/env python3
"""Analyze a WAV file and print sample statistics.

Usage: python scripts/analyze_wav.py <file.wav>
"""
import sys
import os
import wave
import struct
import math
import statistics
import collections
import audioop


def human(n):
    return f"{n:,}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "rime_tts_localtest_1.wav"
    if not os.path.exists(path):
        print("File not found:", path)
        return 2

    with wave.open(path, "rb") as w:
        nchannels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        comp = w.getcomptype()
        frames = w.readframes(nframes)

    print("File:", path)
    print("Channels:", nchannels, "SampleWidth(bytes):", sampwidth, "Rate(Hz):", framerate)
    print("Frames:", human(nframes), "Duration(s):", f"{nframes/framerate:.3f}")

    # Convert to mono if needed
    mono = frames
    if nchannels > 1:
        try:
            mono = audioop.tomono(frames, sampwidth, 0.5, 0.5)
        except Exception:
            # Fallback: take first channel by slicing
            if sampwidth == 2:
                vals = struct.unpack(f"<{nframes * nchannels}h", frames)
                mono_vals = vals[0::nchannels]
                mono = struct.pack(f"<{len(mono_vals)}h", *mono_vals)
            else:
                raise

    # Normalize sample width to 2 bytes (16-bit) for analysis
    if sampwidth != 2:
        try:
            mono2 = audioop.lin2lin(mono, sampwidth, 2)
        except Exception:
            print("Unsupported sample width for conversion:", sampwidth)
            return 3
    else:
        mono2 = mono

    nsamples = len(mono2) // 2
    samples = struct.unpack(f"<{nsamples}h", mono2)

    print("Samples:", human(nsamples))

    mn = min(samples)
    mx = max(samples)
    mean = sum(samples) / nsamples
    med = statistics.median(samples)
    rms = math.sqrt(sum(s * s for s in samples) / nsamples)
    uniq = len(set(samples))
    print(f"min={mn} max={mx} mean={mean:.2f} median={med} rms={rms:.2f} uniq_values={uniq}")

    # Average absolute sample-to-sample diff
    diffs = [abs(samples[i + 1] - samples[i]) for i in range(nsamples - 1)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    print("avg sample diff:", f"{avg_diff:.2f}")

    # Per-offset mean abs diff using 20ms frame size (useful for telephony)
    MOD = int(round(framerate * 0.02)) if framerate else 160
    per_offset = []
    for k in range(MOD):
        idxs = range(k, nsamples - 1, MOD)
        total = 0
        count = 0
        for i in idxs:
            total += abs(samples[i + 1] - samples[i])
            count += 1
        per_offset.append(total / count if count else 0)

    top = sorted(list(enumerate(per_offset)), key=lambda x: -x[1])[:6]
    print("Top offsets by mean abs diff (offset, value):")
    for off, val in top:
        print(f"  {off} -> {val:.2f}")

    # Byte-level histogram (on 16-bit values) — show most common values
    ctr = collections.Counter(samples)
    most = ctr.most_common(8)
    print("Top sample values (val, count, pct):")
    for val, cnt in most:
        print(f"  {val}  {cnt}  {cnt/nsamples*100:.2f}%")

    # Quick sanity checks
    if MOD and nsamples % MOD != 0:
        print("WARNING: total samples not multiple of 20ms frame size — potential misalignment")
    if uniq < 100:
        print("NOTICE: very few unique PCM values (could be low-bit-depth or clipping)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
