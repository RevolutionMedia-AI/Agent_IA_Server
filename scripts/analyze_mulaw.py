#!/usr/bin/env python3
import sys
import os
import struct
import collections
import math
import statistics
import audioop


def human(n):
    return f"{n:,}"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "rime_tts_localtest_1.mulaw"
    if not os.path.exists(path):
        print("File not found:", path)
        return 2
    data = open(path, "rb").read()
    n = len(data)
    print("File:", path)
    print("Bytes:", n)
    print("20ms chunks (160 mu-law bytes):", n // 160, "remainder:", n % 160)

    ctr = collections.Counter(data)
    most = ctr.most_common(12)
    print("Top mu-law byte values (hex, count, pct):")
    for val, cnt in most:
        print(f"  {hex(val)}  {cnt}  {cnt/n*100:.2f}%")
    print("Counts: 0x00=", ctr.get(0, 0), " 0xFF=", ctr.get(0xFF, 0))

    try:
        pcm = audioop.ulaw2lin(data, 2)
    except Exception as e:
        print("ulaw2lin error:", e)
        return 3

    nsamples = len(pcm) // 2
    samples = struct.unpack(f"<{nsamples}h", pcm)
    print("Samples:", nsamples)

    mn = min(samples)
    mx = max(samples)
    mean = sum(samples) / nsamples
    med = statistics.median(samples)
    rms = math.sqrt(sum(s * s for s in samples) / nsamples)
    uniq = len(set(samples))
    print(f"min={mn} max={mx} mean={mean:.2f} median={med} rms={rms:.2f} uniq_values={uniq}")

    # Average absolute sample-to-sample diff
    diffs = [abs(samples[i + 1] - samples[i]) for i in range(nsamples - 1)]
    avg_diff = sum(diffs) / len(diffs)
    print("avg sample diff:", f"{avg_diff:.2f}")

    # Per-offset mean abs diff modulo CHUNK (160 samples)
    MOD = 160
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

    print("First 20 samples:", samples[:20])
    print("Last 20 samples:", samples[-20:])

    # Quick sanity checks
    if n % 160 != 0:
        print("WARNING: total bytes not multiple of 160 — potential misalignment")
    if uniq < 100:
        print("NOTICE: very few unique PCM values (could be low-bit-depth or clipping)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
