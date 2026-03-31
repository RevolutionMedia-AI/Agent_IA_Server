#!/usr/bin/env python3
import sys
import os
import struct
import audioop
import math
import statistics


def analyze_file(path):
    data = open(path, 'rb').read()
    n = len(data)
    print(f"File: {path}")
    print(f" Bytes: {n}")
    print(f" 20ms chunks (160 mu-law bytes): {n // 160} remainder: {n % 160}")
    return data


def compare(a_path, b_path):
    a = analyze_file(a_path)
    b = analyze_file(b_path)

    print()
    print("Byte-level comparison:")
    print(" Equal bytes (full file):", a == b)
    minlen = min(len(a), len(b))
    if a != b:
        same_prefix = sum(1 for i in range(minlen) if a[i] == b[i])
        diff_in_overlap = minlen - same_prefix
        total_diff = diff_in_overlap + abs(len(a) - len(b))
        print(f" Differing bytes (overlap): {diff_in_overlap}")
        print(f" Total differing bytes: {total_diff} ({total_diff / max(len(a), len(b)) * 100:.2f}%)")
        diffs = [i for i in range(minlen) if a[i] != b[i]]
        print(" First differing indices (up to 20):", diffs[:20])
    else:
        print(" Files are identical at byte level.")

    # Convert to PCM16 using ulaw -> 2-byte samples
    try:
        a_pcm = audioop.ulaw2lin(a, 2)
        b_pcm = audioop.ulaw2lin(b, 2)
    except Exception as e:
        print("ulaw2lin conversion error:", e)
        return

    a_ns = len(a_pcm) // 2
    b_ns = len(b_pcm) // 2
    a_samps = struct.unpack(f"<{a_ns}h", a_pcm)
    b_samps = struct.unpack(f"<{b_ns}h", b_pcm)

    print()
    print("PCM sample counts:", a_ns, b_ns)
    def stats(arr):
        n = len(arr)
        mn = min(arr)
        mx = max(arr)
        mean = sum(arr) / n
        rms = math.sqrt(sum(x * x for x in arr) / n)
        med = statistics.median(arr)
        return mn, mx, mean, med, rms

    a_stats = stats(a_samps)
    b_stats = stats(b_samps)
    print("PCM A stats min/max/mean/median/rms:", a_stats)
    print("PCM B stats min/max/mean/median/rms:", b_stats)

    # Compare overlapping samples
    min_ns = min(a_ns, b_ns)
    diffs = [a_samps[i] - b_samps[i] for i in range(min_ns)]
    mse = sum(d * d for d in diffs) / min_ns
    mae = sum(abs(d) for d in diffs) / min_ns
    print()
    print(f"PCM diff MSE: {mse:.2f} MAE: {mae:.2f} RMS diff: {math.sqrt(mse):.2f}")

    diff_idxs = [i for i in range(min_ns) if a_samps[i] != b_samps[i]]
    print("Differing PCM sample count:", len(diff_idxs), "of", min_ns)
    print("First differing sample indices (up to 20):", diff_idxs[:20])
    if diff_idxs:
        print("Sample pairs (idx, A, B, A-B) for first 10 diffs:")
        for i in diff_idxs[:10]:
            print(f" {i}: {a_samps[i]} {b_samps[i]} {a_samps[i]-b_samps[i]}")


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else 'rime_tts_localtest_1.mulaw'
    b = sys.argv[2] if len(sys.argv) > 2 else 'twilio_out_localtest_1.mulaw'
    if not os.path.exists(a) or not os.path.exists(b):
        print('Missing input files:', a, b)
        sys.exit(2)
    compare(a, b)
