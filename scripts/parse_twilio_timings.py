#!/usr/bin/env python3
"""Parse Twilio timing JSONL files created by playback_service and summarize jitter/latency.

Usage: python scripts/parse_twilio_timings.py twilio_out_sessionid_generation.timings.jsonl
"""
import sys
import os
import json
import statistics


def analyze_file(path):
    if not os.path.exists(path):
        print('File not found:', path)
        return 2
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                print('Skipping invalid json line:', e)
    if not items:
        print('No timing records in', path)
        return 1

    send_elapsed = [it.get('send_elapsed', 0.0) for it in items]
    wait_applied = [it.get('wait_applied_s', 0.0) for it in items]
    total_frame_time = [s + w for s, w in zip(send_elapsed, wait_applied)]
    frames = len(items)

    def stats(arr):
        return {
            'count': len(arr),
            'min': min(arr),
            'max': max(arr),
            'mean': statistics.mean(arr),
            'median': statistics.median(arr),
            'stdev': statistics.pstdev(arr) if len(arr) > 1 else 0.0,
        }

    se_stats = stats(send_elapsed)
    wa_stats = stats(wait_applied)
    tf_stats = stats(total_frame_time)

    print('Timing file:', path)
    print('Frames:', frames)
    print('\nSend elapsed (s):', f"mean={se_stats['mean']:.5f} med={se_stats['median']:.5f} min={se_stats['min']:.5f} max={se_stats['max']:.5f} stdev={se_stats['stdev']:.5f}")
    print('Wait applied (s):', f"mean={wa_stats['mean']:.5f} med={wa_stats['median']:.5f} min={wa_stats['min']:.5f} max={wa_stats['max']:.5f} stdev={wa_stats['stdev']:.5f}")
    print('Total per-frame time (s):', f"mean={tf_stats['mean']:.5f} med={tf_stats['median']:.5f} min={tf_stats['min']:.5f} max={tf_stats['max']:.5f} stdev={tf_stats['stdev']:.5f}")

    expected_s = 0.02
    print('\nExpected per-frame time (target): 0.020000 s (20 ms)')
    print('Avg delta vs target (s):', f"{tf_stats['mean'] - expected_s:.5f}")
    print('Pct frames with positive wait applied:', f"{sum(1 for w in wait_applied if w>0)/frames*100:.2f}%")

    # Top slow frames by total time
    top = sorted(enumerate(total_frame_time), key=lambda x: -x[1])[:10]
    print('\nTop slow frames (index, total_s):')
    for idx, val in top:
        print(' ', idx, f'{val:.6f}')

    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python scripts/parse_twilio_timings.py <timings.jsonl>')
        sys.exit(2)
    sys.exit(analyze_file(sys.argv[1]))
