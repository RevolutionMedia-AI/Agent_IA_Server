#!/usr/bin/env python3
"""Compare two WAV files (PCM) and print sample-level differences."""
import sys
import wave
import struct
import math


def load_wav(path):
    with wave.open(path, 'rb') as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes(), w.getcomptype())
        frames = w.readframes(w.getnframes())
    return params, frames


def pcm_stats(frames, sampwidth):
    if sampwidth != 2:
        raise RuntimeError('Only 16-bit supported')
    ns = len(frames) // 2
    samples = struct.unpack(f'<{ns}h', frames)
    return samples


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else 'rime_tts_localtest_1.mulaw.wav'
    b = sys.argv[2] if len(sys.argv) > 2 else 'twilio_out_localtest_1.mulaw.wav'

    pa, fa = load_wav(a)
    pb, fb = load_wav(b)

    print('A:', a, 'params=', pa)
    print('B:', b, 'params=', pb)

    if pa[:3] != pb[:3]:
        print('WARNING: WAV params differ (channels/sampwidth/framerate)')

    if fa == fb:
        print('PCM frames are byte-identical')
        return 0

    try:
        sa = pcm_stats(fa, pa[1])
        sb = pcm_stats(fb, pb[1])
    except Exception as e:
        print('Error reading PCM samples:', e)
        return 2

    minlen = min(len(sa), len(sb))
    diffs = 0
    mse = 0
    mae = 0
    first_diffs = []
    for i in range(minlen):
        d = sa[i] - sb[i]
        if d != 0:
            diffs += 1
            if len(first_diffs) < 10:
                first_diffs.append((i, sa[i], sb[i], d))
        mse += d * d
        mae += abs(d)

    mse = mse / minlen if minlen else float('nan')
    mae = mae / minlen if minlen else float('nan')

    print('Samples A,B:', len(sa), len(sb))
    print('Differing samples (overlap):', diffs, 'of', minlen)
    print(f'MSE: {mse:.2f} MAE: {mae:.2f} RMS diff: {math.sqrt(mse):.2f}')
    if first_diffs:
        print('First diffs (idx, A, B, A-B):')
        for t in first_diffs:
            print(' ', t)

    if len(sa) != len(sb):
        print('Note: lengths differ by', abs(len(sa) - len(sb)), 'samples')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
