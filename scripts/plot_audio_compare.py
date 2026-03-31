#!/usr/bin/env python3
"""Generate comparison plots for two WAV files: waveforms, spectrograms, histograms, and diff spectrogram.

Usage: python scripts/plot_audio_compare.py <a.wav> <b.wav>
"""
import sys
import os
import wave
import struct
import math

def load_wav(path):
    import audioop
    import numpy as _np

    with wave.open(path, 'rb') as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        fr = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if sw != 2:
        raw = audioop.lin2lin(raw, sw, 2)
        sw = 2

    arr = _np.frombuffer(raw, dtype='<i2')
    if nch > 1:
        arr = arr.reshape(-1, nch)[:, 0].copy()
    return arr.astype(_np.int16), fr


def rms(x):
    import numpy as _np
    return float(_np.sqrt(_np.mean(_np.asarray(x, dtype=_np.float64) ** 2)))


def main():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print('Missing dependency:', e)
        print('Install with: pip install numpy matplotlib')
        return 2

    a_path = sys.argv[1] if len(sys.argv) > 1 else 'rime_out_converted.wav'
    b_path = sys.argv[2] if len(sys.argv) > 2 else 'twilio_out_localtest_1.mulaw.wav'

    if not os.path.exists(a_path) or not os.path.exists(b_path):
        print('Missing input files:', a_path, b_path)
        return 3

    a, fa = load_wav(a_path)
    b, fb = load_wav(b_path)

    print('Loaded:', a_path, 'samples=', len(a), 'fs=', fa)
    print('Loaded:', b_path, 'samples=', len(b), 'fs=', fb)

    if fa != fb:
        print('Warning: sample rates differ: ', fa, fb)

    fs = fa

    # Trim to min length for direct comparisons
    minlen = min(len(a), len(b))
    a_t = a[:minlen]
    b_t = b[:minlen]
    diff = a_t.astype('int32') - b_t.astype('int32')

    sig_rms = rms(a_t)
    diff_rms = rms(diff)
    snr_db = float('inf') if diff_rms == 0 else 20.0 * math.log10(sig_rms / diff_rms)

    print(f'SNR (A vs B): {snr_db if diff_rms != 0 else "inf"} dB')
    print(f'Samples: A={len(a)} B={len(b)} trimmed={minlen} diff_rms={diff_rms:.2f} sig_rms={sig_rms:.2f}')

    t_a = np.arange(len(a)) / fs
    t_b = np.arange(len(b)) / fs
    t = np.arange(minlen) / fs

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Waveforms
    axes[0, 0].plot(t_a, a / 32768.0, color='C0')
    axes[0, 0].set_title(f'Waveform A: {os.path.basename(a_path)}')
    axes[0, 1].plot(t_b, b / 32768.0, color='C1')
    axes[0, 1].set_title(f'Waveform B: {os.path.basename(b_path)}')

    # Spectrograms
    NFFT = 512
    noverlap = NFFT // 2
    axes[1, 0].specgram(a, NFFT=NFFT, Fs=fs, noverlap=noverlap, cmap='magma')
    axes[1, 0].set_title('Spectrogram A')
    axes[1, 1].specgram(b, NFFT=NFFT, Fs=fs, noverlap=noverlap, cmap='magma')
    axes[1, 1].set_title('Spectrogram B')

    # Histograms
    axes[2, 0].hist(a, bins=120, color='C0', alpha=0.8)
    axes[2, 0].set_title('Amplitude histogram A')
    axes[2, 1].hist(b, bins=120, color='C1', alpha=0.8)
    axes[2, 1].set_title('Amplitude histogram B')

    plt.tight_layout()
    out_png = 'audio_compare.png'
    fig.savefig(out_png, dpi=150)
    print('Wrote', out_png)

    # Diff spectrogram
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
    Pxx, freqs, bins, im = ax2.specgram(diff, NFFT=NFFT, Fs=fs, noverlap=noverlap, cmap='inferno')
    ax2.set_title('Diff spectrogram (A-B)')
    fig2.tight_layout()
    out_diff = 'diff_spectrogram.png'
    fig2.savefig(out_diff, dpi=150)
    print('Wrote', out_diff)

    # Save numeric summary
    with open('audio_compare_stats.txt', 'w') as f:
        f.write(f'A: {a_path}\nB: {b_path}\n')
        f.write(f'samples A={len(a)} B={len(b)} trimmed={minlen}\n')
        f.write(f'sig_rms={sig_rms:.2f} diff_rms={diff_rms:.2f} snr_db={snr_db}\n')

    print('Wrote audio_compare_stats.txt')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
