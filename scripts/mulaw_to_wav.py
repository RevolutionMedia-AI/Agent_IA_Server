#!/usr/bin/env python3
"""Convert a raw mu-law (.mulaw) file (8 kHz, mono) to a 16-bit WAV file.

Usage: python scripts/mulaw_to_wav.py in.mulaw out.wav
"""
import sys
import os
import audioop
import wave


def main():
    if len(sys.argv) < 3:
        print("Usage: mulaw_to_wav.py in.mulaw out.wav")
        return 2
    inp = sys.argv[1]
    out = sys.argv[2]
    if not os.path.exists(inp):
        print("Input not found:", inp)
        return 3
    data = open(inp, 'rb').read()
    try:
        pcm = audioop.ulaw2lin(data, 2)
    except Exception as e:
        print("ulaw2lin error:", e)
        return 4

    with wave.open(out, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(pcm)

    print(f"Wrote WAV: {out} samples={len(pcm)//2}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
