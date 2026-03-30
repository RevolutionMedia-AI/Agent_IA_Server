#!/usr/bin/env python3
import sys
import wave
import audioop
import os
from pathlib import Path


def convert(src: str, dst: str, target_rate: int = 8000) -> int:
    src_path = Path(src)
    dst_path = Path(dst)

    if not src_path.exists():
        print(f"error: source file not found: {src}", file=sys.stderr)
        return 2

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(src_path), 'rb') as inf:
        nchannels = inf.getnchannels()
        sampwidth = inf.getsampwidth()
        orig_rate = inf.getframerate()
        comptype = inf.getcomptype()

        print(f"Source: {src} channels={nchannels} sampwidth={sampwidth} rate={orig_rate} comptype={comptype}")

        if sampwidth not in (1, 2, 3, 4):
            print(f"Unsupported sample width: {sampwidth}")
            return 3

        with wave.open(str(dst_path), 'wb') as outf:
            outf.setnchannels(1)
            outf.setsampwidth(2)
            outf.setframerate(target_rate)

            state = None
            CHUNK_FRAMES = 4096

            while True:
                frames = inf.readframes(CHUNK_FRAMES)
                if not frames:
                    break

                # Convert sample width if needed
                if sampwidth != 2:
                    frames = audioop.lin2lin(frames, sampwidth, 2)

                # Convert to mono if needed
                if nchannels != 1:
                    try:
                        frames = audioop.tomono(frames, 2, 0.5, 0.5)
                    except TypeError:
                        # Fallback: use integer weights then scale down
                        frames = audioop.tomono(frames, 2, 1, 1)
                        frames = audioop.mul(frames, 2, 0.5)

                # Resample to target_rate
                converted, state = audioop.ratecv(frames, 2, 1, orig_rate, target_rate, state)
                if converted:
                    outf.writeframes(converted)

            # Flush residual
            try:
                converted, state = audioop.ratecv(b"", 2, 1, orig_rate, target_rate, state)
                if converted:
                    outf.writeframes(converted)
            except Exception:
                pass

    print(f"Converted {src} -> {dst} ({orig_rate}Hz -> {target_rate}Hz)")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: convert_to_8k.py <src.wav> <dst.wav>")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2]
    sys.exit(convert(src, dst))
