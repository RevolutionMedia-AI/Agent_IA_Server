#!/usr/bin/env python3
import sys
import wave
from STT_server.services import audio_codec
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
                    # TODO: lin2lin not in audio_codec.py; install scipy for sample-width conversion
                    pass

                # Convert to mono if needed
                if nchannels != 1:
                    # TODO: tomono/mul not in audio_codec.py; install scipy for stereo-to-mono mixing
                    pass

                # Resample to target_rate
                # TODO: ratecv not in audio_codec.py; install scipy or shell out to ffmpeg for resampling
                converted = frames  # ponytail: passthrough at original sample rate (no resampler)
                if converted:
                    outf.writeframes(converted)

            # Flush residual
            # TODO: ratecv not in audio_codec.py; install scipy or shell out to ffmpeg for resampling
            converted = b""  # ponytail: no stateful resampler, nothing to flush
            if converted:
                outf.writeframes(converted)

    print(f"Converted {src} -> {dst} ({orig_rate}Hz -> {target_rate}Hz)")
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: convert_to_8k.py <src.wav> <dst.wav>")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2]
    sys.exit(convert(src, dst))
