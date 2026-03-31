#!/usr/bin/env python3
"""Apply RNNoiseFilter to a .mulaw file (frame-by-frame).

Usage:
    python scripts/offline_denoise.py input.mulaw output.mulaw
"""
import argparse
import sys
import os

# Ensure repository root is on sys.path so `STT_server` package imports work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from STT_server.services.rnnoise_filter import RNNoiseFilter


def process_file(input_path: str, output_path: str, frame_bytes: int = 160) -> int:
    denoiser = RNNoiseFilter()
    processed = 0
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        while True:
            chunk = fin.read(frame_bytes)
            if not chunk:
                break
            # If last partial chunk, write through unchanged
            if len(chunk) != frame_bytes:
                fout.write(chunk)
                break
            try:
                out_chunk = denoiser.process_mulaw_frame(chunk)
            except Exception:
                out_chunk = chunk
            fout.write(out_chunk)
            processed += 1
    return processed


def main() -> None:
    p = argparse.ArgumentParser(description="Offline RNNoise denoise for .mulaw files")
    p.add_argument("input", help="Input .mulaw file")
    p.add_argument("output", help="Output .mulaw file")
    p.add_argument("--frame-bytes", type=int, default=160, help="Bytes per frame (default 160)")
    args = p.parse_args()

    try:
        cnt = process_file(args.input, args.output, args.frame_bytes)
        print(f"Processed {cnt} frames -> {args.output}")
    except Exception as e:
        print("Error while processing:", e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
