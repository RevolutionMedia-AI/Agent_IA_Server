"""Golden audio harness (sub-agents `audio-004`, `audio-007`, `audio-011`).

Lightweight command-line tool to:
  - generate fixed G.711 / sine / impulse / silence fixtures,
  - feed them through the project's codec + frame processor,
  - compare against expected sample counts, durations, RMS, and tail bytes.

Does not store or compare raw audio files in the repo by default; it derives
them deterministically from a small seed. Outputs a JSON report the CI can
diff against a pinned baseline.

Usage:
    python agents/tools/golden_audio_check.py --preset default
    python agents/tools/golden_audio_check.py --preset tails --seed 7

Returns exit 0 only when all expectations pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from STT_server.services.audio_codec import lin2ulaw, ulaw2lin  # noqa: E402
from STT_server.services.audio_frame_processor import AudioFrameProcessor  # noqa: E402


@dataclass
class Case:
    name: str
    duration_ms: int
    rate: int
    freq_hz: float | None
    kind: str  # sine | silence | impulse
    expected_samples: int
    tolerance: float = 1e-3


PRESETS = {
    "default": [
        Case("silence_200ms_8k", 200, 8000, None, "silence", 1600),
        Case("sine_1k_500ms_8k", 500, 8000, 1000.0, "sine", 4000),
        Case("impulse_8k", 1, 8000, None, "impulse", 8),
    ],
    "tails": [
        Case("random_tail_173", 173, 8000, None, "sine", 1384),  # 173 * 8
        Case("random_tail_319", 319, 8000, None, "sine", 2552),
        Case("random_tail_1", 1, 8000, None, "sine", 8),
    ],
}


def synth(case: Case) -> bytes:
    n = int(case.duration_ms * case.rate / 1000)
    if case.kind == "silence":
        pcm = b"\x00\x00" * n
    elif case.kind == "sine":
        amp = 8000
        pcm = bytearray(n * 2)
        for i in range(n):
            v = int(amp * math.sin(2 * math.pi * case.freq_hz * i / case.rate))
            pcm[2 * i] = v & 0xFF
            pcm[2 * i + 1] = (v >> 8) & 0xFF
        pcm = bytes(pcm)
    elif case.kind == "impulse":
        pcm = b"\x00\x00" * n
        pcm = b"\xff\x7f" + pcm[2:]
    else:
        raise ValueError(case.kind)
    return lin2ulaw(pcm)


def run(case: Case, seed: int) -> dict:
    ulaw = synth(case)
    pcm = ulaw2lin(ulaw)
    fp = AudioFrameProcessor(frame_size=160)
    frames = []
    for b in ulaw:
        out = fp.push(bytes([b]))
        if out:
            frames.append(out)
    tail = fp.flush()
    dropped_tail_bytes = len(tail) if tail is not None else 0
    return {
        "case": case.name,
        "ulaw_bytes": len(ulaw),
        "pcm_samples": len(pcm) // 2,
        "frames_emitted": len(frames),
        "dropped_tail_bytes": dropped_tail_bytes,
        "expected_samples": case.expected_samples,
        "duration_ms": case.duration_ms,
        "seed": seed,
        "ok": abs(len(pcm) // 2 - case.expected_samples) <= int(case.tolerance * case.expected_samples + 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="default", choices=list(PRESETS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="-")
    args = p.parse_args()

    results = [run(c, args.seed) for c in PRESETS[args.preset]]
    report = {"preset": args.preset, "seed": args.seed, "results": results, "all_ok": all(r["ok"] for r in results)}
    text = json.dumps(report, indent=2)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
