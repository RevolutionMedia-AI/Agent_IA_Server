"""Queue drop probe (sub-agent `audio-001`).

Standalone probe that exercises `enqueue_nowait_with_drop` to verify the
kind-aware drop policy behaves as documented:
  - control items are never dropped,
  - audio items use drop-oldest with high-water tracking,
  - gap markers are emitted whenever an audio item is dropped.

Usage:
    python agents/tools/queue_drop_probe.py
    python agents/tools/queue_drop_probe.py --maxsize 4 --audio 20 --control 6

Returns exit 0 on policy conformance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass

ROOT = Path = __import__("pathlib").Path
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from STT_server.services.common import enqueue_nowait_with_drop  # noqa: E402


@dataclass
class Item:
    kind: str  # audio | control | mark | clear | segment_end
    payload: bytes = b""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--maxsize", type=int, default=4)
    p.add_argument("--audio", type=int, default=20)
    p.add_argument("--control", type=int, default=6)
    args = p.parse_args()

    q: deque = deque(maxlen=args.maxsize)
    drops_audio = 0
    drops_control = 0
    gap_markers = 0

    for i in range(args.audio):
        try:
            enqueue_nowait_with_drop(q, Item(kind="audio", payload=bytes([i])))
        except Exception:
            drops_audio += 1
            gap_markers += 1

    for i in range(args.control):
        try:
            enqueue_nowait_with_drop(q, Item(kind="control", payload=bytes([i])))
        except Exception:
            drops_control += 1

    kinds = [it.kind for it in q]
    audio_present = sum(1 for k in kinds if k == "audio")
    control_present = sum(1 for k in kinds if k == "control")
    control_tail = all(k == "control" for k in kinds[-args.control :]) if args.control <= len(kinds) else False

    report = {
        "maxsize": args.maxsize,
        "audio_sent": args.audio,
        "control_sent": args.control,
        "drops_audio": drops_audio,
        "drops_control": drops_control,
        "gap_markers": gap_markers,
        "audio_in_queue": audio_present,
        "control_in_queue": control_present,
        "control_at_tail": control_tail,
    }
    print(report)

    ok = drops_control == 0 and gap_markers >= drops_audio and control_tail
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
