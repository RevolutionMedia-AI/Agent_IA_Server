"""Audio metrics collector (sub-agent `audio-012`).

Provides a single sink for per-call audio observability. The collector stores
metadata only — never raw audio content, transcripts, tool payloads, or
credentials. Used by every audio sub-agent to feed the per-call summary.

Schema (all keys optional but present when applicable):

    {
        "call_id": str,
        "tenant_id": str | None,
        "agent_id": str | None,
        "format": "pcmu/8000/1",
        "bytes_in": int,
        "frames_in": int,
        "bytes_out": int,
        "frames_out": int,
        "seq_gaps": int,
        "dupes": int,
        "reorders": int,
        "queue_high_water": {queue_name: int, ...},
        "drop_count": {kind: int, ...},
        "tail_bytes_dropped": {provider: int, ...},
        "resample_path": "scipy" | "fallback",
        "pacing_drift_ms_p50": float,
        "pacing_drift_ms_p95": float,
        "bargein_count": int,
        "bargein_staleness_ms": float,
        "mark_rtt_ms_p50": float,
        "mark_rtt_ms_p95": float,
        "generation_transitions": int,
        "vad_preset": str,
        "vad_preset_version": int,
    }

Invariants:
    - Never accepts binary blobs; all values are scalars or dicts of scalars.
    - Emits JSONL to `data/audio_metrics/<call_id>.jsonl` (one line per call).
    - Counts only — no audio content ever written.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(os.environ.get("AUDIO_METRICS_DIR", "data/audio_metrics"))

# Maximum scalar lengths we will accept — guards against accidentally piping
# raw bytes through a field name.
MAX_STR = 256
MAX_INT = 2**48


@dataclass
class AudioCallSummary:
    call_id: str
    tenant_id: str | None = None
    agent_id: str | None = None
    format: str = "pcmu/8000/1"
    bytes_in: int = 0
    frames_in: int = 0
    bytes_out: int = 0
    frames_out: int = 0
    seq_gaps: int = 0
    dupes: int = 0
    reorders: int = 0
    queue_high_water: dict[str, int] = field(default_factory=dict)
    drop_count: dict[str, int] = field(default_factory=dict)
    tail_bytes_dropped: dict[str, int] = field(default_factory=dict)
    resample_path: str = "scipy"
    pacing_drift_ms_p50: float = 0.0
    pacing_drift_ms_p95: float = 0.0
    bargein_count: int = 0
    bargein_staleness_ms: float = 0.0
    mark_rtt_ms_p50: float = 0.0
    mark_rtt_ms_p95: float = 0.0
    generation_transitions: int = 0
    vad_preset: str = "default"
    vad_preset_version: int = 1
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


class AudioMetricsCollector:
    """Thread-safe collector for one process."""

    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._summaries: dict[str, AudioCallSummary] = {}

    def start(self, call_id: str, tenant_id: str | None = None, agent_id: str | None = None) -> AudioCallSummary:
        with self._lock:
            s = AudioCallSummary(call_id=call_id, tenant_id=tenant_id, agent_id=agent_id)
            self._summaries[call_id] = s
            return s

    def get(self, call_id: str) -> AudioCallSummary | None:
        with self._lock:
            return self._summaries.get(call_id)

    def finalize(self, call_id: str) -> AudioCallSummary:
        with self._lock:
            s = self._summaries.pop(call_id, None)
        if s is None:
            raise KeyError(call_id)
        s.ended_at = time.time()
        path = self.root / f"{_safe(call_id)}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(s)) + "\n")
        return s

    def incr(self, call_id: str, field_name: str, by: int = 1) -> None:
        """Increment a scalar counter on the summary; bounded to MAX_INT."""
        s = self.get(call_id)
        if s is None:
            return
        cur = int(getattr(s, field_name, 0)) + by
        setattr(s, field_name, max(0, min(MAX_INT, cur)))

    def incr_dict(self, call_id: str, field_name: str, key: str, by: int = 1) -> None:
        s = self.get(call_id)
        if s is None:
            return
        d: dict[str, int] = getattr(s, field_name)
        d[_safe(key)] = d.get(_safe(key), 0) + by

    def record_high_water(self, call_id: str, queue: str, value: int) -> None:
        s = self.get(call_id)
        if s is None:
            return
        d: dict[str, int] = s.queue_high_water
        d[_safe(queue)] = max(d.get(_safe(queue), 0), int(value))

    def to_dict(self, call_id: str) -> dict[str, Any]:
        s = self.get(call_id) or AudioCallSummary(call_id=call_id)
        return asdict(s)


def _safe(name: str) -> str:
    """Restrict field keys to printable ASCII ≤ MAX_STR."""
    out = "".join(c if 32 <= ord(c) < 127 else "_" for c in name)
    return out[:MAX_STR] or "_"


# Convenience singleton for fast-path callers; tests should construct their
# own collector with a temporary directory.
_default: AudioMetricsCollector | None = None


def default() -> AudioMetricsCollector:
    global _default
    if _default is None:
        _default = AudioMetricsCollector()
    return _default


if __name__ == "__main__":  # smoke
    c = AudioMetricsCollector(root="data/audio_metrics")
    s = c.start("call-smoke", tenant_id="t1", agent_id="a1")
    c.incr("call-smoke", "bytes_in", 160)
    c.incr_dict("call-smoke", "drop_count", "audio", 1)
    c.record_high_water("call-smoke", "stt_audio_queue", 17)
    out = c.finalize("call-smoke")
    assert out.bytes_in == 160
    assert out.drop_count["audio"] == 1
    assert out.queue_high_water["stt_audio_queue"] == 17
    print("ok", out.call_id, out.bytes_in, out.queue_high_water)
