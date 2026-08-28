"""Audio capture helpers for the 2026-08-28 forensic A/B capture.

Pinpoints where audio artifacts appear by capturing:

  A — bytes from Inworld (post base64-decode, BEFORE AudioFrameProcessor)
  B — bytes to Twilio   (inside send_twilio_media, BEFORE base64-encode)

Storage target:
  - When ``DATABASE_URL`` is set (production / Railway): writes go to
    Postgres via an async background queue so DB latency never sits
    on the live WS send.
  - When ``DATABASE_URL`` is unset (local dev): writes go to
    per-call/per-gen files under ``TTS_AUDIO_CAPTURE_DIR`` (same
    layout as before — fallback for devs without a DB handy).

Master switch: ``TTS_AUDIO_CAPTURE_DIR``. Set to any non-empty value
to enable; unset (or empty) to disable (no-op). The DB-vs-file
backend is picked automatically from ``DATABASE_URL`` — the operator
only flips ONE env var to turn the whole forensic chain on.

Best-effort: a write failure logs once and disables capture for the
relevant scope (a (call, gen, stage) tuple in DB mode, the whole
callSid in file mode). NEVER crash the live path.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import IO


log = logging.getLogger("stt_server.audio_capture")

# ─── File mode (local-dev fallback) ──────────────────────────────

# Per (callSid, generation) open file handles. Key = (callSid, gen),
# value = {"inworld": binary fh, "inworld_jsonl": text fh,
#          "twilio": binary fh, "twilio_jsonl": text fh}. Lazy:
# only stages that were actually written open handles.
_HANDLES: dict[tuple[str, int], dict[str, IO]] = {}
_HANDLES_LOCK = threading.Lock()

_ERROR_LOGGED: set[tuple[str, int, str]] = set()
_DIR_FAILED: set[str] = set()


def _capture_dir() -> str:
    return os.environ.get("TTS_AUDIO_CAPTURE_DIR", "").strip()


def _use_db() -> bool:
    # ponytail: keep JSON-only deployments free of psycopg2 import.
    from STT_server.db import is_postgres
    return is_postgres()


def _paths(call_sid: str, generation: int, capture_dir: str) -> dict[str, Path]:
    base = Path(capture_dir) / call_sid
    gen = f"gen-{generation}"
    return {
        "inworld": base / f"{gen}-inworld.mulaw",
        "inworld_jsonl": base / f"{gen}-inworld.jsonl",
        "twilio": base / f"{gen}-twilio.mulaw",
        "twilio_jsonl": base / f"{gen}-twilio.jsonl",
    }


def _open_locked(
    call_sid: str, generation: int, stage: str, capture_dir: str
) -> tuple[IO, IO] | None:
    if call_sid in _DIR_FAILED:
        return None
    key = (call_sid, generation)
    handles = _HANDLES.setdefault(key, {})
    bin_key = stage
    json_key = f"{stage}_jsonl"
    bin_fh = handles.get(bin_key)
    json_fh = handles.get(json_key)
    if bin_fh is not None and json_fh is not None:
        return bin_fh, json_fh
    try:
        base = Path(capture_dir) / call_sid
        base.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log.warning(
            "[AUDIO_CAPTURE] mkdir failed for call_sid=%s dir=%s err=%s; "
            "disabling capture for this call",
            call_sid, capture_dir, exc,
        )
        _DIR_FAILED.add(call_sid)
        return None
    paths = _paths(call_sid, generation, capture_dir)
    try:
        if bin_fh is None:
            bin_fh = open(paths[bin_key], "ab", buffering=0)
            handles[bin_key] = bin_fh
        if json_fh is None:
            json_fh = open(paths[json_key], "a", encoding="utf-8", buffering=1)
            handles[json_key] = json_fh
        return bin_fh, json_fh
    except Exception as exc:
        log.warning(
            "[AUDIO_CAPTURE] open failed for call_sid=%s gen=%d stage=%s err=%s; "
            "disabling capture for this (call, gen, stage)",
            call_sid, generation, stage, exc,
        )
        _ERROR_LOGGED.add((call_sid, generation, stage))
        return None


def _write_file(
    call_sid: str,
    generation: int,
    stage: str,
    mulaw_bytes: bytes,
    sha: str,
    seg_idx: int | None,
    capture_dir: str,
) -> None:
    err_key = (call_sid, generation, stage)
    if err_key in _ERROR_LOGGED:
        return
    with _HANDLES_LOCK:
        result = _open_locked(call_sid, generation, stage, capture_dir)
    if result is None:
        return
    bin_fh, json_fh = result
    try:
        bin_fh.write(mulaw_bytes)
    except Exception as exc:
        log.warning(
            "[AUDIO_CAPTURE] write failed for call_sid=%s gen=%d stage=%s err=%s; "
            "disabling capture for this (call, gen, stage)",
            call_sid, generation, stage, exc,
        )
        _ERROR_LOGGED.add(err_key)
        return
    record = {
        "session": call_sid,
        "gen": generation,
        "stage": stage,
        "seg": seg_idx,
        "byte_count": len(mulaw_bytes),
        "sha256": sha,
        "ts": time.time(),
    }
    try:
        json_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        json_fh.flush()
    except Exception as exc:
        log.warning(
            "[AUDIO_CAPTURE] jsonl write failed for call_sid=%s gen=%d stage=%s err=%s; "
            "disabling sidecar for this (call, gen, stage)",
            call_sid, generation, stage, exc,
        )
        _ERROR_LOGGED.add(err_key)


# ─── DB mode (async background writer) ───────────────────────────

# Records the worker drains into Postgres. Daemon thread, single
# worker: order preservation across the queue is guaranteed and a
# single insert path is easier to reason about. maxsize 20k covers
# 6 s × 50 frames/s × 2 sides × 2 writes each ≈ 1200 records/call
# with plenty of headroom for barge-in replays.
_capture_queue: queue.Queue = queue.Queue(maxsize=20000)
_worker_started = False
_worker_lock = threading.Lock()


def _ensure_worker() -> None:
    """Lazy-start a daemon thread that drains the capture queue into
    Postgres. Safe to call from any thread, idempotent."""
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(
            target=_drain_worker,
            name="audio-capture-writer",
            daemon=True,
        )
        t.start()
        _worker_started = True


def _drain_worker() -> None:
    """Pop records from the queue and INSERT into audio_capture.
    Never raises: a failed insert logs once and continues."""
    from STT_server.db_audio_capture import insert_capture
    while True:
        record = _capture_queue.get()
        try:
            insert_capture(**record)
        except Exception:
            log.exception(
                "[AUDIO_CAPTURE] worker insert failed call_sid=%s gen=%d stage=%s",
                record.get("call_sid"),
                record.get("generation"),
                record.get("stage"),
            )
        finally:
            _capture_queue.task_done()


def _enqueue_db(
    call_sid: str,
    generation: int,
    stage: str,
    mulaw_bytes: bytes,
    sha: str,
    seg_idx: int | None,
) -> None:
    _ensure_worker()
    record = {
        "call_sid": call_sid,
        "generation": generation,
        "stage": stage,
        "seg": seg_idx,
        "byte_count": len(mulaw_bytes),
        "sha256": sha,
        "payload": mulaw_bytes,
    }
    try:
        _capture_queue.put_nowait(record)
    except queue.Full:
        log.warning(
            "[AUDIO_CAPTURE] queue full call_sid=%s gen=%d stage=%s; dropping",
            call_sid, generation, stage,
        )


# ─── Public API ──────────────────────────────────────────────────


def _capture(
    call_sid: str,
    generation: int,
    mulaw_bytes: bytes,
    stage: str,
    seg_idx: int | None,
    capture_dir: str | None,
) -> None:
    # ponytail: keep this fast in the no-op case (the default).
    if capture_dir is None:
        capture_dir = _capture_dir()
    if not capture_dir or not call_sid or not mulaw_bytes:
        return
    sha = hashlib.sha256(mulaw_bytes).hexdigest()
    if _use_db():
        _enqueue_db(call_sid, generation, stage, mulaw_bytes, sha, seg_idx)
        return
    _write_file(call_sid, generation, stage, mulaw_bytes, sha, seg_idx, capture_dir)


def capture_a(
    call_sid: str,
    generation: int,
    mulaw_bytes: bytes,
    seg_idx: int = 0,
    capture_dir: str | None = None,
) -> None:
    """Append raw Inworld μ-law bytes (post base64-decode, BEFORE
    AudioFrameProcessor) to the A capture for this (callSid, gen).

    No-op when ``TTS_AUDIO_CAPTURE_DIR`` is unset (the default)."""
    _capture(call_sid, generation, mulaw_bytes, "inworld", seg_idx, capture_dir)


def capture_b(
    call_sid: str,
    generation: int,
    mulaw_bytes: bytes,
    capture_dir: str | None = None,
) -> None:
    """Append a single 160-byte Twilio frame (BEFORE base64-encode
    inside ``send_twilio_media``) to the B capture for this
    (callSid, gen).

    No-op when ``TTS_AUDIO_CAPTURE_DIR`` is unset."""
    _capture(call_sid, generation, mulaw_bytes, "twilio", None, capture_dir)


def close_all() -> None:
    """Flush + close capture files (file mode) AND wait briefly for
    the DB queue to drain (DB mode). Idempotent.

    Called from session cleanup so the last frame of the call lands
    in storage before the session row closes."""
    with _HANDLES_LOCK:
        for handles in _HANDLES.values():
            for fh in handles.values():
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
        _HANDLES.clear()
        _DIR_FAILED.clear()
        _ERROR_LOGGED.clear()
    # Drain the DB queue with a bounded wait. The worker is a daemon
    # thread so anything left after the cap is silently dropped when
    # the process exits — acceptable for a forensic diagnostic.
    if _worker_started:
        deadline = time.monotonic() + 2.0
        while not _capture_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)


def capture_queue_size() -> int:
    """Diagnostic: how many records are pending the DB worker.
    Useful for the call summary."""
    return _capture_queue.qsize()


if __name__ == "__main__":
    # Smoke: file-mode layout + bytes + JSONL. DB mode is exercised
    # by the unit tests with a mocked insert function.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TTS_AUDIO_CAPTURE_DIR"] = tmp
        capture_a("CA-test-A", 1, b"\xff" * 160, seg_idx=0)
        capture_a("CA-test-A", 1, b"\x01" * 320, seg_idx=1)
        capture_b("CA-test-A", 1, b"\x00" * 160)
        capture_b("CA-test-A", 1, b"\x02" * 160)
        capture_a("CA-test-A", 2, b"\x99" * 160)
        capture_b("CA-test-A", 2, b"\xaa" * 160)
        capture_a("CA-test-B", 1, b"\x55" * 160)
        capture_b("CA-test-B", 1, b"\x66" * 160)
        close_all()

        base_a = Path(tmp) / "CA-test-A"
        a1 = base_a / "gen-1-inworld.mulaw"
        a1j = base_a / "gen-1-inworld.jsonl"
        b1 = base_a / "gen-1-twilio.mulaw"
        b1j = base_a / "gen-1-twilio.jsonl"
        assert a1.read_bytes() == b"\xff" * 160 + b"\x01" * 320
        assert b1.read_bytes() == b"\x00" * 160 + b"\x02" * 160
        assert (base_a / "gen-2-inworld.mulaw").read_bytes() == b"\x99" * 160
        assert (base_a / "gen-2-twilio.mulaw").read_bytes() == b"\xaa" * 160
        base_b = Path(tmp) / "CA-test-B"
        assert (base_b / "gen-1-inworld.mulaw").read_bytes() == b"\x55" * 160
        assert (base_b / "gen-1-twilio.mulaw").read_bytes() == b"\x66" * 160

        lines = a1j.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        rec0 = json.loads(lines[0])
        rec1 = json.loads(lines[1])
        assert rec0["session"] == "CA-test-A"
        assert rec0["gen"] == 1
        assert rec0["stage"] == "inworld"
        assert rec0["seg"] == 0
        assert rec0["byte_count"] == 160
        assert rec0["sha256"] == hashlib.sha256(b"\xff" * 160).hexdigest()
        assert rec1["seg"] == 1
        assert rec1["byte_count"] == 320

        lines_b = b1j.read_text(encoding="utf-8").strip().splitlines()
        for r in (json.loads(x) for x in lines_b):
            assert r["stage"] == "twilio"
            assert r["seg"] is None

        print("audio_capture: OK")
