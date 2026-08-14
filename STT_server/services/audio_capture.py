"""Audio capture helpers for the 2026-08-14 A/B test.

The 2026-08-14 audio review concluded the artifacts are NOT in
WhatsApp / AMR but are introduced by the live pipeline. To pinpoint
WHERE, the operator needs three captures per call:

  A — bytes the TTS adapter produced (post-resample, post-μ-law
      encode, in the exact order it emitted them).
  B — bytes the runtime sends to Twilio (post-frame_proc, the
      exact 160-byte frames paced out).
  C — the AMR recording from Twilio / carrier.

This module writes A and B to disk when ``TTS_AUDIO_CAPTURE_DIR``
is set. C comes from Twilio's recording system and is unrelated
to the runtime.

Design rules:
  - Best-effort. A write failure logs once and disables capture
    for the rest of the call. NEVER crash the live path on a
    disk-full or permission error.
  - Per-callSid append-only. A new callSid opens a new file. The
    operator diffs them with ffmpeg / sox offline.
  - No locks. Append mode on POSIX is atomic for O(1024)-byte
    writes; Windows append mode is similarly OK for the sizes we
    emit (~160 B per frame). If concurrent writes become an issue
    we add a per-callSid lock, but the current call model is one
    writer per call.
  - Lazy-import pathlib inside the function so import time
    stays cheap when the env var is unset (the common case).
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import IO


log = logging.getLogger("stt_server.audio_capture")

# Per-callSid open file handles. Key = callSid, value = {"A":
# open handle, "B": open handle}. Cleared on close (operator
# drops the file via process exit; not a hot-path concern).
_HANDLES: dict[str, dict[str, IO[bytes]]] = {}
_HANDLES_LOCK = threading.Lock()

# Tracks callSids we've already logged a write error for, so a
# disk-full situation doesn't produce 8000 log lines per call.
_ERROR_LOGGED: set[str] = set()


def _open_locked(call_sid: str, capture_dir: str) -> dict[str, IO[bytes]] | None:
    """Open (lazily) the per-call A and B capture files. Returns
    None when the env var is unset, the directory is missing, or
    a previous write already failed for this callSid.

    Caller MUST hold ``_HANDLES_LOCK``.
    """
    if not capture_dir:
        return None
    if call_sid in _ERROR_LOGGED:
        return None
    existing = _HANDLES.get(call_sid)
    if existing is not None:
        return existing
    try:
        base = Path(capture_dir)
        base.mkdir(parents=True, exist_ok=True)
        handles: dict[str, IO[bytes]] = {}
        for label in ("A", "B"):
            path = base / f"{label}_inworld_{call_sid}.mulaw" if label == "A" else base / f"{label}_twilio_{call_sid}.mulaw"
            # 'ab' = append-binary, buffering=0 so writes hit disk
            # before the function returns (a slow disk doesn't
            # backpressure the WS send).
            handles[label] = open(path, "ab", buffering=0)
        _HANDLES[call_sid] = handles
        log.info(
            "[AUDIO_CAPTURE] enabled for call_sid=%s dir=%s "
            "files: %s, %s",
            call_sid, capture_dir,
            base / f"A_inworld_{call_sid}.mulaw",
            base / f"B_twilio_{call_sid}.mulaw",
        )
        return handles
    except Exception as exc:
        log.warning(
            "[AUDIO_CAPTURE] open failed for call_sid=%s dir=%s err=%s; "
            "disabling capture for this call",
            call_sid, capture_dir, exc,
        )
        _ERROR_LOGGED.add(call_sid)
        return None


def capture_a(call_sid: str, mulaw_bytes: bytes, capture_dir: str | None = None) -> None:
    """Append *mulaw_bytes* (post-resample, post-μ-law) to the A
    capture file for this call. No-op when capture is disabled.
    """
    _capture(call_sid, mulaw_bytes, "A", capture_dir)


def capture_b(call_sid: str, mulaw_bytes: bytes, capture_dir: str | None = None) -> None:
    """Append *mulaw_bytes* (a single 160-byte Twilio frame, exactly
    what is being base64-encoded and sent on the WS) to the B
    capture file for this call. No-op when capture is disabled.
    """
    _capture(call_sid, mulaw_bytes, "B", capture_dir)


def _capture(call_sid: str, mulaw_bytes: bytes, label: str, capture_dir: str | None) -> None:
    # ponytail: keep this fast in the no-op case (the default).
    # The check is one os.environ access; everything else is gated.
    if capture_dir is None:
        capture_dir = os.environ.get("TTS_AUDIO_CAPTURE_DIR", "").strip()
    if not capture_dir or not call_sid or not mulaw_bytes:
        return
    with _HANDLES_LOCK:
        handles = _open_locked(call_sid, capture_dir)
        if handles is None:
            return
        fh = handles.get(label)
        if fh is None:
            return
    # Write OUTSIDE the lock so a slow disk doesn't block the WS send.
    try:
        fh.write(mulaw_bytes)
    except Exception as exc:
        log.warning(
            "[AUDIO_CAPTURE] write failed for call_sid=%s label=%s err=%s; "
            "disabling capture for this call",
            call_sid, label, exc,
        )
        with _HANDLES_LOCK:
            _ERROR_LOGGED.add(call_sid)
            try:
                fh.close()
            except Exception:
                pass
            _HANDLES.pop(call_sid, None)


def close_all() -> None:
    """Flush + close every open capture file. Called from
    session_runtime.cleanup_session so the B file sees the last
    frame of the call. Idempotent."""
    with _HANDLES_LOCK:
        for sid, handles in list(_HANDLES.items()):
            for label, fh in handles.items():
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
        _HANDLES.clear()


if __name__ == "__main__":
    # Smoke: capture to a tmp dir, verify the file gets the bytes.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TTS_AUDIO_CAPTURE_DIR"] = tmp
        capture_a("CA-test-A", b"\xff" * 160)
        capture_b("CA-test-A", b"\x00" * 160)
        capture_a("CA-test-A", b"\x01" * 320)
        capture_b("CA-test-A", b"\x02" * 160)
        # Different callSid opens a separate file pair.
        capture_a("CA-test-B", b"\xaa" * 160)
        capture_b("CA-test-B", b"\xbb" * 160)
        # Force-flush.
        close_all()
        a_path = Path(tmp) / "A_inworld_CA-test-A.mulaw"
        b_path = Path(tmp) / "B_twilio_CA-test-A.mulaw"
        assert a_path.read_bytes() == b"\xff" * 160 + b"\x01" * 320
        assert b_path.read_bytes() == b"\x00" * 160 + b"\x02" * 160
        # Other call's files are separate.
        a_b = Path(tmp) / "A_inworld_CA-test-B.mulaw"
        b_b = Path(tmp) / "B_twilio_CA-test-B.mulaw"
        assert a_b.read_bytes() == b"\xaa" * 160
        assert b_b.read_bytes() == b"\xbb" * 160
        print("audio_capture: OK")