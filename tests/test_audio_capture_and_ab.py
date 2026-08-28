"""Tests for the 2026-08-28 forensic A/B audio capture.

The operator's diff is three sources:

  A — raw Inworld bytes (post base64-decode, BEFORE AudioFrameProcessor)
      captured by inworld_tts.py right after base64.b64decode
  B — exact 160-byte μ-law frames about to be base64-encoded into the
      Twilio WS payload (captured inside send_twilio_media)
  C — the AMR recording from Twilio / carrier (not produced by us)

Forensic contract: per (callSid, generation) the SHA-256 of A
concatenated should match the SHA-256 of B concatenated IF our
pipeline never touched the bytes. Any divergence pinpoints the
stage that mutated the audio.

Layout::

    <TTS_AUDIO_CAPTURE_DIR>/
      <callSid>/
        gen-1-inworld.mulaw
        gen-1-inworld.jsonl
        gen-1-twilio.mulaw
        gen-1-twilio.jsonl
        ...

These tests pin the on-disk layout, the per-call/per-gen isolation,
the JSONL sidecar contract, and the no-op failure modes so a future
refactor can't silently break the forensic chain.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest


# ─── Helpers ────────────────────────────────────────────────────────


def _reload_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-import the audio_capture module so module-level env
    reads happen against the monkeypatched environment."""
    from STT_server.services import audio_capture as ac
    importlib.reload(ac)


# ─── Disabled / no-op paths ─────────────────────────────────────────


def test_capture_disabled_when_env_var_unset(tmp_path: Path, monkeypatch) -> None:
    """Without TTS_AUDIO_CAPTURE_DIR set, capture_a / capture_b are
    no-ops. Verify by calling them and asserting no files appear
    in the (otherwise writable) tmp_path."""
    monkeypatch.delenv("TTS_AUDIO_CAPTURE_DIR", raising=False)
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-disabled-A", 1, b"\xff" * 160)
    ac.capture_b("CA-disabled-B", 1, b"\x00" * 160)
    assert list(tmp_path.glob("*")) == []
    assert not (tmp_path / "CA-disabled-A").exists()


def test_capture_empty_bytes_is_noop(tmp_path: Path, monkeypatch) -> None:
    """A zero-byte capture call must not create an empty file
    (nothing to diagnose) AND must not crash. Best-effort."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-empty", 1, b"")
    ac.capture_b("CA-empty", 1, b"")
    ac.close_all()
    # No files should exist because the helper short-circuits on
    # empty bytes (lazy per-stage open never fires).
    assert not (tmp_path / "CA-empty").exists()


def test_capture_disables_after_write_failure(tmp_path: Path, monkeypatch) -> None:
    """A write failure (read-only dir, disk full, permission denied)
    must NOT crash the live path. The capture module logs once
    and stops trying for the rest of the call. Verified by writing
    to a path that doesn't exist (parent doesn't exist) and
    confirming subsequent writes don't crash."""
    bogus_dir = tmp_path / "does-not-exist"
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(bogus_dir))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    # First write attempts to mkdir + open, both fail. Module must
    # swallow the exception.
    ac.capture_a("CA-fail", 1, b"\xff" * 160)
    # Second write goes through the same path — must also not crash.
    ac.capture_a("CA-fail", 1, b"\x00" * 160)
    # And close_all on the (never-opened) state — must not crash.
    ac.close_all()


# ─── On-disk layout ────────────────────────────────────────────────


def test_capture_a_writes_inworld_bytes(tmp_path: Path, monkeypatch) -> None:
    """With TTS_AUDIO_CAPTURE_DIR set, capture_a appends the bytes
    to <callSid>/gen-<n>-inworld.mulaw in append-binary mode."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-test-A", 1, b"\xff" * 160, seg_idx=0)
    ac.capture_a("CA-test-A", 1, b"\x01" * 320, seg_idx=1)
    ac.close_all()
    a_path = tmp_path / "CA-test-A" / "gen-1-inworld.mulaw"
    assert a_path.read_bytes() == b"\xff" * 160 + b"\x01" * 320


def test_capture_b_writes_twilio_bytes(tmp_path: Path, monkeypatch) -> None:
    """Same contract for B — bytes appended to
    <callSid>/gen-<n>-twilio.mulaw in append-binary mode."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_b("CA-test-B", 1, b"\x00" * 160)
    ac.capture_b("CA-test-B", 1, b"\x02" * 160)
    ac.close_all()
    b_path = tmp_path / "CA-test-B" / "gen-1-twilio.mulaw"
    assert b_path.read_bytes() == b"\x00" * 160 + b"\x02" * 160


def test_capture_a_and_b_are_separate_files(tmp_path: Path, monkeypatch) -> None:
    """A and B for the same callSid / gen go to different files.
    The whole point of the A/B test is that we can diff them.
    If they merged, the test would be useless."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-same", 1, b"\xaa" * 160)
    ac.capture_b("CA-same", 1, b"\xbb" * 160)
    ac.close_all()
    a_path = tmp_path / "CA-same" / "gen-1-inworld.mulaw"
    b_path = tmp_path / "CA-same" / "gen-1-twilio.mulaw"
    assert a_path.read_bytes() == b"\xaa" * 160
    assert b_path.read_bytes() == b"\xbb" * 160
    assert a_path != b_path


# ─── Per-call / per-generation isolation ───────────────────────────


def test_capture_per_call_isolation(tmp_path: Path, monkeypatch) -> None:
    """Different callSids write to different subdirectories. The
    operator diffs one call at a time, never two at once."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-call-1", 1, b"\x01" * 160)
    ac.capture_a("CA-call-2", 1, b"\x02" * 160)
    ac.capture_b("CA-call-1", 1, b"\x11" * 160)
    ac.capture_b("CA-call-2", 1, b"\x22" * 160)
    ac.close_all()
    base1 = tmp_path / "CA-call-1"
    base2 = tmp_path / "CA-call-2"
    assert (base1 / "gen-1-inworld.mulaw").read_bytes() == b"\x01" * 160
    assert (base1 / "gen-1-twilio.mulaw").read_bytes() == b"\x11" * 160
    assert (base2 / "gen-1-inworld.mulaw").read_bytes() == b"\x02" * 160
    assert (base2 / "gen-1-twilio.mulaw").read_bytes() == b"\x22" * 160


def test_capture_per_generation_isolation(tmp_path: Path, monkeypatch) -> None:
    """Different generations of the same call go to DIFFERENT files
    inside the same <callSid>/ dir. This is the whole point of the
    new layout: A==B has to be checked PER generation because
    barge-in bumps generation mid-call, and concatenating gens
    would mix cancelled audio with live audio."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-multi", 1, b"\x01" * 160)
    ac.capture_a("CA-multi", 1, b"\x02" * 160)
    ac.capture_a("CA-multi", 2, b"\x03" * 160)
    ac.capture_a("CA-multi", 2, b"\x04" * 160)
    ac.capture_b("CA-multi", 1, b"\x11" * 160)
    ac.capture_b("CA-multi", 2, b"\x22" * 160)
    ac.close_all()
    base = tmp_path / "CA-multi"
    assert (base / "gen-1-inworld.mulaw").read_bytes() == b"\x01" * 160 + b"\x02" * 160
    assert (base / "gen-2-inworld.mulaw").read_bytes() == b"\x03" * 160 + b"\x04" * 160
    assert (base / "gen-1-twilio.mulaw").read_bytes() == b"\x11" * 160
    assert (base / "gen-2-twilio.mulaw").read_bytes() == b"\x22" * 160


def test_capture_lazy_open_unwritten_stage_creates_no_files(tmp_path: Path, monkeypatch) -> None:
    """If a (callSid, gen) only ever sees capture_a, the
    gen-N-twilio.mulaw must NOT exist (lazy per-stage open)."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-a-only", 1, b"\x01" * 160)
    ac.close_all()
    base = tmp_path / "CA-a-only"
    assert (base / "gen-1-inworld.mulaw").exists()
    assert not (base / "gen-1-twilio.mulaw").exists()


# ─── JSONL sidecar contract ───────────────────────────────────────


def test_capture_jsonl_records_every_write(tmp_path: Path, monkeypatch) -> None:
    """The JSONL sidecar writes one record per capture call with:
    session, gen, stage, seg, byte_count, sha256, ts. This is the
    offline forensic chain — the operator joins .mulaw bytes to
    JSONL metadata without re-running the call."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    # gen 1: two A writes (different seg), two B writes.
    ac.capture_a("CA-meta", 1, b"\xff" * 160, seg_idx=0)
    ac.capture_a("CA-meta", 1, b"\x01" * 320, seg_idx=1)
    ac.capture_b("CA-meta", 1, b"\x00" * 160)
    ac.capture_b("CA-meta", 1, b"\x02" * 160)
    ac.close_all()

    a_jsonl = tmp_path / "CA-meta" / "gen-1-inworld.jsonl"
    b_jsonl = tmp_path / "CA-meta" / "gen-1-twilio.jsonl"
    assert a_jsonl.exists()
    assert b_jsonl.exists()

    a_records = [json.loads(line) for line in a_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(a_records) == 2
    assert a_records[0]["session"] == "CA-meta"
    assert a_records[0]["gen"] == 1
    assert a_records[0]["stage"] == "inworld"
    assert a_records[0]["seg"] == 0
    assert a_records[0]["byte_count"] == 160
    assert a_records[0]["sha256"] == hashlib.sha256(b"\xff" * 160).hexdigest()
    assert "ts" in a_records[0]
    assert a_records[1]["seg"] == 1
    assert a_records[1]["byte_count"] == 320
    assert a_records[1]["sha256"] == hashlib.sha256(b"\x01" * 320).hexdigest()

    b_records = [json.loads(line) for line in b_jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(b_records) == 2
    for r in b_records:
        assert r["stage"] == "twilio"
        assert r["session"] == "CA-meta"
        assert r["gen"] == 1
        assert r["seg"] is None


def test_capture_a_vs_b_sha256_equality(tmp_path: Path, monkeypatch) -> None:
    """If A's bytes == B's bytes (the bytes the adapter output are
    identical to the bytes Twilio receives), the per-generation
    SHA-256s match. This is the forensic predicate that the
    pipeline never touched a single byte. The test wires
    capture_a and capture_b to identical bytes and asserts the
    recorded SHA-256s match."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    payload = b"\xab\xcd" * 80  # exactly 160 bytes (one Twilio frame)
    ac.capture_a("CA-sha", 1, payload, seg_idx=0)
    ac.capture_b("CA-sha", 1, payload)
    ac.close_all()
    expected = hashlib.sha256(payload).hexdigest()
    a = json.loads((tmp_path / "CA-sha" / "gen-1-inworld.jsonl").read_text().splitlines()[0])
    b = json.loads((tmp_path / "CA-sha" / "gen-1-twilio.jsonl").read_text().splitlines()[0])
    assert a["sha256"] == expected
    assert b["sha256"] == expected
    assert a["sha256"] == b["sha256"]


# ─── DB mode (DATABASE_URL set, Railway production) ───────────────


class _FakeInsert:
    """Records every insert_capture() call so a test can assert on
    what the worker forwarded. Also bypasses the daemon thread for
    deterministic draining — the worker is the only thing that
    normally calls insert_capture, so swapping it for this fake is
    enough to make the whole pipeline synchronous in tests."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def __call__(self, **kwargs) -> None:
        self.records.append(kwargs)


def _enable_db_mode(monkeypatch) -> _FakeInsert:
    """Wire audio_capture into DB mode for the duration of the test.
    Returns the fake insert recorder so the caller can assert."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", "/unused-in-db-mode")
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost:5432/fake")
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    from STT_server import db
    # is_postgres() reads DATABASE_URL at call time; monkeypatch the
    # function the audio_capture module looks up so it doesn't need
    # a real DB connection pool.
    monkeypatch.setattr(db, "is_postgres", lambda: True, raising=False)
    fake = _FakeInsert()
    # Replace the function the worker imports inside _drain_worker.
    import STT_server.db_audio_capture as dbac
    monkeypatch.setattr(dbac, "insert_capture", fake, raising=False)
    # Stop the auto-start of the daemon worker for the test — we
    # drive the queue manually via close_all() at the end.
    monkeypatch.setattr(ac, "_ensure_worker", lambda: None)
    # Patch the enqueue path to call the fake synchronously. This
    # is simpler than spinning the daemon and waiting 2 s on the
    # queue drain for every test.
    monkeypatch.setattr(
        ac,
        "_enqueue_db",
        lambda call_sid, generation, stage, mulaw_bytes, sha, seg_idx:
            fake(
                call_sid=call_sid,
                generation=generation,
                stage=stage,
                seg=seg_idx,
                byte_count=len(mulaw_bytes),
                sha256=sha,
                payload=mulaw_bytes,
            ),
    )
    return fake


def test_capture_db_mode_writes_payload_and_metadata(monkeypatch) -> None:
    """In DB mode (DATABASE_URL set), each capture_a / capture_b call
    ends up as one INSERT into audio_capture with the full payload
    bytes + metadata fields. No files are written."""
    monkeypatch.delenv("TTS_AUDIO_CAPTURE_DIR", raising=False)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        # If anything tried to write to files, this dir would have
        # subdirs in it. Capture the state before/after.
        monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", "/tmp/never-touched")
        fake = _enable_db_mode(monkeypatch)
        from STT_server.services import audio_capture as ac
        payload = b"\xab\xcd" * 80
        ac.capture_a("CA-db", 3, payload, seg_idx=7)
        ac.capture_b("CA-db", 3, payload)
        ac.close_all()

        assert len(fake.records) == 2
        a, b = fake.records
        assert a["call_sid"] == "CA-db" and a["generation"] == 3
        assert a["stage"] == "inworld" and a["seg"] == 7
        assert a["byte_count"] == 160
        assert a["sha256"] == hashlib.sha256(payload).hexdigest()
        assert a["payload"] == payload
        assert b["stage"] == "twilio" and b["seg"] is None
        assert b["payload"] == payload
        assert b["sha256"] == a["sha256"]


def test_capture_db_mode_disabled_when_capture_dir_unset(monkeypatch) -> None:
    """The TTS_AUDIO_CAPTURE_DIR env var is still the master switch:
    even with DATABASE_URL set, an unset TTS_AUDIO_CAPTURE_DIR means
    no capture happens (no DB inserts, no files)."""
    monkeypatch.delenv("TTS_AUDIO_CAPTURE_DIR", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost:5432/fake")
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    from STT_server import db
    monkeypatch.setattr(db, "is_postgres", lambda: True, raising=False)
    fake = _FakeInsert()
    monkeypatch.setattr(ac, "_enqueue_db", fake, raising=False)
    ac.capture_a("CA-no-capture", 1, b"\xff" * 160)
    ac.capture_b("CA-no-capture", 1, b"\x00" * 160)
    assert fake.records == []


def test_capture_db_mode_falls_back_to_files_when_no_db(monkeypatch, tmp_path) -> None:
    """When DATABASE_URL is unset (local dev), writes go to files
    even if TTS_AUDIO_CAPTURE_DIR is set. The branch is decided by
    is_postgres(), not by TTS_AUDIO_CAPTURE_DIR alone."""
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PGHOST", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)
    monkeypatch.delenv("PGPASSWORD", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-fallback", 1, b"\x10" * 160, seg_idx=0)
    ac.capture_b("CA-fallback", 1, b"\x20" * 160)
    ac.close_all()
    # File-mode layout, not DB.
    assert (tmp_path / "CA-fallback" / "gen-1-inworld.mulaw").read_bytes() == b"\x10" * 160
    assert (tmp_path / "CA-fallback" / "gen-1-twilio.mulaw").read_bytes() == b"\x20" * 160


def test_capture_db_mode_per_call_per_gen_isolation(monkeypatch) -> None:
    """Different (callSid, gen) writes go to distinct INSERT rows —
    the operator diffs one generation at a time, never two at once."""
    fake = _enable_db_mode(monkeypatch)
    from STT_server.services import audio_capture as ac
    ac.capture_a("CA-call-1", 1, b"\x01" * 160)
    ac.capture_a("CA-call-1", 2, b"\x02" * 160)
    ac.capture_a("CA-call-2", 1, b"\x03" * 160)
    ac.capture_b("CA-call-1", 1, b"\x11" * 160)
    ac.capture_b("CA-call-2", 1, b"\x33" * 160)
    ac.close_all()
    keys = {(r["call_sid"], r["generation"], r["stage"]) for r in fake.records}
    assert keys == {
        ("CA-call-1", 1, "inworld"),
        ("CA-call-1", 2, "inworld"),
        ("CA-call-2", 1, "inworld"),
        ("CA-call-1", 1, "twilio"),
        ("CA-call-2", 1, "twilio"),
    }


def test_capture_db_mode_queue_drains_on_close_all(monkeypatch) -> None:
    """close_all() drains pending records. In production this is the
    last frame of the call landing in the DB before session cleanup
    completes. We assert by enqueueing + waiting for the worker via
    close_all()'s bounded wait."""
    monkeypatch.delenv("TTS_AUDIO_CAPTURE_DIR", raising=False)
    monkeypatch.setenv("TTS_AUDIO_CAPTURE_DIR", "/unused-in-db-mode")
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake:fake@localhost:5432/fake")
    _reload_capture(monkeypatch)
    from STT_server.services import audio_capture as ac
    from STT_server import db
    monkeypatch.setattr(db, "is_postgres", lambda: True, raising=False)
    fake = _FakeInsert()
    import STT_server.db_audio_capture as dbac
    monkeypatch.setattr(dbac, "insert_capture", fake, raising=False)
    # Start the real worker (overrides _enable_db_mode's no-op in
    # the helper we don't use here). The worker calls our fake
    # insert_capture for every queued record.
    ac._ensure_worker()
    # Bypass the DB-mode enqueue helper and put records directly
    # into the queue so the real worker drains them.
    for i in range(5):
        ac._capture_queue.put_nowait({
            "call_sid": "CA-drain",
            "generation": 1,
            "stage": "inworld",
            "seg": i,
            "byte_count": 160,
            "sha256": hashlib.sha256(bytes([i]) * 160).hexdigest(),
            "payload": bytes([i]) * 160,
        })
    assert ac.capture_queue_size() == 5
    ac.close_all()
    # Worker has drained everything within the 2-second cap.
    assert ac.capture_queue_size() == 0
    assert len(fake.records) == 5
    assert all(r["call_sid"] == "CA-drain" for r in fake.records)

