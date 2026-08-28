"""Postgres-backed audio capture for the 2026-08-28 forensic A/B test.

One INSERT per capture write (one Inworld chunk for stage='inworld',
one 160-byte Twilio frame for stage='twilio'). The raw mu-law bytes
go in the BYTEA column so the operator can SELECT them out and
diff against the AMR recording from Twilio.

Schema (db/migrations/015_audio_capture.sql):
  audio_capture(
    id              BIGSERIAL PRIMARY KEY,
    call_sid        TEXT NOT NULL,
    generation      INTEGER NOT NULL,
    stage           TEXT NOT NULL CHECK (stage IN ('inworld','twilio')),
    seg             INTEGER,
    byte_count      INTEGER NOT NULL,
    sha256          TEXT NOT NULL,
    payload         BYTEA NOT NULL,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )
"""
from __future__ import annotations

import logging

from STT_server.db import get_conn, is_postgres


log = logging.getLogger("stt_server.db_audio_capture")


def insert_capture(
    call_sid: str,
    generation: int,
    stage: str,
    seg: int | None,
    byte_count: int,
    sha256: str,
    payload: bytes,
) -> None:
    """Insert one capture row.

    No-op when Postgres is not configured — callers are expected to
    gate on ``is_postgres()`` before invoking. The function is the
    hot path for the async writer in audio_capture._drain_worker;
    any exception here propagates so the worker can log + continue.
    """
    if not is_postgres():
        return
    import psycopg2
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audio_capture
                  (call_sid, generation, stage, seg, byte_count, sha256, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    call_sid,
                    generation,
                    stage,
                    seg,
                    byte_count,
                    sha256,
                    psycopg2.Binary(payload),
                ),
            )


def list_captures(call_sid: str, generation: int | None = None) -> list[dict]:
    """Return all capture rows for a call (optionally filtered to one
    generation). Useful for the forensic A/B diff script — concatenate
    the payload bytes per (callSid, gen, stage) and compare SHA-256s."""
    if not is_postgres():
        return []
    with get_conn() as conn:
        with conn.cursor() as cur:
            if generation is None:
                cur.execute(
                    """
                    SELECT id, call_sid, generation, stage, seg, byte_count,
                           sha256, payload, ts
                    FROM audio_capture
                    WHERE call_sid = %s
                    ORDER BY generation, stage, seg, id
                    """,
                    (call_sid,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, call_sid, generation, stage, seg, byte_count,
                           sha256, payload, ts
                    FROM audio_capture
                    WHERE call_sid = %s AND generation = %s
                    ORDER BY stage, seg, id
                    """,
                    (call_sid, generation),
                )
            rows = cur.fetchall()
    import psycopg2
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        # RealDictCursor already gives dicts, but payload came back as
        # memoryview/bytes; keep it as bytes for caller convenience.
        if isinstance(d.get("payload"), (memoryview, bytes)):
            d["payload"] = bytes(d["payload"])
        out.append(d)
    return out


__all__ = ["insert_capture", "list_captures"]
