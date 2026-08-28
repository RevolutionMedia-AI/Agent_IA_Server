import base64
import logging
import time
from fastapi import WebSocket

log = logging.getLogger("stt_server")


def _seq_state(session, key, default=0):
    """Lazy-init per-session state on session (duck-typed; no CallSession change)."""
    if not hasattr(session, key):
        setattr(session, key, default)
    return getattr(session, key)


def track_twilio_sequence(session, event) -> None:
    """Parse sequenceNumber/timestamp from a Twilio `media` event and update
    session-attached counters (gaps, dupes, reorders, jitter). Shape-agnostic
    across `media.chunk` vs `media.payload` — these fields live on the envelope.

    ponytail: the previous version compared sequenceNumber across ALL
    event types (media, mark, start, stop). Twilio assigns its own
    monotonic sequenceNumber to each WebSocket message, but the
    media-stream counter is continuous across the whole session —
    every media, mark, start and stop carries one. The detector was
    reading the same counter and seeing a +1 jump between a `media`
    event and the next `mark` event, logging it as a sequence gap.
    The logs from 2026-08-14 show gaps in lock-step with mark acks
    (prev=3893, got=3895, missing=1) and never with actual missing
    audio — the count was tracking Twilio's WS heartbeat, not our
    audio. Filter to media events so the counter means what its
    name says: "how many audio frames did Twilio fail to send us?".
    """
    event_type = event.get("event")
    if event_type != "media":
        return

    # ponytail: hotfix for production crash — initialize ALL counters at the
    # start of every call so the log.warning on the gap branch can read
    # _twilio_seq_dupes / _twilio_seq_reorders without AttributeError when
    # the very first anomaly observed is a gap. Without this, the gap
    # branch would raise before the f-string finished rendering.
    _seq_state(session, "_twilio_seq_dupes", 0)
    _seq_state(session, "_twilio_seq_reorders", 0)
    _seq_state(session, "_twilio_seq_gaps", 0)
    _seq_state(session, "_twilio_first_seq", -1)

    seq = int(event.get("sequenceNumber", -1) or -1)
    prev = _seq_state(session, "_twilio_last_seq", -1)
    if seq >= 0:
        if prev >= 0:
            if seq == prev:
                session._twilio_seq_dupes += 1
            elif seq < prev:
                session._twilio_seq_reorders += 1
            elif seq > prev + 1:
                gap = seq - prev - 1
                session._twilio_seq_gaps += gap
                log.warning(
                    "[SEQ_GAP] session=%s stream=%s prev=%s got=%s missing=%d (running gaps=%d dupes=%d reorders=%d)",
                    session.session_key, getattr(session, "stream_sid", "?"),
                    prev, seq, gap, session._twilio_seq_gaps,
                    session._twilio_seq_dupes, session._twilio_seq_reorders,
                )
        else:
            session._twilio_first_seq = seq
        session._twilio_last_seq = seq

    ts = int(event.get("media", {}).get("timestamp", -1) or -1)
    now_mono = time.monotonic()
    prev_ts = _seq_state(session, "_twilio_last_chunk_ts", None)
    if ts >= 0 and prev_ts is not None:
        gap_ms = (now_mono - prev_ts) * 1000
        if gap_ms > 100:
            log.debug("[JITTER] session=%s media-arrival gap=%.1fms", session.session_key, gap_ms)
    session._twilio_last_chunk_ts = now_mono


def summarize_twilio_sequence(session) -> None:
    """One-line summary for the `stop` event. Call from the inbound stop handler."""
    gaps = getattr(session, "_twilio_seq_gaps", 0)
    dupes = getattr(session, "_twilio_seq_dupes", 0)
    reorders = getattr(session, "_twilio_seq_reorders", 0)
    first = getattr(session, "_twilio_first_seq", -1)
    last = getattr(session, "_twilio_last_seq", -1)
    log.info(
        "[SEQ_SUMMARY] session=%s stream=%s first=%s last=%s gaps=%d dupes=%d reorders=%d",
        session.session_key, getattr(session, "stream_sid", "?"),
        first, last, gaps, dupes, reorders,
    )


async def send_twilio_media(
    ws: WebSocket,
    stream_sid: str,
    mulaw_audio: bytes,
    call_sid: str = "",
    generation: int = 0,
) -> None:
    # ponytail: 2026-08-28 forensic A/B capture — write the exact
    # 160-byte μ-law frame about to be base64-encoded + sent on
    # the WS. Pair with the A capture (raw Inworld bytes) in
    # inworld_tts.py: sha256(A) == sha256(B) per generation means
    # the pipeline never touched the bytes; any divergence
    # pinpoints the stage that mutated the audio. No-op when
    # TTS_AUDIO_CAPTURE_DIR is unset (the default). Best-effort:
    # a write failure logs once and disables capture for this
    # (call, gen, stage) — NEVER blocks the WS send.
    if call_sid and mulaw_audio:
        from STT_server.services.audio_capture import capture_b
        capture_b(call_sid, generation, mulaw_audio)
    payload = base64.b64encode(mulaw_audio).decode("ascii")
    await ws.send_json(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }
    )


async def send_twilio_mark(ws: WebSocket, stream_sid: str, mark_name: str) -> None:
    await ws.send_json(
        {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": mark_name},
        }
    )


async def send_twilio_clear(ws: WebSocket, stream_sid: str) -> None:
    await ws.send_json({"event": "clear", "streamSid": stream_sid})
