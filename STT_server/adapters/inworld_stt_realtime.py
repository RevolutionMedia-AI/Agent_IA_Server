"""Inworld STT adapter (realtime, bidirectional WebSocket).

Streams the session's mu-law 8 kHz Twilio audio into Inworld's
/stt/v1/transcribe:streamBidirectional WS endpoint after
converting to LINEAR16 PCM at 16 kHz (Inworld's minimum required
rate for raw PCM streams). Emits transcripts in the same shape
the rest of the pipeline expects.

Auth: same Basic scheme as Inworld TTS - the api_key is the
Base64 credential and ships verbatim in the Authorization header.

Endpoint: https://docs.inworld.ai/api-reference/sttAPI/speechtotext/transcribe-stream-websocket
Model:    inworld/inworld-stt-1 (first-party)
"""
import asyncio
import audioop
import base64
import json
import logging
import struct

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    STT_RECONNECT_BASE_DELAY_MS,
    STT_RECONNECT_MAX_ATTEMPTS,
    STT_RECONNECT_MAX_DELAY_MS,
)
from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider

log = logging.getLogger("stt_server")

INWORLD_WS_URL = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"
INWORLD_STT_SAMPLE_RATE = 16000  # raw PCM requires >=8000; 16k matches the doc default
INWORLD_DEFAULT_MODEL = "inworld/inworld-stt-1"


def _mulaw_8k_to_pcm16_16k(mulaw: bytes) -> bytes:
    """Convert mu-law 8 kHz mono bytes from Twilio to LINEAR16 PCM 16 kHz mono.

    Inworld's WS only accepts LINEAR16 / MP3 / OGG_OPUS / FLAC / AUTO_DETECT
    for streaming; MP3/OGG_OPUS/FLAC aren't supported for streaming per the
    docs, so LINEAR16 is the choice. The docs warn that 8 kHz telephony
    audio has lower transcription quality - we don't upsample by stretching,
    we interpolate at the mid-points so the rate is correct without
    hallucinating high-frequency content.
    """
    pcm_8k = audioop.ulaw2lin(mulaw, 2)  # PCM16 @ 8 kHz mono
    n_8k = len(pcm_8k) // 2
    if n_8k == 0:
        return b""
    samples_8k = struct.unpack(f"<{n_8k}h", pcm_8k)
    # Linear 2x upsample with mid-point averaging.
    samples_16k = [0] * (2 * n_8k)
    for i, s in enumerate(samples_8k):
        samples_16k[2 * i] = s
        if i + 1 < n_8k:
            samples_16k[2 * i + 1] = (s + samples_8k[i + 1]) >> 1
        else:
            samples_16k[2 * i + 1] = s
    return struct.pack(f"<{len(samples_16k)}h", *samples_16k)


def _resolve_api_key(session: CallSession) -> str:
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "inworld") if user_id else {}
    return (creds.get("api_key") or "").strip()


def _resolve_model(session: CallSession) -> str:
    return getattr(session, "stt_model", None) or INWORLD_DEFAULT_MODEL


async def _inworld_audio_sender(ws, session: CallSession, model_id: str, language: str) -> None:
    """First sends transcribeConfig, then pumps Twilio audio as LINEAR16 chunks."""
    await ws.send(json.dumps({
        "transcribeConfig": {
            "modelId": model_id,
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": INWORLD_STT_SAMPLE_RATE,
            "numberOfChannels": 1,
            "language": language,
        }
    }))

    while True:
        try:
            chunk = await asyncio.wait_for(session.stt_audio_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            if session.closed:
                return
            # No audio flowing (likely during TTS playback muting). Don't
            # ping Inworld with empty data - just wait for the next frame.
            continue
        if chunk is None:
            # ponytail: None is the "session ended" sentinel that
            # session_runtime enqueues before tearing the session down.
            try:
                await ws.send(json.dumps({"closeStream": {}}))
            except Exception:
                return
            return
        if not chunk:
            continue
        pcm = _mulaw_8k_to_pcm16_16k(chunk)
        if not pcm:
            continue
        await ws.send(json.dumps({
            "audioChunk": {"content": base64.b64encode(pcm).decode("ascii")},
        }))


async def run_realtime_stt(session: CallSession, on_transcript, on_failure) -> None:
    """Mirror deepgram_stt_realtime.run_realtime_stt - same signature so
    the per-session STT dispatcher in STT_Server.py can route either
    provider's adapter in. The on_transcript callback expects the
    TurnManager's transcript-event shape.
    """
    api_key = _resolve_api_key(session)
    if not api_key:
        return

    model_id = _resolve_model(session)
    language = (session.preferred_language or DEFAULT_CALL_LANGUAGE or "en").strip().lower()
    auth_header_value = f"Basic {api_key}"
    connect_kwargs = {"ping_interval": 20, "ping_timeout": 20}

    attempt = 0
    while not session.closed:
        sender_task: asyncio.Task | None = None
        received_any_result = False
        try:
            try:
                conn = websockets.connect(
                    INWORLD_WS_URL,
                    additional_headers={"Authorization": auth_header_value},
                    **connect_kwargs,
                )
            except TypeError:
                # websockets < 10 / older API
                conn = websockets.connect(
                    INWORLD_WS_URL,
                    extra_headers={"Authorization": auth_header_value},
                    **connect_kwargs,
                )

            async with conn as ws:
                sender_task = asyncio.create_task(
                    _inworld_audio_sender(ws, session, model_id, language)
                )
                log.info(
                    "[INWORLD_STT] WS connected for %s (model=%s lang=%s)",
                    session.session_key, model_id, language,
                )

                while not session.closed:
                    try:
                        raw = await ws.recv()
                    except ConnectionClosed:
                        if not received_any_result:
                            log.warning(
                                "[INWORLD_STT] WS closed without transcripts in %s",
                                session.session_key,
                            )
                        break

                    if isinstance(raw, bytes):
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    if "transcription" in msg:
                        p = msg["transcription"] or {}
                        text = (p.get("transcript") or "").strip()
                        is_final = bool(p.get("isFinal"))
                        if text:
                            received_any_result = True
                            attempt = 0
                            await on_transcript({
                                "text": text,
                                "language": language,
                                "is_final": is_final,
                                # ponytail: Inworld doesn't have a separate
                                # speech_final signal - is_final is the
                                # closest analog, same as Deepgram's
                                # interim_results=true path.
                                "speech_final": is_final,
                                "source": "inworld_realtime",
                            })
                        continue

                    if "speechStarted" in msg or "speechStopped" in msg:
                        # ponytail: VAD events for barge-in. The TurnManager
                        # does its own barge-in via session-level VAD
                        # (audio_ingest.py), so we accept and ignore these
                        # here. Future: hook session.interrupt_current_turn
                        # off speechStarted if caller needs server-side VAD.
                        continue

                    if "usage" in msg:
                        # Per docs: "Coming soon" - not populated yet.
                        continue

                    if "error" in msg:
                        log.error(
                            "[INWORLD_STT] WS error in %s: %s",
                            session.session_key, msg["error"],
                        )
                        break

                if session.closed:
                    return

                if not received_any_result:
                    if sender_task is not None:
                        sender_task.cancel()
                        try:
                            await sender_task
                        except asyncio.CancelledError:
                            pass
                        sender_task = None
                    log.info(
                        "[INWORLD_STT] dropped without results for %s, retrying",
                        session.session_key,
                    )
                    continue
                break
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "unknown")
            log.warning(
                "[INWORLD_STT] handshake rejected in %s status=%s",
                session.session_key, status,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[INWORLD_STT] error in %s", session.session_key)
        finally:
            if sender_task is not None:
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass

        attempt += 1
        if attempt > STT_RECONNECT_MAX_ATTEMPTS:
            await on_failure(session)
            return

        delay_ms = min(
            STT_RECONNECT_BASE_DELAY_MS * (2 ** (attempt - 1)),
            STT_RECONNECT_MAX_DELAY_MS,
        )
        log.warning(
            "[INWORLD_STT] reconnecting %s in %sms (attempt %s/%s)",
            session.session_key, delay_ms, attempt, STT_RECONNECT_MAX_ATTEMPTS,
        )
        await asyncio.sleep(delay_ms / 1000.0)
