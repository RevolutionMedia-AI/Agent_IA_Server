"""Rime realtime STT adapter.

Streams the session's mu-law 8 kHz mono Twilio audio into Rime's
WebSocket STT endpoint after converting to LINEAR16 PCM at 16 kHz
(Rime's documented minimum for streaming). Emits transcripts in the
same shape the rest of the pipeline expects.

Endpoint: wss://users-ws.rime.ai/v1/stt
Auth:     Authorization: Bearer <api_key>
Send:     {"type": "config", ...} once at start, then
          {"type": "audio", "data": "<base64>"} per chunk
Recv:     {"type": "transcript", "text", "isFinal", "speechFinal"} |
          {"type": "error", "message"}

ponytail: Rime's public docs at the time of writing don't expose a
stable STT WS endpoint URL — the REST models endpoint at
users-ws.rime.ai/v1/models lists mist-v2 as an STT model, and the
TTS adapter uses users-ws.rime.ai/ws3. The STT path `/v1/stt` is
the natural sibling of the TTS path; if Rime's actual STT endpoint
differs, only the constant below needs to change. Auth + protocol
shape follow the same conventions as the TTS adapter.
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
from STT_server.domain.language import normalize_supported_language
from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider

log = logging.getLogger("stt_server")


RIME_STT_WS_URL = "wss://users-ws.rime.ai/v1/stt"
RIME_STT_SAMPLE_RATE = 16000
RIME_DEFAULT_MODEL = "mist-v2"


def _mulaw_8k_to_pcm16_16k(mulaw: bytes) -> bytes:
    """Convert Twilio's mu-law 8 kHz mono to LINEAR16 PCM 16 kHz mono.

    Mirrors inworld_stt_realtime._mulaw_8k_to_pcm16_16k — same
    mid-point 2x upsample so the rate matches Rime's minimum without
    inventing high-frequency content that 8 kHz telephony audio
    doesn't carry.
    """
    pcm_8k = audioop.ulaw2lin(mulaw, 2)
    n_8k = len(pcm_8k) // 2
    if n_8k == 0:
        return b""
    samples_8k = struct.unpack(f"<{n_8k}h", pcm_8k)
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
    creds = resolve_provider(user_id, "rime") if user_id else {}
    return (creds.get("api_key") or "").strip()


def _resolve_model(session: CallSession) -> str:
    return getattr(session, "stt_model", None) or RIME_DEFAULT_MODEL


async def _rime_audio_sender(ws, session: CallSession, model_id: str, language: str) -> None:
    """Send config once, then pump Twilio audio as LINEAR16 @ 16 kHz."""
    await ws.send(json.dumps({
        "type": "config",
        "modelId": model_id,
        "audioFormat": {
            "encoding": "linear16",
            "sampleRate": RIME_STT_SAMPLE_RATE,
            "channels": 1,
        },
        "language": language,
    }))

    while True:
        try:
            chunk = await asyncio.wait_for(session.stt_audio_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            if session.closed:
                return
            # No audio flowing. Rime doesn't need keepalive pings here —
            # wait for the next real chunk.
            continue
        if chunk is None:
            # End-of-session sentinel.
            try:
                await ws.close()
            except Exception:
                pass
            return
        if not chunk:
            continue
        pcm = _mulaw_8k_to_pcm16_16k(chunk)
        if not pcm:
            continue
        try:
            await ws.send(json.dumps({
                "type": "audio",
                "data": base64.b64encode(pcm).decode("ascii"),
            }))
        except Exception:
            return


async def run_realtime_stt(session: CallSession, on_transcript, on_failure) -> None:
    """Mirror inworld_stt_realtime.run_realtime_stt signature."""
    api_key = _resolve_api_key(session)
    if not api_key:
        return

    model_id = _resolve_model(session)
    language = normalize_supported_language(session.preferred_language)
    connect_kwargs = {"ping_interval": 20, "ping_timeout": 20}

    attempt = 0
    while not session.closed:
        sender_task: asyncio.Task | None = None
        received_any_result = False
        try:
            try:
                conn = websockets.connect(
                    RIME_STT_WS_URL,
                    additional_headers={"Authorization": f"Bearer {api_key}"},
                    **connect_kwargs,
                )
            except TypeError:
                conn = websockets.connect(
                    RIME_STT_WS_URL,
                    extra_headers={"Authorization": f"Bearer {api_key}"},
                    **connect_kwargs,
                )

            async with conn as ws:
                sender_task = asyncio.create_task(
                    _rime_audio_sender(ws, session, model_id, language)
                )
                log.info(
                    "[RIME_STT] WS connected for %s (model=%s lang=%s)",
                    session.session_key, model_id, language,
                )

                while not session.closed:
                    try:
                        raw = await ws.recv()
                    except ConnectionClosed:
                        if not received_any_result:
                            log.warning(
                                "[RIME_STT] WS closed without transcripts in %s",
                                session.session_key,
                            )
                        break

                    if isinstance(raw, bytes):
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    mtype = msg.get("type")
                    if mtype == "transcript":
                        text = (msg.get("text") or "").strip()
                        if not text:
                            continue
                        is_final = bool(msg.get("isFinal"))
                        speech_final = bool(msg.get("speechFinal", is_final))
                        received_any_result = True
                        attempt = 0
                        await on_transcript({
                            "text": text,
                            "language": language,
                            "is_final": is_final,
                            "speech_final": speech_final,
                            "source": "rime_realtime",
                        })
                        continue

                    if mtype == "error":
                        log.error(
                            "[RIME_STT] error in %s: %s",
                            session.session_key,
                            msg.get("message") or msg,
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
                        "[RIME_STT] dropped without results for %s, retrying",
                        session.session_key,
                    )
                    continue
                break
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "unknown")
            log.warning(
                "[RIME_STT] handshake rejected in %s status=%s",
                session.session_key, status,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[RIME_STT] error in %s", session.session_key)
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
            "[RIME_STT] reconnecting %s in %sms (attempt %s/%s)",
            session.session_key, delay_ms, attempt, STT_RECONNECT_MAX_ATTEMPTS,
        )
        await asyncio.sleep(delay_ms / 1000.0)
