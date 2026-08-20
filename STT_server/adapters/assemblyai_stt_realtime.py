"""AssemblyAI realtime STT adapter.

Streams the session's mu-law 8 kHz mono Twilio audio into AssemblyAI's
WebSocket realtime endpoint. Zero transcoding: Twilio's native format
matches AssemblyAI's preferred ``encoding=pcm_mulaw, sample_rate=8000``
exactly, so chunks are forwarded as-is (base64-wrapped).

Endpoint: wss://api.assemblyai.com/v2/realtime/ws
Auth:     Authorization header = raw api_key (no Bearer prefix per
          AssemblyAI docs; verified against /v2/models which uses the
          same pattern in credentials_resolver._fetch_assemblyai_models)
Send:     {"type": "data", "audio_data": "<base64>"}
Recv:     {"type": "Begin"} | {"type": "Turn", "transcript", "end_of_turn", ...}
        | {"type": "Termination"} | {"type": "Error", "error": ...}

The `turn_is_formatted` flag is set on the server side (we open the
socket with format_turns=true) so transcripts arrive already punctuated.
"""
import asyncio
import base64
import json
import logging
import urllib.parse

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    STT_RECONNECT_BASE_DELAY_MS,
    STT_RECONNECT_MAX_ATTEMPTS,
    STT_RECONNECT_MAX_DELAY_MS,
    TWILIO_SR,
)
from STT_server.domain.language import normalize_supported_language
from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_for_session

log = logging.getLogger("stt_server")


ASSEMBLYAI_WS_URL = "wss://api.assemblyai.com/v2/realtime/ws"
ASSEMBLYAI_DEFAULT_MODEL = "best"
# ponytail: AssemblyAI's accepted language codes for the `language`
# query param. Anything outside this list falls back to "auto" so the
# server picks instead of the WS handshake 400-ing us.
_ASSEMBLYAI_LANG_MAP = {
    "en": "en_us",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "ja": "ja",
    "zh": "zh",
    "ko": "ko",
    "ru": "ru",
    "hi": "hi",
    "pl": "pl",
    "tr": "tr",
    "uk": "uk",
}


def _build_url(language_hint: str | None) -> str:
    """Build the realtime WS URL with Twilio-matching audio params."""
    params = {
        "sample_rate": str(TWILIO_SR),
        "encoding": "pcm_mulaw",
        "format_turns": "true",
        # ponytail: confidence below 0.7 means AssemblyAI is still
        # waiting for more audio before closing the turn. 0.7 is the
        # value they document for live captioning; tighter cuts lose
        # mid-sentence hesitations.
        "end_of_turn_confidence_threshold": "0.7",
    }
    normalized = normalize_supported_language(language_hint)
    code = _ASSEMBLYAI_LANG_MAP.get(normalized)
    if code:
        params["language"] = code
    qs = urllib.parse.urlencode(params)
    return f"{ASSEMBLYAI_WS_URL}?{qs}"


def _resolve_api_key(session: CallSession) -> str:
    creds = resolve_for_session(session, "stt", "assemblyai")
    return (creds.get("api_key") or "").strip()


def _resolve_model(session: CallSession) -> str:
    # ponytail: model is currently a no-op for AssemblyAI (the realtime
    # endpoint ignores model_id; only the model param on /v2/transcript
    # matters, which we're not using). Keep it on the session so the
    # UI doesn't lie when the user picks "Universal-3.5 Pro Realtime".
    return getattr(session, "stt_model", None) or ASSEMBLYAI_DEFAULT_MODEL


async def _assemblyai_audio_sender(ws, session: CallSession) -> None:
    """Forward Twilio's mu-law 8 kHz mono chunks verbatim.

    No transcoding — AssemblyAI's WS accepts pcm_mulaw@8000 when
    opened with the matching query params, and Twilio hands us exactly
    that. Each chunk becomes a JSON {"type": "data", "audio_data":
    "<b64>"} frame.
    """
    while True:
        try:
            chunk = await asyncio.wait_for(session.stt_audio_queue.get(), timeout=5.0)
        except asyncio.TimeoutError:
            if session.closed:
                return
            # No audio flowing (TTS playback muting). Don't spam AssemblyAI
            # with empty payloads — the WS will idle until we send the
            # next real chunk. Keepalive is implicit in the WS ping.
            continue
        if chunk is None:
            # End-of-session sentinel from session_runtime. Close the
            # upstream side cleanly so AssemblyAI flushes its final Turn.
            try:
                await ws.close()
            except Exception:
                pass
            return
        if not chunk:
            continue
        try:
            await ws.send(json.dumps({
                "type": "data",
                "audio_data": base64.b64encode(chunk).decode("ascii"),
            }))
        except Exception:
            return


async def run_realtime_stt(session: CallSession, on_transcript, on_failure) -> None:
    """Mirror deepgram_stt_realtime.run_realtime_stt signature so
    STT_Server.py's per-session dispatch can route either provider.
    """
    api_key = _resolve_api_key(session)
    if not api_key:
        return

    url = _build_url(session.preferred_language)
    model_id = _resolve_model(session)
    connect_kwargs = {"ping_interval": 20, "ping_timeout": 20}
    language = normalize_supported_language(session.preferred_language)

    attempt = 0
    while not session.closed:
        # On reconnect, flush stale buffered audio so AssemblyAI
        # processes current speech instead of backlog from before the
        # last turn-end.
        if attempt > 0:
            drained = 0
            while not session.stt_audio_queue.empty():
                try:
                    session.stt_audio_queue.get_nowait()
                    drained += 1
                except asyncio.QueueEmpty:
                    break
            if drained:
                log.info(
                    "[ASSEMBLYAI_STT] drained %d stale chunks on reconnect for %s",
                    drained, session.session_key,
                )

        sender_task: asyncio.Task | None = None
        received_any_result = False
        try:
            try:
                conn = websockets.connect(
                    url,
                    additional_headers={"Authorization": api_key},
                    **connect_kwargs,
                )
            except TypeError:
                conn = websockets.connect(
                    url,
                    extra_headers={"Authorization": api_key},
                    **connect_kwargs,
                )

            async with conn as ws:
                sender_task = asyncio.create_task(_assemblyai_audio_sender(ws, session))
                log.info(
                    "[ASSEMBLYAI_STT] WS connected for %s model=%s lang=%s",
                    session.session_key, model_id, language,
                )

                while not session.closed:
                    try:
                        raw = await ws.recv()
                    except ConnectionClosed:
                        if not received_any_result:
                            log.warning(
                                "[ASSEMBLYAI_STT] WS closed without transcripts in %s",
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
                    if mtype == "Begin":
                        # Session start info (id, expires_at). Nothing
                        # actionable for us; log at debug.
                        log.debug(
                            "[ASSEMBLYAI_STT] session begin %s: %s",
                            session.session_key, msg,
                        )
                        continue

                    if mtype == "Turn":
                        text = (msg.get("transcript") or "").strip()
                        end_of_turn = bool(msg.get("end_of_turn"))
                        if text:
                            received_any_result = True
                            attempt = 0
                            await on_transcript({
                                "text": text,
                                "language": language,
                                "is_final": end_of_turn,
                                # ponytail: AssemblyAI doesn't emit a
                                # separate speech_final vs is_final
                                # distinction — end_of_turn is the
                                # single "turn finished" signal. Map
                                # both flags to the same value so the
                                # TurnManager treats it like Deepgram's
                                # speech_final=true path.
                                "speech_final": end_of_turn,
                                "source": "assemblyai_realtime",
                            })
                        continue

                    if mtype == "Termination":
                        # Audio duration summary; natural close path.
                        log.info(
                            "[ASSEMBLYAI_STT] session terminated %s: audio_ms=%s",
                            session.session_key,
                            msg.get("audio_duration_ms"),
                        )
                        break

                    if mtype == "Error":
                        log.error(
                            "[ASSEMBLYAI_STT] error in %s: %s",
                            session.session_key,
                            msg.get("error") or msg,
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
                        "[ASSEMBLYAI_STT] dropped without results for %s, retrying",
                        session.session_key,
                    )
                    continue
                break
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "unknown")
            log.warning(
                "[ASSEMBLYAI_STT] handshake rejected in %s status=%s",
                session.session_key, status,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[ASSEMBLYAI_STT] error in %s", session.session_key)
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
            "[ASSEMBLYAI_STT] reconnecting %s in %sms (attempt %s/%s)",
            session.session_key, delay_ms, attempt, STT_RECONNECT_MAX_ATTEMPTS,
        )
        await asyncio.sleep(delay_ms / 1000.0)
