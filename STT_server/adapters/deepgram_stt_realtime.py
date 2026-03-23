import asyncio
import json
import logging
import urllib.parse

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from STT_server.config import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_DETECT_LANGUAGE,
    DEEPGRAM_STT_ENDPOINTING_MS,
    DEEPGRAM_STT_LANGUAGE_HINT,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_STT_PUNCTUATE,
    DEEPGRAM_STT_SMART_FORMAT,
    STT_RECONNECT_BASE_DELAY_MS,
    STT_RECONNECT_MAX_ATTEMPTS,
    STT_RECONNECT_MAX_DELAY_MS,
    TWILIO_CHANNELS,
    TWILIO_SR,
)
from STT_server.domain.language import infer_supported_language_from_text, normalize_deepgram_language, normalize_supported_language
from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")


def extract_deepgram_stream_result(payload: dict, fallback_language: str | None = None) -> tuple[str, str, bool, bool]:
    fallback = normalize_supported_language(fallback_language)
    channel = payload.get("channel") or {}
    alternatives = channel.get("alternatives") or []
    if not alternatives:
        return "", fallback, False, False

    alternative = alternatives[0] or {}
    transcript = (alternative.get("transcript") or "").strip()
    detected_language = normalize_deepgram_language(
        alternative.get("detected_language") or channel.get("detected_language")
    )

    if not detected_language:
        languages = alternative.get("languages") or channel.get("languages") or []
        if languages:
            detected_language = normalize_deepgram_language(languages[0])

    if not detected_language and transcript:
        detected_language = infer_supported_language_from_text(transcript, fallback=fallback)

    return transcript, detected_language or fallback, bool(payload.get("is_final")), bool(payload.get("speech_final"))


def build_deepgram_realtime_url(language_hint: str | None = None) -> str:
    hint = normalize_deepgram_language(language_hint or DEEPGRAM_STT_LANGUAGE_HINT)
    params = {
        "model": DEEPGRAM_STT_MODEL,
        "encoding": "mulaw",
        "sample_rate": str(TWILIO_SR),
        "channels": str(TWILIO_CHANNELS),
        "interim_results": "true",
        "punctuate": str(DEEPGRAM_STT_PUNCTUATE).lower(),
        "smart_format": str(DEEPGRAM_STT_SMART_FORMAT).lower(),
        "endpointing": str(DEEPGRAM_STT_ENDPOINTING_MS),
        "vad_events": "true",
    }

    if hint:
        params["language"] = hint
    elif DEEPGRAM_STT_DETECT_LANGUAGE:
        params["detect_language"] = "true"

    return f"wss://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"


def build_deepgram_realtime_candidates(language_hint: str | None = None) -> list[dict[str, str]]:
    hint = normalize_deepgram_language(language_hint or DEEPGRAM_STT_LANGUAGE_HINT)
    candidate_models: list[str] = []
    for model in (DEEPGRAM_STT_MODEL, "nova-3", "nova-2", "phonecall"):
        if model and model not in candidate_models:
            candidate_models.append(model)

    base_params = {
        "encoding": "mulaw",
        "sample_rate": str(TWILIO_SR),
        "channels": str(TWILIO_CHANNELS),
        "interim_results": "true",
        "punctuate": str(DEEPGRAM_STT_PUNCTUATE).lower(),
        "smart_format": str(DEEPGRAM_STT_SMART_FORMAT).lower(),
        "endpointing": str(DEEPGRAM_STT_ENDPOINTING_MS),
    }

    candidates: list[dict[str, str]] = []

    for model in candidate_models:
        params = {**base_params, "model": model, "vad_events": "true"}
        if hint:
            params["language"] = hint
        elif DEEPGRAM_STT_DETECT_LANGUAGE:
            params["detect_language"] = "true"
        candidates.append(params)

        no_vad_params = {key: value for key, value in params.items() if key != "vad_events"}
        candidates.append(no_vad_params)

        neutral_params = {key: value for key, value in no_vad_params.items() if key not in {"language", "detect_language"}}
        candidates.append(neutral_params)

    unique_candidates: list[dict[str, str]] = []
    seen = set()
    for params in candidates:
        signature = tuple(sorted(params.items()))
        if signature in seen:
            continue
        seen.add(signature)
        unique_candidates.append(params)

    return unique_candidates


def build_deepgram_realtime_url_from_params(params: dict[str, str]) -> str:
    return f"wss://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"


async def deepgram_audio_sender(dg_ws, session: CallSession) -> None:
    while True:
        chunk = await session.stt_audio_queue.get()
        if chunk is None:
            await dg_ws.send(json.dumps({"type": "Finalize"}))
            return
        if chunk:
            await dg_ws.send(chunk)


async def run_realtime_stt(session: CallSession, on_transcript, on_failure) -> None:
    if not DEEPGRAM_API_KEY:
        return

    connect_kwargs = {"ping_interval": 20, "ping_timeout": 20}
    attempt = 0

    while not session.closed:
        sender_task: asyncio.Task | None = None
        last_invalid_status: InvalidStatus | None = None
        try:
            for params in build_deepgram_realtime_candidates(session.preferred_language):
                url = build_deepgram_realtime_url_from_params(params)
                try:
                    try:
                        realtime_connection = websockets.connect(
                            url,
                            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                            **connect_kwargs,
                        )
                    except TypeError:
                        realtime_connection = websockets.connect(
                            url,
                            extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                            **connect_kwargs,
                        )

                    async with realtime_connection as dg_ws:
                        attempt = 0
                        sender_task = asyncio.create_task(deepgram_audio_sender(dg_ws, session))
                        log.info(
                            "Deepgram realtime conectado en %s con params=%s",
                            session.session_key,
                            params,
                        )

                        while not session.closed:
                            try:
                                raw_message = await dg_ws.recv()
                            except ConnectionClosed:
                                break

                            if isinstance(raw_message, bytes):
                                continue

                            payload = json.loads(raw_message)
                            message_type = payload.get("type")

                            if message_type == "Results":
                                transcript, language, is_final, speech_final = extract_deepgram_stream_result(
                                    payload,
                                    fallback_language=session.preferred_language,
                                )
                                if transcript:
                                    await on_transcript(
                                        {
                                            "text": transcript,
                                            "language": language,
                                            "is_final": is_final or speech_final,
                                            "speech_final": speech_final,
                                        }
                                    )
                                continue

                            if message_type == "UtteranceEnd" and session.current_transcript:
                                await on_transcript(
                                    {
                                        "text": session.current_transcript,
                                        "language": session.preferred_language,
                                        "is_final": True,
                                        "speech_final": True,
                                    }
                                )
                                continue

                            if message_type == "Error":
                                log.error("Deepgram realtime error en %s: %s", session.session_key, payload)
                                break

                        if session.closed:
                            return

                        break
                except InvalidStatus as exc:
                    last_invalid_status = exc
                    log.warning(
                        "Handshake Deepgram rechazado en %s status=%s params=%s",
                        session.session_key,
                        getattr(getattr(exc, "response", None), "status_code", "unknown"),
                        params,
                    )
                    continue
            else:
                if last_invalid_status is not None:
                    raise last_invalid_status
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Error en run_realtime_stt para %s", session.session_key)
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

        delay_ms = min(STT_RECONNECT_BASE_DELAY_MS * (2 ** (attempt - 1)), STT_RECONNECT_MAX_DELAY_MS)
        log.warning(
            "Reconectando Deepgram realtime para %s en %sms (intento %s/%s)",
            session.session_key,
            delay_ms,
            attempt,
            STT_RECONNECT_MAX_ATTEMPTS,
        )
        await asyncio.sleep(delay_ms / 1000.0)