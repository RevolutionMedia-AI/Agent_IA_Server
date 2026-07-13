"""Inworld TTS adapter.

Streams TTS via Inworld's /tts/v1/voice:stream endpoint and emits
mu-law 8 kHz audio chunks suitable for the live call pipeline.

Auth uses Inworld's Basic scheme: the api_key the user pastes in
the modal IS already a Base64-encoded credential, so we send it
verbatim in the Authorization header. See
https://docs.inworld.ai/api-reference/introduction for the auth
contract and https://docs.inworld.ai/tts/tts for the streaming
endpoint contract.

We ask for audioEncoding=MULAW + sampleRateHertz=8000 so the
bytes Inworld returns are already in the format Twilio consumes.
No PCM-to-mu-law step, no resample, no WAV wrap. The bytes go
straight into the session's audio emit and to Twilio.

Models: inworld-tts-2 (flagship, 200ms), inworld-tts-1.5-max
(15 langs, 200ms), inworld-tts-1.5-mini (15 langs, 120ms).
Default to 1.5-mini for low latency.
"""
import asyncio
import base64
import json
import logging
import time
import urllib.error
import urllib.request

from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider

log = logging.getLogger("stt_server")

DEFAULT_MODEL_ID = "inworld-tts-2"
DEFAULT_VOICE_ID = "Dennis"


def _resolve_api_key(session: CallSession) -> str:
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "inworld") if user_id else {}
    return (creds.get("api_key") or "").strip()


async def stream_tts_segment(
    session: CallSession,
    text: str,
    generation: int,
    emit_item,
) -> tuple[float | None, float]:
    api_key = _resolve_api_key(session)
    if not api_key:
        raise RuntimeError("Inworld API key not configured.")

    started = time.perf_counter()
    voice_id = getattr(session, "voice_id", None) or DEFAULT_VOICE_ID
    model_id = getattr(session, "model_id", None) or DEFAULT_MODEL_ID

    body = json.dumps({
        "text": text,
        "voiceId": voice_id,
        "modelId": model_id,
        "audioConfig": {
            "audioEncoding": "MULAW",
            "sampleRateHertz": 8000,
        },
    }).encode("utf-8")

    url = "https://api.inworld.ai/tts/v1/voice:stream"
    headers = {
        "Authorization": f"Basic {api_key}",
        "Content-Type": "application/json",
    }

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None

    def producer() -> None:
        nonlocal ttfb_ms
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                for raw_line in resp:
                    line = raw_line.strip() if isinstance(raw_line, bytes) else raw_line.encode().strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        msg = obj["error"].get("message", "Inworld stream error")
                        loop.call_soon_threadsafe(
                            emit_item,
                            {"type": "error", "generation": generation, "message": f"Inworld TTS: {msg}"},
                        )
                        return
                    result = obj.get("result") or {}
                    b64_audio = result.get("audioContent")
                    if not b64_audio:
                        continue
                    try:
                        audio = base64.b64decode(b64_audio, validate=False)
                    except Exception as exc:
                        log.warning("[INWORLD_TTS] bad base64 chunk: %s", exc)
                        continue
                    if not audio:
                        continue
                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started) * 1000
                    loop.call_soon_threadsafe(
                        emit_item,
                        {"type": "audio", "generation": generation, "data": audio},
                    )
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            loop.call_soon_threadsafe(
                emit_item,
                {"type": "error", "generation": generation, "message": f"Inworld TTS error {exc.code}: {err_body}"},
            )
        except urllib.error.URLError as exc:
            loop.call_soon_threadsafe(
                emit_item,
                {"type": "error", "generation": generation, "message": f"Inworld TTS connection error: {exc}"},
            )
        finally:
            loop.call_soon_threadsafe(
                emit_item,
                {"type": "segment_end", "generation": generation},
            )

    await asyncio.to_thread(producer)
    total_ms = (time.perf_counter() - started) * 1000
    return ttfb_ms, total_ms


def fetch_preview(
    text: str,
    voice_id: str,
    model_id: str,
    api_key: str,
) -> bytes:
    """One-shot TTS preview using the non-streaming endpoint.

    Returns raw mu-law 8 kHz bytes - no WAV header (the caller
    wraps them via tts_preview._wrap_mulaw_as_wav_pcm16 so the FE
    <audio> element can decode them).
    """
    body = json.dumps({
        "text": text,
        "voiceId": voice_id or DEFAULT_VOICE_ID,
        "modelId": model_id or DEFAULT_MODEL_ID,
        "audioConfig": {"audioEncoding": "MULAW", "sampleRateHertz": 8000},
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.inworld.ai/tts/v1/voice",
        data=body,
        headers={"Authorization": f"Basic {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # ponytail: surface Inworld's actual error body so the FE log
        # shows "voice X not found" instead of an opaque "HTTP 400".
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            pass
        raise RuntimeError(
            f"Inworld TTS error {exc.code}: {err_body or exc.reason}"
        ) from exc
    b64_audio = (payload.get("audioContent") or "").strip()
    if not b64_audio:
        # Some error responses come back 200 with an `error` field;
        # surface that too.
        err = payload.get("error") or {}
        if err:
            raise RuntimeError(f"Inworld TTS: {err.get('message') or err}")
        return b""
    try:
        return base64.b64decode(b64_audio, validate=False)
    except Exception as exc:
        raise RuntimeError(f"Inworld: invalid audio payload ({exc})") from exc
