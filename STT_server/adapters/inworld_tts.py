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

# ponytail: inworld-tts-2 was deprecated by Inworld. The docstring
# above already listed inworld-tts-1.5-mini as the intended default
# (low-latency, 15 langs); the constant was just never updated to
# match. Use 1.5-mini.
DEFAULT_MODEL_ID = "inworld-tts-1.5-mini"
DEFAULT_VOICE_ID = "Dennis"


def _summarize_mulaw_chunk(audio: bytes) -> dict:
    """Cheap shape check on a mu-law 8 kHz byte stream.

    - bytes: raw length (8 kHz mu-law = 1 byte/ms, so bytes/8 = ms)
    - peak: max absolute value of decoded samples (255 = saturation)
    - dc: sum of decoded samples over chunk size; non-zero DC =
      net positive/negative bias, often the cause of a constant
      hiss on the caller's line
    - zc: zero-crossings (sign flips). For real speech we expect
      hundreds per second; a near-zero count = digital noise / clip
    """
    if not audio:
        return {"bytes": 0, "peak": 0, "dc": 0, "zc": 0}
    # ponytail: G.711 mu-law decodes to signed 16-bit linear with
    # audioop.ulaw2lin. Doing it over thousands of bytes per chunk
    # would be slow in pure Python — audioop is C-backed and fast
    # enough at 8 kHz that we can afford it for the first-chunk
    # diagnostic.
    import audioop
    try:
        pcm = audioop.ulaw2lin(audio, 2)
    except Exception:
        return {"bytes": len(audio), "peak": 0, "dc": 0, "zc": 0}
    n = len(pcm) // 2
    if n == 0:
        return {"bytes": len(audio), "peak": 0, "dc": 0, "zc": 0}
    samples = pcm[:n * 2]
    # Quick stats via struct
    import struct
    fmt = "<" + "h" * n
    try:
        vals = struct.unpack(fmt, samples)
    except Exception:
        return {"bytes": len(audio), "peak": 0, "dc": 0, "zc": 0}
    peak = max(abs(v) for v in vals) if vals else 0
    dc = sum(vals)
    zc = sum(1 for a, b in zip(vals, vals[1:]) if (a >= 0) != (b >= 0))
    return {"bytes": len(audio), "peak": peak, "dc": dc, "zc": zc}


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
    # ponytail: inworld TTS needs a real voice name in voiceId
    # (e.g. "Aarav", "Dennis"), NOT a model id. The FE was setting
    # voice_id to the model name ("inworld-tts-2") which Inworld
    # rejects with 404 "Unknown voice". When the value we get
    # looks like a model id (starts with "inworld-"), fall back
    # to the default voice. The agent's actual model is still
    # sent in modelId — the user just has to set voice_id to a
    # real voice name in the FE (separate field from the model
    # dropdown).
    _configured_voice = getattr(session, "voice_id", None) or DEFAULT_VOICE_ID
    if _configured_voice.startswith("inworld-") and _configured_voice != DEFAULT_VOICE_ID:
        log.warning(
            "[INWORLD_TTS] session=%s voice_id=%r looks like a model "
            "id, not a voice name. Falling back to %r. The FE should "
            "send a real Inworld voice name in session.voice_id "
            "(e.g. Aarav, Dennis, Hank).",
            getattr(session, "session_key", "?"),
            _configured_voice, DEFAULT_VOICE_ID,
        )
        voice_id = DEFAULT_VOICE_ID
    else:
        voice_id = _configured_voice
    # ponytail: H1 from the call-flow audit. The dataclass has
    # `tts_model` (set in STT_Server.py:401 from the agent's
    # tts_model), not `model_id`. The previous getattr returned None
    # always, silently dropping the per-agent model selection and
    # falling back to the default. Read tts_model instead.
    model_id = getattr(session, "tts_model", None) or DEFAULT_MODEL_ID
    # ponytail: surface what the BE is actually sending to Inworld.
    # The user hit "Unknown voice: inworld-tts-2 not found!" — the
    # voice_id is being populated with a model name instead of a
    # real Inworld voice name. Logged here so the next time this
    # happens, the operator sees the exact values without having
    # to add a breakpoint.
    log.info(
        "[INWORLD_TTS] session=%s voice_id=%r model_id=%r key_present=%s",
        getattr(session, "session_key", "?"),
        voice_id, model_id, bool(api_key),
    )

    # ponytail: per-agent speed override (006_agent_runtime_params.sql).
    # Inworld accepts `speakingRate` in audioConfig (0.5..2.0). The schema
    # CHECK constraint in the migration mirrors that range, but the
    # adapter clamps as a second line of defense in case a future code
    # path writes a bad value (e.g. defaults reset).
    _speed = getattr(session, "tts_speed", None)
    speaking_rate = max(0.5, min(2.0, _speed if _speed is not None else 1.0))
    body = json.dumps({
        "text": text,
        "voiceId": voice_id,
        "modelId": model_id,
        "audioConfig": {
            "audioEncoding": "MULAW",
            "sampleRateHertz": 8000,
            "speakingRate": speaking_rate,
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
                    # ponytail: one-shot diagnostic on the FIRST chunk
                    # of each turn. Verifies the format Inworld actually
                    # returns against what we requested (mu-law 8 kHz).
                    # A wrong sample rate (e.g. 16 kHz delivered as 8 kHz
                    # mu-law) produces the "constant hiss" the user
                    # reported — half the audible bandwidth, half the
                    # playback speed, plus aliased artifacts.
                    audio_stats = _summarize_mulaw_chunk(audio)
                    if not getattr(stream_tts_segment, "_logged_format", False):
                        stream_tts_segment._logged_format = True
                        log.warning(
                            "[INWORLD_TTS] first chunk session=%s gen=%s "
                            "bytes=%d duration_ms=%.1f peak_amplitude=%d "
                            "dc_offset_estimate=%d zero_crossings=%d",
                            getattr(session, "session_key", "?"),
                            generation,
                            audio_stats["bytes"],
                            audio_stats["bytes"] / 8.0,  # 8 kHz mu-law = 1 byte/ms
                            audio_stats["peak"],
                            audio_stats["dc"],
                            audio_stats["zc"],
                        )
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
