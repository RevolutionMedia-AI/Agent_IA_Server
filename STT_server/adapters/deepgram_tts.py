import asyncio
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from STT_server.config import DEEPGRAM_TTS_ENCODING, DEEPGRAM_TTS_SAMPLE_RATE
from STT_server.domain.language import get_tts_model, infer_supported_language_from_text
from STT_server.domain.session import CallSession
from STT_server.services._instrumentation import Stages
from STT_server.services.audio_frame_processor import AudioFrameProcessor
from STT_server.services.credentials_resolver import resolve_provider
from STT_server.services.thread_pool import to_thread as _to_thread
from STT_server.services.wait_signals import mark_stage as _mark_stage


log = logging.getLogger("stt_server")


async def stream_tts_segment(session: CallSession, text: str, generation: int, emit_item) -> tuple[float | None, float]:
    frame_proc = AudioFrameProcessor(emit_silence_tail=False)
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "deepgram")
    api_key = creds.get("api_key")
    if not api_key:
        raise RuntimeError("Deepgram no configurado. Sube tu key en Settings → API o en el campo inline de ModalAgents.")

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    tts_language = session.preferred_language if session.preferred_language else infer_supported_language_from_text(text, fallback="en")
    # ponytail: pass the provider so get_tts_model() returns the right
    # default (was hardcoded to ElevenLabs' voice id regardless of
    # provider; see 006_agent_runtime_params.sql commit message).
    model = get_tts_model(tts_language, provider="deepgram")
    params = urllib.parse.urlencode(
        {
            "model": model,
            "encoding": DEEPGRAM_TTS_ENCODING,
            "sample_rate": DEEPGRAM_TTS_SAMPLE_RATE,
            "container": "none",
        }
    )
    url = f"https://api.deepgram.com/v1/speak?{params}"
    payload = json.dumps({"text": text}).encode("utf-8")
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }

    def producer() -> None:
        nonlocal ttfb_ms

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break

                    if ttfb_ms is None:
                        ttfb_ms = (time.perf_counter() - started_at) * 1000
                        # ponytail: stamp TTS-first-byte the first time
                        # audio bytes are emitted. Cheap and idempotent.
                        _mark_stage(session, Stages.TTS_FIRST_BYTE)

                    for frame in frame_proc.feed(chunk):
                        loop.call_soon_threadsafe(
                            emit_item,
                            {"type": "audio", "generation": generation, "data": frame},
                        )
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            loop.call_soon_threadsafe(
                emit_item,
                {
                    "type": "error",
                    "generation": generation,
                    "message": f"Deepgram TTS error {exc.code}: {body}",
                },
            )
        except urllib.error.URLError as exc:
            loop.call_soon_threadsafe(
                emit_item,
                {
                    "type": "error",
                    "generation": generation,
                    "message": f"Deepgram TTS connection error: {exc}",
                },
            )
        finally:
            for frame in frame_proc.flush():
                loop.call_soon_threadsafe(
                    emit_item,
                    {"type": "audio", "generation": generation, "data": frame},
                )
            loop.call_soon_threadsafe(emit_item, {"type": "segment_end", "generation": generation})

    await _to_thread(producer)
    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms