import asyncio
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from STT_server.config import DEEPGRAM_API_KEY, DEEPGRAM_TTS_ENCODING, DEEPGRAM_TTS_SAMPLE_RATE
from STT_server.domain.language import get_tts_model, infer_supported_language_from_text
from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")


async def stream_tts_segment(session: CallSession, text: str, generation: int, emit_item) -> tuple[float | None, float]:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("Deepgram no configurado")

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    tts_language = infer_supported_language_from_text(text, fallback=session.preferred_language)
    model = get_tts_model(tts_language)
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
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
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

                    loop.call_soon_threadsafe(
                        emit_item,
                        {"type": "audio", "generation": generation, "data": chunk},
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
            loop.call_soon_threadsafe(emit_item, {"type": "segment_end", "generation": generation})

    await asyncio.to_thread(producer)
    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms