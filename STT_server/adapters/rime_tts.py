import asyncio
import base64
import json
import logging
import time
import urllib.error
import urllib.request

from STT_server.config import RIME_API_KEY, RIME_TTS_MODEL_ID
from STT_server.domain.language import get_tts_model, infer_supported_language_from_text, normalize_supported_language
from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")

RIME_TTS_URL = "https://users.rime.ai/v1/rime-tts"


async def stream_tts_segment(session: CallSession, text: str, generation: int, emit_item) -> tuple[float | None, float]:
    if not RIME_API_KEY:
        raise RuntimeError("RIME_API_KEY no configurada")

    loop = asyncio.get_running_loop()
    ttfb_ms: float | None = None
    started_at = time.perf_counter()
    tts_language = session.preferred_language if session.preferred_language else infer_supported_language_from_text(text, fallback="en")
    speaker = get_tts_model(tts_language)

    lang_code = "eng" if normalize_supported_language(tts_language) == "en" else "spa"

    payload_dict = {
        "speaker": speaker,
        "text": text,
        "modelId": RIME_TTS_MODEL_ID,
        "lang": lang_code,
        "audioFormat": "mulaw",
        "samplingRate": 8000,
    }
    log.info("Rime TTS request: speaker=%s model=%s lang=%s text_len=%d", speaker, RIME_TTS_MODEL_ID, lang_code, len(text))
    payload = json.dumps(payload_dict).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {RIME_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    def producer() -> None:
        nonlocal ttfb_ms

        req = urllib.request.Request(RIME_TTS_URL, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read()

                if ttfb_ms is None:
                    ttfb_ms = (time.perf_counter() - started_at) * 1000

                data = json.loads(raw)
                audio_b64 = data.get("audioContent", "")
                audio_bytes = base64.b64decode(audio_b64)

                for i in range(0, len(audio_bytes), 4096):
                    chunk = audio_bytes[i : i + 4096]
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
            log.error("Rime TTS HTTP %d — headers: %s — body: %s", exc.code, dict(exc.headers), body)
            loop.call_soon_threadsafe(
                emit_item,
                {
                    "type": "error",
                    "generation": generation,
                    "message": f"Rime TTS error {exc.code}: {body}",
                },
            )
        except urllib.error.URLError as exc:
            loop.call_soon_threadsafe(
                emit_item,
                {
                    "type": "error",
                    "generation": generation,
                    "message": f"Rime TTS connection error: {exc}",
                },
            )
        finally:
            loop.call_soon_threadsafe(emit_item, {"type": "segment_end", "generation": generation})

    await asyncio.to_thread(producer)
    total_ms = (time.perf_counter() - started_at) * 1000
    return ttfb_ms, total_ms
