import asyncio
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import wave

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    DEEPGRAM_API_KEY,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_STT_PUNCTUATE,
    DEEPGRAM_STT_SMART_FORMAT,
    STT_TIMEOUT_SEC,
    TWILIO_CHANNELS,
    TWILIO_SR,
)
from STT_server.domain.language import (
    infer_supported_language_from_text,
    normalize_deepgram_language,
    normalize_supported_language,
)


def pcm16_to_wav_bytes(pcm16_audio: bytes, sample_rate: int = TWILIO_SR, channels: int = TWILIO_CHANNELS) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_audio)
    return buffer.getvalue()


def extract_deepgram_transcript(payload: dict, fallback_language: str | None = None) -> tuple[list[str], str]:
    fallback = normalize_supported_language(fallback_language)
    results = payload.get("results") or {}
    channels = results.get("channels") or []
    if not channels:
        return [], fallback

    channel = channels[0] or {}
    alternatives = channel.get("alternatives") or []
    if not alternatives:
        return [], fallback

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

    return ([transcript] if transcript else []), detected_language or fallback


def transcribe_sync(pcm16_audio: bytes, language_hint: str | None = None) -> tuple[list[str], str]:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("Deepgram no configurado. Define DEEPGRAM_API_KEY.")

    fallback_language = normalize_supported_language(language_hint or DEFAULT_CALL_LANGUAGE)
    if not pcm16_audio:
        return [], fallback_language

    params = {
        "model": DEEPGRAM_STT_MODEL,
        "punctuate": str(DEEPGRAM_STT_PUNCTUATE).lower(),
        "smart_format": str(DEEPGRAM_STT_SMART_FORMAT).lower(),
    }

    hint = normalize_deepgram_language(language_hint)
    if hint:
        params["language"] = hint
    else:
        params["language"] = "multi"

    url = f"https://api.deepgram.com/v1/listen?{urllib.parse.urlencode(params)}"
    wav_audio = pcm16_to_wav_bytes(pcm16_audio)
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
        "Accept": "application/json",
    }
    timeout = STT_TIMEOUT_SEC if STT_TIMEOUT_SEC > 0 else 45
    request = urllib.request.Request(url, data=wav_audio, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Deepgram STT error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Deepgram STT connection error: {exc}") from exc

    return extract_deepgram_transcript(payload, fallback_language=fallback_language)


async def transcribe_block(pcm16_audio: bytes, language_hint: str | None = None) -> tuple[list[str], str]:
    return await asyncio.to_thread(transcribe_sync, pcm16_audio, language_hint)
