"""Pre-generate static greeting WAV files at startup.

Why: the operator reported "el saludo llega después de 28 Segundos
de iniciada la llamada". The 28 s breakdown (Twilio WS handshake +
Twilio-to-phone path + TTS TTFB) is mostly outside our control, but
the most expensive part — opening the WebSocket before Twilio can
play anything — IS in our control. The fix: pre-generate the
default greeting audio at startup and serve it via TwiML ``<Play>``
in the /voice response. Twilio plays the file as soon as the call
connects (no backend handshake), so the caller hears audio within
~500 ms — long before the WS opens and the in-band TTS greeting
would otherwise start.

Format: 8 kHz mono mu-law (G.711) WAV. Twilio's ``<Play>`` accepts WAV
with mu-law encoding. The 44-byte RIFF header + raw mu-law samples
is the simplest representation Twilio can stream.

Triggered:
  - At module import (lifespan startup). Best-effort: if the runtime
    has no TTS provider configured (or the call raises), we log and
    continue — the in-band greeting still fires as a fallback.
  - On the first /voice request for each language (lazy generation
    in case the TTS provider was unavailable at boot).
"""
from __future__ import annotations

import asyncio
import logging
import struct
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("stt_server.greeting")


# ponytail: 8 kHz mono mu-law (G.711). 44-byte RIFF/WAVE header that
# matches every WAV decoder (Twilio, ffmpeg, sox, etc.). `data` chunk
# size is filled in at write time.
_WAV_HEADER = struct.pack(
    "<4sI4s4sIHHIIHH4sI",
    b"RIFF",
    0,  # placeholder: total file size - 8, patched at write
    b"WAVE",
    b"fmt ",
    16,  # fmt chunk size
    7,   # format code: 7 = mu-law
    1,   # channels
    8000,  # sample rate
    8000,  # byte rate (8000 samples/sec * 1 byte/sample)
    1,   # block align
    8,   # bits per sample
    b"data",
    0,  # placeholder: data chunk size, patched at write
)


def _wrap_mulaw_in_wav(mulaw_bytes: bytes) -> bytes:
    """Wrap raw mu-law bytes in a 44-byte RIFF/WAVE header.

    Returns a bytes buffer that can be served as a WAV file via
    HTTP. The header's two size fields are patched in before
    return.
    """
    data_size = len(mulaw_bytes)
    riff_size = 36 + data_size  # 44-byte header total = 36 bytes preamble + data
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        riff_size,
        b"WAVE",
        b"fmt ",
        16,
        7,
        1,
        8000,
        8000,
        1,
        8,
        b"data",
        data_size,
    )
    return header + mulaw_bytes


def static_greeting_path(static_dir: Path, lang: str) -> Path:
    """Where the pre-generated greeting WAV lives for a given language."""
    return static_dir / f"greeting_{lang}.wav"


async def _generate_greeting_wav(
    text: str,
    voice_id: str,
    lang: str,
    out_path: Path,
    api_key: str,
    tts_provider: str,
    timeout_sec: float = 30.0,
) -> bool:
    """Generate one greeting WAV file via the configured TTS provider.

    Returns True on success, False if the TTS provider was unreachable
    or raised. Never raises — the caller treats False as "file absent,
    fall back to in-band greeting".
    """
    if not api_key:
        log.info(
            "[greeting] skip pre-generation for lang=%s: no TTS API key",
            lang,
        )
        return False

    from STT_server.adapters.tts_dispatcher import stream_tts_segment

    # Synthetic session — only the fields the TTS adapters actually
    # read. tts_provider / voice_id / language flow through the
    # dispatch priority chain.
    from STT_server.domain.session import CallSession

    session = CallSession(session_key=f"warmup-{lang}")
    session.tts_provider = tts_provider
    session.voice_id = voice_id
    session.preferred_language = lang
    session.user_id = None  # platform-owned

    chunks: list[bytes] = []
    first_chunk_at: float | None = None
    started_at = time.perf_counter()

    def _on_item(item: dict) -> None:
        nonlocal first_chunk_at
        if item.get("type") == "audio":
            data = item.get("data") or b""
            if data:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                chunks.append(data)

    try:
        await asyncio.wait_for(
            stream_tts_segment(session, text, 0, _on_item),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        log.warning(
            "[greeting] TTS timeout (%.1fs) for lang=%s; skipping pre-gen", timeout_sec, lang
        )
        return False
    except Exception as exc:
        log.warning(
            "[greeting] TTS failed for lang=%s (%s); skipping pre-gen", lang, exc
        )
        return False

    if not chunks:
        log.warning("[greeting] no audio chunks for lang=%s; skipping pre-gen", lang)
        return False

    mulaw = b"".join(chunks)
    wav = _wrap_mulaw_in_wav(mulaw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(wav)
    log.info(
        "[greeting] pre-generated %s (%.1f KB mulaw, %.1fs TTFB, %.1fs total) "
        "for lang=%s voice=%s",
        out_path,
        len(mulaw) / 1024,
        (first_chunk_at or started_at) - started_at,
        time.perf_counter() - started_at,
        lang,
        voice_id,
    )
    return True


def pregenerate_greeting_at_startup(
    static_dir: Path,
    texts: dict[str, str],
    voice_ids: dict[str, str],
    api_key_resolver,
    tts_provider: str,
) -> None:
    """Spawn background tasks to pre-generate the static greeting WAVs.

    Called from the FastAPI lifespan at boot. Best-effort: each
    language runs as its own task; if one fails, the others still try.
    The function returns immediately; tasks run in the background and
    the /voice handler will block briefly waiting for the file (with a
    timeout) if a call comes in before pre-generation finishes.

    * ``texts[lang]`` — the greeting text per language.
    * ``voice_ids[lang]`` — the voice id per language.
    * ``api_key_resolver`` — callable (no args) returning the resolved
      TTS API key, or '' if unavailable.
    """
    api_key = api_key_resolver()
    if not api_key:
        log.info("[greeting] no TTS API key at boot; skipping pre-generation")
        return

    async def _run_all() -> None:
        tasks = []
        for lang, text in texts.items():
            voice_id = voice_ids.get(lang, "")
            out_path = static_greeting_path(static_dir, lang)
            tasks.append(
                asyncio.create_task(
                    _generate_greeting_wav(
                        text=text,
                        voice_id=voice_id,
                        lang=lang,
                        out_path=out_path,
                        api_key=api_key,
                        tts_provider=tts_provider,
                    ),
                    name=f"prewarm-greeting-{lang}",
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ok = sum(1 for r in results if r is True)
        log.info("[greeting] pre-generation complete: %d/%d languages ready", ok, len(results))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not in an async context — fire-and-forget on the default loop.
        # The startup path runs inside FastAPI lifespan which IS async,
        # so this branch is only hit by tests that import the module.
        log.debug("[greeting] no running loop; skipping background pre-gen")
        return
    loop.create_task(_run_all(), name="prewarm-greeting-all")


# ── Lazy fallback (first call wins) ───────────────────────────────────────────────


_GREETING_GENERATION_LOCKS: dict[str, asyncio.Lock] = {}


async def ensure_greeting_available(
    lang: str,
    static_dir: Path,
    text: str,
    voice_id: str,
    api_key: str,
    tts_provider: str,
    timeout_sec: float = 8.0,
) -> bool:
    """Block until the static greeting WAV for ``lang`` exists on disk,
    generating it on-the-fly if pre-generation didn't run.

    Returns True if the file is ready (either pre-generated or
    generated now), False if generation timed out / failed. Called
    from the /voice handler so the first inbound call pays the
    pre-generation latency (≤ timeout_sec) instead of waiting for
    Twilio's WS handshake.
    """
    path = static_greeting_path(static_dir, lang)
    if path.exists() and path.stat().st_size > 44:
        return True

    lock = _GREETING_GENERATION_LOCKS.setdefault(lang, asyncio.Lock())
    async with lock:
        if path.exists() and path.stat().st_size > 44:
            return True
        try:
            return await asyncio.wait_for(
                _generate_greeting_wav(
                    text=text,
                    voice_id=voice_id,
                    lang=lang,
                    out_path=path,
                    api_key=api_key,
                    tts_provider=tts_provider,
                ),
                timeout=timeout_sec,
            )
        except asyncio.TimeoutError:
            return False


if __name__ == "__main__":
    # Smoke: verify the WAV header is well-formed for empty + non-empty
    # mu-law payloads. Twilio's <Play> will refuse to serve a file with a
    # bad header — the 44-byte preamble is non-negotiable.
    empty = _wrap_mulaw_in_wav(b"")
    assert empty[:4] == b"RIFF"
    assert empty[8:12] == b"WAVE"
    assert empty[12:16] == b"fmt "
    assert empty[20:22] == b"\x07\x00"  # format code 7 = mu-law, little-endian
    assert empty[36:40] == b"data"
    # RIFF size = total_size - 8 = 36 + data_size. With 0 data, RIFF size = 36.
    assert struct.unpack("<I", empty[4:8])[0] == 36
    # data size = 0
    assert struct.unpack("<I", empty[40:44])[0] == 0
    print("empty WAV: OK")

    nonempty = _wrap_mulaw_in_wav(b"\xff\x00\x7f" * 100)
    assert struct.unpack("<I", nonempty[4:8])[0] == 36 + 300
    assert struct.unpack("<I", nonempty[40:44])[0] == 300
    assert len(nonempty) == 44 + 300
    print("nonempty WAV: OK")
    print("greeting: OK")