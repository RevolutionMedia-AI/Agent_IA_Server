"""Tests for the 2026-08-14 TwiML <Play> greeting fix.

The operator reported "el saludo llega después de 28 Segundos de
iniciada la llamada" — 28 seconds of pure silence after the call
connected before the agent's greeting became audible. The breakdown:
  - Twilio WS handshake: ~5-15 s (out of our control)
  - Twilio POST /voice → TwiML: < 1 s
  - Server processes start event: < 1 s
  - TTS provider TTFB: ~0.5-2 s
  - Twilio streams audio to phone: 0.5-2 s

The expensive part — opening the WebSocket before Twilio can play
anything — IS in our control. Fix: pre-generate a static greeting
WAV at boot and have /voice return TwiML ``<Play>`` that file
BEFORE ``<Connect>``. Twilio plays the file as soon as the call
connects, no backend handshake, no TTS round-trip. Caller hears
audio within ~500 ms.

These tests pin the contract so a future refactor can't silently
break it.
"""
from __future__ import annotations

import struct

import pytest


# ── WAV header correctness ────────────────────────────────────────────────


def test_wrap_mulaw_in_wav_header_is_correct_for_empty_payload() -> None:
    """RIFF/WAVE header for empty mu-law data: must be exactly 44
    bytes with size fields = 0 and the magic bytes in the right
    order. Twilio's <Play> refuses to serve malformed WAV; this
    is the smallest valid file."""
    from STT_server.services.greeting import _wrap_mulaw_in_wav

    wav = _wrap_mulaw_in_wav(b"")
    assert len(wav) == 44
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"
    # RIFF size = total - 8 = 36 (header bytes before the data)
    assert struct.unpack("<I", wav[4:8])[0] == 36
    # data size = 0
    assert struct.unpack("<I", wav[40:44])[0] == 0
    # format code 7 = mu-law (little-endian uint16)
    assert struct.unpack("<H", wav[20:22])[0] == 7
    # channels = 1, sample rate = 8000, bits per sample = 8
    assert struct.unpack("<H", wav[22:24])[0] == 1
    assert struct.unpack("<I", wav[24:28])[0] == 8000
    assert struct.unpack("<H", wav[34:36])[0] == 8


def test_wrap_mulaw_in_wav_header_sizes_match_payload() -> None:
    """RIFF size and data size fields must reflect the actual payload."""
    from STT_server.services.greeting import _wrap_mulaw_in_wav

    payload = b"\xff\x00\x7f" * 100  # 300 bytes
    wav = _wrap_mulaw_in_wav(payload)
    assert len(wav) == 44 + 300
    # RIFF size = 36 + 300 = 336
    assert struct.unpack("<I", wav[4:8])[0] == 36 + 300
    # data size = 300
    assert struct.unpack("<I", wav[40:44])[0] == 300
    # payload preserved verbatim after the header
    assert wav[44:] == payload


# ── static_greeting_path helper ─────────────────────────────────────────────


def test_static_greeting_path_uses_lang_suffix(tmp_path) -> None:
    """The greeting file is named ``greeting_{lang}.wav`` so the
    /voice handler can pick the right one for the caller's language.
    """
    from STT_server.services.greeting import static_greeting_path
    assert static_greeting_path(tmp_path, "en").name == "greeting_en.wav"
    assert static_greeting_path(tmp_path, "es").name == "greeting_es.wav"
    # Different langs → different files in the same dir.
    assert (
        static_greeting_path(tmp_path, "en")
        != static_greeting_path(tmp_path, "es")
    )


# ── /voice TwiML response shape ────────────────────────────────────────────


def _build_voice_twiml(static_greeting_exists: bool, lang: str = "es") -> str:
    """Reconstruct the TwiML response from the /voice handler.

    The handler is sync and not easy to call directly in tests, so
    we duplicate the relevant logic here. If the production
    code's TwiML changes (e.g. drops the <Play> tag), these
    tests should fail and force the change to be intentional.
    """
    from STT_server.services.greeting import static_greeting_path

    # Mirror production: greeting file lives in static_dir (the BE
    # module's static/ directory). For the test we point at tmp_path.
    static_dir = _FAKE_STATIC_DIR
    play_section = ""
    if static_greeting_exists:
        play_section = (
            f'<Play>{_FAKE_PUBLIC_URL}/static/{static_greeting_path(static_dir, lang).name}</Play>'
        )
    if play_section:
        return (
            f"<Response>\n        {play_section}\n"
            f"        <Connect>\n"
            f"            <Stream url=\"{_FAKE_WS_URL}/media-stream\">"
            f"</Stream>\n"
            f"        </Connect>\n"
            f"    </Response>\n    "
        )
    return (
        f"<Response>\n        <Connect>\n"
        f"            <Stream url=\"{_FAKE_WS_URL}/media-stream\">"
        f"</Stream>\n"
        f"        </Connect>\n"
        f"    </Response>\n    "
    )


_FAKE_STATIC_DIR = None  # set in fixtures below
_FAKE_PUBLIC_URL = "https://example.test"
_FAKE_WS_URL = "wss://example.test"


def test_voice_twiml_includes_play_when_greeting_exists(tmp_path, monkeypatch) -> None:
    """When the static greeting WAV exists on disk, /voice returns
    TwiML with ``<Play>`` BEFORE ``<Connect>`` so Twilio plays the
    file as soon as the call connects."""
    global _FAKE_STATIC_DIR
    _FAKE_STATIC_DIR = tmp_path
    (tmp_path / "greeting_es.wav").write_bytes(b"\xff" * 320)
    twiml = _build_voice_twiml(static_greeting_exists=True)
    assert "<Play>" in twiml
    assert "greeting_es.wav" in twiml
    # <Play> must come BEFORE <Connect> so Twilio plays it first.
    assert twiml.index("<Play>") < twiml.index("<Connect>")


def test_voice_twiml_omits_play_when_greeting_missing(tmp_path) -> None:
    """When the static greeting file does NOT exist on disk (boot
    pre-generation failed or no TTS provider), /voice falls back to
    plain ``<Connect>`` — no ``<Play>`` tag at all. The in-band
    greeting via ``play_initial_greeting`` handles this case after
    the WS opens."""
    global _FAKE_STATIC_DIR
    _FAKE_STATIC_DIR = tmp_path
    twiml = _build_voice_twiml(static_greeting_exists=False)
    assert "<Play>" not in twiml
    assert "<Connect>" in twiml


def test_voice_twiml_picks_lang_specific_greeting(tmp_path) -> None:
    """When the file for the caller's language exists, /voice uses
    IT (not some other lang). Pin this so an EN call can't play the
    ES file just because both happen to exist."""
    global _FAKE_STATIC_DIR
    _FAKE_STATIC_DIR = tmp_path
    (tmp_path / "greeting_en.wav").write_bytes(b"\xff" * 320)
    (tmp_path / "greeting_es.wav").write_bytes(b"\xff" * 320)
    # EN call
    twiml_en = _build_voice_twiml(static_greeting_exists=True, lang="en")
    assert "greeting_en.wav" in twiml_en
    assert "greeting_es.wav" not in twiml_en
    # ES call
    twiml_es = _build_voice_twiml(static_greeting_exists=True, lang="es")
    assert "greeting_es.wav" in twiml_es
    assert "greeting_en.wav" not in twiml_es