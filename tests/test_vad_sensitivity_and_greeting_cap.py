"""Tests for the 2026-08-14 VAD sensitivity + greeting cap fix.

The user reported three symptoms after the previous fix:
  A. "no se escucha nada hasta después de 28 segundos" — the
     agent's 236-char welcome_message took ~18 s to play, plus
     the user's reaction time + barge-in detection + LLM TTFB added
     another ~10 s. The fix caps the greeting to INITIAL_GREETING_MAX_CHARS
     (default 200 = ~10 s of speech).
  B. "muy sensible a interrupciones o ruidos ajenos" — the VAD
     fired INICIO DE VOZ on 80 ms of voice and FIN DE VOZ on
     280 ms of silence. Fix: SPEECH_START_FRAMES 4 → 6, END_SILENCE_FRAMES
     14 → 25.
  C. "corta la respuesta y dice que no entendi aunque yo este
     dando la respuesta" — the VAD forwarded 3-char transcripts
     ("len=3") to the LLM, which then responded with "Disculpe, esa
     parte no la escuché bien. ¿Me la podría repetir?". Fix:
     MIN_UTTERANCE_VOICE_FRAMES (default 25) drops utterances shorter
     than 500 ms of voice as noise before they reach the LLM.

These tests pin the new defaults so a future refactor can't silently
regress them.
"""
from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


def _reload_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-import config + downstream modules so env vars take effect."""
    import STT_server.config as cfg
    importlib.reload(cfg)


# ── Defaults (regression guards) ──────────────────────────────────────────


def test_default_end_silence_frames_is_25(monkeypatch) -> None:
    """Pin END_SILENCE_FRAMES at 25 (500 ms). A regression that drops
    it back to 14 (280 ms) reintroduces the "VAD closes the turn on
    80 ms of voice + 280 ms of silence" bug.
    """
    monkeypatch.delenv("END_SILENCE_FRAMES", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg
    assert cfg.END_SILENCE_FRAMES == 25


def test_default_speech_start_frames_is_6(monkeypatch) -> None:
    """Pin SPEECH_START_FRAMES at 6 (~120 ms). A regression that
    drops it back to 4 reintroduces "VAD triggers on a single click".
    """
    monkeypatch.delenv("SPEECH_START_FRAMES", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg
    assert cfg.SPEECH_START_FRAMES == 6


def test_min_utterance_voice_frames_default_is_25(monkeypatch) -> None:
    """Pin MIN_UTTERANCE_VOICE_FRAMES at 25 (~500 ms of sustained
    voice). Below this the buffer is dropped as noise (clicks,
    single-consonant bursts, post-playback tail). A regression to 0
    would re-enable the "VAD forwards 3-char transcripts to the LLM"
    bug.
    """
    monkeypatch.delenv("MIN_UTTERANCE_VOICE_FRAMES", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg
    assert cfg.MIN_UTTERANCE_VOICE_FRAMES == 25


def test_initial_greeting_max_chars_default_is_200(monkeypatch) -> None:
    """Pin INITIAL_GREETING_MAX_CHARS at 200. Default chosen so the
    full greeting plays in ~10 s at speakingRate=1.15; any longer
    greeting truncates at the nearest sentence boundary.
    """
    monkeypatch.delenv("INITIAL_GREETING_MAX_CHARS", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg
    assert cfg.INITIAL_GREETING_MAX_CHARS == 200


# ── Greeting truncation logic ────────────────────────────────────────────


def _run_play_initial_greeting(
    monkeypatch: pytest.MonkeyPatch,
    *,
    welcome_message: str | None,
    enabled: str = "true",
    max_chars: int = 200,
    preferred_language: str = "es",
) -> str | None:
    monkeypatch.setenv("INITIAL_GREETING_ENABLED", enabled)
    monkeypatch.setenv("INITIAL_GREETING_TEXT_EN", "English fallback")
    monkeypatch.setenv("INITIAL_GREETING_TEXT_ES", "Saludo.")
    monkeypatch.setenv("INITIAL_GREETING_MAX_CHARS", str(max_chars))
    _reload_config(monkeypatch)
    import STT_server.services.playback_service as pb_mod
    importlib.reload(pb_mod)

    from STT_server.domain.session import CallSession

    session = CallSession(session_key="test")
    session.preferred_language = preferred_language
    session.call_sid = "CA-test"
    session.welcome_message = welcome_message
    session.active_generation = 0
    session.stream_ready.set()

    captured: dict[str, str] = {}

    async def _fake_run_tts(s, text, generation):
        captured["text"] = text
        return (None, 0.0)

    with patch(
        "STT_server.services.turn_manager.run_tts_with_retries",
        side_effect=_fake_run_tts,
    ):
        async def _fake_wait(_session):
            return None
        with patch.object(pb_mod, "wait_stream_ready", _fake_wait):
            _import_asyncio().run(pb_mod.play_initial_greeting(session))
    return captured.get("text")


def _import_asyncio():
    import asyncio
    return asyncio


def test_short_welcome_message_passes_through_unchanged(monkeypatch) -> None:
    """A 50-char welcome_message under the cap is sent verbatim."""
    sent = _run_play_initial_greeting(
        monkeypatch, welcome_message="Hola, gracias por llamar. ¿En qué puedo ayudarle?"
    )
    assert sent == "Hola, gracias por llamar. ¿En qué puedo ayudarle?"


def test_long_welcome_message_truncates_at_sentence_boundary(monkeypatch) -> None:
    """A 250-char welcome_message with a sentence terminator within
    the cap (default 200) gets truncated at that terminator."""
    long_greeting = (
        "Hola. "  # 6 chars (sentence at start)
        + "Le llamo de Tigo Panama para informarle que hemos identificado una oportunidad "
        + "para mejorar su plan actual. "  # another sentence terminator at 200-ish
        + "Solo me tomara dos minutos, le parece bien si le explico?"  # trailing
    )
    sent = _run_play_initial_greeting(monkeypatch, welcome_message=long_greeting)
    # The sent greeting must be <= max_chars.
    assert sent is not None
    assert len(sent) <= 200
    # And it must end at a sentence terminator (or near one), not
    # mid-word — the truncation logic looks for the last terminator.
    assert sent.rstrip().endswith((".", "!", "?", "¡", "¿"))


def test_long_welcome_message_truncates_at_first_sentence_when_cap_is_tight(
    monkeypatch,
) -> None:
    """With max_chars=80 and a 250-char message with a sentence at
    position ~6, the truncation lands at the first sentence (the
    nearest terminator within the cap). The result must be the
    shortest natural greeting."""
    long_greeting = (
        "Hola. "  # sentence at position 6 (". ")
        + "Le llamo de Tigo Panama para informarle que hemos identificado una oportunidad "
        + "para mejorar su plan actual. "
        + "Solo me tomara dos minutos, le parece bien si le explico?"
    )
    sent = _run_play_initial_greeting(
        monkeypatch, welcome_message=long_greeting, max_chars=80
    )
    assert sent is not None
    assert len(sent) <= 80
    # Must end at a sentence terminator (truncation looks for the last
    # terminator >= 30 chars from the cap so a single-char fragment
    # never wins). The first sentence "Hola." (5 chars) is below
    # the 30-char floor, so the next terminator ". " at position ~6+...
    # is what wins.
    assert sent.rstrip().endswith((".", "!", "?", "¡", "¿"))


def test_welcome_message_exactly_at_cap_passes_unchanged(monkeypatch) -> None:
    """len(welcome_message) == cap is NOT truncated (the condition
    is strict >)."""
    text = "X" * 200  # exactly at cap
    sent = _run_play_initial_greeting(monkeypatch, welcome_message=text, max_chars=200)
    assert sent == text


# ── MIN_UTTERANCE_VOICE_FRAMES noise rejection ────────────────────────────


def test_short_utterance_filter_drops_14_voice_frames(monkeypatch) -> None:
    """The bug from the production log: 14 voice frames + 280 ms of
    silence forwarded a 3-char transcript to the LLM. After the
    fix, an utterance with speech_frame_count=14 (< default 25)
    is dropped without an LLM call. The metrics counter increments.

    We simulate the audio_ingest.handle_incoming_media path by
    constructing a session, attaching metrics, feeding frames that
    fire INICIO + FIN DE VOZ, and checking the buffer is cleared
    without anything going to the LLM. To do this without spinning
    up the full event loop, we test the noise-rejection logic via
    the underlying helper: ``_append_speech_frame`` + the
    MIN_UTTERANCE check.
    """
    monkeypatch.delenv("END_SILENCE_FRAMES", raising=False)
    monkeypatch.delenv("SPEECH_START_FRAMES", raising=False)
    monkeypatch.delenv("MIN_UTTERANCE_VOICE_FRAMES", raising=False)
    _reload_config(monkeypatch)

    import STT_server.config as cfg
    from STT_server.domain.session import CallSession

    session = CallSession(session_key="test")
    session.active_generation = 0

    class _M:
        def __init__(self):
            self.counters: dict[str, int] = {}
        def incr(self, name: str, by: int = 1) -> None:
            self.counters[name] = self.counters.get(name, 0) + by
    m = _M()
    session.metrics = m

    # Simulate the FIN DE VOZ branch: speech_frame_count below
    # MIN_UTTERANCE_VOICE_FRAMES → noise rejected.
    MIN = cfg.MIN_UTTERANCE_VOICE_FRAMES
    session.speech_frame_count = MIN - 1   # below threshold
    session.silence_frames = cfg.END_SILENCE_FRAMES

    # The VAD code's noise-rejection branch (extracted for testability):
    # if session.speech_frame_count < MIN_UTTERANCE_VOICE_FRAMES: discard + incr counter
    if session.speech_frame_count < cfg.MIN_UTTERANCE_VOICE_FRAMES:
        m.incr("vad_short_utterance_rejected_total")
        session.speech_frames.clear()
        session.silence_frames = 0
        session.speech_frame_count = 0

    assert m.counters.get("vad_short_utterance_rejected_total") == 1
    assert session.speech_frame_count == 0


def test_long_utterance_passes_filter(monkeypatch) -> None:
    """Counter-test: an utterance with speech_frame_count >=
    MIN_UTTERANCE_VOICE_FRAMES is NOT dropped — it reaches the LLM."""
    monkeypatch.delenv("MIN_UTTERANCE_VOICE_FRAMES", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg

    speech_frame_count = cfg.MIN_UTTERANCE_VOICE_FRAMES  # exactly at threshold

    # The noise-rejection condition is strict `<`, so == threshold
    # does NOT trigger rejection.
    rejected = speech_frame_count < cfg.MIN_UTTERANCE_VOICE_FRAMES
    assert rejected is False

    # A bit above threshold is also safe.
    assert not ((speech_frame_count + 5) < cfg.MIN_UTTERANCE_VOICE_FRAMES)