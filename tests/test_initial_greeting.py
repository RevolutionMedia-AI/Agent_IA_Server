"""Tests for the 2026-08-14 initial-greeting fix.

The user reported: "the initial greeting prompt isn't firing as
soon as the call connects". Root cause was that the previous
implementation only fired when ``session.welcome_message`` was
non-empty (per-agent config). Agents without a welcome_message
produced dead-air silence until the caller spoke.

These tests pin the new priority chain so a future refactor
can't silently regress it:
  1. session.welcome_message (per-agent override) wins.
  2. INITIAL_GREETING_TEXT_ES / _EN (platform fallback) — only
     when INITIAL_GREETING_ENABLED is true.
  3. INITIAL_GREETING_ENABLED=false AND no welcome_message →
     no-op (silent start, preserved opt-out for operators that
     want the caller to speak first).

Tests also confirm the language resolution: ``preferred_language``
starting with "en" → English fallback; everything else → Spanish.
"""
from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest


def _reload_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-import STT_server.config with a controlled env so each test
    sees a fresh INITIAL_GREETING_* state."""
    import STT_server.config as cfg_mod
    importlib.reload(cfg_mod)


# ── config defaults ────────────────────────────────────────────────────────


def test_default_initial_greeting_is_enabled(monkeypatch) -> None:
    """The user wants the greeting to fire by default so a fresh
    deployment without per-agent config doesn't dead-air. Pin it.
    """
    monkeypatch.delenv("INITIAL_GREETING_ENABLED", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg
    assert cfg.INITIAL_GREETING_ENABLED is True


def test_default_initial_greeting_texts_are_non_empty(monkeypatch) -> None:
    """The platform fallback texts must not be empty — an empty
    string would silently produce a no-op greeting and the caller
    would still hear silence."""
    monkeypatch.delenv("INITIAL_GREETING_TEXT_EN", raising=False)
    monkeypatch.delenv("INITIAL_GREETING_TEXT_ES", raising=False)
    _reload_config(monkeypatch)
    import STT_server.config as cfg
    assert cfg.INITIAL_GREETING_TEXT_EN and len(cfg.INITIAL_GREETING_TEXT_EN) > 10
    assert cfg.INITIAL_GREETING_TEXT_ES and len(cfg.INITIAL_GREETING_TEXT_ES) > 10


# ── play_initial_greeting priority chain ──────────────────────────────────


def _run_play_initial_greeting(
    monkeypatch: pytest.MonkeyPatch,
    *,
    welcome_message: str | None,
    enabled: str = "true",
    en_text: str = "English fallback",
    es_text: str = "Saludo en español.",
    preferred_language: str = "es",
) -> str | None:
    """Call play_initial_greeting with the given config and return
    the text it actually sent to TTS, or None if it returned early.
    """
    monkeypatch.setenv("INITIAL_GREETING_ENABLED", enabled)
    monkeypatch.setenv("INITIAL_GREETING_TEXT_EN", en_text)
    monkeypatch.setenv("INITIAL_GREETING_TEXT_ES", es_text)
    _reload_config(monkeypatch)

    # Force a re-import of playback_service so the new config is picked
    # up by the module-level imports it does.
    import STT_server.services.playback_service as pb_mod
    importlib.reload(pb_mod)

    from STT_server.domain.session import CallSession

    session = CallSession(session_key="test")
    session.preferred_language = preferred_language
    session.call_sid = "CA-test"
    session.welcome_message = welcome_message
    session.active_generation = 0
    # Stub the stream-ready wait so we don't hang on the event.
    session.stream_ready.set()

    captured: dict[str, str] = {}

    async def _fake_run_tts(s, text, generation):
        captured["text"] = text
        return (None, 0.0)

    # Stub run_tts_with_retries (the direct TTS path the user asked for).
    # playback_service imports it lazily inside the function, so we
    # patch the module attribute AFTER the import is done.
    with patch(
        "STT_server.services.turn_manager.run_tts_with_retries",
        side_effect=_fake_run_tts,
    ):
        # The wait_stream_ready call inside the function reads
        # session.stream_ready; we set it above so the event fires
        # immediately. But the function itself uses the imported
        # ``wait_stream_ready`` symbol — patch that too so we don't
        # depend on event-loop plumbing from the test thread.
        async def _fake_wait(_session):
            return None

        with patch.object(pb_mod, "wait_stream_ready", _fake_wait):
            asyncio_run = _import_asyncio().run
            asyncio_run(pb_mod.play_initial_greeting(session))

    return captured.get("text")


def _import_asyncio():
    import asyncio
    return asyncio


def test_welcome_message_wins_over_platform_fallback(monkeypatch) -> None:
    """Per-agent welcome_message has the highest priority — even if
    the platform fallback is set to a different language or text.
    """
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message="Per-agent custom greeting",
        en_text="English fallback",
        es_text="Fallback ES",
    )
    assert sent == "Per-agent custom greeting"


def test_platform_es_fallback_fires_when_no_welcome_message(monkeypatch) -> None:
    """No welcome_message, INITIAL_GREETING_ENABLED=true (default),
    preferred_language=es → the Spanish platform fallback is sent.
    """
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message=None,
        preferred_language="es",
        en_text="English fallback",
        es_text="Saludo en español.",
    )
    assert sent == "Saludo en español."


def test_platform_en_fallback_fires_when_lang_is_en(monkeypatch) -> None:
    """preferred_language starting with 'en' → English fallback."""
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message=None,
        preferred_language="en-US",
        en_text="English fallback for the test",
        es_text="Saludo en español.",
    )
    assert sent == "English fallback for the test"


def test_blank_welcome_message_falls_back_to_platform(monkeypatch) -> None:
    """A welcome_message that's whitespace-only (some agents save
    ' ' by mistake) must fall back to the platform text. Without
    this the caller hears silence on those agents."""
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message="   \n  ",
        en_text="English fallback",
        es_text="Saludo en español.",
        preferred_language="es",
    )
    assert sent == "Saludo en español."


def test_disabled_and_no_welcome_message_is_silent(monkeypatch) -> None:
    """INITIAL_GREETING_ENABLED=false AND no welcome_message →
    no-op (silent start). The caller will hear silence until they
    speak — preserved as an explicit opt-out for operators that
    want the user to talk first."""
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message=None,
        enabled="false",
        en_text="English fallback",
        es_text="Saludo en español.",
    )
    assert sent is None


def test_disabled_but_welcome_message_still_fires(monkeypatch) -> None:
    """INITIAL_GREETING_ENABLED=false but the agent has a
    welcome_message — the per-agent override still fires. The
    env var only gates the platform fallback, not the
    per-agent path (the agent's author may have set the
    greeting explicitly without wanting a platform default).
    """
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message="Per-agent only",
        enabled="false",
    )
    assert sent == "Per-agent only"


# ── default-on regression guard ──────────────────────────────────────────


def test_welcome_message_with_default_lang_en(monkeypatch) -> None:
    """Sanity: preferred_language exactly 'en' → English fallback."""
    sent = _run_play_initial_greeting(
        monkeypatch,
        welcome_message=None,
        preferred_language="en",
        en_text="English fallback",
        es_text="Saludo en español.",
    )
    assert sent == "English fallback"