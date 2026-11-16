"""Regression test for the per-user credential toggle.

The bug: the resolver only read the per-user layer when the agent's
"use my own key" toggle was ON, and the schema defaults that toggle to
FALSE. So with the default, a user who saved their OpenAI key got the
PLATFORM key instead, and their saved credential was never used.

The toggle means "use my own key", not "ignore my key": the fallback is
to the platform env vars "when no per-user key is configured"
(009_agent_use_own_key.sql).
"""
from __future__ import annotations

import pytest

from STT_server.services import credentials_resolver as cr


class _Session:
    """A session with the toggle in the given state."""

    def __init__(self, use_own_key: bool):
        self.user_id = "u1"
        self.tts_use_own_key = use_own_key
        self.stt_use_own_key = use_own_key
        self.llm_use_own_key = use_own_key


@pytest.fixture
def user_key(monkeypatch: pytest.MonkeyPatch):
    """Pretend the user stored a key for ElevenLabs."""
    monkeypatch.setattr(
        cr, "_read_per_user", lambda user_id, provider_id: {"api_key": "sk-USUARIO"}
    )
    monkeypatch.setattr(cr, "_read_platform", lambda provider_id: {"api_key": "sk-PLATAFORMA"})


def test_the_toggle_off_still_uses_the_users_key(user_key):
    """The default: the user's key wins, the platform fills the gaps."""
    creds = cr.resolve_for_session(_Session(use_own_key=False), "tts", "elevenlabs")
    assert creds.get("api_key") == "sk-USUARIO"


def test_the_toggle_on_uses_the_users_key(user_key):
    creds = cr.resolve_for_session(_Session(use_own_key=True), "tts", "elevenlabs")
    assert creds.get("api_key") == "sk-USUARIO"


def test_the_platform_key_fills_the_gaps(monkeypatch: pytest.MonkeyPatch):
    """A field the user did not set still comes from the platform."""
    monkeypatch.setattr(
        cr, "_read_per_user", lambda user_id, provider_id: {"api_key": "sk-USUARIO"}
    )
    monkeypatch.setattr(
        cr,
        "_read_platform",
        lambda provider_id: {"api_key": "sk-PLATAFORMA", "voice_id": "platform-voice"},
    )
    creds = cr.resolve_for_session(_Session(use_own_key=False), "tts", "elevenlabs")
    assert creds.get("api_key") == "sk-USUARIO"
    assert creds.get("voice_id") == "platform-voice"


def test_without_a_user_key_the_platform_key_is_used(monkeypatch: pytest.MonkeyPatch):
    """A user with no stored key still gets the platform key."""
    monkeypatch.setattr(cr, "_read_per_user", lambda user_id, provider_id: {})
    monkeypatch.setattr(cr, "_read_platform", lambda provider_id: {"api_key": "sk-PLATAFORMA"})
    creds = cr.resolve_for_session(_Session(use_own_key=False), "tts", "elevenlabs")
    assert creds.get("api_key") == "sk-PLATAFORMA"
