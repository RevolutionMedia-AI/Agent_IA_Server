"""Tests for STT_server.services.audio_errors.

The hierarchy is the load-bearing piece of the recoverable-audio
design. The contract callers depend on:

  1. ``RecoverableAudioError`` and ``FatalAudioError`` are siblings,
     not parent/child. A catch on one must NOT swallow the other.
  2. Both inherit from a common base so a single ``except AudioError``
     can catch both where it makes sense (e.g. the per-call summary
     log line) without forcing the choice between them.
  3. A bare ``Exception`` catch that doesn't list ``AudioError`` will
     also catch them — this is the global ``media_stream`` handler
     and it MUST keep firing ``play_error_and_hangup`` on a
     ``FatalAudioError``. Covered indirectly by the inheritance
     chain (any Exception matches ``Exception``).
"""
from __future__ import annotations

import pytest

from STT_server.services.audio_errors import (
    AudioError,
    FatalAudioError,
    RecoverableAudioError,
)


# ── inheritance ─────────────────────────────────────────────────────────────


def test_recoverable_is_audio_error() -> None:
    """Catching ``AudioError`` catches both subclasses."""
    assert issubclass(RecoverableAudioError, AudioError)


def test_fatal_is_audio_error() -> None:
    assert issubclass(FatalAudioError, AudioError)


def test_recoverable_is_not_fatal() -> None:
    """Sibling, not parent. Catching Fatal must NOT silently swallow a
    Recoverable; this is the whole point of having two classes.
    """
    assert not issubclass(RecoverableAudioError, FatalAudioError)


def test_fatal_is_not_recoverable() -> None:
    assert not issubclass(FatalAudioError, RecoverableAudioError)


# ── message passthrough ─────────────────────────────────────────────────────


def test_recoverable_carries_message() -> None:
    exc = RecoverableAudioError("twilio ws closed: 1006")
    assert "1006" in str(exc)


def test_fatal_carries_message() -> None:
    exc = FatalAudioError("SPEECH_FRAMES_MAX is not defined")
    assert "SPEECH_FRAMES_MAX" in str(exc)


# ─- contract with the global handler ──────────────────────────────────────


def test_media_stream_global_handler_still_catches_fatal() -> None:
    """The ``media_stream`` global ``except Exception`` must still
    catch ``FatalAudioError`` — that's the visible-failure gate the
    user explicitly asked us to keep. If a future refactor makes
    ``FatalAudioError`` an instance of something that escapes a bare
    ``Exception`` catch, this test will fail and warn the author.
    """
    try:
        raise FatalAudioError("boom")
    except Exception as exc:
        assert isinstance(exc, FatalAudioError)
        assert isinstance(exc, AudioError)


def test_recoverable_can_be_distinguished_from_fatal_by_handler() -> None:
    """A handler that wants to keep the call alive on recoverable and
    crash on fatal can do ``except (RecoverableAudioError, FatalAudioError)
    as exc:`` and inspect ``type(exc)`` — no ambiguity allowed."""
    recoverable = RecoverableAudioError("x")
    fatal = FatalAudioError("y")
    assert type(recoverable) is RecoverableAudioError
    assert type(fatal) is FatalAudioError
    assert type(recoverable) is not type(fatal)