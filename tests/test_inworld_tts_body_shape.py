"""Tests for the Inworld TTS body shape.

Three guarantees pinned:

  1. `deliveryMode` is a TOP-LEVEL field in the JSON body (NOT
     inside `audioConfig`). Inworld's official HTTP schema documents
     it at the root of the request. Re-nesting it inside audioConfig
     would 4xx the next call.

  2. The fake `Steering` flag is gone. Inworld's HTTP API for
     `/tts/v1/voice` and `/tts/v1/voice:stream` does NOT accept an
     `audioConfig.Steering` field — the previous version set it
     for tts-2 only and Inworld silently ignored it. Removing it
     brings the wire payload back into contract with the docs.

  3. Default `speakingRate` is 1.0 when no per-agent override is
     set (was 1.15 — bumped up because the LLM hint now asks for
     real <break /> pauses; a fast speaking rate would compress
     them into rushed-stops). When `session.tts_speed > 1.10`,
     the adapter logs a warning so the markup work doesn't get
     silently undone by an old setting.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from STT_server.adapters.inworld_tts import DEFAULT_MODEL_ID


def _make_session(**overrides):
    """Build a CallSession stand-in with just the attributes
    `stream_tts_segment` reads."""
    s = MagicMock()
    s.session_key = "test"
    s.voice_id = overrides.get("voice_id", "Mateo")
    s.tts_model = overrides.get("tts_model", DEFAULT_MODEL_ID)
    s.tts_speed = overrides.get("tts_speed", None)
    s.preferred_language = overrides.get("preferred_language", "es")
    s.user_id = "user-test"
    s._stage_timer = None
    s.active_generation = 0
    s.use_own_key = True
    return s


@pytest.fixture(autouse=True)
def _mock_resolve_credentials(monkeypatch):
    """`stream_tts_segment` reads the Inworld API key via
    `resolve_for_session`. Without mocking, the function raises
    RuntimeError before any HTTP call."""
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.resolve_for_session",
        lambda session, *a, **kw: {"api_key": "fake-test-key"},
    )


def _make_fake_urlopen(captured):
    """Build a fake urlopen that captures the request and returns
    a minimal Inworld streaming response with one audio chunk."""
    def fake_urlopen(req, *a, **kw):
        captured.append(req.data)
        r = MagicMock()
        r.__enter__ = lambda self: self
        r.__exit__ = lambda *a: False
        r.read = lambda *_: b'{"result":{"audioContent":"AAAA"}}'
        return r
    return fake_urlopen


# ── deliveryMode placement ──────────────────────────────────────


@pytest.mark.asyncio
async def test_delivery_mode_is_top_level_not_inside_audio_config(monkeypatch):
    from STT_server.adapters.inworld_tts import stream_tts_segment

    captured = []
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.urllib.request.urlopen",
        _make_fake_urlopen(captured),
    )

    session = _make_session()
    gen = AsyncMock()
    await stream_tts_segment(session, "Hola.", 0, gen)

    assert len(captured) == 1, f"expected 1 request, got {len(captured)}"
    body = json.loads(captured[0].decode("utf-8"))
    assert "deliveryMode" in body, (
        "deliveryMode missing — Inworld expects it at the root of the "
        "request body, not nested in audioConfig."
    )
    assert body["deliveryMode"] == "BALANCED", (
        f"expected deliveryMode=BALANCED, got {body['deliveryMode']!r}"
    )
    assert "deliveryMode" not in body["audioConfig"], (
        "deliveryMode MUST NOT be nested inside audioConfig — "
        "Inworld documents it at the request root."
    )


# ── fake Steering field gone ─────────────────────────────────────


@pytest.mark.asyncio
async def test_steering_field_is_not_in_body(monkeypatch):
    """The previous version set `audioConfig.Steering: true` for
    tts-2 only. Inworld silently ignored it (the field isn't in
    the schema). Removing it makes the wire payload match the
    contract. Steering is now controlled inline via markup like
    `[speak calmly, professionally.]`."""
    from STT_server.adapters.inworld_tts import stream_tts_segment

    captured = []
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.urllib.request.urlopen",
        _make_fake_urlopen(captured),
    )

    session = _make_session(tts_model="inworld-tts-2")
    gen = AsyncMock()
    await stream_tts_segment(session, "Hola.", 0, gen)

    body = json.loads(captured[0].decode("utf-8"))
    assert "Steering" not in body.get("audioConfig", {}), (
        "Steering field is not part of Inworld's HTTP API and was "
        "silently ignored. Re-nesting it here is a regression."
    )
    assert "Steering" not in body, (
        "Steering at the root is also invalid — Inworld controls "
        "steering via inline markup ([speak calmly, professionally.])."
    )


# ── speakingRate defaults ───────────────────────────────────────


@pytest.mark.asyncio
async def test_default_speaking_rate_is_one(monkeypatch):
    """When `session.tts_speed` is None (no per-agent override),
    the request must use 1.0. The previous default of 1.15 was
    set before the LLM hint asked for real <break /> pauses —
    1.15 compresses those pauses into rushed-stops. Per-agent
    overrides via the agent row still apply on top."""
    from STT_server.adapters.inworld_tts import stream_tts_segment

    captured = []
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.urllib.request.urlopen",
        _make_fake_urlopen(captured),
    )

    session = _make_session(tts_speed=None)
    gen = AsyncMock()
    await stream_tts_segment(session, "Hola.", 0, gen)

    body = json.loads(captured[0].decode("utf-8"))
    assert body["audioConfig"]["speakingRate"] == 1.0, (
        f"expected default speakingRate=1.0, got {body['audioConfig']['speakingRate']!r}. "
        f"The old default 1.15 was set before the markup work; with real "
        f"<break /> pauses the agent sounds rushed instead of natural."
    )


@pytest.mark.asyncio
async def test_speaking_rate_above_one_ten_logs_warning(monkeypatch):
    """When an operator sets tts_speed > 1.10, the adapter logs a
    warning that the markup work may be silently undone. The actual
    rate still goes out in the body — we don't override the
    operator's choice, just nudge them toward the A/B test."""
    from STT_server.adapters.inworld_tts import stream_tts_segment

    captured = []
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.urllib.request.urlopen",
        _make_fake_urlopen(captured),
    )

    session = _make_session(tts_speed=1.25)
    gen = AsyncMock()
    with patch("STT_server.adapters.inworld_tts.log") as mock_log:
        await stream_tts_segment(session, "Hola.", 0, gen)

    body = json.loads(captured[0].decode("utf-8"))
    assert body["audioConfig"]["speakingRate"] == 1.25, (
        f"operator choice (1.25) must be honored in the body, got "
        f"{body['audioConfig']['speakingRate']!r}"
    )
    # The warning must have fired on the log. Search the call
    # args list for the speakingRate > 1.10 message.
    warning_calls = [
        call for call in mock_log.info.call_args_list
        if call.args and "speakingRate=" in str(call.args[0])
        and ">1.10" in str(call.args[0])
    ]
    assert warning_calls, (
        "expected a warning log for speakingRate > 1.10; "
        f"got log.info calls: {mock_log.info.call_args_list}"
    )


@pytest.mark.asyncio
async def test_speaking_rate_at_or_below_one_ten_does_not_log_warning(monkeypatch):
    """No warning when tts_speed ≤ 1.10 — the threshold the
    adapter logs at. Operators at 1.0–1.05 (the recommended range)
    don't get false-positive warnings."""
    from STT_server.adapters.inworld_tts import stream_tts_segment

    captured = []
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.urllib.request.urlopen",
        _make_fake_urlopen(captured),
    )

    session = _make_session(tts_speed=1.0)
    gen = AsyncMock()
    with patch("STT_server.adapters.inworld_tts.log") as mock_log:
        await stream_tts_segment(session, "Hola.", 0, gen)

    # Verify no speakingRate > 1.10 warning was emitted.
    warning_calls = [
        call for call in mock_log.info.call_args_list
        if call.args and "speakingRate=" in str(call.args[0])
        and ">1.10" in str(call.args[0])
    ]
    assert not warning_calls, (
        f"unexpected warning for speakingRate=1.0: "
        f"{[str(c.args[0]) for c in mock_log.info.call_args_list]}"
    )


# ── model id handling ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tts_model_passed_through_verbatim(monkeypatch):
    """The adapter doesn't override tts_model — it passes whatever
    the agent row stored. Default is 1.5-mini (kept that way per
    operator decision; the TTS-2 A/B test is opt-in via the FE)."""
    from STT_server.adapters.inworld_tts import stream_tts_segment

    captured = []
    monkeypatch.setattr(
        "STT_server.adapters.inworld_tts.urllib.request.urlopen",
        _make_fake_urlopen(captured),
    )

    session = _make_session(tts_model="inworld-tts-2")
    gen = AsyncMock()
    await stream_tts_segment(session, "Hola.", 0, gen)

    body = json.loads(captured[0].decode("utf-8"))
    assert body["modelId"] == "inworld-tts-2"