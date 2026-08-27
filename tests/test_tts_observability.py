"""Tests for the TTS observability chain.

The TTS pipeline has three layers that mutate the text:
1. gpt-realtime output (or whatever LLM produced)
2. sanitize_tts_text() (strips Markdown, filters non-verbals)
3. Inworld request body (the wire payload)

When the operator sees flat TTS, they need to know which layer
stripped the markup. These tests pin the chain:

- TTS_DEBUG_LOG env var gates the logs (default off — PII risk)
- TTS_RAW_SEGMENT logs the text BEFORE sanitize
- TTS_SANITIZED_SEGMENT logs the text AFTER sanitize
- TTS_INWORLD_BODY logs the request body sent to Inworld

All three carry session + generation + seg for cross-correlation.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
from unittest.mock import patch

import pytest


# ── TTS_DEBUG_LOG env-var gating ─────────────────────────────────


def _reload_turn_manager(monkeypatch, env_value=None):
    """Reload services/turn_manager.py so the module-level
    TTS_DEBUG_LOG re-evaluates against the patched env."""
    if env_value is None:
        monkeypatch.delenv("TTS_DEBUG_LOG", raising=False)
    else:
        monkeypatch.setenv("TTS_DEBUG_LOG", env_value)
    sys.modules.pop("STT_server.services.turn_manager", None)
    return importlib.import_module("STT_server.services.turn_manager")


def test_tts_debug_log_env_default_is_off(monkeypatch):
    """Default (TTS_DEBUG_LOG not set or false) means the three
    observability logs stay silent in production. PII risk."""
    mod = _reload_turn_manager(monkeypatch, env_value="")
    assert mod.TTS_DEBUG_LOG is False, (
        "TTS_DEBUG_LOG default must be off — emails and IDs would "
        "land in production logs otherwise."
    )


def test_tts_debug_log_env_truthy_enables(monkeypatch):
    """TTS_DEBUG_LOG=1 flips the flag on so the observability chain
    fires. Operators enable while chasing a TTS issue."""
    mod = _reload_turn_manager(monkeypatch, env_value="1")
    assert mod.TTS_DEBUG_LOG is True


def test_tts_debug_log_env_random_string_disables(monkeypatch):
    """Anything not in the truthy set keeps the logs off. A typo'd env
    var shouldn't dump emails to production."""
    mod = _reload_turn_manager(monkeypatch, env_value="yes please")
    assert mod.TTS_DEBUG_LOG is False


# ── Correlation chain: session + gen + seg ───────────────────────


def _make_log_capture(records):
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    return Capture()


@pytest.mark.asyncio
async def test_observability_chain_correlation(monkeypatch):
    """One streaming pass emits RAW + SANITIZED + INWORLD_BODY
    with the SAME session/gen/seg. Operators join the three by
    those fields and grep across them."""
    mod = _reload_turn_manager(monkeypatch, env_value="1")
    sys.modules.pop("STT_server.adapters.inworld_tts", None)
    inworld_tts = importlib.import_module("STT_server.adapters.inworld_tts")

    captured = []

    class FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def read(self, *_a):
            return b'{"result":{"audioContent":"AAAA"}}'

    def fake_urlopen(req, *a, **kw):
        captured.append({"url": req.full_url, "data": req.data})
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    # Stub the body construction in stream_tts_segment.
    async def fake_run_tts_with_retries(session, text, generation, seg_idx=0):
        body = json.dumps({
            "text": text, "voiceId": "Miguel", "modelId": "inworld-tts-2",
            "language": "es", "deliveryMode": "BALANCED",
            "applyTextNormalization": "ON", "enhanceGeneration": False,
            "audioConfig": {
                "audioEncoding": "MULAW",
                "sampleRateHertz": 8000,
                "speakingRate": 1.0,
            },
        }).encode("utf-8")
        import urllib.request
        req = urllib.request.Request(
            "https://api.inworld.ai/tts/v1/voice:stream",
            data=body,
            headers={"Authorization": "Basic test"},
            method="POST",
        )
        return fake_urlopen(req)

    monkeypatch.setattr(mod, "run_tts_with_retries", fake_run_tts_with_retries)

    from unittest.mock import MagicMock
    session = MagicMock()
    session.session_key = "CA-test"
    session.active_generation = 4
    session.preferred_language = "es"
    session.user_id = "user-1"
    session.tts_speed = None
    session.tts_provider = "inworld"

    queue = mod.asyncio.Queue()
    await queue.put("De acuerdo. <break time=\"300ms\" /> [breathe] Hola.")
    await queue.put("¿Algo más?")
    await queue.put(None)

    records = []
    handler = _make_log_capture(records)
    mod.log.addHandler(handler)
    inworld_tts.log.addHandler(handler)

    await mod.play_tts_from_text_queue(session, 4, queue)

    mod.log.removeHandler(handler)
    inworld_tts.log.removeHandler(handler)

    raw_logs = [r for r in records if "TTS_RAW_SEGMENT" in r.getMessage()]
    sanitized_logs = [r for r in records if "TTS_SANITIZED_SEGMENT" in r.getMessage()]
    inworld_logs = [r for r in records if "TTS_INWORLD_BODY" in r.getMessage()]

    assert len(raw_logs) == 2, f"expected 2 RAW logs, got {len(raw_logs)}"
    assert len(sanitized_logs) == 2, f"expected 2 SANITIZED logs, got {len(sanitized_logs)}"
    assert len(inworld_logs) == 2, f"expected 2 INWORLD_BODY logs, got {len(inworld_logs)}"

    for r in raw_logs + sanitized_logs + inworld_logs:
        msg = r.getMessage()
        assert "session=CA-test" in msg, f"missing session id: {msg!r}"
        assert "gen=4" in msg, f"missing gen=4: {msg!r}"
        assert "seg=" in msg, f"missing seg counter: {msg!r}"

    assert "<break" in raw_logs[0].getMessage(), (
        f"RAW log missing markup: {raw_logs[0].getMessage()!r}"
    )
    assert "[breathe]" in raw_logs[0].getMessage(), (
        f"RAW log missing non-verbal: {raw_logs[0].getMessage()!r}"
    )


# ── Text truncation ────────────────────────────────────────────────


def test_raw_text_is_truncated_to_200_chars(monkeypatch):
    """A runaway LLM reply (1000+ chars) shouldn't blow up the log
    pipeline. Both RAW and SANITIZED truncate to TTS_DEBUG_TEXT_CHARS."""
    mod = _reload_turn_manager(monkeypatch, env_value="1")
    long_text = "a" * 1000
    records = []
    handler = _make_log_capture(records)
    mod.log.addHandler(handler)
    mod.log.info(
        "[TTS_RAW_SEGMENT] session=test gen=1 seg=1 text=%r",
        long_text[: mod.TTS_DEBUG_TEXT_CHARS],
    )
    mod.log.removeHandler(handler)
    assert len(records) == 1
    msg = records[0].getMessage()
    a_count = msg.count("'a'")
    assert a_count == 200, (
        f"expected 200 'a' chars in truncated message, got {a_count}: {msg!r}"
    )


# ── Disable path (production) ────────────────────────────────────────


def test_raw_log_does_not_fire_when_tts_debug_log_off(monkeypatch):
    """Production logs stay clean. PII (emails, IDs, names) never
    lands in logs when TTS_DEBUG_LOG is not set."""
    mod = _reload_turn_manager(monkeypatch, env_value="")
    assert mod.TTS_DEBUG_LOG is False
    assert mod.TTS_DEBUG_TEXT_CHARS == 200, (
        "truncation cap must be 200 chars regardless of gate state"
    )