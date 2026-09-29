"""Tests for the defensive None handling in execute_call_transfer.

The 2026-09-22 production incident showed transfer_call returning
None (Twilio SDK edge case / version mismatch), which crashed the
Realtime dispatcher with AttributeError: 'NoneType' object has no
attribute 'get'. The fix normalises the return value to a dict so
the caller always sees success=False with a diagnostic.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _reload_executor():
    if "STT_server.services.tool_executor" in __import__("sys").modules:
        del __import__("sys").modules["STT_server.services.tool_executor"]
    return importlib.import_module("STT_server.services.tool_executor")


# ponytail: execute_call_transfer does
# ``from STT_server.adapters.twilio_api import transfer_call`` INSIDE
# the function. Patch at the source module so the import resolves to
# the mock. Patching the destination module attribute gives the
# lazy import a name to bind to.
def _patch_transfer(monkeypatch, return_value):
    fake = AsyncMock(return_value=return_value)
    monkeypatch.setattr(
        "STT_server.adapters.twilio_api.transfer_call", fake
    )
    return fake


@pytest.mark.asyncio
async def test_execute_call_transfer_handles_none_response(monkeypatch):
    """When transfer_call returns None (Twilio SDK edge case), the
    executor must NOT crash with AttributeError. Instead it raises
    ToolExecutionError with a diagnostic that mentions the type
    mismatch so the operator can grep logs.
    """
    te = _reload_executor()
    _patch_transfer(monkeypatch, None)

    with pytest.raises(te.ToolExecutionError) as excinfo:
        await te.execute_call_transfer(
            account_sid="AC-test",
            auth_token="token-test",
            call_sid="CA-test",
            destination="+526649070770",
            tool_name="Reception",
            timeout_sec=20,
        )
    # Diagnostic message must mention the type mismatch so the operator
    # can identify the SDK anomaly vs a Twilio 4xx.
    msg = str(excinfo.value)
    assert "NoneType" in msg or "non-dict" in msg


@pytest.mark.asyncio
async def test_execute_call_transfer_handles_string_response(monkeypatch):
    """If transfer_call returns a non-dict (string, None, list, etc.)
    the executor normalises and raises ToolExecutionError.
    """
    te = _reload_executor()
    _patch_transfer(monkeypatch, "call_xyz123")

    with pytest.raises(te.ToolExecutionError):
        await te.execute_call_transfer(
            account_sid="AC-test",
            auth_token="token-test",
            call_sid="CA-test",
            destination="+526649070770",
            tool_name="Reception",
        )


@pytest.mark.asyncio
async def test_execute_call_transfer_success_path_still_works(monkeypatch):
    """The defensive fix must NOT regress the happy path: a valid
    dict with success=True should pass through unchanged.
    """
    te = _reload_executor()
    _patch_transfer(monkeypatch, {"success": True, "callsid": "CA-test"})

    result = await te.execute_call_transfer(
        account_sid="AC-test",
        auth_token="token-test",
        call_sid="CA-test",
        destination="+526649070770",
        tool_name="Reception",
    )
    assert result["success"] is True
    assert result["callsid"] == "CA-test"


@pytest.mark.asyncio
async def test_execute_call_transfer_explicit_failure_surfaces(monkeypatch):
    """A success=False dict from transfer_call becomes a clear
    ToolExecutionError so the LLM can recover gracefully.
    """
    te = _reload_executor()
    _patch_transfer(monkeypatch, {"success": False, "error": "21408"})

    with pytest.raises(te.ToolExecutionError) as excinfo:
        await te.execute_call_transfer(
            account_sid="AC-test",
            auth_token="token-test",
            call_sid="CA-test",
            destination="+526649070770",
            tool_name="Reception",
        )
    assert "21408" in str(excinfo.value)
    assert "rejected by Twilio" in str(excinfo.value)


# ── The real transfer_call, not a mock of it ─────────────────────────────
# Every test above patches transfer_call out, so none of them can catch
# a defect INSIDE it. That is exactly how this shipped: transfer_call
# was missing its `return await ... _to_thread(_transfer)`, so the
# coroutine fell off the end and returned None on every call, failing
# 100% of human transfers with
#   "call_transfer 'Recepcion' rejected by Twilio: transfer_call
#    returned non-dict: NoneType"
# The 2026-09-22 note above blamed a "Twilio SDK edge case"; it was
# our missing return. These tests drive the real function with only the
# Twilio SDK client faked.


def _fake_twilio_client(monkeypatch, *, raise_on_update=None):
    """Install a fake twilio client and return the recorded update call."""
    recorded = {}

    class _Calls:
        def __init__(self, sid):
            self.sid = sid

        def update(self, **kwargs):
            if raise_on_update is not None:
                raise raise_on_update
            recorded.update(kwargs)
            return {"sid": self.sid, "status": "in-progress"}

    class _Client:
        def __init__(self, *_a, **_k):
            pass

        def calls(self, sid):
            return _Calls(sid)

    monkeypatch.setattr(
        "STT_server.adapters.twilio_api._get_twilio_client",
        lambda *_a, **_k: _Client(),
    )
    return recorded


@pytest.mark.asyncio
async def test_real_transfer_call_returns_a_dict(monkeypatch):
    """The regression that took human transfer offline: the function must
    RETURN the result of the threaded Twilio call. A coroutine that
    falls off its end returns None and every transfer fails."""
    import importlib
    api = importlib.import_module("STT_server.adapters.twilio_api")

    recorded = _fake_twilio_client(monkeypatch)
    result = await api.transfer_call(
        account_sid="AC-test", auth_token="tok", call_sid="CA-test",
        destination="+526649070770", timeout_sec=20,
        action_url="https://x.test/voice/transfer-fallback?remaining=",
    )
    assert isinstance(result, dict), (
        f"transfer_call returned {type(result).__name__}; a missing return "
        "here fails every human transfer"
    )
    assert result["success"] is True
    assert result["destination"] == "+526649070770"
    # and the Dial TwiML actually reached the Twilio client
    assert "twiml" in recorded
    assert "<Dial" in recorded["twiml"]
    assert "+526649070770" in recorded["twiml"]


@pytest.mark.asyncio
async def test_real_transfer_call_returns_failure_dict_not_none(monkeypatch):
    """A Twilio-side rejection must surface as success=False with the
    error, not as None."""
    import importlib
    api = importlib.import_module("STT_server.adapters.twilio_api")

    _fake_twilio_client(monkeypatch, raise_on_update=RuntimeError("21211"))

    result = await api.transfer_call(
        account_sid="AC-test", auth_token="tok", call_sid="CA-test",
        destination="+15005550006", timeout_sec=20,
    )
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "21211" in result["error"]


@pytest.mark.asyncio
async def test_real_hangup_call_still_returns_a_dict(monkeypatch):
    """hangup_call shares the file with transfer_call; guard that the fix
    did not disturb it, since its own return was what masked the missing
    one next door."""
    import importlib
    api = importlib.import_module("STT_server.adapters.twilio_api")

    recorded = _fake_twilio_client(monkeypatch)
    result = await api.hangup_call(
        account_sid="AC-test", auth_token="tok", call_sid="CA-test",
    )
    assert isinstance(result, dict)
    assert result["success"] is True
    assert recorded.get("status") == "completed"

