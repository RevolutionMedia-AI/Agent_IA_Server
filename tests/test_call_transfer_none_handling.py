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

