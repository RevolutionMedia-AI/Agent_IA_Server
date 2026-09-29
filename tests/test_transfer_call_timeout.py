"""transfer_call must cap the Twilio REST call that redirects a live call.

Without the cap, a hung HTTP request keeps the coroutine alive long
after the operator hung up — blocking cleanup and leaking a task. 30s is
twice Twilio's own 15s HTTP timeout so the SDK gets room to retry once.

Also the regression guard for a defect that could only hide here:
transfer_call was missing its `return await ... _to_thread(_transfer)`,
so the coroutine fell off the end and returned None on EVERY call,
failing 100% of human transfers with
  "call_transfer 'Recepcion' rejected by Twilio: transfer_call
   returned non-dict: NoneType"
while the AI apologised to the caller. It was written off as a Twilio
SDK quirk because the surrounding tests all mocked transfer_call out
instead of calling it. The sibling tests in
test_call_transfer_none_handling.py now drive the real function.
"""
from __future__ import annotations

import asyncio
import importlib

import pytest


def _twilio_api():
    return importlib.import_module("STT_server.adapters.twilio_api")


@pytest.mark.asyncio
async def test_transfer_call_times_out_on_a_hung_twilio_call(monkeypatch):
    """asyncio.wait_for must wrap the threaded Twilio call.

    The stub is an ASYNC function on purpose: a sync stub would block the
    event loop, so wait_for could never fire and the test would instead
    fall through into a real network call to Twilio.
    """
    ta = _twilio_api()
    monkeypatch.setattr(ta, "TRANSFER_CALL_TIMEOUT_SEC", 0.25)

    never = asyncio.Event()

    async def hung_thread(func, *args, **kwargs):
        await never.wait()
        return func(*args, **kwargs)  # pragma: no cover - never reached

    monkeypatch.setattr(ta, "_to_thread", hung_thread)

    with pytest.raises(asyncio.TimeoutError):
        await ta.transfer_call(
            account_sid="AC-test",
            auth_token="token-test",
            call_sid="CA-test",
            destination="+526649070770",
        )
    never.set()


@pytest.mark.asyncio
async def test_transfer_call_returns_the_threaded_result(monkeypatch):
    """The cap must not swallow the result. This is the assertion the
    mock-only tests could not make: transfer_call returns a dict, the
    Twilio client is reached, and the Dial TwiML is what we built.

    The stub must be async because asyncio.wait_for needs an awaitable.
    _get_twilio_client is faked so the real body of _transfer runs —
    TwiML construction included — without a network call.
    """
    ta = _twilio_api()
    monkeypatch.setattr(ta, "TRANSFER_CALL_TIMEOUT_SEC", 5.0)

    recorded = {}

    class _Calls:
        def __init__(self, sid):
            self.sid = sid

        def update(self, **kwargs):
            recorded.update(kwargs)

    class _Client:
        def __init__(self, *_a, **_k):
            pass

        def calls(self, sid):
            return _Calls(sid)

    monkeypatch.setattr(
        ta, "_get_twilio_client", lambda *_a, **_k: _Client()
    )

    async def passthrough_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(ta, "_to_thread", passthrough_thread)

    result = await ta.transfer_call(
        account_sid="AC-test",
        auth_token="token-test",
        call_sid="CA-test",
        destination="+526649070770",
    )
    assert isinstance(result, dict), (
        f"transfer_call returned {type(result).__name__}; a missing return "
        "here fails every human transfer"
    )
    assert result["success"] is True
    assert "<Dial" in recorded.get("twiml", "")
    assert "+526649070770" in recorded.get("twiml", "")


def test_default_timeout_is_twice_twilios_http_timeout():
    """Pinned so a well-meaning 'just lower it to 10s' change is a
    visible diff: below ~30s we start cutting off Twilio's own retry."""
    ta = _twilio_api()
    assert ta.TRANSFER_CALL_TIMEOUT_SEC == 30.0
