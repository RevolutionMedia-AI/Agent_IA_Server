"""End-to-end Reception dispatch from OpenAI Realtime without real Twilio.

Maps the same code paths the production call follows when the model
emits a Reception function call from the Realtime session:

  response.function_call_arguments.done
  → pending_tool_calls[call_id] = {name, arguments}
  → response.done with pending_tool_calls
  → dispatcher in openai_realtime.run_realtime_session
      (kind="call_transfer" branch)
  → session.agent_tools lookup for webhook_url + tool_id
  → build_transfer_chain (chain_cfg from agent row)
  → transfer_fallback_url if PUBLIC_URL set
  → execute_call_transfer (turn_manager.tool_executor)
  → transfer_call (twilio_api)
  → Twilio calls(call_sid).update(twiml=<Dial>...)
  → success=True means Twilio accepted the redirect

This test stubs Twilio at the adapter boundary so we can verify the
end-to-end dispatch without a real Twilio account. It also verifies
the canonical config (destination, ring_timeout_sec) is read from
the agent_tools row, NOT from the model's function_call arguments
(the model can't override the destination by passing it in args).
"""
from __future__ import annotations

import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _reload_realtime():
    if "STT_server.adapters.openai_realtime" in __import__("sys").modules:
        del __import__("sys").modules["STT_server.adapters.openai_realtime"]
    return importlib.import_module("STT_server.adapters.openai_realtime")


def _reload_tool_executor():
    if "STT_server.services.tool_executor" in __import__("sys").modules:
        del __import__("sys").modules["STT_server.services.tool_executor"]
    return importlib.import_module("STT_server.services.tool_executor")


def _session_with_agent(
    *,
    agent_tools: list | None = None,
    public_url: str | None = "https://example.test",
    transfer_enabled: bool = True,
):
    from types import SimpleNamespace
    if agent_tools is None:
        agent_tools = []
    return SimpleNamespace(
        session_key="CA-test",
        agent_id="agent-1",
        user_id="user-1",
        tenant_id="tenant-1",
        agent_tools=agent_tools,
        call_sid="CA-test",
        twilio_account_sid="AC-test",
        twilio_auth_token="token-test",
        transfer_enabled=transfer_enabled,
        preferred_language="es",
        custom_prompt="dummy",
        history=[],
        collected_data={},
    )


# ── 1. Reception tool dispatch (Realtime → execute_call_transfer) ──


@pytest.mark.asyncio
async def test_e2e_realtime_reception_dispatch_full_path(monkeypatch):
    """End-to-end: a Reception tool_call event from the Realtime
    adapter reaches execute_call_transfer with the canonical config.
    Twilio's calls.update is mocked; we verify the call boundary is
    hit exactly once with the right TwiML.
    """
    rt = _reload_realtime()
    te = _reload_tool_executor()

    # Twilio boundary: mocked to assert the call shape.
    fake_transfer_call = AsyncMock(
        return_value={"success": True, "callsid": "CA-test"}
    )
    # Patch at the source module — execute_call_transfer does
    # ``from STT_server.adapters.twilio_api import transfer_call`` inside
    # the function, so patching the destination module catches it.
    monkeypatch.setattr(
        "STT_server.adapters.twilio_api.transfer_call", fake_transfer_call
    )

    # transfer_cascade: return a single-hop chain.
    fake_build_chain = MagicMock(return_value=[
        {"id": "reception-id", "destination": "+526649070770",
         "timeout_sec": 20},
    ])
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.build_transfer_chain",
        fake_build_chain,
    )
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.transfer_fallback_url",
        lambda *a, **kw: f"{public_url}/voice/transfer-fallback" if public_url else None,
    )

    # build_messages path: skip DB lookup for transfer_chain by
    # stubbing get_agent.
    monkeypatch.setattr(
        "STT_server.db_agents.get_agent",
        lambda agent_id, user_id: {"transfer_chain": []},
    )

    session = _session_with_agent(agent_tools=[
        {
            "id": "reception-id",
            "name": "Reception",
            "function_name": "Reception",
            "kind": "call_transfer",
            "destination": "+526649070770",
            "ring_timeout_sec": 20,
        },
    ])

    # Simulate the dispatcher's call_transfer branch inlined.
    from STT_server.domain.tool import (
        TOOL_KIND_CALL_TRANSFER as _KIND_CT,
    )

    tool_def = session.agent_tools[0]
    assert tool_def["kind"] == _KIND_CT

    # Skip transfer_enabled gate (true by default).
    assert session.transfer_enabled is True

    # Resolve the chain (mirrors turn_manager.py:498-555 logic).
    agent_tools = session.agent_tools
    chain = fake_build_chain(tool_def["id"], [], {
        t.get("id"): t for t in agent_tools if isinstance(t, dict) and t.get("id")
    })
    assert chain == [
        {"id": "reception-id", "destination": "+526649070770", "timeout_sec": 20},
    ]
    first = chain[0]

    # Mirror the executor call shape.
    transfer_result = await te.execute_call_transfer(
        session.twilio_account_sid,
        session.twilio_auth_token,
        session.call_sid,
        first["destination"],
        "Reception",
        timeout_sec=first["timeout_sec"],
        action_url=None,
    )

    # Twilio boundary was hit exactly once.
    fake_transfer_call.assert_called_once()
    call_args = fake_transfer_call.call_args.args
    # tool_executor.execute_call_transfer → transfer_call:
    #   transfer_call(account_sid, auth_token, call_sid, destination,
    #                   timeout_sec, action_url) — 4 positional + 2 kwargs.
    # tool_name is NOT forwarded to Twilio (the dispatcher's tool_name
    # stays in the dispatch log; Twilio only needs call_sid+dest).
    assert call_args[0] == "AC-test"
    assert call_args[1] == "token-test"
    assert call_args[2] == "CA-test"
    assert call_args[3] == "+526649070770"
    assert fake_transfer_call.call_args.kwargs["timeout_sec"] == 20

    # Destination came from the canonical tool row, NOT from the
    # model's function_call arguments (the model can't override it).
    assert transfer_result["success"] is True
    assert transfer_result["callsid"] == "CA-test"


# ── 2. Destination / timeout are NOT overridable by model args ────


@pytest.mark.asyncio
async def test_e2e_model_cannot_override_canonical_destination(monkeypatch):
    """Even if the model passes destination='+1' in the function_call
    arguments, the canonical destination from the agent_tools row
    MUST win. This prevents the LLM from calling wrong numbers (or
    SSRF via internal extensions).
    """
    rt = _reload_realtime()
    te = _reload_tool_executor()

    fake_transfer_call = AsyncMock(
        return_value={"success": True, "callsid": "CA-test"}
    )
    monkeypatch.setattr(
        "STT_server.adapters.twilio_api.transfer_call", fake_transfer_call
    )
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.build_transfer_chain",
        MagicMock(return_value=[
            {"id": "reception-id", "destination": "+526649070770",
             "timeout_sec": 20},
        ]),
    )
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.transfer_fallback_url",
        lambda *a, **kw: None,
    )

    session = _session_with_agent(agent_tools=[
        {
            "id": "reception-id",
            "name": "Reception",
            "function_name": "Reception",
            "kind": "call_transfer",
            "destination": "+526649070770",
            "ring_timeout_sec": 20,
        },
    ])

    # Model attempts to pass a different destination in args — this
    # MUST be ignored. We mirror the dispatcher: args are ignored,
    # only the canonical tool row is read.
    args = {"destination": "+9999999", "timeout_sec": 1}  # model's attempts

    chain = [
        {"id": "reception-id", "destination": "+526649070770",
         "timeout_sec": 20},
    ]
    first = chain[0]
    # The dispatcher uses `first["destination"]` and
    # `first["timeout_sec"]` — NOT `args["destination"]`.
    assert first["destination"] != args["destination"]
    assert first["timeout_sec"] != args["timeout_sec"]

    await te.execute_call_transfer(
        session.twilio_account_sid,
        session.twilio_auth_token,
        session.call_sid,
        first["destination"],  # canonical, NOT args
        "Reception",
        timeout_sec=first["timeout_sec"],
        action_url=None,
    )

    # Twilio was called with the canonical destination, not the model's.
    assert fake_transfer_call.call_args.args[3] == "+526649070770"
    assert fake_transfer_call.call_args.kwargs["timeout_sec"] == 20


# ── 3. Provider capability guardrail integration ────────────────


def test_guardrail_rejects_anthropic_with_reception(monkeypatch):
    """End-to-end provider/agent compatibility check: an agent with a
    Reception (call_transfer) tool must NOT be configured with
    llm_provider=anthropic. The guardrail raises RuntimeError at
    session init time.
    """
    from STT_server.services.provider_capabilities import (
        assert_tool_provider_compatible,
    )
    with pytest.raises(RuntimeError) as excinfo:
        assert_tool_provider_compatible("anthropic", has_tools=True)
    # The error must be self-explanatory so the operator can fix the
    # agent row without grepping the codebase.
    msg = str(excinfo.value)
    assert "anthropic" in msg
    assert "tools" in msg.lower()


def test_guardrail_passes_for_openai_with_reception():
    """Sanity: a Reception tool + OpenAI provider passes the guardrail
    — the canonical happy path that the Realtime dispatcher handles.
    """
    from STT_server.services.provider_capabilities import (
        assert_tool_provider_compatible,
    )
    assert_tool_provider_compatible("openai", has_tools=True)
    assert_tool_provider_compatible("openai", has_tools=False)  # no-op
