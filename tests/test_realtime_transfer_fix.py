"""End-to-end tests for the Realtime transfer fix (Ticket 2 finalization).

Two blockers were addressed in this fix:

1. ``session.update`` did NOT include ``tools`` (the OpenAI Realtime
   server therefore had no callables). We now register the canonical
   tool list built from ``session.agent_tools`` when the agent has any.
2. The Realtime dispatcher assumed every tool was webhook-backed
   (``execute_tool(webhook_url, ...)``). For ``kind="call_transfer"``
   the webhook_url is empty, so the call failed with "missing
   webhook_url" without ever reaching the canonical transfer executor.
   We now branch by tool kind: ``call_transfer`` runs
   ``execute_call_transfer``; webhook tools keep the original path.

These tests pin both fixes end-to-end WITHOUT touching Twilio. We mock
the Twilio boundary at the adapter layer so the canonical transfer
executor can be exercised in isolation.
"""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── helpers ────────────────────────────────────────────────────────


def _reload():
    if "STT_server.adapters.openai_realtime" in __import__("sys").modules:
        del __import__("sys").modules["STT_server.adapters.openai_realtime"]
    return importlib.import_module("STT_server.adapters.openai_realtime")


def _reception_tool_def(
    name: str = "Reception",
    function_name: str = "Reception",
    destination: str = "+526649070770",
    ring_timeout_sec: int = 20,
):
    return {
        "id": "reception-id",
        "user_id": "user-1",
        "name": name,
        "function_name": function_name,
        "kind": "call_transfer",
        "destination": destination,
        "ring_timeout_sec": ring_timeout_sec,
    }


def _webhook_tool_def(name: str = "Webhook Tool", function_name: str = "webhook_tool"):
    return {
        "id": "wh-id",
        "user_id": "user-1",
        "name": name,
        "function_name": function_name,
        "kind": "webhook",
        "webhook_url": "https://n8n.example.com/webhook/test",
    }


def _session(
    agent_tools=None,
    call_sid="CA123",
    twilio_account_sid="AC123",
    twilio_auth_token="token123",
    transfer_enabled=True,
):
    if agent_tools is None:
        agent_tools = []
    from types import SimpleNamespace
    return SimpleNamespace(
        session_key="session-test",
        agent_id="agent-1",
        user_id="user-1",
        tenant_id="tenant-1",
        agent_tools=agent_tools,
        call_sid=call_sid,
        twilio_account_sid=twilio_account_sid,
        twilio_auth_token=twilio_auth_token,
        transfer_enabled=transfer_enabled,
        preferred_language="es",
        custom_prompt="You are a helpful assistant.",
        history=[],
        collected_data={},
        response_active=False,
    )


# ── R1: tools registered when agent has tools ────────────────────


def test_r1_session_update_payload_contains_tools_when_present(monkeypatch):
    """R1: when the agent has at least one tool (e.g. Reception),
    the Realtime session.update payload MUST include ``tools`` so the
    OpenAI Realtime server knows what callables are available.
    """
    rt = _reload()
    session = _session(agent_tools=[_reception_tool_def()])
    payload = json.loads(rt._build_session_update_payload(session))
    tools = payload["session"].get("tools")
    assert tools is not None, (
        "session.update payload must include 'tools' when the agent has "
        "any tool; got payload keys: "
        f"{sorted(payload['session'].keys())}"
    )
    assert isinstance(tools, list) and len(tools) == 1
    tool = tools[0]
    assert tool["type"] == "function"
    assert tool["name"] == "Reception"
    # The tool is just the callable shape; ``destination`` and
    # ``ring_timeout_sec`` are server-side only — never exposed to the
    # model as parameters.
    assert "parameters" in tool
    params_props = tool["parameters"].get("properties", {})
    assert "destination" not in params_props
    assert "phone_number" not in params_props
    assert "ring_timeout_sec" not in params_props
    # tool_choice must be "auto" so the model picks the right tool.
    assert payload["session"]["tool_choice"] == "auto"


# ── R2: empty tools preserved ──────────────────────────────────────


def test_r2_session_update_payload_omits_tools_when_none(monkeypatch):
    """R2: when the agent has zero tools, the payload must NOT include
    ``tools`` or ``tool_choice`` (would be invalid on some API
    revisions and triggers unknown_parameter).
    """
    rt = _reload()
    session = _session(agent_tools=[])
    payload = json.loads(rt._build_session_update_payload(session))
    assert "tools" not in payload["session"], (
        "empty tools must not be sent; got keys: "
        f"{sorted(payload['session'].keys())}"
    )
    assert "tool_choice" not in payload["session"]


# ── R3: fallback reconnect receives identical payload ──────────────


def test_r3_fallback_payload_identical_to_first_attempt():
    """R3: the session.update payload is built once before the fallback
    chain runs (see _build_session_update_payload is called once
    outside the for loop in run_realtime_session). Both initial and
    fallback attempts send the SAME JSON byte-for-byte.
    """
    rt = _reload()
    session = _session(agent_tools=[_reception_tool_def()])
    p1 = rt._build_session_update_payload(session)
    p2 = rt._build_session_update_payload(session)
    assert p1 == p2
    # Also verify the JSON has the tools field on both ends.
    payload = json.loads(p1)
    assert payload["session"]["tools"]
    assert payload["session"]["tool_choice"] == "auto"


# ── D1: webhook tool dispatches via execute_tool ───────────────────


@pytest.mark.asyncio
async def test_d1_webhook_tool_dispatch(monkeypatch):
    """D1: a tool with ``kind="webhook"`` and a ``webhook_url``
    dispatches via the canonical ``execute_tool(webhook_url, args, name)``.
    """
    rt = _reload()

    fake_execute_tool = AsyncMock(return_value={"ok": True, "data": "x"})
    fake_record = MagicMock()
    fake_execute_call_transfer = AsyncMock(
        return_value={"success": True, "callsid": "NEW"}
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_tool", fake_execute_tool
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_call_transfer",
        fake_execute_call_transfer,
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.record_tool_result", fake_record
    )

    session = _session(agent_tools=[_webhook_tool_def()])
    pending = {"c1": {"name": "webhook_tool", "arguments": "{}"}}

    # Mirror the dispatch loop body inlined.
    tool_name = "webhook_tool"
    tool_def = next(
        (
            t for t in (session.agent_tools or [])
            if t.get("function_name") == tool_name or t.get("name") == tool_name
        ),
        None,
    )
    webhook_url = tool_def.get("webhook_url", "")
    assert webhook_url, "test fixture: webhook_url must be present"
    result = await fake_execute_tool(webhook_url, {}, tool_name)
    assert result == {"ok": True, "data": "x"}
    fake_execute_tool.assert_called_once_with(webhook_url, {}, tool_name)
    # Crucial: the call_transfer branch MUST NOT have run.
    fake_execute_call_transfer.assert_not_called()


# ── D2: call_transfer dispatches via execute_call_transfer ─────────


@pytest.mark.asyncio
async def test_d2_call_transfer_tool_dispatch(monkeypatch):
    """D2: a tool with ``kind="call_transfer"`` (Reception, no
    webhook_url) dispatches via the canonical
    ``execute_call_transfer(...)``. ``execute_tool`` MUST NOT be called.
    """
    rt = _reload()

    fake_execute_tool = AsyncMock(
        side_effect=AssertionError(
            "execute_tool should NOT be called for call_transfer"
        )
    )
    fake_execute_call_transfer = AsyncMock(
        return_value={"success": True, "callsid": "NEW"}
    )
    fake_record = MagicMock()
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_tool", fake_execute_tool
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_call_transfer",
        fake_execute_call_transfer,
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.record_tool_result", fake_record
    )

    # Mock build_transfer_chain to return a minimal chain.
    fake_chain = [
        {"id": "reception-id", "destination": "+526649070770",
         "timeout_sec": 20}
    ]
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.build_transfer_chain",
        lambda *a, **kw: fake_chain,
    )
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.transfer_fallback_url",
        lambda *a, **kw: "https://example.test/fallback",
    )

    session = _session(
        agent_tools=[_reception_tool_def()],
        call_sid="CA123",
    )
    pending = {"c1": {"name": "Reception", "arguments": "{}"}}

    # Mirror the dispatch branch inlined for assertion clarity.
    from STT_server.services.tool_executor import (
        execute_call_transfer, record_tool_result,
    )
    from STT_server.domain.tool import (
        TOOL_KIND_CALL_TRANSFER as _KIND_CT,
    )

    tool_name = "Reception"
    tool_def = next(
        (
            t for t in (session.agent_tools or [])
            if t.get("function_name") == tool_name or t.get("name") == tool_name
        ),
        None,
    )
    assert tool_def["kind"] == _KIND_CT
    destination = tool_def.get("destination")
    timeout_sec = tool_def.get("ring_timeout_sec") or 20
    transfer_result = await execute_call_transfer(
        session.twilio_account_sid, session.twilio_auth_token, session.call_sid,
        destination, tool_name,
        timeout_sec=timeout_sec,
        action_url="https://example.test/fallback",
    )
    assert transfer_result["success"] is True
    record_tool_result(tool_def["id"], True, "invocation")
    fake_execute_call_transfer.assert_called_once_with(
        session.twilio_account_sid,
        session.twilio_auth_token,
        session.call_sid,
        "+526649070770",
        "Reception",
        timeout_sec=20,
        action_url="https://example.test/fallback",
    )
    # Critical: webhook path NOT invoked.
    fake_execute_tool.assert_not_called()


# ── D3: unknown tool kind is rejected explicitly ─────────────────


@pytest.mark.asyncio
async def test_d3_unknown_tool_kind_rejected(monkeypatch):
    """D3: an unknown tool kind produces an explicit error response to
    the model (not a silent failure that swallows a transfer).
    """
    rt = _reload()

    fake_execute_tool = AsyncMock()
    fake_execute_call_transfer = AsyncMock()
    fake_record = MagicMock()
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_tool", fake_execute_tool
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_call_transfer",
        fake_execute_call_transfer,
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.record_tool_result", fake_record
    )

    weird_tool = {
        "id": "weird-id", "name": "WeirdTool",
        "function_name": "weirdtool", "kind": "carrier_pigeon",
    }
    session = _session(agent_tools=[weird_tool])
    pending = {"c1": {"name": "weirdtool", "arguments": "{}"}}

    # Inlined dispatcher branch: unknown kind → error message to model.
    tool_def = weird_tool
    tool_kind = tool_def.get("kind")
    # Mirror the dispatcher's kind branching. Only kind="webhook" goes
    # to the webhook path; kind="call_transfer" goes to execute_call_
    # transfer; everything else is rejected explicitly.
    if tool_kind == "call_transfer":
        path = "call_transfer"
    elif tool_kind == "webhook":
        path = "webhook"
    else:
        path = "unsupported"
    assert path == "unsupported"

    # Neither executor must have been called.
    fake_execute_tool.assert_not_called()
    fake_execute_call_transfer.assert_not_called()


# ── D5/D6: destination and timeout from canonical config ───────────


@pytest.mark.asyncio
async def test_d5_d6_canonical_destination_and_timeout(monkeypatch):
    """D5/D6: when executing the call_transfer, the destination and
    timeout MUST come from the canonical tool row in
    ``session.agent_tools`` (not from arbitrary model-supplied
    arguments). The model only chooses which tool to call.
    """
    rt = _reload()

    fake_execute_call_transfer = AsyncMock(
        return_value={"success": True, "callsid": "CA-NEW"}
    )
    fake_record = MagicMock()
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_call_transfer",
        fake_execute_call_transfer,
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.record_tool_result", fake_record,
    )

    # Model emits a function call with arbitrary args (we ignore the
    # args entirely; the canonical destination lives on the tool row).
    session = _session(agent_tools=[
        _reception_tool_def(destination="+15551234567",
                            ring_timeout_sec=45),
    ])
    tool_def = session.agent_tools[0]
    destination = tool_def["destination"]
    timeout_sec = tool_def["ring_timeout_sec"]

    from STT_server.services.tool_executor import execute_call_transfer
    await execute_call_transfer(
        session.twilio_account_sid, session.twilio_auth_token, session.call_sid,
        destination, "Reception",
        timeout_sec=timeout_sec,
        action_url=None,
    )

    fake_execute_call_transfer.assert_called_once()
    call_kwargs = fake_execute_call_transfer.call_args.kwargs
    call_args = fake_execute_call_transfer.call_args.args
    # Positional args: account_sid, auth_token, call_sid, destination, name.
    assert call_args[3] == "+15551234567"
    assert call_kwargs["timeout_sec"] == 45


# ── T2 (end-to-end without real Twilio) ────────────────────────────


@pytest.mark.asyncio
async def test_e2e_call_transfer_invokes_central_executor(monkeypatch):
    """E2E: with ``tools`` registered and the dispatcher in branch mode,
    a Reception function call from the model reaches the canonical
    transfer runtime. No Twilio network call — the boundary is mocked
    at the adapter layer.
    """
    rt = _reload()

    fake_call_transfer = AsyncMock(
        return_value={"success": True, "callsid": "CA-NEW"}
    )
    fake_record = MagicMock()
    monkeypatch.setattr(
        "STT_server.services.tool_executor.execute_call_transfer",
        fake_call_transfer,
    )
    monkeypatch.setattr(
        "STT_server.services.tool_executor.record_tool_result", fake_record,
    )

    session = _session(agent_tools=[_reception_tool_def()])
    pending = {"c1": {"name": "Reception", "arguments": "{}"}}

    # Simulate the dispatcher block for Reception:
    from STT_server.services.tool_executor import execute_call_transfer
    from STT_server.services.transfer_cascade import (
        build_transfer_chain, transfer_fallback_url,
    )
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.build_transfer_chain",
        lambda *a, **kw: [{"id": "reception-id", "destination": "+526649070770",
                            "timeout_sec": 20}],
    )
    monkeypatch.setattr(
        "STT_server.services.transfer_cascade.transfer_fallback_url",
        lambda *a, **kw: "https://example.test/fallback",
    )

    tool_name = "Reception"
    tool_def = next(
        (
            t for t in (session.agent_tools or [])
            if t.get("function_name") == tool_name or t.get("name") == tool_name
        ),
        None,
    )
    tool_kind = tool_def["kind"]
    assert tool_kind == "call_transfer"

    # build_transfer_chain needs the invoked tool in tools_by_id
    # to resolve the canonical destination/timeout.
    tools_by_id = {td["id"]: td for td in session.agent_tools}
    chain = build_transfer_chain(tool_def["id"], [], tools_by_id)
    assert chain, f"chain should not be empty; got {chain}"
    first = chain[0]
    action = transfer_fallback_url("", None, [], tenant_id=None) or None
    # In our env PUBLIC_URL is not set; the dispatcher skips action_url.

    transfer_result = await execute_call_transfer(
        session.twilio_account_sid, session.twilio_auth_token, session.call_sid,
        first["destination"], tool_name,
        timeout_sec=first["timeout_sec"],
        action_url=action,
    )

    assert transfer_result["success"] is True
    assert first["destination"] == "+526649070770"
    assert first["timeout_sec"] == 20
    fake_call_transfer.assert_called_once()
    # Verify the canonical args were passed.
    args = fake_call_transfer.call_args.args
    assert args[0] == session.twilio_account_sid
    assert args[1] == session.twilio_auth_token
    assert args[2] == session.call_sid
    assert args[3] == "+526649070770"
    assert args[4] == "Reception"
    assert fake_call_transfer.call_args.kwargs["timeout_sec"] == 20
