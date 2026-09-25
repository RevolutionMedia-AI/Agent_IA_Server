"""Regression: the order-number-asked-twice escalation directive in the
Realtime / LLM provider prompts must reference the operator-named
call_transfer tool (its ``function_name``), NOT a hardcoded
``TRANSFER_AGENT``.

Previously the directive was hardcoded; if the operator named the
human-routing tool anything other than ``TRANSFER_AGENT`` the
model couldn't find a matching callable and answered with text only.
"""

from __future__ import annotations

import importlib

import pytest


def _reload_realtime():
    """Re-import the module so module-level prompt helpers are fresh."""
    if "STT_server.adapters.openai_realtime" in __import__("sys").modules:
        del __import__("sys").modules["STT_server.adapters.openai_realtime"]
    return importlib.import_module("STT_server.adapters.openai_realtime")


def _reload_llm():
    if "STT_server.adapters.openai_llm" in __import__("sys").modules:
        del __import__("sys").modules["STT_server.adapters.openai_llm"]
    return importlib.import_module("STT_server.adapters.openai_llm")


def _session(
    agent_id: str = "agent-1",
    user_id: str = "user-1",
    custom_prompt: str = "You are a helpful assistant.",
    agent_tools: list | None = None,
    history: list | None = None,
):
    """Build a SimpleNamespace that mimics the relevant CallSession fields
    the prompts read (custom_prompt, agent_tools, history)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        session_key="session-test",
        agent_id=agent_id,
        user_id=user_id,
        preferred_language="es",
        custom_prompt=custom_prompt,
        agent_tools=agent_tools or [],
        collected_data={},
        history=history or [],
    )


# ── Realtime provider prompt selection ────────────────────────────────


@pytest.mark.parametrize(
    "tool_name,tool_kind,function_name,name,expect_match",
    [
        # Single call_transfer tool: directive cites its function_name.
        (
            "Reception", "call_transfer", "Reception", "Reception",
            "Reception",
        ),
        # Operator named it differently; function_name is the
        # OpenAI-safe sanitised form.
        (
            "human-routing", "call_transfer", "human_routing",
            "Human Routing!",
            "human_routing",
        ),
        # No call_transfer tool: directive omitted (no false reference).
        ("n/a", None, None, None, None),
        # Multiple call_transfer tools: first match wins.
        (
            "two", "call_transfer", "first_match", "Alpha", "first_match",
        ),
    ],
    ids=["reception", "human-routing-sanitised", "no-call-transfer-tool",
          "multiple-call-transfer"],
)
def test_realtime_order_loop_directive_uses_real_call_transfer_tool(
    tool_name, tool_kind, function_name, name, expect_match
):
    """The order-number-asked-twice escalation in the Realtime prompt
    must cite the operator's actual call_transfer tool. If no such
    tool exists the directive is omitted (no false reference).
    """
    rt = _reload_realtime()
    agent_tools = []
    if tool_kind == "call_transfer":
        agent_tools.append({
            "id": "t1",
            "name": name,
            "function_name": function_name,
            "kind": "call_transfer",
            "destination": "+526649070770",
        })

    session = _session(
        agent_id="agent-1",
        agent_tools=agent_tools,
        history=[
            {"role": "assistant", "content": "Could you give me the order number?"},
            {"role": "assistant", "content": "What is the order number?"},
        ],
    )

    prompt = rt._build_instructions(session)

    if expect_match is None:
        # No call_transfer tool: no escalation directive.
        assert "live agent" not in prompt, (
            "escalation directive must be omitted when no call_transfer "
            f"tool is available (would instruct the model to invoke a "
            f"non-existent callable); got: {prompt}"
        )
    else:
        assert expect_match in prompt, (
            f"escalation directive must cite the operator-named "
            f"call_transfer tool {expect_match!r}; got prompt: {prompt}"
        )
        # Sanity: the previous hardcoded string is gone.
        assert "TRANSFER_AGENT" not in prompt, (
            "hardcoded TRANSFER_AGENT token must not appear in the "
            "prompt — the model searches for the literal callable name"
        )


# ── Central LLM provider prompt selection ────────────────────────────


def _find_system_with_order_loop_directive(messages):
    """Find the system message in build_messages() output that contains
    the order-number escalation."""
    for m in messages:
        if m.get("role") != "system":
            continue
        if "order number" in m.get("content", "") and "live agent" in m.get("content", ""):
            return m["content"]
    return None


@pytest.mark.parametrize(
    "tool_name,tool_kind,function_name,name,expect_match",
    [
        ("Reception", "call_transfer", "Reception", "Reception", "Reception"),
        ("human-routing", "call_transfer", "human_routing",
         "Human Routing!", "human_routing"),
        ("n/a", None, None, None, None),
    ],
    ids=["reception", "human-routing-sanitised", "no-call-transfer-tool"],
)
def test_llm_order_loop_directive_uses_real_call_transfer_tool(
    tool_name, tool_kind, function_name, name, expect_match
):
    """Same fix in openai_llm.build_messages — the order-number
    escalation system message cites the real call_transfer tool, not
    the hardcoded ``TRANSFER_AGENT``.
    """
    llm = _reload_llm()
    agent_tools = []
    if tool_kind == "call_transfer":
        agent_tools.append({
            "id": "t1",
            "name": name,
            "function_name": function_name,
            "kind": "call_transfer",
            "destination": "+526649070770",
        })

    session = _session(
        agent_id="agent-1",
        agent_tools=agent_tools,
        history=[
            {"role": "assistant", "content": "Could you give me the order number?"},
            {"role": "assistant", "content": "What is the order number?"},
        ],
    )

    messages = llm.build_messages(session, "what's next?")

    if expect_match is None:
        assert _find_system_with_order_loop_directive(messages) is None, (
            "no call_transfer tool → order-number escalation system "
            "message must be omitted"
        )
    else:
        content = _find_system_with_order_loop_directive(messages)
        assert content is not None, (
            f"order-number escalation directive must be present when "
            f"a call_transfer tool is available; got messages={messages}"
        )
        assert expect_match in content, (
            f"escalation directive must cite the operator's "
            f"call_transfer tool {expect_match!r}; got: {content}"
        )
        assert "TRANSFER_AGENT" not in content
