"""Single source of truth for runtime LLM provider capabilities.

The BE has two LLM call paths:

  1. The Realtime WebSocket path (``STT_server.adapters.openai_realtime``)
     which streams audio to OpenAI's Realtime API and lets the model
     emit function_call events.

  2. The streaming chat.completions path
     (``STT_server.adapters.openai_llm.stream_llm_reply_sync``) which
     the central pipeline (``process_transcripts``) uses for non-
     realtime sessions.

Function calling / tools support is provider-specific:

  - openai Realtime: tools (flat shape).
  - openai / MiniMax chat.completions: tools (nested shape).
  - Anthropic Messages API: supports tools (their own shape) — NOT
    wired up yet.
  - Gemini generateContent: supports tools (their own shape) — NOT
    wired up yet.

Before this module existed, the LLM adapter hardcoded which
providers could carry tools in their call bodies
(``_anthropic_stream_sync`` / ``_gemini_stream_sync`` silently dropped
them). Operators who set ``llm_provider="anthropic"`` on an agent
configured with a ``call_transfer`` tool could see the LLM appear to
ignore the tool — the model never saw it.

This registry is the single source of truth for:

  1. ``supports_tools(provider) -> bool`` — does this provider's call
     path actually pass tools to the model? Used by the guardrail.
  2. ``tool_call_format(provider) -> str`` — the schema shape the
     provider expects (``flat`` for Realtime / ``nested`` for
     chat.completions). Future use; not wired up to schema
     translation yet.
  3. ``supports_realtime(provider) -> bool`` — does this provider
     have a Realtime path? Only OpenAI today.
  4. ``transfer_path_for(provider) -> str`` — which dispatcher fires
     for call_transfer tools on this provider path
     (``realtime`` / ``central``). Both paths ultimately call
     ``execute_call_transfer``; the difference is where the tool_call
     event is parsed.

The guardrail ``assert_tool_provider_compatible(provider, has_tools)``
lives in ``session_runtime`` and refuses to start a call with
tools=True + provider without function calling support — that
configuration can never reach a tool_call and the operator was
debugging "AI says it will transfer but does not".

If a future provider needs different wiring (e.g. Anthropic's
``input_schema`` instead of ``parameters``), edit ONLY this module.
Both adapters (``openai_realtime`` and ``openai_llm``) and the
session_runtime guardrail read from here.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


# Provider id -> capabilities. Keys mirror ``session.llm_provider``
# values used in ``openai_llm._session_provider``.
#
# Notes on additions:
# - ``tool_call_format`` is informational today (no shape translation).
#   ``flat`` matches OpenAI Realtime and Tools API.
#   ``nested`` matches chat.completions ``{"type": "function",
#   "function": {...}}``.
# - ``transfer_path_for`` selects which dispatcher emits the
#   tool_call. Both paths converge on
#   ``STT_server.services.tool_executor.execute_call_transfer``.
_PROVIDER_CAPABILITIES: dict[str, dict] = {
    "openai": {
        "supports_tools": True,
        "tool_call_format": "flat",          # Realtime + chat-completions both OK
        "supports_realtime": True,
        "transfer_path_for": "realtime",     # Realtime dispatcher fires
    },
    "minimax": {
        "supports_tools": True,
        "tool_call_format": "flat",          # OpenAI-compatible API
        "supports_realtime": False,
        "transfer_path_for": "central",      # chat-completions path via turn_manager
    },
    "anthropic": {
        "supports_tools": False,             # NOT wired up in this adapter
        "tool_call_format": "anthropic",     # Messages API supports tools
        "supports_realtime": False,
        "transfer_path_for": None,           # no working transfer path
    },
    "gemini": {
        "supports_tools": False,             # NOT wired up in this adapter
        "tool_call_format": "gemini",
        "supports_realtime": False,
        "transfer_path_for": None,
    },
}


def capabilities(provider: str) -> Mapping[str, object]:
    """Read-only view of the capability dict for ``provider``.

    Returns an empty MappingProxyType for unknown providers so callers
    can safely ``.get()`` without catching AttributeError. The empty
    view reports ``supports_tools=False`` so the guardrail refuses
    to start a call with tools=True on an unknown provider.
    """
    return MappingProxyType(_PROVIDER_CAPABILITIES.get(provider) or {})


def supports_tools(provider: str) -> bool:
    """True iff ``provider``'s adapter actually passes ``tools`` to
    the model API. False for both unknown providers and providers
    whose adapter has not been wired up (anthropic, gemini).
    """
    return bool(capabilities(provider).get("supports_tools", False))


def supports_realtime(provider: str) -> bool:
    """True iff ``provider`` has a Realtime audio-in/audio-out path.
    Only OpenAI today; MiniMax, Anthropic, Gemini do not.
    """
    return bool(capabilities(provider).get("supports_realtime", False))


def transfer_path_for(provider: str) -> str | None:
    """Which path fires ``execute_call_transfer`` for this provider:

      - ``"realtime"`` — openai_realtime dispatcher handles
        ``response.function_call_arguments.done`` events.
      - ``"central"`` — turn_manager's streaming-LLM dispatcher
        handles ``delta.tool_calls`` events.
      - ``None`` — provider has no working transfer path (tools never
        reach a transfer-capable dispatcher).
    """
    val = capabilities(provider).get("transfer_path_for")
    return val if isinstance(val, str) else None


def tool_call_format(provider: str) -> str | None:
    """Schema shape the provider expects for the ``tools`` field:
    ``flat`` (Realtime), ``nested`` (chat.completions),
    ``anthropic``, ``gemini``, or ``None`` for unsupported.
    """
    val = capabilities(provider).get("tool_call_format")
    return val if isinstance(val, str) else None


def assert_tool_provider_compatible(provider: str, has_tools: bool) -> None:
    """Guardrail: refuse to start a session whose tools and provider
    can't meet. Tools without a tool-calling path produce the
    "AI says it will transfer but does not" bug — the model never sees
    the callable.

    Raises ``RuntimeError`` on incompatible configurations. Caller is
    responsible for catching and surfacing as a 4xx/5xx with a clear
    error to the operator. The check is best-effort: if the provider is
    unknown (``supports_tools=False``) and the agent has no tools,
    this is a no-op.
    """
    if not has_tools:
        return
    if supports_tools(provider):
        return
    # Tools exist but the provider's adapter can't pass them. The
    # operator MUST either switch provider or remove tools. We do
    # NOT auto-fallback to a different provider — silently masking
    # the mismatch caused the original 2026-09-22 production bug
    # where Reception was configured but never reached the model.
    raise RuntimeError(
        f"agent has tools configured but llm_provider={provider!r} does "
        f"not support tool calling. Tools will be invisible to the model. "
        f"Either remove tools from the agent, or change the agent's "
        f"llm_provider to one with tools support "
        f"(currently: openai, minimax). provider capabilities: "
        f"{dict(capabilities(provider))}"
    )


def list_supported_providers() -> list[str]:
    """Convenience for diagnostics / tests."""
    return sorted(_PROVIDER_CAPABILITIES.keys())
