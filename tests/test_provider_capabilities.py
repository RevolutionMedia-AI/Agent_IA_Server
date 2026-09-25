"""Tests for STT_server.services.provider_capabilities.

Single source of truth for runtime LLM provider capabilities:

  - supports_tools(provider)         — does the adapter pass tools?
  - supports_realtime(provider)      — has the provider a Realtime path?
  - transfer_path_for(provider)       — which dispatcher fires?
  - assert_tool_provider_compatible() — guardrail for misconfiguration
"""

from __future__ import annotations

import pytest

from STT_server.services.provider_capabilities import (
    _PROVIDER_CAPABILITIES,
    assert_tool_provider_compatible,
    capabilities,
    list_supported_providers,
    supports_realtime,
    supports_tools,
    tool_call_format,
    transfer_path_for,
)


# ── registry shape ─────────────────────────────────────────────


def test_registry_covers_known_providers():
    """Every provider that the LLM adapter can dispatch to has an
    entry in the registry. Adding a new provider without registering
    here would silently get ``supports_tools=False`` and fail the
    guardrail — the operator sees a runtime error instead of a silent
    tool-not-called bug.
    """
    assert set(_PROVIDER_CAPABILITIES.keys()) == {
        "openai", "minimax", "anthropic", "gemini",
    }


def test_list_supported_providers():
    """Sorted convenience for diagnostics / ops dashboards."""
    assert list_supported_providers() == [
        "anthropic", "gemini", "minimax", "openai",
    ]


def test_capabilities_returns_empty_for_unknown_provider():
    """``capabilities`` returns an empty MappingProxyType for unknown
    providers. Empty caps give ``supports_tools=False`` (via the
    public ``supports_tools`` accessor, NOT direct dict access which
    would raise KeyError on an empty proxy).
    """
    caps = capabilities("does-not-exist")
    # Use the public accessors that handle the empty case.
    assert supports_tools("does-not-exist") is False
    assert supports_realtime("does-not-exist") is False
    assert transfer_path_for("does-not-exist") is None
    assert tool_call_format("does-not-exist") is None
    # Direct caps.get returns None on empty proxy (not False); the
    # public functions translate that to the right default.
    assert caps.get("supports_tools") is None
    assert len(caps) == 0


# ── supports_tools ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openai", True),
        ("minimax", True),
        ("anthropic", False),
        ("gemini", False),
        ("unknown", False),
    ],
)
def test_supports_tools(provider, expected):
    assert supports_tools(provider) is expected


# ── supports_realtime ───────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openai", True),
        ("minimax", False),
        ("anthropic", False),
        ("gemini", False),
    ],
)
def test_supports_realtime(provider, expected):
    assert supports_realtime(provider) is expected


# ── transfer_path_for ──────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openai", "realtime"),
        ("minimax", "central"),
        ("anthropic", None),
        ("gemini", None),
    ],
)
def test_transfer_path_for(provider, expected):
    assert transfer_path_for(provider) == expected


# ── tool_call_format ───────────────────────────────────────────


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("openai", "flat"),
        ("minimax", "flat"),
        ("anthropic", "anthropic"),
        ("gemini", "gemini"),
    ],
)
def test_tool_call_format(provider, expected):
    assert tool_call_format(provider) == expected


# ── assert_tool_provider_compatible guardrail ─────────────────


def test_guardrail_noop_when_no_tools():
    """No tools -> no compatibility check. Empty agent is fine for
    any provider, even ones without tool calling.
    """
    # Must not raise for any provider when there are no tools.
    for provider in ["openai", "anthropic", "gemini", "unknown"]:
        assert_tool_provider_compatible(provider, has_tools=False)


def test_guardrail_passes_when_provider_supports_tools():
    """Tools + provider with tool calling -> no error."""
    assert_tool_provider_compatible("openai", has_tools=True)
    assert_tool_provider_compatible("minimax", has_tools=True)


def test_guardrail_raises_when_tools_and_provider_lacks_tools():
    """The original 2026-09-22 production bug:
    agent has Reception (call_transfer), but llm_provider=anthropic.
    The Anthropic adapter doesn't send tools to the Messages API, so
    the model never sees Reception, never emits a tool_call, never
    reaches execute_call_transfer. Caller's job: fix the agent
    config (switch provider or remove tools).
    """
    with pytest.raises(RuntimeError) as excinfo:
        assert_tool_provider_compatible("anthropic", has_tools=True)
    msg = str(excinfo.value)
    # The error message MUST include the provider id so the operator
    # can grep logs and find the offending agent row.
    assert "anthropic" in msg
    assert "Reception" not in msg  # provider error, not tool-specific
    # And MUST explain the fix path (switch provider or remove tools).
    assert "llm_provider" in msg or "remove tools" in msg


def test_guardrail_raises_for_unknown_provider_with_tools():
    """If the operator set a typo'd llm_provider (e.g. 'anthropicc'),
    the registry returns empty caps -> supports_tools=False ->
    guardrail raises. We do NOT silently fall back to a different
    provider — silent degradation was the original bug.
    """
    with pytest.raises(RuntimeError):
        assert_tool_provider_compatible("anthropicc", has_tools=True)


def test_guardrail_does_not_swallow_misconfiguration():
    """The RuntimeError must propagate. If we accidentally swallowed
    it, the misconfiguration would silently fall through to the
    model — same root cause as the original incident.
    """
    try:
        assert_tool_provider_compatible("gemini", has_tools=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "guardrail must raise on gemini + tools (gemini lacks tools)"
        )
