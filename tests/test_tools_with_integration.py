"""Tool executor: integration-aware body, forbidden-key strip, URL redaction.

Covers:
  * execute_tool_call builds a body with server-injected
    integration_id / provider / action when the tool has an integration
  * The LLM's `arguments` are stripped of action / provider /
    integration_id / webhook_url / credentials before posting
  * tool without integration still works (legacy path) — action +
    arguments only, no integration_id / provider
  * error messages that bubble up to the caller don't include the
    full webhook URL when the URL is a tokenized generic_webhook path
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_execute_tool_call_injects_server_fields(monkeypatch):
    """Stub the executor and verify the body shape."""
    captured: list[dict] = []

    class FakeExecutor:
        async def execute(self, url, arguments, tool_name):
            captured.append({"url": url, "arguments": arguments, "tool_name": tool_name})
            return {"ok": True}

    # Patch the singleton factory so execute_tool_call uses our fake.
    from STT_server.services import tool_executor as te
    monkeypatch.setattr(te, "get_tool_executor", lambda: FakeExecutor())

    # Set the n8n router URL so the integration's webhook resolution
    # returns a non-empty URL. Without this the executor bails with
    # "no webhook URL" — which is the right behavior for a missing
    # config but not what this test wants to assert.
    monkeypatch.setenv("INTEGRATIONS_N8N_WEBHOOK", "https://n8n.example/webhook/integrations")

    # Patch get_integration so the integration lookup returns our row.
    fake_integration = {
        "id": "int-abc12345",
        "provider": "zendesk",
        "agent_id": "__shared__",
        "configuration": {"subdomain": "acme"},
        "credentials_encrypted": None,
    }
    from STT_server import db_integrations
    monkeypatch.setattr(
        db_integrations, "get_integration",
        lambda integration_id, user_id: fake_integration if integration_id == fake_integration["id"] else None,
    )

    tool = {
        "id": "tool-1234abcd",
        "name": "Find customer",
        "function_name": "find_customer",
        "integration_id": "int-abc12345",
        "action": "find_customer",
        "webhook_url": "https://n8n.example/webhook/legacy-should-not-be-used",
    }

    result = await te.execute_tool_call(
        tool=tool,
        user_id="user-test-001",
        llm_arguments={"email": "caller@example.com", "action": "INJECTED_BY_LLM"},
    )
    assert result == {"ok": True}
    assert len(captured) == 1
    payload = captured[0]["arguments"]
    # Server-injected
    assert payload["integration_id"] == "int-abc12345"
    assert payload["provider"] == "zendesk"
    assert payload["action"] == "find_customer"
    assert payload["tool_name"] == "find_customer"
    # LLM-provided value — NOT the LLM's injected "action" key
    assert payload["arguments"] == {"email": "caller@example.com"}
    assert "action" not in payload["arguments"]


@pytest.mark.asyncio
async def test_execute_tool_call_legacy_path_without_integration(monkeypatch):
    captured: list[dict] = []

    class FakeExecutor:
        async def execute(self, url, arguments, tool_name):
            captured.append({"url": url, "arguments": arguments})
            return {"ok": True}

    from STT_server.services import tool_executor as te
    monkeypatch.setattr(te, "get_tool_executor", lambda: FakeExecutor())

    tool = {
        "id": "tool-legacy01",
        "name": "Legacy webhook",
        "function_name": "legacy_webhook",
        "webhook_url": "https://example.com/webhook",
        # no integration_id, no action
    }
    result = await te.execute_tool_call(
        tool=tool, user_id="u", llm_arguments={"x": 1},
    )
    payload = captured[0]["arguments"]
    # No server-injected integration fields
    assert "integration_id" not in payload
    assert "provider" not in payload
    # tool_name present, arguments present
    assert payload["tool_name"] == "legacy_webhook"
    assert payload["arguments"] == {"x": 1}


@pytest.mark.asyncio
async def test_execute_tool_call_strips_all_forbidden_keys(monkeypatch):
    captured: list[dict] = []

    class FakeExecutor:
        async def execute(self, url, arguments, tool_name):
            captured.append({"arguments": arguments})
            return {}

    from STT_server.services import tool_executor as te
    monkeypatch.setattr(te, "get_tool_executor", lambda: FakeExecutor())

    tool = {
        "id": "tool-1",
        "function_name": "f",
        "webhook_url": "https://example.com/wh",
    }
    await te.execute_tool_call(
        tool=tool, user_id="u",
        llm_arguments={
            "legit_key": "ok",
            "action": "DROP_ME",
            "provider": "DROP_ME",
            "integration_id": "DROP_ME",
            "webhook_url": "DROP_ME",
            "credentials": {"DROP": "ME"},
        },
    )
    payload = captured[0]["arguments"]
    assert payload["arguments"] == {"legit_key": "ok"}
    for forbidden in ("action", "provider", "integration_id", "webhook_url", "credentials"):
        assert forbidden not in payload["arguments"]


def test_redact_url_keeps_host_and_hides_path_token():
    """Generic webhook URLs sometimes embed a token in the path. The
    redaction must keep the host (so the operator can recognise the
    endpoint) and hide the path token."""
    from STT_server.services.tool_executor import _redact_url
    out = _redact_url("https://hooks.example.com/secret/token-abc123/path")
    assert out.startswith("https://hooks.example.com/")
    assert "secret" not in out
    assert "abc123" not in out


def test_redact_url_handles_unparseable():
    from STT_server.services.tool_executor import _redact_url
    assert _redact_url("") == ""
    assert _redact_url("not a url at all") == "****"