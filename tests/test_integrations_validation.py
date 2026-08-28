"""Shared/private matrix + action validation when creating a tool that
references an integration.

Per the agreed contract:
  Private tool → Shared integration:        OK
  Private tool → Same private integration:  OK
  Private tool → Other agent's private:     422
  Shared tool  → Shared integration:        OK
  Shared tool  → Private integration:       422

Action validation:
  * Action id must match ^[a-z0-9_]+$
  * Action must be a registered action for the provider's catalog
  * generic_webhook accepts any well-formed id
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_test_fn(monkeypatch):
    def fake(test_fn_path, config, creds):
        return True, "ok"
    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake)
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake, raising=False)


async def _seed_integration(client, headers, agent_id: str = "__shared__") -> str:
    resp = await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "zendesk",
            "name": "Acme",
            "agent_id": agent_id,
            "configuration": {"subdomain": "acme"},
            "credentials": {"email": "admin@acme.com", "api_token": "a" * 25},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


async def test_private_tool_uses_shared_integration(client, auth_token, fake_test_fn):
    """agent-123 tool binding to __shared__ integration: ok."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_integration(client, headers, agent_id="__shared__")
    resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "Find customer",
            "description": "Lookup",
            "kind": "webhook",
            "integration_id": iid,
            "action": "find_customer",
            "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["integration_id"] == iid
    assert body["action"] == "find_customer"


async def test_private_tool_uses_own_private_integration(client, auth_token, fake_test_fn):
    """agent-123 tool binding to its own private integration: ok."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_integration(client, headers, agent_id="agent-123")
    resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "Find customer",
            "description": "Lookup",
            "kind": "webhook",
            "integration_id": iid,
            "action": "find_customer",
            "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
        },
    )
    assert resp.status_code == 200


async def test_private_tool_blocked_from_other_agent_integration(client, auth_token, fake_test_fn):
    """agent-X tool binding to agent-Y's private integration: 422."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_integration(client, headers, agent_id="agent-Y")
    resp = await client.post(
        "/agents/agent-X/tools",
        headers=headers,
        json={
            "name": "Find customer",
            "description": "Lookup",
            "kind": "webhook",
            "integration_id": iid,
            "action": "find_customer",
            "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
        },
    )
    assert resp.status_code == 422
    assert "scoped to agent" in resp.json()["detail"]


async def test_shared_tool_blocked_from_private_integration(client, auth_token, fake_test_fn):
    """Shared tool binding to a private integration: 422. The matrix
    only goes one direction (shared → private is rejected)."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_integration(client, headers, agent_id="agent-123")
    resp = await client.post(
        "/tools",
        headers=headers,
        json={
            "name": "Find customer",
            "description": "Lookup",
            "kind": "webhook",
            "integration_id": iid,
            "action": "find_customer",
            "parameters": {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
        },
    )
    assert resp.status_code == 422
    assert "scoped to agent" in resp.json()["detail"]


async def test_action_must_be_registered_for_provider(client, auth_token, fake_test_fn):
    """find_foo is not a Zendesk action: 422."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_integration(client, headers)
    resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "Find foo",
            "description": "Lookup",
            "kind": "webhook",
            "integration_id": iid,
            "action": "find_foo",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "find_foo" in detail
    assert "Allowed" in detail


async def test_action_must_match_id_pattern(client, auth_token, fake_test_fn):
    """Action id with spaces / dashes fails the AgentTool-level regex
    check (400) — catches it before the catalog lookup runs."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_integration(client, headers)
    resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "Bad",
            "description": "X",
            "kind": "webhook",
            "integration_id": iid,
            "action": "find customer",  # has a space
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    # AgentTool.validate runs first (400). The catalog check (422) only
    # fires for well-formed ids that aren't in the provider's action
    # list — see test_action_must_be_registered_for_provider above.
    assert resp.status_code == 400
    assert "action" in resp.json()["detail"]


async def test_generic_webhook_accepts_any_well_formed_action(client, auth_token, fake_test_fn):
    """generic_webhook has no fixed actions — the operator defines the
    action per tool. Just needs to match the regex."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    resp = await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "generic_webhook",
            "name": "My Hook",
            "configuration": {"webhook_url": "https://example.com/hook"},
            "credentials": {},
        },
    )
    assert resp.status_code == 200
    iid = resp.json()["id"]
    tool_resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "Whatever",
            "description": "X",
            "kind": "webhook",
            "integration_id": iid,
            "action": "my_custom_action",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    assert tool_resp.status_code == 200


async def test_call_transfer_cannot_reference_integration(client, auth_token):
    """kind=call_transfer is the platform's call-redirect action. It
    has nothing to do with a third-party integration."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "Transfer",
            "description": "X",
            "kind": "call_transfer",
            "destination": "+15071234567",
            "integration_id": "int-anything",
        },
    )
    assert resp.status_code == 400
    assert "call_transfer tools cannot reference an integration" in resp.json()["detail"]


async def test_integration_not_found_returns_404(client, auth_token, fake_test_fn):
    headers = {"Authorization": f"Bearer {auth_token}"}
    resp = await client.post(
        "/agents/agent-123/tools",
        headers=headers,
        json={
            "name": "X",
            "description": "X",
            "kind": "webhook",
            "integration_id": "int-nonexistent",
            "action": "find_customer",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    )
    assert resp.status_code == 404


async def test_admin_skip_preflight_requires_admin(monkeypatch, client, auth_token, fake_test_fn):
    """skip_preflight is admin-only. Without ADMIN_USER_IDS, the
    field is rejected with 403 even though we have a valid user
    token."""
    # skip_preflight is a body field on IntegrationCreate. Sending
    # it as part of a create call triggers require_admin() inside the
    # route. With no admin configured (the default), this 403s.
    resp = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "acme"},
            "credentials": {"email": "admin@acme.com", "api_token": "a" * 25},
            "skip_preflight": True,
        },
    )
    assert resp.status_code == 403
    assert "admin" in resp.json()["detail"]


async def test_admin_skip_preflight_accepted_when_admin_set(monkeypatch, client, auth_token, fake_test_fn):
    monkeypatch.setenv("ADMIN_USER_IDS", "user-test-001")
    # Re-import the dep so it reads the fresh env var.
    import importlib
    from STT_server.routes import api as api_mod
    importlib.reload(api_mod)
    # Now a create call with skip_preflight passes even though the
    # fake runner is monkeypatched to return True (we'd never know
    # either way; the point is the gate passed).
    resp = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "acme"},
            "credentials": {"email": "admin@acme.com", "api_token": "a" * 25},
            "skip_preflight": True,
        },
    )
    assert resp.status_code == 200