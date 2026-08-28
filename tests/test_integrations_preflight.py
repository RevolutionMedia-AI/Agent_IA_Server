"""Preflight endpoint behaviour.

We replace run_integration_test to avoid the network, then exercise:
  * valid creds + stub returning True → valid:true
  * valid creds + stub returning False → valid:false
  * invalid field shape → 422 before preflight runs
  * provider with test_fn=None → valid:false with "Test not yet implemented"
"""
from __future__ import annotations

import pytest


@pytest.fixture
def stub_runner(monkeypatch):
    """Inject a controllable runner. The fixture returns the list so
    tests can assert calls + customise the return per-test."""
    state = {"return": (True, "ok"), "calls": []}

    def fake(test_fn_path: str, config: dict, creds: dict) -> tuple[bool, str]:
        state["calls"].append((test_fn_path, config, creds))
        return state["return"]

    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake)
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake, raising=False)
    return state


async def test_preflight_success(client, auth_token, stub_runner):
    resp = await client.post(
        "/integrations/preflight",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "configuration": {"subdomain": "acme"},
            "credentials": {
                "email": "admin@acme.com",
                "api_token": "a" * 25,
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["connection_status"] == "connected"
    assert len(stub_runner["calls"]) == 1


async def test_preflight_failure(client, auth_token, stub_runner):
    stub_runner["return"] = (False, "401 unauthorized")
    resp = await client.post(
        "/integrations/preflight",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "configuration": {"subdomain": "acme"},
            "credentials": {
                "email": "admin@acme.com",
                "api_token": "a" * 25,
            },
        },
    )
    body = resp.json()
    assert body["valid"] is False
    assert body["message"] == "401 unauthorized"
    assert body["connection_status"] == "failed"


async def test_preflight_invalid_fields_rejected(client, auth_token, stub_runner):
    """Bad email — the preflight endpoint returns 200 with valid:false
    (it's a check endpoint, not a 4xx contract). The stub never ran
    because validation caught the bad shape first."""
    resp = await client.post(
        "/integrations/preflight",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "configuration": {"subdomain": "acme"},
            "credentials": {
                "email": "not-an-email",
                "api_token": "a" * 25,
            },
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert "email" in body["message"].lower() or "format" in body["message"].lower()
    # The stub never ran — validation caught the bad shape first
    assert stub_runner["calls"] == []


async def test_preflight_unknown_provider(client, auth_token, stub_runner):
    resp = await client.post(
        "/integrations/preflight",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "made_up",
            "configuration": {},
            "credentials": {},
        },
    )
    body = resp.json()
    assert body["valid"] is False
    assert "Unknown provider" in body["message"]


async def test_preflight_provider_without_test_fn(client, auth_token, stub_runner):
    # Salesforce has test_fn=None → returns "Test not yet implemented"
    resp = await client.post(
        "/integrations/preflight",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "salesforce",
            "configuration": {"instance_url": "https://acme.my.salesforce.com"},
            "credentials": {"access_token": "a" * 25},
        },
    )
    body = resp.json()
    assert body["valid"] is False
    assert "not yet implemented" in body["message"]
    # stub never ran (test_fn was None)
    assert stub_runner["calls"] == []


async def test_create_integration_blocked_by_preflight_failure(client, auth_token, stub_runner):
    """POST /integrations runs the preflight internally and 422s on
    failure so we never persist a connection that can't reach the
    provider."""
    stub_runner["return"] = (False, "HTTP 401")
    resp = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "acme"},
            "credentials": {"email": "admin@acme.com", "api_token": "a" * 25},
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "preflight" in detail
    assert detail["preflight"]["valid"] is False