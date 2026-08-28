"""Internal credential endpoint for n8n.

The endpoint requires the INTEGRATIONS_N8N_TOKEN service token and is
the only way to fetch plaintext credentials. Even the integration's
owner can NOT use their user token here — they can only Replace via
PUT /integrations/{id}.

Covers:
  * no token          → 401
  * wrong token       → 401
  * right token       → 200 with the decrypted credentials
  * unknown id        → 404
  * missing env token → 503 (fail closed; never accidentally work)
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import pytest


@pytest.fixture
def service_token(monkeypatch) -> str:
    """Set INTEGRATIONS_N8N_TOKEN for this test, return the value."""
    tok = secrets.token_urlsafe(32)
    monkeypatch.setenv("INTEGRATIONS_N8N_TOKEN", tok)
    return tok


@pytest.fixture
def fake_test_fn(monkeypatch):
    """Skip live preflight on the integration create call."""
    def fake(test_fn_path, config, creds):
        return True, "fake ok"
    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake)
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake, raising=False)


async def _seed_zendesk(client, headers) -> str:
    resp = await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "acme"},
            "credentials": {
                "email": "admin@acme.com",
                "api_token": "abcdef0123456789abcdef0123456789",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


async def test_internal_endpoint_requires_token(client, auth_token, fake_test_fn, service_token):
    """No Authorization header at all → 401.

    ponytail: the test must request the service_token fixture so the
    server has INTEGRATIONS_N8N_TOKEN configured; otherwise the env
    check fires first and returns 503 (fail-closed). With the env
    set, missing-header is a 401.
    """
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_zendesk(client, headers)
    resp = await client.post(f"/internal/integrations/{iid}/credentials")
    assert resp.status_code == 401


async def test_internal_endpoint_rejects_wrong_token(client, auth_token, fake_test_fn, service_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_zendesk(client, headers)
    resp = await client.post(
        f"/internal/integrations/{iid}/credentials",
        headers={"Authorization": "Bearer not-the-right-token"},
    )
    assert resp.status_code == 401


async def test_internal_endpoint_rejects_user_token(client, auth_token, fake_test_fn, service_token):
    """Even the integration's owner can NOT use their user bearer to
    pull credentials. This is the core of the no-reveal contract."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_zendesk(client, headers)
    # The user token is well-formed but it's NOT the service token.
    resp = await client.post(
        f"/internal/integrations/{iid}/credentials",
        headers=headers,
    )
    assert resp.status_code == 401


async def test_internal_endpoint_returns_decrypted_with_service_token(
    client, auth_token, fake_test_fn, service_token,
):
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed_zendesk(client, headers)
    resp = await client.post(
        f"/internal/integrations/{iid}/credentials",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["integration_id"] == iid
    assert body["provider"] == "zendesk"
    assert body["configuration"]["subdomain"] == "acme"
    assert body["credentials"]["email"] == "admin@acme.com"
    assert body["credentials"]["api_token"] == "abcdef0123456789abcdef0123456789"


async def test_internal_endpoint_404_on_unknown_id(
    client, auth_token, fake_test_fn, service_token,
):
    resp = await client.post(
        "/internal/integrations/int-nonexistent/credentials",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert resp.status_code == 404


async def test_internal_endpoint_fails_closed_when_env_token_missing(
    client, auth_token, fake_test_fn, monkeypatch,
):
    """No INTEGRATIONS_N8N_TOKEN configured → 503. We never let an
    unconfigured env silently accept calls (e.g. empty == empty)."""
    monkeypatch.delenv("INTEGRATIONS_N8N_TOKEN", raising=False)
    resp = await client.post(
        "/internal/integrations/int-anything/credentials",
        headers={"Authorization": "Bearer some-token"},
    )
    assert resp.status_code == 503
    assert "INTEGRATIONS_N8N_TOKEN" in resp.json()["detail"]