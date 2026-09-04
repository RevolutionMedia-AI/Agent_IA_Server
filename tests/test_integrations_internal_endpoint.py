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


async def _seed_provider(client, headers, provider: str, credentials: dict) -> str:
    """Seed an OAuth integration row through the SAME persistence path
    the OAuth callback uses (encrypt_credentials + _write_integrations_file).
    We patch decrypt_credentials to identity-return so the test only
    covers the credentials-for-n8n FILTER (not the encryption layer).
    """
    from STT_server.db_integrations import (
        _read_integrations_file, _write_integrations_file,
    )
    from STT_server.security.credentials import encrypt_credentials

    integ_id = f"int-{provider}-test"
    user_id = "user-test-001"
    now = "2099-01-01T00:00:00Z"
    config = (
        {"calendar_id": "ops@example.com", "timezone": "America/Tijuana"}
        if provider == "google_calendar"
        else {"instance_url": "https://example.my.salesforce.com"}
    )
    encrypted_blob = encrypt_credentials(credentials)
    rows = _read_integrations_file()
    rows = [
        r for r in rows
        if not (r.get("id") == integ_id and r.get("user_id") == user_id)
    ]
    rows.append({
        "id": integ_id,
        "user_id": user_id,
        "provider": provider,
        "name": f"Test {provider}",
        "configuration": config,
        "credentials_encrypted": encrypted_blob,
        "credentials_cipher": "fernet-v1",
        "connection_status": "connected",
        "scope": credentials.get("scope", ""),
        "assignments": [],
        "created_at": now,
        "updated_at": now,
    })
    _write_integrations_file(rows)

    # ponytail: the FILTER under test is _credentials_for_n8n, which
    # runs AFTER decrypt. Identity-return keeps the test focused on
    # the filter (what reaches n8n) rather than the Fernet layer.
    from STT_server.routes import api as api_mod
    api_mod.decrypt_credentials = lambda blob: credentials
    return integ_id


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


async def test_internal_endpoint_google_calendar_returns_access_token(
    client, auth_token, fake_test_fn, service_token, monkeypatch,
):
    """Regression 2026-09-04: the BE shipped a credentials-for-n8n
    filter that hard-coded ``provider == "salesforce"``. Every other
    provider (Google Calendar among them) hit the empty ``{}`` branch
    and n8n's calendar workflow had nothing to authenticate with.

    This test asserts that the bearer reaches n8n exactly the way the
    BE advertises — config + bearer only, refresh_token stays BE-side.
    """
    headers = {"Authorization": f"Bearer {auth_token}"}
    integ_id = await _seed_provider(client, headers, "google_calendar", {
        "access_token": "GOOGLE_BEARER",
        "refresh_token": "GOOGLE_REFRESH_SHOULD_NOT_LEAK",
        "expires_at": "2099-01-01T00:00:00Z",
        "scope": "openid email calendar.events",
    })

    from STT_server.services import oauth_providers
    monkeypatch.setattr(oauth_providers, "is_token_expiring", lambda *_a, **_kw: False)

    resp = await client.post(
        f"/internal/integrations/{integ_id}/credentials",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "google_calendar"
    assert body["credentials"] == {"access_token": "GOOGLE_BEARER"}, (
        "Google Calendar must hit the SAME filter branch as Salesforce "
        "(bearer only — refresh_token stays BE-side)"
    )
    assert "refresh_token" not in body["credentials"]
    assert body["configuration"]["calendar_id"] == "ops@example.com"
    assert body["configuration"]["timezone"] == "America/Tijuana"


@pytest.mark.parametrize("provider,bearer_key", [
    ("salesforce", "SF_BEARER"),
    ("google_calendar", "GOOGLE_BEARER"),
])
async def test_internal_endpoint_filters_only_access_token(
    client, auth_token, fake_test_fn, service_token, monkeypatch,
    provider, bearer_key,
):
    """White-box version of the credentials-for-n8n contract: BOTH
    Salesforce AND Google Calendar expose ONLY ``access_token``. Any
    other key (``refresh_token``, ``expires_at``, ``scope``, raw
    provider blob) is dropped at the BE→n8n boundary.

    Parametrised so a regression in one provider is caught by the
    other (the filter looks the same on both branches today).
    """
    headers = {"Authorization": f"Bearer {auth_token}"}
    integ_id = await _seed_provider(client, headers, provider, {
        "access_token": bearer_key,
        "refresh_token": f"{provider}_REFRESH_SHOULD_NOT_LEAK",
        "expires_at": "2099-01-01T00:00:00Z",
        "scope": "calendar.events",
        "client_secret": "DO_NOT_LEAK",
    })
    from STT_server.services import oauth_providers
    monkeypatch.setattr(oauth_providers, "is_token_expiring", lambda *_a, **_kw: False)

    resp = await client.post(
        f"/internal/integrations/{integ_id}/credentials",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["credentials"] == {"access_token": bearer_key}, (
        f"provider={provider} returned unexpected shape: {body['credentials']}"
    )