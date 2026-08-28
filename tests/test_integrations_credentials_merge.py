"""Credentials merge semantics on PUT /integrations/{id}.

Contract (from the user's plan):
  * missing field   → keep existing
  * "" (empty)      → keep existing (NOT clear; clearing needs an
                       explicit revoke op we haven't built)
  * non-empty value → replace

The wire endpoint hides credentials, so we exercise the merge by
re-fetching via the internal endpoint with the service token and
asserting the plaintext is what we expect.
"""
from __future__ import annotations

import secrets

import pytest


@pytest.fixture
def fake_test_fn(monkeypatch):
    def fake(test_fn_path, config, creds):
        return True, "ok"
    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake)
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake, raising=False)


@pytest.fixture
def service_token(monkeypatch) -> str:
    tok = secrets.token_urlsafe(32)
    monkeypatch.setenv("INTEGRATIONS_N8N_TOKEN", tok)
    return tok


async def _seed(client, headers) -> str:
    resp = await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "acme"},
            "credentials": {
                "email": "admin@acme.com",
                "api_token": "ORIGINAL_TOKEN_xxxxxxxxxxxxxxxxxxxx",
            },
        },
    )
    assert resp.status_code == 200
    return resp.json()["id"]


async def _internal_get(client, iid, service_token) -> dict:
    resp = await client.post(
        f"/internal/integrations/{iid}/credentials",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert resp.status_code == 200
    return resp.json()


async def test_empty_string_keeps_existing_value(client, auth_token, fake_test_fn, service_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed(client, headers)
    # Empty email, no api_token at all (missing) → both keep
    resp = await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"credentials": {"email": ""}},
    )
    assert resp.status_code == 200
    body = await _internal_get(client, iid, service_token)
    assert body["credentials"]["email"] == "admin@acme.com"
    assert body["credentials"]["api_token"] == "ORIGINAL_TOKEN_xxxxxxxxxxxxxxxxxxxx"


async def test_missing_field_keeps_existing_value(client, auth_token, fake_test_fn, service_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed(client, headers)
    # Only send api_token — email should keep its original
    resp = await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"credentials": {"api_token": "NEW_TOKEN_yyyyyyyyyyyyyyyyyyyy"}},
    )
    assert resp.status_code == 200
    body = await _internal_get(client, iid, service_token)
    assert body["credentials"]["email"] == "admin@acme.com"  # kept
    assert body["credentials"]["api_token"] == "NEW_TOKEN_yyyyyyyyyyyyyyyyyyyy"


async def test_non_empty_string_replaces(client, auth_token, fake_test_fn, service_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed(client, headers)
    resp = await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"credentials": {
            "email": "new-admin@acme.com",
            "api_token": "REPLACED_TOKEN_zzzzzzzzzzzzzzzzzzzz",
        }},
    )
    assert resp.status_code == 200
    body = await _internal_get(client, iid, service_token)
    assert body["credentials"]["email"] == "new-admin@acme.com"
    assert body["credentials"]["api_token"] == "REPLACED_TOKEN_zzzzzzzzzzzzzzzzzzzz"


async def test_configuration_replace_full_dict(client, auth_token, fake_test_fn, service_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    iid = await _seed(client, headers)
    # configuration is REPLACE not merge — the FE re-sends the full
    # object on every save.
    resp = await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"configuration": {"subdomain": "new-subdomain"}},
    )
    assert resp.status_code == 200
    body = await _internal_get(client, iid, service_token)
    assert body["configuration"]["subdomain"] == "new-subdomain"
    # credentials untouched
    assert body["credentials"]["email"] == "admin@acme.com"