"""Dynamics 365 CONNECTION plumbing: OAuth, tenant discovery, URL handling.

Scope change. This file held 31 test functions covering the Dataverse
executor -- the action dispatcher, the CRUD mappers and the whole Field
Service block. None of that exists any more: the backend stores the OAuth
connection and hands the bearer to n8n, and n8n calls Dataverse itself.

What survives is the half that is still ours, extracted verbatim:
  * the OAuth authorize URL and its multitenant authority
  * environment URL normalisation and validation
  * instance discovery and the tenant-specific token exchange
  * rejecting a consumed / expired oauth_state

The executor tests went with the executor, as did
tests/test_integrations_executor.py, whose subject was the dispatcher
registry.
"""
import json
from urllib.parse import parse_qs, urlparse

import pytest


GUID = "11111111-1111-1111-1111-111111111111"
ENV_URL = "https://contoso.api.crm.dynamics.com"


@pytest.fixture(autouse=True)
def dynamics_env(monkeypatch, data_dir):
    monkeypatch.setenv("DYNAMICS365_CLIENT_ID", "dynamics-client")
    monkeypatch.setenv("DYNAMICS365_CLIENT_SECRET", "dynamics-secret")
    monkeypatch.setenv(
        "DYNAMICS365_REDIRECT_URI",
        "https://backend.test/integrations/dynamics365/oauth/callback",
    )
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://frontend.test")
    from STT_server import db_integrations
    from STT_server.services import oauth_providers
    monkeypatch.setattr(db_integrations, "_INTEGRATIONS_FILE", data_dir / "integrations.json")
    oauth_providers._OAUTH_PROVIDERS.clear()
    yield
    oauth_providers._OAUTH_PROVIDERS.clear()


class Response:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.status = 200
        self.headers = headers or {}

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_oauth_provider_builds_multitenant_authorization_url():
    from STT_server.services.oauth_providers import build_authorize_url, get_oauth_config
    url = build_authorize_url(get_oauth_config("dynamics365"), "state", code_verifier="v" * 64)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.netloc == "login.microsoftonline.com"
    assert "/organizations/oauth2/v2.0/authorize" in parsed.path
    assert query["client_id"] == ["dynamics-client"]
    assert query["state"] == ["state"]
    assert "offline_access" in query["scope"][0]
    assert query["code_challenge_method"] == ["S256"]


def test_environment_url_validation():
    from STT_server.services.dynamics365 import normalize_environment_url
    assert normalize_environment_url(f"{ENV_URL}/api/data/v9.2/") == ENV_URL
    with pytest.raises(ValueError):
        normalize_environment_url("http://169.254.169.254/latest/meta-data")


def test_environment_token_uses_discovered_tenant_authority():
    from STT_server.services.dynamics365 import oauth_config_for_tenant
    config = oauth_config_for_tenant(GUID)
    assert config.token_url == f"https://login.microsoftonline.com/{GUID}/oauth2/v2.0/token"


async def test_oauth_callback_discovers_environment_and_encrypts_tokens(client, auth_token, monkeypatch):
    from STT_server.services import oauth_providers

    create = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"provider": "dynamics365", "name": "Contoso Dynamics", "configuration": {}, "credentials": {}},
    )
    integration_id = create.json()["id"]
    start = await client.get(
        f"/integrations/{integration_id}/oauth/start",
        headers={"Authorization": f"Bearer {auth_token}"},
        follow_redirects=False,
    )
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    responses = iter([
        Response({"access_token": "discovery-access", "refresh_token": "refresh", "expires_in": 3600, "scope": "offline_access"}),
        Response({"value": [{"ApiUrl": ENV_URL, "FriendlyName": "Contoso", "Id": GUID, "EnvironmentId": GUID, "TenantId": GUID, "State": 0}]}),
        Response({"access_token": "environment-access", "refresh_token": "refresh-2", "expires_in": 3600, "scope": f"{ENV_URL}/.default"}),
    ])
    monkeypatch.setattr(oauth_providers.urllib.request, "urlopen", lambda *a, **k: next(responses))

    callback = await client.get(
        f"/integrations/dynamics365/oauth/callback?code=code&state={state}",
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert f"connected={integration_id}" in callback.headers["location"]
    row = (await client.get(
        f"/integrations/{integration_id}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )).json()
    assert row["connection_status"] == "connected"
    assert row["configuration"]["environment_url"] == ENV_URL
    assert "credentials_encrypted" not in row
    from STT_server.db_integrations import get_integration
    from STT_server.security.credentials import decrypt_credentials
    stored = get_integration(integration_id, "user-test-001")
    assert decrypt_credentials(stored["credentials_encrypted"])["access_token"] == "environment-access"


async def test_oauth_callback_rejects_invalid_state(client):
    response = await client.get(
        "/integrations/dynamics365/oauth/callback?code=code&state=tampered",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "oauth_invalid_state" in response.headers["location"]


def test_expired_oauth_state_is_rejected_and_cleared():
    from STT_server import db_integrations
    row = db_integrations.create_integration(
        "user-test-001",
        {"provider": "dynamics365", "name": "Expired", "agent_id": "__shared__", "configuration": {}},
    )
    db_integrations.start_oauth_flow(row["id"], "user-test-001", "expired-state")
    rows = db_integrations._read_integrations_file()
    rows[0]["oauth_state_expires_at"] = "2000-01-01T00:00:00Z"
    db_integrations._write_integrations_file(rows)
    assert db_integrations.consume_oauth_state("expired-state") is None
    assert db_integrations.get_integration(row["id"], "user-test-001")["oauth_state_hash"] is None


async def test_microsoft_error_callback_marks_integration_failed(client, auth_token):
    from STT_server import db_integrations
    created = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"provider": "dynamics365", "name": "Denied", "configuration": {}, "credentials": {}},
    )
    integration_id = created.json()["id"]
    start = await client.get(
        f"/integrations/{integration_id}/oauth/start",
        headers={"Authorization": f"Bearer {auth_token}"},
        follow_redirects=False,
    )
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    callback = await client.get(
        "/integrations/dynamics365/oauth/callback",
        params={"error": "access_denied", "error_description": "user declined", "state": state},
        follow_redirects=False,
    )
    assert callback.status_code == 302
    assert "error=oauth_access_denied" in callback.headers["location"]
    assert "error_description" not in callback.headers["location"]
    row = db_integrations.get_integration(integration_id, "user-test-001")
    assert row["connection_status"] == "failed"
    assert row["credentials_encrypted"] is None


async def test_multiple_environments_require_selection_before_connected(
    client, auth_token, monkeypatch,
):
    from STT_server import db_integrations
    from STT_server.services import oauth_providers
    created = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"provider": "dynamics365", "name": "Multiple", "configuration": {}, "credentials": {}},
    )
    integration_id = created.json()["id"]
    start = await client.get(
        f"/integrations/{integration_id}/oauth/start",
        headers={"Authorization": f"Bearer {auth_token}"},
        follow_redirects=False,
    )
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    sandbox_url = "https://contoso-sandbox.crm4.dynamics.com"
    responses = iter([
        Response({"access_token": "discovery-access", "refresh_token": "refresh", "expires_in": 3600}),
        Response({"value": [
            {"ApiUrl": ENV_URL, "FriendlyName": "Production", "Id": GUID, "EnvironmentId": GUID, "TenantId": GUID, "State": 0},
            {"ApiUrl": sandbox_url, "FriendlyName": "Sandbox", "Id": "22222222-2222-2222-2222-222222222222", "EnvironmentId": "22222222-2222-2222-2222-222222222222", "TenantId": "33333333-3333-3333-3333-333333333333", "State": 0},
        ]}),
        Response({"access_token": "sandbox-access", "refresh_token": "refresh-2", "expires_in": 3600}),
    ])
    monkeypatch.setattr(oauth_providers.urllib.request, "urlopen", lambda *a, **k: next(responses))
    callback = await client.get(
        f"/integrations/dynamics365/oauth/callback?code=code&state={state}",
        follow_redirects=False,
    )
    assert callback.status_code == 302
    pending = db_integrations.get_integration(integration_id, "user-test-001")
    assert pending["connection_status"] == "pending"
    assert "environment_url" not in pending["configuration"]

    selected = await client.put(
        f"/integrations/{integration_id}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"configuration": {"environment_url": sandbox_url}},
    )
    assert selected.status_code == 200, selected.text
    connected = db_integrations.get_integration(integration_id, "user-test-001")
    assert connected["connection_status"] == "connected"
    assert connected["configuration"]["environment_url"] == sandbox_url
    assert connected["configuration"]["tenant_id"] == "33333333-3333-3333-3333-333333333333"
    from STT_server.security.credentials import decrypt_credentials
    assert decrypt_credentials(connected["credentials_encrypted"])["access_token"] == "sandbox-access"


async def test_reconnect_reuses_owned_integration_row(client, auth_token):
    from STT_server import db_integrations
    row = db_integrations.create_integration(
        "user-test-001",
        {"provider": "dynamics365", "name": "Reconnect", "agent_id": "__shared__", "configuration": {}},
    )
    db_integrations.mark_integration_status(row["id"], "user-test-001", "failed")
    response = await client.get(
        f"/integrations/{row['id']}/oauth/start",
        headers={"Authorization": f"Bearer {auth_token}"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    refreshed = db_integrations.get_integration(row["id"], "user-test-001")
    assert refreshed["id"] == row["id"]
    assert refreshed["connection_status"] == "pending"


async def test_disconnect_clears_dynamics_tokens(client, auth_token):
    from STT_server import db_integrations
    from STT_server.security.credentials import encrypt_credentials
    row = db_integrations.create_integration(
        "user-test-001",
        {
            "provider": "dynamics365",
            "name": "Disconnect",
            "agent_id": "__shared__",
            "configuration": {"environment_url": ENV_URL, "tenant_id": GUID},
        },
        credentials_encrypted=encrypt_credentials({
            "access_token": "access", "refresh_token": "refresh", "expires_at": "2099-01-01T00:00:00Z",
        }),
    )
    db_integrations.mark_integration_status(row["id"], "user-test-001", "connected")
    response = await client.post(
        f"/integrations/{row['id']}/disconnect",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert response.status_code == 200, response.text
    disconnected = db_integrations.get_integration(row["id"], "user-test-001")
    assert disconnected["connection_status"] == "disconnected"
    assert disconnected["credentials_encrypted"] is None


def test_discovery_normalizes_multiple_environments(monkeypatch):
    from STT_server.services import dynamics365
    monkeypatch.setattr(dynamics365, "_json_request", lambda *a, **k: ({"value": [
        {"ApiUrl": ENV_URL, "FriendlyName": "Production", "Id": GUID, "State": 0},
        {"ApiUrl": "https://sandbox.crm4.dynamics.com", "FriendlyName": "Sandbox", "Id": GUID, "State": 0},
    ]}, {}))
    result = dynamics365.discover_environments("token")
    assert [row["name"] for row in result] == ["Production", "Sandbox"]


