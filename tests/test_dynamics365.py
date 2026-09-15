from __future__ import annotations

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


def test_catalog_exposes_all_real_dynamics_actions():
    from STT_server.services.integrations_catalog import get_integration_provider_spec
    from STT_server.services.integrations_executor import supported_actions
    spec = get_integration_provider_spec("dynamics365")
    assert spec.auth_type == "oauth"
    assert spec.capabilities == ("sales", "customer_service", "field_service")
    assert len(spec.actions) == 43
    assert {"find_customer", "create_contact", "resolve_case", "close_opportunity_won", "complete_task"} <= {a.id for a in spec.actions}
    assert all(a.parameters_schema.get("additionalProperties") is False for a in spec.actions)
    assert set(supported_actions("dynamics365")) == {action.id for action in spec.actions}


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


@pytest.mark.parametrize(
    ("action", "arguments", "expected_method", "expected_path"),
    [
        ("create_contact", {"last_name": "Perez", "email": "a@example.com"}, "POST", "contacts"),
        ("update_customer", {"customer_id": GUID, "email": "b@example.com"}, "PATCH", f"contacts({GUID})"),
        ("create_account", {"name": "RevolutionMedia"}, "POST", "accounts"),
        ("update_account", {"account_id": GUID, "phone": "555"}, "PATCH", f"accounts({GUID})"),
        ("create_case", {"title": "Help"}, "POST", "incidents"),
        ("create_opportunity", {"name": "Renewal"}, "POST", "opportunities"),
        ("log_call", {"subject": "Follow up"}, "POST", "phonecalls"),
        ("create_task", {"subject": "Call customer"}, "POST", "tasks"),
    ],
)
def test_mutating_actions_map_to_dataverse(monkeypatch, action, arguments, expected_method, expected_path):
    from STT_server.services import dynamics365
    calls = []
    monkeypatch.setattr(
        dynamics365.Dynamics365Client,
        "request",
        lambda self, method, path, body=None, query=None: (calls.append((method, path, body)) or ({}, {})),
    )
    ok, data, error = dynamics365.execute_dynamics_action(
        action,
        {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}},
        {"access_token": "token", "refresh_token": "refresh"},
        arguments,
    )
    assert ok is True and error is None
    assert calls[0][0:2] == (expected_method, expected_path)
    assert data


@pytest.mark.parametrize(
    ("action", "arguments", "entity"),
    [
        ("find_customer", {"query": "test@example.com"}, "contacts"),
        ("find_account", {"query": "RevolutionMedia"}, "accounts"),
        ("find_lead", {"query": "Juan"}, "leads"),
        ("get_case", {"case_id": GUID}, f"incidents({GUID})"),
        ("get_opportunity", {"opportunity_id": GUID}, f"opportunities({GUID})"),
    ],
)
def test_read_actions_map_to_dataverse(monkeypatch, action, arguments, entity):
    from STT_server.services import dynamics365
    calls = []
    monkeypatch.setattr(
        dynamics365.Dynamics365Client,
        "request",
        lambda self, method, path, body=None, query=None: (calls.append(path) or ({"value": []}, {})),
    )
    ok, _, error = dynamics365.execute_dynamics_action(
        action,
        {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}},
        {"access_token": "token"},
        arguments,
    )
    assert ok is True and error is None
    assert entity in calls


def test_client_retries_once_after_401(monkeypatch):
    from STT_server.services import dynamics365
    calls = []

    def request(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 1:
            raise dynamics365.DataverseError("expired", 401)
        return {"UserId": GUID}, {}

    monkeypatch.setattr(dynamics365, "_json_request", request)
    monkeypatch.setattr(
        dynamics365.Dynamics365Client,
        "_refresh",
        lambda self: self.credentials.update(access_token="new-token"),
    )
    client = dynamics365.Dynamics365Client(
        {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}},
        {"access_token": "old-token", "refresh_token": "refresh"},
    )
    assert client.request("GET", "WhoAmI")[0]["UserId"] == GUID
    assert calls == ["old-token", "new-token"]


def test_client_marks_reauth_required_after_second_401(monkeypatch):
    from STT_server.services import dynamics365
    monkeypatch.setattr(
        dynamics365,
        "_json_request",
        lambda *a, **k: (_ for _ in ()).throw(dynamics365.DataverseError("expired", 401)),
    )
    monkeypatch.setattr(dynamics365.Dynamics365Client, "_refresh", lambda self: None)
    marked = []
    monkeypatch.setattr(dynamics365.Dynamics365Client, "_mark_failed", lambda self, message: marked.append(message))
    client = dynamics365.Dynamics365Client(
        {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}},
        {"access_token": "old-token", "refresh_token": "refresh"},
    )
    with pytest.raises(dynamics365.DataverseError):
        client.request("GET", "WhoAmI")
    assert marked == ["Microsoft session expired; reconnect the integration"]


def test_executor_rejects_invalid_action_and_backend_fields():
    from STT_server.services.integrations_executor import execute_action
    row = {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}}
    assert execute_action("dynamics365", "not_real", row, {}, {})[0] is False
    result = execute_action("dynamics365", "find_customer", row, {}, {"query": "x", "access_token": "bad"})
    assert result[0] is False
    assert "backend-managed" in result[2]


async def test_integration_ownership_and_tokens_are_not_exposed(client, auth_token, other_user_token):
    created = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"provider": "dynamics365", "name": "Private", "configuration": {}, "credentials": {}},
    )
    integration_id = created.json()["id"]
    forbidden = await client.get(
        f"/integrations/{integration_id}",
        headers={"Authorization": f"Bearer {other_user_token}"},
    )
    assert forbidden.status_code == 404
    assert "credentials" not in json.dumps(created.json())


async def test_internal_executor_envelope_and_service_token(client, auth_token, monkeypatch):
    from STT_server import db_integrations
    from STT_server.security.credentials import encrypt_credentials
    service_token = "n8n-service-token"
    monkeypatch.setenv("INTEGRATIONS_N8N_TOKEN", service_token)
    row = db_integrations.create_integration(
        "user-test-001",
        {"provider": "dynamics365", "name": "Runtime", "agent_id": "__shared__", "configuration": {"environment_url": ENV_URL}},
        credentials_encrypted=encrypt_credentials({"access_token": "token", "refresh_token": "refresh", "expires_at": "2099-01-01T00:00:00Z"}),
    )
    db_integrations.mark_integration_status(row["id"], "user-test-001", "connected")
    from STT_server.services import dynamics365
    monkeypatch.setattr(dynamics365.Dynamics365Client, "request", lambda *a, **k: ({"value": []}, {}))
    response = await client.post(
        f"/internal/integrations/{row['id']}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={"action": "find_account", "arguments": {"query": "Acme"}},
    )
    assert response.status_code == 200
    assert response.json() == {"success": True, "action": "find_account", "data": {"records": [], "count": 0}, "error": None}
    credentials_response = await client.post(
        f"/internal/integrations/{row['id']}/credentials",
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert credentials_response.status_code == 403
    assert "credentials never leave" in credentials_response.json()["detail"]
    denied = await client.post(
        f"/internal/integrations/{row['id']}/execute",
        json={"action": "find_account", "arguments": {"query": "Acme"}},
    )
    assert denied.status_code == 401


async def test_internal_executor_rejects_incomplete_dynamics_connection(
    client, auth_token, monkeypatch,
):
    from STT_server import db_integrations
    service_token = "n8n-service-token"
    monkeypatch.setenv("INTEGRATIONS_N8N_TOKEN", service_token)
    row = db_integrations.create_integration(
        "user-test-001",
        {
            "provider": "dynamics365",
            "name": "Needs environment",
            "agent_id": "__shared__",
            "configuration": {"environments": [{"environment_url": ENV_URL}]},
            "connection_status": "pending",
        },
    )
    response = await client.post(
        f"/internal/integrations/{row['id']}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={"action": "find_customer", "arguments": {"query": "test@example.com"}},
    )
    assert response.status_code == 200
    assert response.json()["error"] == "environment_required"


async def test_environment_selection_rejects_cross_tenant_url(client, auth_token):
    from STT_server import db_integrations
    row = db_integrations.create_integration(
        "user-test-001",
        {"provider": "dynamics365", "name": "Selection", "agent_id": "__shared__", "configuration": {"environments": [{"environment_url": ENV_URL, "name": "Contoso"}]}},
    )
    response = await client.put(
        f"/integrations/{row['id']}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"configuration": {"environment_url": "https://evil.crm.dynamics.com"}},
    )
    assert response.status_code == 422


# ── Field Service ─────────────────────────────────────────────────────

FS_INTEGRATION = {
    "id": "int-fs-1",
    "user_id": "user-1",
    "configuration": {"environment_url": ENV_URL, "field_service_available": True},
}
FS_CREDS = {"access_token": "token", "refresh_token": "refresh"}


def _fs_calls(monkeypatch, handler):
    from STT_server.services import dynamics365
    calls = []
    def request(self, method, path, body=None, query=None):
        calls.append((method, path, body, query))
        return handler(method, path, body, query)
    monkeypatch.setattr(dynamics365.Dynamics365Client, "request", request)
    return calls


def test_field_service_actions_registered_once():
    from STT_server.services.dynamics365 import DYNAMICS365_EXECUTORS, FIELD_SERVICE_ACTIONS
    from STT_server.services.integrations_catalog import get_integration_provider_spec
    from STT_server.services.integrations_executor import supported_actions
    assert len(FIELD_SERVICE_ACTIONS) == 16
    assert len(DYNAMICS365_EXECUTORS) == 43
    spec_ids = {a.id for a in get_integration_provider_spec("dynamics365").actions}
    assert set(FIELD_SERVICE_ACTIONS) <= spec_ids
    assert set(supported_actions("dynamics365")) == spec_ids
    assert "field_service" in get_integration_provider_spec("dynamics365").capabilities


def test_capability_probe_404_means_not_available(monkeypatch):
    from STT_server.services import dynamics365
    def request(self, method, path, body=None, query=None):
        raise dynamics365.DataverseError("Resource not found for the segment 'msdyn_workorders'", 404)
    monkeypatch.setattr(dynamics365.Dynamics365Client, "request", request)
    integration = {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}}
    ok, _, error = dynamics365.execute_dynamics_action(
        "find_asset", integration, FS_CREDS, {"query": "HVAC"})
    assert ok is False
    assert isinstance(error, str)
    assert error.startswith("CAPABILITY_NOT_AVAILABLE: ")
    assert integration["configuration"]["field_service_available"] is False


def test_capability_cached_false_skips_http(monkeypatch):
    from STT_server.services import dynamics365
    monkeypatch.setattr(
        dynamics365.Dynamics365Client, "request",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("probe must not run")),
    )
    integration = {"id": "int-1", "user_id": "user-1",
                   "configuration": {"environment_url": ENV_URL, "field_service_available": False}}
    ok, _, error = dynamics365.execute_dynamics_action(
        "find_asset", integration, FS_CREDS, {"query": "HVAC"})
    assert ok is False
    assert error.startswith("CAPABILITY_NOT_AVAILABLE: ")


def test_capability_probe_success_caches_true(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({"value": [{"msdyn_workorderid": GUID}]}, {}))
    integration = {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}}
    ok, data, error = dynamics365.execute_dynamics_action(
        "find_asset", integration, FS_CREDS, {"query": "HVAC"})
    assert ok is True and error is None
    assert calls[0][1] == "msdyn_workorders"
    assert integration["configuration"]["field_service_available"] is True


def test_capability_probe_forwards_non_404_errors(monkeypatch):
    from STT_server.services import dynamics365
    def request(self, method, path, body=None, query=None):
        raise dynamics365.DataverseError("Forbidden", 403)
    monkeypatch.setattr(dynamics365.Dynamics365Client, "request", request)
    integration = {"id": "int-1", "user_id": "user-1", "configuration": {"environment_url": ENV_URL}}
    ok, _, error = dynamics365.execute_dynamics_action(
        "find_asset", integration, FS_CREDS, {"query": "HVAC"})
    assert ok is False
    assert "CAPABILITY_NOT_AVAILABLE" not in (error or "")
    assert "field_service_available" not in integration["configuration"]


@pytest.mark.parametrize(
    ("action", "arguments", "expected_method", "expected_path"),
    [
        ("find_asset", {"query": "HVAC-1"}, "GET", "msdyn_customerassets"),
        ("get_asset", {"asset_id": GUID}, "GET", f"msdyn_customerassets({GUID})"),
        ("create_asset", {"name": "HVAC Unit 01", "asset_tag": "SN-1"}, "POST", "msdyn_customerassets"),
        ("update_asset", {"asset_id": GUID, "name": "New"}, "PATCH", f"msdyn_customerassets({GUID})"),
        ("create_work_order", {"description": "AC not cooling"}, "POST", "msdyn_workorders"),
        ("get_work_order", {"work_order_id": GUID}, "GET", f"msdyn_workorders({GUID})"),
        ("get_work_orders", {}, "GET", "msdyn_workorders"),
        ("update_work_order", {"work_order_id": GUID, "description": "Updated"}, "PATCH", f"msdyn_workorders({GUID})"),
        ("find_service_agreement", {"query": "Maintenance"}, "GET", "msdyn_agreements"),
        ("get_service_agreements", {}, "GET", "msdyn_agreements"),
    ],
)
def test_field_service_crud_maps_to_dataverse(monkeypatch, action, arguments, expected_method, expected_path):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({"value": []}, {}))
    ok, data, error = dynamics365.execute_dynamics_action(
        action, dict(FS_INTEGRATION), FS_CREDS, arguments)
    assert ok is True and error is None, error
    assert calls[-1][0:2] == (expected_method, expected_path)
    assert data


def test_find_asset_filters_name_tag_and_account(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({"value": []}, {}))
    dynamics365.execute_dynamics_action(
        "find_asset", dict(FS_INTEGRATION), FS_CREDS,
        {"query": "HVAC-1", "account_id": GUID})
    filt = calls[-1][3]["$filter"]
    assert "contains(msdyn_name,'HVAC-1')" in filt
    assert "contains(msdyn_assettag,'HVAC-1')" in filt
    assert f"_msdyn_account_value eq {GUID}" in filt


def test_create_asset_binds_lookups_without_serial_or_install_date(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({}, {}))
    dynamics365.execute_dynamics_action(
        "create_asset", dict(FS_INTEGRATION), FS_CREDS,
        {"name": "HVAC Unit 01", "asset_tag": "SN-1", "account_id": GUID, "product_id": GUID})
    body = calls[-1][2]
    assert body["msdyn_name"] == "HVAC Unit 01"
    assert body["msdyn_assettag"] == "SN-1"
    assert body["msdyn_account@odata.bind"] == f"/accounts({GUID})"
    assert body["msdyn_product@odata.bind"] == f"/products({GUID})"
    assert "serial_number" not in body and "install_date" not in body
    ok, _, error = dynamics365.execute_dynamics_action(
        "create_asset", dict(FS_INTEGRATION), FS_CREDS, {"asset_tag": "SN-1"})
    assert ok is False and "name" in error


def test_update_asset_rejects_empty_patch():
    from STT_server.services import dynamics365
    ok, _, error = dynamics365.execute_dynamics_action(
        "update_asset", dict(FS_INTEGRATION), FS_CREDS, {"asset_id": GUID})
    assert ok is False and "at least one field" in error.lower()


def test_create_work_order_never_writes_autonumber_and_defaults_unscheduled(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({}, {}))
    dynamics365.execute_dynamics_action(
        "create_work_order", dict(FS_INTEGRATION), FS_CREDS,
        {"description": "AC not cooling", "service_account_id": GUID, "asset_id": GUID,
         "scheduled_start": "2026-09-15T10:00:00-07:00", "scheduled_end": "2026-09-15T11:00:00-07:00"})
    body = calls[-1][2]
    assert "msdyn_name" not in body
    assert body["msdyn_instructions"] == "AC not cooling"
    assert body["msdyn_serviceaccount@odata.bind"] == f"/accounts({GUID})"
    assert body["msdyn_customerasset@odata.bind"] == f"/msdyn_customerassets({GUID})"
    assert body["msdyn_timefrompromised"] == "2026-09-15T10:00:00-07:00"
    assert body["msdyn_timetopromised"] == "2026-09-15T11:00:00-07:00"
    assert body["msdyn_systemstatus"] == 690970000


def test_create_work_order_resolves_priority_name(monkeypatch):
    from STT_server.services import dynamics365
    seen = []
    def handler(method, path, body, query):
        seen.append((method, path, query))
        if path == "msdyn_priorities":
            return {"value": [{"msdyn_priorityid": GUID, "msdyn_name": "High"}]}, {}
        return {}, {}
    calls = []
    from STT_server.services import dynamics365 as dyn
    def spy(self, method, path, body=None, query=None):
        calls.append((method, path, body))
        return handler(method, path, body, query)
    monkeypatch.setattr(dyn.Dynamics365Client, "request", spy)
    ok, _, error = dynamics365.execute_dynamics_action(
        "create_work_order", dict(FS_INTEGRATION), FS_CREDS, {"priority": "High"})
    assert ok is True, error
    assert calls[0][1] == "msdyn_priorities"
    assert calls[-1][2]["msdyn_priority@odata.bind"] == f"/msdyn_priorities({GUID})"


def test_get_work_orders_open_filter_excludes_closed_statuses(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({"value": []}, {}))
    dynamics365.execute_dynamics_action("get_work_orders", dict(FS_INTEGRATION), FS_CREDS, {})
    filt = calls[-1][3]["$filter"]
    assert "msdyn_systemstatus ne 690970003" in filt
    assert "msdyn_systemstatus ne 690970004" in filt
    assert "msdyn_systemstatus ne 690970005" in filt
    dynamics365.execute_dynamics_action(
        "get_work_orders", dict(FS_INTEGRATION), FS_CREDS, {"status": "all", "account_id": GUID})
    assert calls[-1][3].get("$filter") == f"_msdyn_serviceaccount_value eq {GUID}"
    ok, _, error = dynamics365.execute_dynamics_action(
        "get_work_orders", dict(FS_INTEGRATION), FS_CREDS, {"status": "bogus"})
    assert ok is False


def test_cancel_and_complete_work_order_use_official_statuses(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({}, {}))
    ok, _, error = dynamics365.execute_dynamics_action(
        "cancel_work_order", dict(FS_INTEGRATION), FS_CREDS,
        {"work_order_id": GUID, "reason": "Customer cancelled"})
    assert ok is True, error
    assert calls[0][1] == "annotations"  # reason saved first
    assert calls[-1][0:2] == ("PATCH", f"msdyn_workorders({GUID})")
    assert calls[-1][2] == {"msdyn_systemstatus": 690970005}
    calls.clear()
    ok, _, error = dynamics365.execute_dynamics_action(
        "complete_work_order", dict(FS_INTEGRATION), FS_CREDS,
        {"work_order_id": GUID, "completion_notes": "Repaired"})
    assert ok is True, error
    assert calls[-1][2] == {"msdyn_systemstatus": 690970003}


def test_get_available_resources_calls_official_action(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(
        monkeypatch,
        lambda m, p, b, q: ({"Resources": [{"ResourceId": GUID, "ResourceName": "Tech"}]}, {}))
    ok, data, error = dynamics365.execute_dynamics_action(
        "get_available_resources", dict(FS_INTEGRATION), FS_CREDS,
        {"start": "2026-09-15T10:00:00-07:00", "end": "2026-09-15T12:00:00-07:00"})
    assert ok is True, error
    assert calls[-1][0:2] == ("POST", "msdyn_SearchResourceAvailability")
    body = calls[-1][2]
    assert body["Version"] == "3" and body["IsWebApi"] is True
    assert body["Requirement"]["msdyn_duration"] == 60
    assert body["Requirement"]["@odata.type"] == "Microsoft.Dynamics.CRM.msdyn_resourcerequirement"
    assert data["count"] == 1 and data["resources"][0]["resource_id"] == GUID


def test_create_booking_binds_resource_status_and_work_order(monkeypatch):
    from STT_server.services import dynamics365
    STATUS = "22222222-2222-2222-2222-222222222222"
    def handler(method, path, body, query):
        if path == "msdyn_fieldservicesettings":
            return {"value": [{"_msdyn_defaultscheduledbookingstatus_value": STATUS}]}, {}
        return {}, {}
    calls = _fs_calls(monkeypatch, handler)
    ok, _, error = dynamics365.execute_dynamics_action(
        "create_booking", dict(FS_INTEGRATION), FS_CREDS,
        {"work_order_id": GUID, "resource_id": GUID,
         "start": "2026-09-15T10:00:00-07:00", "end": "2026-09-15T11:00:00-07:00"})
    assert ok is True, error
    body = calls[-1][2]
    assert body["resource@odata.bind"] == f"/bookableresources({GUID})"
    assert body["bookingstatus@odata.bind"] == f"/bookingstatuses({STATUS})"
    assert body["msdyn_workorder@odata.bind"] == f"/msdyn_workorders({GUID})"
    assert body["duration"] == 60


def test_cancel_booking_resolves_canceled_status_via_fallback(monkeypatch):
    from STT_server.services import dynamics365
    STATUS = "33333333-3333-3333-3333-333333333333"
    def handler(method, path, body, query):
        if path == "msdyn_fieldservicesettings":
            return {"value": [{}]}, {}
        if path == "bookingstatuses":
            assert "msdyn_fieldservicestatus eq 690970005" in query["$filter"]
            return {"value": [{"bookingstatusid": STATUS}]}, {}
        return {}, {}
    calls = _fs_calls(monkeypatch, handler)
    ok, _, error = dynamics365.execute_dynamics_action(
        "cancel_booking", dict(FS_INTEGRATION), FS_CREDS, {"booking_id": GUID})
    assert ok is True, error
    assert calls[-1][2] == {"bookingstatus@odata.bind": f"/bookingstatuses({STATUS})"}


def test_update_booking_requires_a_side_and_recomputes_duration(monkeypatch):
    from STT_server.services import dynamics365
    ok, _, error = dynamics365.execute_dynamics_action(
        "update_booking", dict(FS_INTEGRATION), FS_CREDS, {"booking_id": GUID})
    assert ok is False
    calls = _fs_calls(
        monkeypatch,
        lambda m, p, b, q: ({"starttime": "2026-09-15T10:00:00-07:00",
                             "endtime": "2026-09-15T11:00:00-07:00"}, {}))
    ok, _, error = dynamics365.execute_dynamics_action(
        "update_booking", dict(FS_INTEGRATION), FS_CREDS,
        {"booking_id": GUID, "end": "2026-09-15T12:00:00-07:00"})
    assert ok is True, error
    assert calls[-1][2]["duration"] == 120


def test_service_agreement_filters(monkeypatch):
    from STT_server.services import dynamics365
    calls = _fs_calls(monkeypatch, lambda m, p, b, q: ({"value": []}, {}))
    dynamics365.execute_dynamics_action(
        "find_service_agreement", dict(FS_INTEGRATION), FS_CREDS, {"query": "Maintenance"})
    assert "contains(msdyn_name,'Maintenance')" in calls[-1][3]["$filter"]
    dynamics365.execute_dynamics_action("get_service_agreements", dict(FS_INTEGRATION), FS_CREDS, {})
    assert calls[-1][3]["$filter"] == "statecode eq 0"


def test_field_service_executor_rejects_backend_fields_and_unknown():
    from STT_server.services.integrations_executor import execute_action
    row = dict(FS_INTEGRATION)
    result = execute_action("dynamics365", "create_work_order", row, {},
                            {"description": "x", "environment_url": "https://evil.example"})
    assert result[0] is False and "backend-managed" in result[2]
    assert execute_action("dynamics365", "fly_helicopter", row, {}, {})[0] is False


def test_field_service_schemas_hide_backend_fields():
    from STT_server.services.integrations_catalog import get_integration_provider_spec
    spec = get_integration_provider_spec("dynamics365")
    banned = {"integration_id", "provider", "environment_url", "access_token",
              "refresh_token", "credentials", "tenant_id"}
    for action in spec.actions:
        props = set((action.parameters_schema.get("properties") or {}).keys())
        assert not (props & banned), action.id
        assert action.parameters_schema.get("additionalProperties") is False


async def test_internal_execute_runs_field_service_action(client, monkeypatch):
    from STT_server import db_integrations
    from STT_server.security.credentials import encrypt_credentials
    service_token = "n8n-service-token"
    monkeypatch.setenv("INTEGRATIONS_N8N_TOKEN", service_token)
    row = db_integrations.create_integration(
        "user-test-001",
        {"provider": "dynamics365", "name": "FS Runtime", "agent_id": "__shared__",
         "configuration": {"environment_url": ENV_URL, "field_service_available": True}},
        credentials_encrypted=encrypt_credentials({"access_token": "token", "refresh_token": "refresh", "expires_at": "2099-01-01T00:00:00Z"}),
    )
    db_integrations.mark_integration_status(row["id"], "user-test-001", "connected")
    from STT_server.services import dynamics365
    monkeypatch.setattr(dynamics365.Dynamics365Client, "request", lambda *a, **k: ({"value": []}, {}))
    response = await client.post(
        f"/internal/integrations/{row['id']}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={"action": "find_asset", "arguments": {"query": "HVAC"}},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
