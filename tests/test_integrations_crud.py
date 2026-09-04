"""Integration CRUD round-trip + 409 on dependent tools.

Covers:
  * POST /integrations validates fields + runs preflight (we monkey-patch
    run_integration_test to avoid the network for these tests)
  * GET  /integrations + GET /integrations/{id} never returns credentials
  * PUT  /integrations/{id} merges credentials (missing/empty = keep)
  * DELETE /integrations/{id} returns 409 when a tool still references it
  * DELETE /integrations/{id} succeeds when no tools depend on it
"""
from __future__ import annotations

import pytest


@pytest.fixture
def mock_test_fn(monkeypatch):
    """Replace the live test_fn runner with a controllable stub so
    preflight doesn't hit the network. Returns a list of (config, creds)
    tuples the runner was called with, so tests can assert preflight
    actually ran with the right values."""
    calls: list[tuple[str, dict, dict]] = []

    def fake_run(test_fn_path: str, config: dict, creds: dict) -> tuple[bool, str]:
        calls.append((test_fn_path, config, creds))
        return True, "fake ok"

    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake_run)
    # The route layer resolves run_integration_test from the catalog
    # module — patch it there too.
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake_run, raising=False)
    return calls


async def _create_zendesk(client, headers, mock_test_fn):
    return await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "zendesk",
            "name": "Acme Support",
            "agent_id": "__shared__",
            "configuration": {"subdomain": "acme"},
            "credentials": {
                "email": "admin@acme.com",
                "api_token": "abcdef0123456789abcdef0123456789",
            },
        },
    )


async def test_create_integration_runs_preflight(client, auth_token, mock_test_fn):
    headers = {"Authorization": f"Bearer {auth_token}"}
    resp = await _create_zendesk(client, headers, mock_test_fn)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["provider"] == "zendesk"
    assert body["name"] == "Acme Support"
    # Critical: no credentials in the wire response
    assert "credentials_encrypted" not in body
    assert "credentials_cipher" not in body
    assert "credentials" not in body
    # preflight actually ran with the cleaned values
    assert len(mock_test_fn) == 1
    _, config, creds = mock_test_fn[0]
    assert config["subdomain"] == "acme"
    assert creds["email"] == "admin@acme.com"


async def test_create_integration_rejects_unknown_provider(client, auth_token, mock_test_fn):
    resp = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "made_up",
            "name": "X",
            "configuration": {},
            "credentials": {},
        },
    )
    assert resp.status_code == 422
    assert "Unknown provider" in resp.json()["detail"]


async def test_create_integration_validates_fields(client, auth_token, mock_test_fn):
    # Bad subdomain (has uppercase, hyphen at the end) — pattern rejects
    resp = await client.post(
        "/integrations",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "BadSubdomain-"},
            "credentials": {
                "email": "x@y.com",
                "api_token": "a" * 25,
            },
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    errors = detail.get("errors") if isinstance(detail, dict) else None
    assert errors and errors[0]["field"].startswith("config.")


async def test_get_integration_never_returns_credentials(client, auth_token, mock_test_fn):
    headers = {"Authorization": f"Bearer {auth_token}"}
    create_resp = await _create_zendesk(client, headers, mock_test_fn)
    iid = create_resp.json()["id"]
    resp = await client.get(f"/integrations/{iid}", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "credentials_encrypted" not in body
    assert "credentials_cipher" not in body
    assert body["provider"] == "zendesk"
    assert body["configuration"]["subdomain"] == "acme"


async def test_list_integrations(client, auth_token, mock_test_fn):
    headers = {"Authorization": f"Bearer {auth_token}"}
    await _create_zendesk(client, headers, mock_test_fn)
    await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "generic_webhook",
            "name": "Internal Hook",
            "configuration": {"webhook_url": "https://example.com/hook"},
            "credentials": {},
        },
    )
    resp = await client.get("/integrations", headers=headers)
    assert resp.status_code == 200
    rows = resp.json()["integrations"]
    providers = {r["provider"] for r in rows}
    assert {"zendesk", "generic_webhook"}.issubset(providers)
    for r in rows:
        assert "credentials_encrypted" not in r
        assert "credentials_cipher" not in r


async def test_update_credentials_merge_keep(client, auth_token, mock_test_fn):
    """Empty / missing credential fields keep the stored value."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    create_resp = await _create_zendesk(client, headers, mock_test_fn)
    iid = create_resp.json()["id"]
    # Send empty email and missing api_token — both should keep
    resp = await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"credentials": {"email": "", "api_token": "newtoken_replacement_1234567890"}},
    )
    assert resp.status_code == 200, resp.text
    # We can't read credentials back from the wire (no reveal), but the
    # endpoint should accept the merge without complaint. The
    # credentials_merge logic is exercised more thoroughly in
    # test_integrations_credentials_merge.py.


async def test_update_validation_rejects_bad_config(client, auth_token, mock_test_fn):
    headers = {"Authorization": f"Bearer {auth_token}"}
    create_resp = await _create_zendesk(client, headers, mock_test_fn)
    iid = create_resp.json()["id"]
    resp = await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"configuration": {"subdomain": "BAD!!"}},
    )
    assert resp.status_code == 422


async def test_delete_integration_succeeds_when_no_dependents(client, auth_token, mock_test_fn):
    headers = {"Authorization": f"Bearer {auth_token}"}
    create_resp = await _create_zendesk(client, headers, mock_test_fn)
    iid = create_resp.json()["id"]
    resp = await client.delete(f"/integrations/{iid}", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    # ponytail: 019 prompt-persistence refactor added a `change_log`
    # to the delete response so the FE can show a toast about which
    # agents lost their prompt sections. Empty list when no agents
    # were referencing the integration (the common case).
    assert body["success"] is True
    assert body.get("change_log") == []


async def test_unassign_integration_returns_envelope_with_agent_and_change_log(
    client, auth_token, mock_test_fn,
):
    """Regression: the 019 refactor used `db_get_agent` in the
    unassign endpoint without importing it. Production 500'd with
    `NameError: name 'db_get_agent' is not defined`. The endpoint now
    imports + uses the alias `_get_agent` and returns
    `{integration, agent, change_log}` so the FE can sync the
    System Prompt textarea + surface a toast."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    # Seed an agent via the HTTP layer so the route + db module agree
    # on where the row lives (JSON file vs Postgres).
    created = await client.post(
        "/agents",
        headers=headers,
        json={"name": "Regression Agent"},
    )
    assert created.status_code == 200, created.text
    agent_id = created.json()["id"]
    integ = await _create_zendesk(client, headers, mock_test_fn)
    iid = integ.json()["id"]
    # Promote the integration to shared so the assign endpoint accepts it.
    await client.put(
        f"/integrations/{iid}",
        headers=headers,
        json={"agent_id": "__shared__"},
    )
    # Assign + unassign round-trip. The unassign path used to crash
    # with NameError; now it returns the envelope.
    r1 = await client.post(
        f"/agents/{agent_id}/integrations/{iid}/assign",
        headers=headers,
    )
    assert r1.status_code == 200, r1.text
    assign_body = r1.json()
    assert assign_body["agent"]["id"] == agent_id
    assert isinstance(assign_body["change_log"], list)
    r2 = await client.delete(
        f"/agents/{agent_id}/integrations/{iid}/assign",
        headers=headers,
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["integration"]["id"] == iid
    assert body["agent"]["id"] == agent_id
    assert isinstance(body["change_log"], list)


async def test_delete_integration_succeeds_even_when_tools_still_point_at_it(
    client, auth_token, mock_test_fn,
):
    """Regression for the post-refactor delete-flow contract.

    Pre-refactor: `count(agent_tools WHERE integration_id = ?)` gated
    DELETE /integrations/{id} with a 409 + "N tools depend" message
    that the FE had no UI to bulk-resolve. Operators were stuck.

    Post-refactor: tools carry their own webhook_url + name +
    parameters and dispatch without the integration row, so the
    dependent-tool gate is gone. Delete always succeeds (200);
    the backfill in db_integrations.nullify_stale_tool_integration_pointers
    clears the dangling `integration_id` columns when the operator
    next saves the agent's prompt (or on the next BE boot)."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    create_resp = await _create_zendesk(client, headers, mock_test_fn)
    iid = create_resp.json()["id"]
    tool_resp = await client.post(
        f"/tools",
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
    assert tool_resp.status_code == 200, tool_resp.text
    resp = await client.delete(f"/integrations/{iid}", headers=headers)
    assert resp.status_code == 200, resp.text


def test_nullify_stale_tool_integration_pointers_is_idempotent(tmp_path, monkeypatch):
    """Two backfill runs in a row leave the same final state. JSON path
    covers the local-dev + production-fallback backend."""
    import STT_server.db_integrations as _db_int

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    tools_file = data_dir / "agent_tools.json"
    # ponytail: db_integrations derives the tools file path from its
    # own _DATA_DIR (kept separate from db_tools so the backfill is
    # self-contained). We patch _DATA_DIR; the read happens via
    # _DATA_DIR / "agent_tools.json".
    monkeypatch.setattr(_db_int, "_DATA_DIR", data_dir)

    seed = [
        {"id": "t_1", "agent_id": "agent_1", "name": "a", "description": "",
         "webhook_url": "https://x", "kind": "webhook", "destination": None,
         "parameters": {}, "integration_id": "int_1",
         "assignments": [], "function_name": "a",
         "created_at": "2026-09-04T00:00:00Z", "updated_at": "2026-09-04T00:00:00Z",
         "last_tested_at": None, "last_test_result": None,
         "last_invoked_at": None, "last_invocation_status": None,
         "invocation_count": 0},
        {"id": "t_2", "agent_id": "agent_1", "name": "b", "description": "",
         "webhook_url": "https://y", "kind": "webhook", "destination": None,
         "parameters": {}, "integration_id": None,
         "assignments": [], "function_name": "b",
         "created_at": "2026-09-04T00:00:00Z", "updated_at": "2026-09-04T00:00:00Z",
         "last_tested_at": None, "last_test_result": None,
         "last_invoked_at": None, "last_invocation_status": None,
         "invocation_count": 0},
    ]
    import json
    tools_file.write_text(json.dumps(seed), encoding="utf-8")

    first = _db_int.nullify_stale_tool_integration_pointers()
    second = _db_int.nullify_stale_tool_integration_pointers()
    assert first == 1, "exactly one stale row should be cleared on first run"
    assert second == 0, "second run must be a no-op"
    after = json.loads(tools_file.read_text(encoding="utf-8"))
    assert after[0]["integration_id"] is None
    assert after[1]["integration_id"] is None