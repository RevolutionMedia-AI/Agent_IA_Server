"""Integration tests for the shared-n8n-tools CRUD endpoints in
STT_server/routes/api.py.

Covers: POST/GET/PUT/DELETE /tools, POST /tools/{id}/test, auth
required, cross-user isolation, and validation errors returning 400.

Network calls (Twilio / Inworld / OpenAI) are NOT exercised here —
the test endpoint hits the n8n webhook via tool_executor, which we
mock to avoid any external request.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


VALID_PAYLOAD = {
    "name": "query_orders",
    "description": "Look up an order by number",
    "webhook_url": "https://n8n.example.com/webhook/abc123",
    "filler_phrase": "One moment...",
    "parameters": {
        "type": "object",
        "properties": {
            "order_number": {"type": "string", "description": "The order id"},
        },
        "required": ["order_number"],
    },
}


# ── Auth ──────────────────────────────────────────────────────────────


async def test_list_requires_auth(client: AsyncClient) -> None:
    r = await client.get("/tools")
    assert r.status_code == 401, r.text


async def test_create_requires_auth(client: AsyncClient) -> None:
    r = await client.post("/tools", json=VALID_PAYLOAD)
    assert r.status_code == 401, r.text


# ── CRUD happy path ───────────────────────────────────────────────────


async def test_create_returns_200_with_user_id(
    client: AsyncClient, auth_token: str
) -> None:
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"], "id should be auto-assigned"
    assert body["name"] == "query_orders"
    assert body["user_id"] == "user-test-001", "owner stamped from auth"


async def test_list_returns_only_caller_shared_tools(
    client: AsyncClient, auth_token: str, other_user_token: str
) -> None:
    # Alice creates one
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 200
    # Bob creates one
    other_payload = {**VALID_PAYLOAD, "name": "bob_tool"}
    r = await client.post(
        "/tools", json=other_payload, headers={"Authorization": f"Bearer {other_user_token}"}
    )
    assert r.status_code == 200

    # Alice sees only hers
    r = await client.get("/tools", headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200
    names = [t["name"] for t in r.json()]
    assert names == ["query_orders"], "list must scope to caller's user_id"

    # Bob sees only his
    r = await client.get("/tools", headers={"Authorization": f"Bearer {other_user_token}"})
    names = [t["name"] for t in r.json()]
    assert names == ["bob_tool"], "list must scope to caller's user_id"


async def test_update_modifies_fields(
    client: AsyncClient, auth_token: str
) -> None:
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    tool_id = r.json()["id"]

    new_payload = {**VALID_PAYLOAD, "description": "Updated desc", "name": "query_orders_v2"}
    r = await client.put(
        f"/tools/{tool_id}", json=new_payload,
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["description"] == "Updated desc"
    assert body["name"] == "query_orders_v2"


async def test_update_returns_404_for_other_users_tool(
    client: AsyncClient, auth_token: str, other_user_token: str
) -> None:
    # Alice creates
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    tool_id = r.json()["id"]
    # Bob tries to update it
    new_payload = {**VALID_PAYLOAD, "name": "hijacked"}
    r = await client.put(
        f"/tools/{tool_id}", json=new_payload,
        headers={"Authorization": f"Bearer {other_user_token}"},
    )
    assert r.status_code == 404, "Bob must not be able to mutate Alice's tool"


async def test_delete_removes_tool(
    client: AsyncClient, auth_token: str
) -> None:
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    tool_id = r.json()["id"]
    r = await client.delete(
        f"/tools/{tool_id}", headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 200
    # List is empty now
    r = await client.get("/tools", headers={"Authorization": f"Bearer {auth_token}"})
    assert r.json() == []


async def test_delete_404_for_other_users_tool(
    client: AsyncClient, auth_token: str, other_user_token: str
) -> None:
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    tool_id = r.json()["id"]
    r = await client.delete(
        f"/tools/{tool_id}", headers={"Authorization": f"Bearer {other_user_token}"}
    )
    assert r.status_code == 404


async def test_test_endpoint_hits_webhook(
    client: AsyncClient, auth_token: str
) -> None:
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    tool_id = r.json()["id"]
    with patch(
        "STT_server.services.tool_executor.execute_tool",
        new=AsyncMock(return_value={"ok": True}),
    ), patch(
        # Test now always consults the LLM-driven generator
        # (test_data_generator.py). No OpenAI key wired into the
        # test env, so mock the generator to bypass the upstream
        # SDK + keychain resolution and still exercise the executor
        # path that POSTs to the n8n webhook.
        "STT_server.services.test_data_generator.generate_test_payload",
        new=lambda tool, user_id, model=None: {"order_number": "ORD-42"},
    ):
        r = await client.post(
            f"/tools/{tool_id}/test",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True
    assert r.json()["sent_payload"] == {"order_number": "ORD-42"}


# ── Validation errors → 400 ──────────────────────────────────────────


async def test_create_400_when_name_missing(
    client: AsyncClient, auth_token: str
) -> None:
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "name"}
    r = await client.post(
        "/tools", json=bad, headers={"Authorization": f"Bearer {auth_token}"}
    )
    # pydantic catches missing required field first (422), but the route
    # also validates explicitly so accept either as "client error".
    assert r.status_code in (400, 422), r.text


async def test_create_400_when_webhook_url_invalid(
    client: AsyncClient, auth_token: str
) -> None:
    bad = {**VALID_PAYLOAD, "webhook_url": "not-a-url"}
    r = await client.post(
        "/tools", json=bad, headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 400, r.text
    assert "webhook_url" in r.text.lower()


async def test_create_400_when_parameters_schema_invalid(
    client: AsyncClient, auth_token: str
) -> None:
    bad = {**VALID_PAYLOAD, "parameters": {"type": "object",
            "properties": {"x": {"type": "string", "bogus_field": True}}}}
    r = await client.post(
        "/tools", json=bad, headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 400, r.text
    assert "schema" in r.text.lower() or "unknown" in r.text.lower()


# ── Provider credentials must not surface in /tools or as assignable
#    targets — Settings → API saves each provider's key as an
#    agent_tools row with agent_id="__shared__" but no webhook_url /
#    destination. The Edit Agent modal's marketplace was rendering
#    OpenAI / Inworld / ElevenLabs as if they were callable tools.
#    These tests pin the discriminator on the GET endpoints. ──────────


CREDENTIAL_ROW = {
    "id": "openai",
    "agent_id": "__shared__",
    "name": "OpenAI",
    "description": "",
    "webhook_url": "",
    "destination": None,
    "assignments": [],
    "kind": "webhook",
    "function_name": "openai",
    "credentials": {"api_key": "encrypted-blob"},
    "user_id": "user-test-001",
}


async def test_list_tools_filters_out_provider_credential_rows(
    client: AsyncClient, auth_token: str, data_dir, monkeypatch
) -> None:
    """A credential row sitting in agent_tools.json (id=provider
    name, agent_id='__shared__', empty webhook_url, no destination)
    must NOT show up in the shared tools marketplace — only real
    operator-defined tools with a URL or destination should appear."""
    import json
    import STT_server.db_tools as db_mod
    from pathlib import Path
    # ponytail: the data_dir fixture patches api_mod.TOOLS_FILE but
    # db_tools._AGENT_TOOLS_FILE is the actual path db_get_tool reads.
    # Patch it here so the test writes/reads from data_dir and never
    # touches the production STT_server/data/agent_tools.json file.
    tools_file = data_dir / "agent_tools.json"
    monkeypatch.setattr(db_mod, "_AGENT_TOOLS_FILE", tools_file, raising=False)
    tools_file.write_text(
        json.dumps([CREDENTIAL_ROW]),
        encoding="utf-8",
    )
    r = await client.get("/tools", headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200
    names = [t["name"] for t in r.json()]
    assert names == [], (
        f"credential rows leaked into /tools: {names}. "
        "GET /tools must filter on (webhook_url or destination)."
    )


async def test_list_agent_tools_filters_out_credential_assigned(
    client: AsyncClient, auth_token: str, data_dir, monkeypatch
) -> None:
    """Even if a credential row got assigned to an agent somehow
    (legacy data, manual DB edit, etc.), GET /agents/{id}/tools must
    exclude it from the assigned list — otherwise the FE renders a
    'credential' as an 'Assigned shared' tool with a working
    Unassign button that the user clicks, which is the regression
    we're guarding against."""
    import json
    import STT_server.db_tools as db_mod
    agent_id = "agent-with-credential-attached"
    row = {
        **CREDENTIAL_ROW,
        # Legacy state: someone managed to assign a credential row.
        "assignments": [agent_id],
    }
    tools_file = data_dir / "agent_tools.json"
    monkeypatch.setattr(db_mod, "_AGENT_TOOLS_FILE", tools_file, raising=False)
    tools_file.write_text(json.dumps([row]), encoding="utf-8")
    # Seed the agent row so the auth check passes.
    import STT_server.db_agents as agents_mod
    import STT_server.routes.api as api_mod_agents
    from STT_server.db_agents import create_agent as db_create_agent
    # Mirror the data_dir patching the conftest already does for
    # agents so the auth check finds the row we just seeded.
    agents_file = data_dir / "agents.json"
    monkeypatch.setattr(api_mod_agents, "AGENTS_FILE", str(agents_file), raising=False)
    monkeypatch.setattr(agents_mod, "_AGENTS_FILE", agents_file, raising=False)
    db_create_agent(
        "user-test-001",
        {
            "id": agent_id,
            "name": "Test",
            "tts_provider": "",
            "stt_provider": "",
            "llm_provider": "",
            "system_prompt": "",
        },
    )
    r = await client.get(
        f"/agents/{agent_id}/tools",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    ids = [t["id"] for t in r.json()]
    assert "openai" not in ids, (
        "credential row leaked into /agents/{id}/tools as an "
        "assignable 'shared tool' — must filter on "
        "(webhook_url or destination)."
    )


async def test_assign_rejects_credential_rows(
    client: AsyncClient, auth_token: str, data_dir, monkeypatch
) -> None:
    """POST /agents/{id}/tools/{id}/assign must 400 when the target
    row is a provider credential, not a real tool. Without this
    guard an operator (or a stale FE bookmark) could 'assign' OpenAI
    as a callable function and the agent modal would happily render
    it under 'Assigned shared'."""
    import json
    import STT_server.db_tools as db_mod
    import STT_server.db_agents as agents_mod
    import STT_server.routes.api as api_mod_agents
    agent_id = "agent-x"
    tools_file = data_dir / "agent_tools.json"
    monkeypatch.setattr(db_mod, "_AGENT_TOOLS_FILE", tools_file, raising=False)
    tools_file.write_text(json.dumps([CREDENTIAL_ROW]), encoding="utf-8")
    agents_file = data_dir / "agents.json"
    monkeypatch.setattr(api_mod_agents, "AGENTS_FILE", str(agents_file), raising=False)
    monkeypatch.setattr(agents_mod, "_AGENTS_FILE", agents_file, raising=False)
    from STT_server.db_agents import create_agent as db_create_agent
    db_create_agent(
        "user-test-001",
        {
            "id": agent_id, "name": "Test",
            "tts_provider": "", "stt_provider": "", "llm_provider": "",
            "system_prompt": "",
        },
    )
    r = await client.post(
        f"/agents/{agent_id}/tools/openai/assign",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 400, r.text
    assert "credentials" in r.text.lower() or "settings" in r.text.lower()


async def test_assign_real_tool_still_works(
    client: AsyncClient, auth_token: str
) -> None:
    """Regression guard: the credential-row rejection must not block
    legitimate shared-tool assignments. A real n8n tool (webhook_url
    set) should still be assignable end-to-end."""
    r = await client.post(
        "/tools", json=VALID_PAYLOAD, headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 200
    tool_id = r.json()["id"]
    from STT_server.db_agents import create_agent as db_create_agent
    db_create_agent(
        "user-test-001",
        {
            "id": "agent-y", "name": "Test",
            "tts_provider": "", "stt_provider": "", "llm_provider": "",
            "system_prompt": "",
        },
    )
    r = await client.post(
        f"/agents/agent-y/tools/{tool_id}/assign",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200, r.text