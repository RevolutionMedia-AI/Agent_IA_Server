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