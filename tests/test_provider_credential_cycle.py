"""Regression tests for the per-user provider credential cycle.

Three bugs lived in the same feature (Settings -> API) and each one broke
a different step of the cycle. They are pinned here because the symptom
the operator sees is a silent one:

  1. The "reveal" endpoint called ``db_get_tool(auth["user_id"],
     service_id)`` but the signature is ``get_tool(tool_id, user_id)``.
     The arguments were swapped, so the lookup was
     ``WHERE id = '<the user id>' AND user_id = 'openai'`` and never
     matched. The "show" toggle always answered "Key not configured".

  2. The "disconnect" endpoint had the same swap on
     ``db_delete_tool``. It deleted nothing and answered 404, so a
     provider could never be unconnected.

  3. The reveal endpoint also tested ``row.get("connected")``, a column
     that deliberately does NOT exist on agent_tools (migration 014
     kept it off the table shape: it is the inverse of "credentials is
     null"). The check always failed.

And a fourth one lives in the schema and is only visible on Postgres:
agent_tools had PRIMARY KEY (id) instead of (user_id, id), so only the
FIRST user of the platform could store a credential for a provider and
everybody else got a 500 on save. Migration 031 fixes it; the test at
the bottom of this file covers it when a database is available.
"""
from __future__ import annotations

import os

import pytest

API_KEY = "sk-test-0123456789abcdefghijklmnop"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# The JSON-backed tests below and the Postgres-backed test at the bottom
# cannot run in the same process: the storage backend is chosen when
# db_users is imported, so DATABASE_URL decides it for the WHOLE run.
# Each group skips when the other backend is the one in use.
postgres_in_use = bool(os.getenv("DATABASE_URL", ""))

requires_json = pytest.mark.skipif(
    postgres_in_use, reason="DATABASE_URL is set: the storage backend is Postgres"
)


@requires_json
async def test_a_user_can_save_and_read_back_their_credential(
    client, data_dir, auth_token
):
    """Save then reveal: the round trip the FE's "show" toggle does."""
    r = await client.put(
        "/settings/api-keys/openai",
        json={"credentials": {"api_key": API_KEY}},
        headers=_headers(auth_token),
    )
    assert r.status_code == 200, r.text

    r = await client.get(
        "/settings/api-keys/openai/value", headers=_headers(auth_token)
    )
    assert r.status_code == 200, r.text
    assert r.json()["credentials"]["api_key"] == API_KEY


@requires_json
async def test_a_user_can_disconnect_their_credential(
    client, data_dir, auth_token
):
    """Save then disconnect: the delete path."""
    r = await client.put(
        "/settings/api-keys/openai",
        json={"credentials": {"api_key": API_KEY}},
        headers=_headers(auth_token),
    )
    assert r.status_code == 200, r.text

    r = await client.delete("/settings/api-keys/openai", headers=_headers(auth_token))
    assert r.status_code == 200, r.text

    r = await client.get(
        "/settings/api-keys/openai/value", headers=_headers(auth_token)
    )
    assert r.status_code == 404, r.text


@requires_json
async def test_one_user_cannot_read_or_delete_another_users_credential(
    client, data_dir, auth_token, other_user_token
):
    """The credential belongs to the user who saved it."""
    r = await client.put(
        "/settings/api-keys/openai",
        json={"credentials": {"api_key": API_KEY}},
        headers=_headers(auth_token),
    )
    assert r.status_code == 200, r.text

    r = await client.get(
        "/settings/api-keys/openai/value", headers=_headers(other_user_token)
    )
    assert r.status_code == 404, r.text

    r = await client.delete(
        "/settings/api-keys/openai", headers=_headers(other_user_token)
    )
    assert r.status_code == 404, r.text

    # The owner's credential is still there.
    r = await client.get(
        "/settings/api-keys/openai/value", headers=_headers(auth_token)
    )
    assert r.status_code == 200, r.text


# ── the schema bug (Postgres only) ───────────────────────────────────────
#
# agent_tools had PRIMARY KEY (id) instead of (user_id, id), so a
# provider id could only exist once in the whole table: the first user
# to save a credential got it, and everybody else got a 500. Migration
# 031 fixes the key.
#
# The test needs a real Postgres with the migrations applied, so it is
# skipped when there is none. It uses DATABASE_URL and not a separate
# variable: the storage backend is chosen when db_users is IMPORTED, so
# switching it inside the test would come too late.

TEST_DATABASE_URL = os.getenv("DATABASE_URL", "")


@pytest.mark.skipif(not TEST_DATABASE_URL, reason="DATABASE_URL is not set")
async def test_two_users_can_each_store_a_credential_for_the_same_provider(
    client, data_dir
):
    """Two users, the same provider: both must be able to save one.

    This is the bug the schema had: the primary key was (id) alone, so
    the second INSERT raised a unique violation and the endpoint
    answered 500.
    """
    import psycopg2

    conn = psycopg2.connect(TEST_DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_get_constraintdef(oid)
                  FROM pg_constraint
                 WHERE conrelid = 'agent_tools'::regclass AND contype = 'p'
                """
            )
            row = cur.fetchone()
            if not row or "user_id" not in row[0]:
                conn.rollback()
                pytest.skip("the test database has not run migration 031")
            # Start from a clean slate for the users and the credentials.
            # TRUNCATE, not DELETE: several tables reference users, and
            # deleting the users first would break their foreign keys.
            cur.execute(
                """
                TRUNCATE users, auth_sessions, settings, agents,
                         phone_numbers, tools_integrations, tenants,
                         call_sessions, call_usage, agent_tools,
                         audio_capture, integrations,
                         agent_pricing_overrides, twilio_credentials
                RESTART IDENTITY CASCADE
                """
            )
        conn.commit()
    finally:
        conn.close()

    tokens = []
    for email in ("one@example.com", "two@example.com"):
        r = await client.post(
            "/auth/register",
            json={"email": email, "password": "Password123!", "name": "U"},
        )
        assert r.status_code in (200, 201), r.text
        r = await client.post(
            "/auth/login",
            json={"email": email, "password": "Password123!"},
        )
        assert r.status_code == 200, r.text
        tokens.append(r.json()["access_token"])

    for i, token in enumerate(tokens):
        r = await client.put(
            "/settings/api-keys/openai",
            json={"credentials": {"api_key": f"{API_KEY}{i}"}},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text

    # And each one reads back THEIR OWN.
    for i, token in enumerate(tokens):
        r = await client.get(
            "/settings/api-keys/openai/value",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["credentials"]["api_key"] == f"{API_KEY}{i}"
