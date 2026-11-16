"""Cross-tenant access to the tenant ACTION endpoints.

The three endpoints that act on a tenant's Twilio account took the
``tenant_id`` straight from the path and used it without checking who
owns the tenant:

  POST /tenants/{tenant_id}/configure-webhook
  POST /tenants/{tenant_id}/list-numbers
  POST /tenants/{tenant_id}/call

The CRUD endpoints next to them (GET/PATCH/DELETE /tenants/{id},
/validate-twilio) already ran ``_require_tenant_owner``. These three
did not, so any authenticated caller could:

  - rewrite the voice webhook on somebody else's Twilio number,
  - enumerate the phone numbers of somebody else's subaccount, and
  - place outbound calls billed to somebody else's Twilio account.

The tests assert two things per endpoint:

  1. a caller who does not own the tenant gets 404, and
  2. the Twilio adapter is NEVER reached -- no side effect happens.

Point 2 is the one that matters: a 404 raised after the call would
still have placed the call. The recorder proves the request is
rejected BEFORE the side effect.

And the owner still works, so the guard does not break the
legitimate path.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("PUBLIC_URL", "http://localhost:8080")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_KEY",
    "oahqImB7aGYfEFxfWIJLZzJs27YSYAgr5rHUyc3gIRU=",
)

from httpx import ASGITransport, AsyncClient  # noqa: E402

import STT_server.STT_Server as srv  # noqa: E402  (the REAL app)
from STT_server import db_tenants  # noqa: E402

OWNER_ID = "user-test-001"
OTHER_ID = "user-other-999"

OWNED_TENANT = "tenant-owned-001"
FOREIGN_TENANT = "tenant-foreign-001"

SID = "ACtest0000000000000000000000test"
TOKEN = "twilio-auth-token-abc123"
NUMBER = "+15550001111"


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tenants(data_dir) -> None:
    """Seed one tenant per user. The data_dir fixture redirects the
    JSON store to a tmp dir, so this never touches real data."""
    for tenant_id, owner in ((OWNED_TENANT, OWNER_ID), (FOREIGN_TENANT, OTHER_ID)):
        db_tenants.upsert_tenant({
            "tenant_id": tenant_id,
            "user_id": owner,
            "name": tenant_id,
            "twilio_account_sid": SID,
            "twilio_auth_token": TOKEN,
            "twilio_phone_number": NUMBER,
            "custom_prompt": None,
            "tts_provider": "elevenlabs",
            "preferred_language": "es",
            "webhook_configured": False,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        })


@pytest.fixture
def twilio_calls(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record every Twilio side effect the endpoints attempt."""
    calls: list = []

    async def fake_configure_voice_webhook(account_sid, auth_token, phone_number, webhook_url):
        calls.append(("configure_voice_webhook", account_sid, phone_number, webhook_url))
        return {"success": True}

    async def fake_list_phone_numbers(account_sid, auth_token):
        calls.append(("list_phone_numbers", account_sid))
        return {"phone_numbers": [{"phone_number": NUMBER}]}

    async def fake_make_outbound_call(account_sid, auth_token, from_number, to_number, webhook_url):
        calls.append(("make_outbound_call", account_sid, to_number))
        return {"success": True, "sid": "CA" + "0" * 32}

    monkeypatch.setattr(
        "STT_server.adapters.twilio_api.configure_voice_webhook",
        fake_configure_voice_webhook,
    )
    monkeypatch.setattr(
        "STT_server.adapters.twilio_api.list_phone_numbers",
        fake_list_phone_numbers,
    )
    monkeypatch.setattr(
        "STT_server.adapters.twilio_api.make_outbound_call",
        fake_make_outbound_call,
    )
    return calls


@pytest.fixture
async def real_client():
    """ASGI client bound to the REAL app (the tenant routes live on
    STT_Server.app, not on the small router-only test app)."""
    transport = ASGITransport(app=srv.app)
    async with AsyncClient(transport=transport, base_url="http://localhost:8080") as ac:
        yield ac


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── the three action endpoints ──────────────────────────────────────────

ACTION_ENDPOINTS = [
    ("POST", "/tenants/{tid}/configure-webhook", None),
    ("POST", "/tenants/{tid}/list-numbers", None),
    ("POST", "/tenants/{tid}/call", {"to_number": "+15559998888"}),
]


@pytest.mark.parametrize("verb,path,body", ACTION_ENDPOINTS)
async def test_non_owner_cannot_act_on_a_foreign_tenant(
    real_client, auth_token, other_user_token, tenants, twilio_calls, verb, path, body
):
    """A different authenticated user must be refused, with no side effect."""
    url = path.format(tid=OWNED_TENANT)
    resp = await real_client.request(verb, url, headers=_headers(other_user_token), json=body)

    assert resp.status_code == 404, resp.text
    assert "not found" in resp.text.lower()
    assert twilio_calls == [], f"the request reached Twilio anyway: {twilio_calls}"


@pytest.mark.parametrize("verb,path,body", ACTION_ENDPOINTS)
async def test_owner_can_still_act_on_their_own_tenant(
    real_client, auth_token, other_user_token, tenants, twilio_calls, verb, path, body
):
    """The legitimate path must keep working -- the guard is not a
    blanket refusal."""
    url = path.format(tid=OWNED_TENANT)
    resp = await real_client.request(verb, url, headers=_headers(auth_token), json=body)

    assert resp.status_code == 200, resp.text
    assert len(twilio_calls) == 1, f"expected exactly one Twilio call, got {twilio_calls}"


async def test_admin_cannot_act_on_a_tenant_owned_by_someone_else(
    real_client, auth_token, other_user_token, tenants, twilio_calls
):
    """The admin role only covers tenants with NO owner.

    An admin token must not reach into a tenant another user owns --
    otherwise the role is a blanket bypass of the guard.
    """
    # auth_token's user has role=admin (see the conftest fixture) and
    # owns OWNED_TENANT but NOT FOREIGN_TENANT.
    resp = await real_client.post(
        f"/tenants/{FOREIGN_TENANT}/list-numbers",
        headers=_headers(auth_token),
    )

    assert resp.status_code == 404, resp.text
    assert twilio_calls == [], f"the request reached Twilio anyway: {twilio_calls}"


async def test_tenant_the_other_user_owns_is_reachable_by_them(
    real_client, auth_token, other_user_token, tenants, twilio_calls
):
    """Symmetry: each user reaches their own tenant, nobody else's."""
    resp = await real_client.post(
        f"/tenants/{FOREIGN_TENANT}/list-numbers",
        headers=_headers(other_user_token),
    )

    assert resp.status_code == 200, resp.text
    assert len(twilio_calls) == 1


async def test_unknown_tenant_is_404_and_no_side_effect(
    real_client, auth_token, tenants, twilio_calls
):
    """A tenant that does not exist is a 404, not a 403 -- the two are
    indistinguishable on purpose, so the endpoint does not confirm
    which tenant ids exist."""
    resp = await real_client.post(
        "/tenants/tenant-does-not-exist/list-numbers",
        headers=_headers(auth_token),
    )

    assert resp.status_code == 404, resp.text
    assert twilio_calls == []


async def test_missing_bearer_is_401_not_a_tenant_leak(
    real_client, tenants, twilio_calls
):
    """No token at all: 401, and nothing happens."""
    resp = await real_client.post(f"/tenants/{OWNED_TENANT}/list-numbers")

    assert resp.status_code == 401, resp.text
    assert twilio_calls == []


# ── the guard the fix reuses is already covered for the CRUD side ──────


async def test_crud_endpoints_still_reject_a_non_owner(
    real_client, auth_token, other_user_token, tenants
):
    """Sanity: the pre-existing CRUD guard still behaves. If this ever
    fails, _require_tenant_owner changed underneath the fix."""
    resp = await real_client.get(
        f"/tenants/{OWNED_TENANT}", headers=_headers(other_user_token)
    )
    assert resp.status_code == 404, resp.text

    resp = await real_client.get(
        f"/tenants/{OWNED_TENANT}", headers=_headers(auth_token)
    )
    assert resp.status_code == 200, resp.text
