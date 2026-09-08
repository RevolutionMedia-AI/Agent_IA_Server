"""Regression: PUT /integrations/{id} must preserve OAuth-provider
configuration fields (calendar_id, timezone, etc.) that the catalog
validates against a static-only schema.

The 2026-09-04 production incident: Google Calendar's
configuration editor saved ``calendar_id`` + ``timezone`` and the
Test button kept failing with ``Set the target calendar_id before
running this test`` even though the FE rendered the values. Root
cause: the BE's ``validate_integration_fields`` only whitelisted
fields declared in the spec — for Google Calendar (OAuth), the
``fields`` tuple is empty, so every post-Connect field got dropped
silently on save.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def fake_test_fn(monkeypatch):
    def fake(test_fn_path, config, creds):
        return True, "fake ok"
    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake)
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake, raising=False)


@pytest.fixture(autouse=True)
def _restore_decrypt_credentials():
    """No-op: the conftest autouse fixture already pins
    ``api_mod.decrypt_credentials`` for every test. Kept as a
    placeholder so the file's intent stays self-documenting."""
    yield


async def _seed_gcal(client, headers) -> str:
    """Seed a Google Calendar integration via the public POST endpoint
    so the test exercises the same write path as the operator."""
    # ponytail: the JSON file backend is shared across tests, so we
    # sweep the test user's own rows. Without this, a leftover row
    # from a previous run can shadow the next test that re-seeds the
    # same provider. We deliberately keep rows belonging to OTHER
    # users (e.g. the legacy "int-other-user" fixtures in
    # test_shared_tools_api.py) intact.
    from STT_server.db_integrations import _read_integrations_file, _write_integrations_file
    user_id = headers["Authorization"].split()[-1]
    rows = _read_integrations_file()
    rows = [r for r in rows if r.get("user_id") != user_id]
    _write_integrations_file(rows)
    resp = await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "google_calendar",
            "name": "ACME Calendar",
            "configuration": {},
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


async def test_google_calendar_config_update_preserves_calendar_id_and_timezone(
    client, auth_token, fake_test_fn,
):
    """The post-Connect Google Calendar fields (calendar_id +
    timezone) are NOT in the catalog's static ``fields`` tuple
    because Google uses OAuth. The validator must still preserve
    them on save so the next Test Connection can read the
    configuration."""
    integ_id = await _seed_gcal(client, {"Authorization": f"Bearer {auth_token}"})

    resp = await client.put(
        f"/integrations/{integ_id}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "name": "ACME Calendar",
            "agent_id": "__shared__",
            "configuration": {
                "calendar_id": "kevin.escalante@revolutionmedia.ai",
                "timezone": "America/Tijuana",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # ponytail: 2026-09-04 — both fields MUST round-trip on the
    # response, not be silently dropped by the validator. The endpoint
    # returns ``{integration: ..., change_log: [...]}`` so the FE can
    # surface a "we updated your agent prompts too" toast.
    cfg = (body.get("integration") or {}).get("configuration") or {}
    assert cfg.get("calendar_id") == "kevin.escalante@revolutionmedia.ai"
    assert cfg.get("timezone") == "America/Tijuana"


async def test_google_calendar_config_update_trims_whitespace(
    client, auth_token, fake_test_fn,
):
    integ_id = await _seed_gcal(client, {"Authorization": f"Bearer {auth_token}"})
    resp = await client.put(
        f"/integrations/{integ_id}",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={
            "name": "ACME Calendar",
            "agent_id": "__shared__",
            "configuration": {
                "calendar_id": "  kevin.escalante@revolutionmedia.ai  ",
                "timezone": "\tAmerica/Tijuana\n",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    cfg = (resp.json().get("integration") or {}).get("configuration") or {}
    assert cfg.get("calendar_id") == "kevin.escalante@revolutionmedia.ai"
    assert cfg.get("timezone") == "America/Tijuana"


async def test_zendesk_static_fields_still_validated(client, auth_token, fake_test_fn):
    """The fix for Google Calendar (preserve unknowns) must not break
    the static-provider path. Zendesk's spec has ``subdomain`` +
    ``email`` + ``api_token`` in the whitelist; an out-of-spec string
    field is preserved (trimmed) so the operator can keep an extra
    flag without losing it on save."""
    headers = {"Authorization": f"Bearer {auth_token}"}
    # ponytail: clear leftover rows for this user before seeding
    # Zendesk. The shared JSON backend means a previous run could
    # leave an unrelated row shadowing the lookup below. We keep
    # rows belonging to OTHER users untouched.
    from STT_server.db_integrations import _read_integrations_file, _write_integrations_file
    rows = _read_integrations_file()
    rows = [r for r in rows if r.get("user_id") != auth_token]
    _write_integrations_file(rows)
    # Seed a zendesk row via the public create endpoint.
    create = await client.post(
        "/integrations",
        headers=headers,
        json={
            "provider": "zendesk",
            "name": "Acme",
            "configuration": {"subdomain": "acme"},
        },
    )
    assert create.status_code == 200, create.text
    integ_id = create.json()["id"]

    resp = await client.put(
        f"/integrations/{integ_id}",
        headers=headers,
        json={
            "name": "Acme",
            "agent_id": "__shared__",
            "configuration": {
                "subdomain": "  acme  ",  # whitelist (string) — must trim
                "calendar_id": "ops@acme.com",  # NOT in Zendesk whitelist — preserved
            },
        },
    )
    assert resp.status_code == 200, resp.text
    cfg = (resp.json().get("integration") or {}).get("configuration") or {}
    # Whitelisted: trimmed.
    assert cfg.get("subdomain") == "acme"
    # Not in whitelist: preserved as-is (trimmed) so the operator
    # can keep an extra flag without losing it on save.
    assert cfg.get("calendar_id") == "ops@acme.com"
