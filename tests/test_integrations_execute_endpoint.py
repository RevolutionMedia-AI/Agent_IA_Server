"""E2E test for POST /internal/integrations/{id}/execute.

The endpoint stitches three things together:
  1. Service-token auth (same gate as /credentials).
  2. Credential resolution + refresh-on-read under advisory lock.
  3. Dispatch into services/integrations_executor.

We mock the OAuth + Google HTTP layers here so the test stays
hermetic. End-to-end coverage of the refresh path lives in
test_integrations_oauth_flow.py.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import pytest


@pytest.fixture
def service_token(monkeypatch) -> str:
    tok = secrets.token_urlsafe(32)
    monkeypatch.setenv("INTEGRATIONS_N8N_TOKEN", tok)
    return tok


@pytest.fixture
def fake_test_fn(monkeypatch):
    def fake(test_fn_path, config, creds):
        return True, "fake ok"
    from STT_server.services import integrations_tester
    monkeypatch.setattr(integrations_tester, "run_integration_test", fake)
    from STT_server.routes import api as api_mod
    monkeypatch.setattr(api_mod, "run_integration_test", fake, raising=False)


def _seed_google_calendar(auth_token: str) -> str:
    """Drop an integration row directly into the JSON backend with
    Fernet-encrypted credentials. Mirrors the OAuth callback write
    path so the route sees a real row shape on read.
    """
    from STT_server.db_integrations import (
        _read_integrations_file, _write_integrations_file,
    )
    from STT_server.security.credentials import encrypt_credentials

    integ_id = "int-google-e2e"
    now = "2099-01-01T00:00:00Z"
    encrypted = encrypt_credentials({
        "access_token": "GOOGLE_BEARER",
        "refresh_token": "GOOGLE_REFRESH",
        "scope": "openid email profile calendar.events",
        "expires_at": "2099-01-01T00:00:00Z",
    })
    rows = _read_integrations_file()
    rows = [r for r in rows if not r.get("id") == integ_id]
    rows.append({
        "id": integ_id,
        "user_id": "user-test-001",
        "provider": "google_calendar",
        "name": "E2E Calendar",
        "configuration": {
            "calendar_id": "cal-team@example.com",
            "timezone": "America/Tijuana",
        },
        "credentials_encrypted": encrypted,
        "credentials_cipher": "fernet-v1",
        "connection_status": "connected",
        "scope": "calendar.events",
        "assignments": [],
        "created_at": now,
        "updated_at": now,
    })
    _write_integrations_file(rows)
    return integ_id


async def test_execute_check_availability_happy_path(
    client, auth_token, fake_test_fn, service_token, monkeypatch,
):
    integ_id = _seed_google_calendar(auth_token)

    fake_resp = {
        "calendars": {
            "cal-team@example.com": {
                "busy": [
                    {"start": "2026-09-08T14:00:00Z", "end": "2026-09-08T14:30:00Z"},
                ],
            }
        }
    }

    from STT_server.services import integrations_executor as exec_mod
    monkeypatch.setattr(exec_mod, "_google_http", lambda *a, **kw: fake_resp)

    resp = await client.post(
        f"/internal/integrations/{integ_id}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={
            "action": "check_availability",
            "arguments": {
                "datetime": "2026-09-08T14:00:00",
                "duration_minutes": 30,
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["action"] == "check_availability"
    assert body["error"] is None
    assert body["data"]["available"] is False
    assert body["data"]["conflicts"] == 1
    assert body["data"]["busy"][0]["start"].startswith("2026-09-08T14:00:00")


async def test_execute_create_appointment_happy_path(
    client, auth_token, fake_test_fn, service_token, monkeypatch,
):
    integ_id = _seed_google_calendar(auth_token)

    freebusy_resp = {"calendars": {"cal-team@example.com": {"busy": []}}}
    event_resp = {
        "id": "evt-new-1",
        "htmlLink": "https://calendar.google.com/event?eid=evt-new-1",
        "conferenceData": {
            "entryPoints": [
                {"entryPointType": "video", "uri": "https://meet.google.com/abc-defg-hij"}
            ]
        },
        "start": {"dateTime": "2026-09-08T14:00:00Z"},
        "end": {"dateTime": "2026-09-08T14:30:00Z"},
        "summary": "Test (test@example.com)",
        "status": "confirmed",
    }

    from STT_server.services import integrations_executor as exec_mod
    call_log: list = []

    def fake_http(method, url, token, body=None, timeout=8.0):
        call_log.append((method, url, body))
        if method == "POST" and url.endswith("/freeBusy"):
            return freebusy_resp
        return event_resp

    monkeypatch.setattr(exec_mod, "_google_http", fake_http)

    resp = await client.post(
        f"/internal/integrations/{integ_id}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={
            "action": "create_appointment",
            "arguments": {
                "name": "Test",
                "email": "test@example.com",
                "datetime": "2026-09-08T14:00:00",
                "duration_minutes": 30,
                "notes": "first call",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["action"] == "create_appointment"
    assert body["data"]["id"] == "evt-new-1"
    assert body["data"]["meet_link"] == "https://meet.google.com/abc-defg-hij"
    # ponytail: two HTTP calls (freebusy + events.insert) in order.
    # The events.insert URL carries BOTH ``conferenceDataVersion=1`` and
    # ``sendUpdates=all`` so the Meet is actually created and the
    # attendee gets a calendar invite. Order in the query string is
    # implementation-defined; we just check both flags are present.
    assert len(call_log) == 2
    assert call_log[0][0] == "POST" and call_log[0][1].endswith("/freeBusy")
    insert_url = call_log[1][1]
    assert call_log[1][0] == "POST" and "/calendars/" in insert_url
    assert "conferenceDataVersion=1" in insert_url
    assert "sendUpdates=all" in insert_url


async def test_execute_create_appointment_rejects_slot_taken(
    client, auth_token, fake_test_fn, service_token, monkeypatch,
):
    integ_id = _seed_google_calendar(auth_token)

    busy = [{"start": "2026-09-08T14:00:00Z", "end": "2026-09-08T14:30:00Z"}]

    from STT_server.services import integrations_executor as exec_mod
    call_log: list = []
    events_insert_called = {"v": False}

    def fake_http(method, url, token, body=None, timeout=8.0):
        call_log.append(method)
        if method == "POST" and url.endswith("/freeBusy"):
            return {"calendars": {"cal-team@example.com": {"busy": busy}}}
        events_insert_called["v"] = True
        return {"id": "should-not-create"}

    monkeypatch.setattr(exec_mod, "_google_http", fake_http)

    resp = await client.post(
        f"/internal/integrations/{integ_id}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={
            "action": "create_appointment",
            "arguments": {
                "name": "Test",
                "email": "test@example.com",
                "datetime": "2026-09-08T14:00:00",
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "slot_taken"
    assert body["data"]["conflicts"] == 1
    # ponytail: freebusy is the only call that landed — events.insert
    # must not have fired.
    assert call_log == ["POST"]
    assert events_insert_called["v"] is False


async def test_execute_rejects_unknown_action(
    client, auth_token, fake_test_fn, service_token, monkeypatch,
):
    integ_id = _seed_google_calendar(auth_token)

    from STT_server.services import integrations_executor as exec_mod
    monkeypatch.setattr(exec_mod, "_google_http", lambda *a, **kw: {})

    resp = await client.post(
        f"/internal/integrations/{integ_id}/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={"action": "nuke_planet", "arguments": {}},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is False
    assert "unsupported" in body["error"]


async def test_execute_requires_service_token(client, auth_token, fake_test_fn, service_token):
    integ_id = _seed_google_calendar(auth_token)
    # No Authorization header → 401.
    resp = await client.post(
        f"/internal/integrations/{integ_id}/execute",
        json={"action": "check_availability", "arguments": {}},
    )
    assert resp.status_code == 401


async def test_execute_rejects_user_token(client, auth_token, fake_test_fn, service_token):
    integ_id = _seed_google_calendar(auth_token)
    resp = await client.post(
        f"/internal/integrations/{integ_id}/execute",
        headers={"Authorization": f"Bearer {auth_token}"},  # user token, not service
        json={"action": "check_availability", "arguments": {}},
    )
    assert resp.status_code == 401


async def test_execute_404_on_unknown_id(
    client, auth_token, fake_test_fn, service_token,
):
    resp = await client.post(
        "/internal/integrations/int-does-not-exist/execute",
        headers={"Authorization": f"Bearer {service_token}"},
        json={"action": "check_availability", "arguments": {}},
    )
    assert resp.status_code == 404
