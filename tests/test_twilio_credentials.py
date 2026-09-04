"""Round-trip tests for the /twilio-credentials endpoints + the DB helper.

The endpoint powers Settings → API → Twilio sub-accounts. Two contracts
to lock down:
  1. The CRUD flow (create → list → update → delete) persists credentials
     via the new ``twilio_credentials`` table with the SID + Token
     encrypted at rest and the SID's last-4 stored plaintext for the
     list view.
  2. The Test endpoint reflects the live Twilio API verdict in
     ``status`` + ``last_test_message`` so the Settings card stops
     showing "Connected" after a SID/Token pair goes stale.

ponytail: the Twilio SDK is mocked so tests don't actually call Twilio.
"""
from __future__ import annotations

import json
import os
import pathlib

# ponytail: see test_settings_api_keys.py — credential encryption refuses
# to start without a Fernet key in production. Set one before the import.
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_KEY",
    "3caLHixTmxCJ1OAQEK11TEn4k5soMyJhybJIyAFVMfk=",  # sample Fernet key for tests
)

import secrets
from datetime import datetime, timedelta, timezone

import pytest


# ── Fixtures (mirror test_settings_api_keys.py) ────────────────────────────


@pytest.fixture
def data_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    import STT_server.db_users as db_users
    from STT_server.routes import api as api_mod
    from STT_server import db_settings

    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    settings_dir = d / "settings"
    settings_dir.mkdir(exist_ok=True)
    users_file = d / "users.json"
    sessions_file = d / "sessions.json"

    monkeypatch.setattr(api_mod, "DATA_DIR", str(d), raising=False)
    monkeypatch.setattr(api_mod, "SETTINGS_DIR", settings_dir, raising=False)
    monkeypatch.setattr(db_users, "USERS_FILE", users_file, raising=False)
    monkeypatch.setattr(db_users, "SESSIONS_FILE", sessions_file, raising=False)
    monkeypatch.setattr(db_settings, "SETTINGS_DIR", settings_dir, raising=False)
    monkeypatch.setattr(db_settings, "DATA_DIR", d, raising=False)

    # ponytail: keep the twilio_credentials file inside the tmp dir.
    from STT_server import db_twilio_credentials as db_twcreds
    monkeypatch.setattr(db_twcreds, "CREDENTIALS_FILE", d / "twilio_credentials.json", raising=False)

    users_file.write_text(json.dumps([{
        "id": "user-admin-001",
        "name": "Admin",
        "email": "admin@revolutionmedia.ai",
        "password_hash": "",
        "role": "admin",
    }]), encoding="utf-8")

    token = secrets.token_urlsafe(32)
    sessions_file.write_text(json.dumps({
        token: {
            "user_id": "user-admin-001",
            "email": "admin@revolutionmedia.ai",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        },
    }), encoding="utf-8")

    return d


@pytest.fixture
def auth_token(data_dir: pathlib.Path) -> str:
    sessions_file = data_dir / "sessions.json"
    payload = json.loads(sessions_file.read_text(encoding="utf-8"))
    return next(iter(payload.keys()))


@pytest.fixture
def stub_twilio(monkeypatch: pytest.MonkeyPatch):
    """Stub the live Twilio validator so tests don't reach the network.
    ponytail: the route imports the function lazily inside the endpoint
    body (`from STT_server.adapters.twilio_api import
    validate_twilio_credentials`), so monkeypatching
    `STT_server.routes.api.validate_twilio_credentials` only works if the
    symbol was already imported at module top. Patch the source module
    instead — it's the one the lazy import resolves to at call time."""
    from STT_server.adapters import twilio_api

    async def _fake_validate(account_sid: str, auth_token: str):
        if account_sid == "AC" + "0" * 32:
            return {"valid": False, "message": "Twilio rejected the credentials: auth token"}
        return {"valid": True, "message": "account status: active", "account_status": "active"}

    monkeypatch.setattr(twilio_api, "validate_twilio_credentials", _fake_validate)
    return _fake_validate


VALID_SID = "AC" + "a" * 32
VALID_TOKEN = "deadbeef" * 4  # 32 chars


def test_create_list_update_delete_roundtrip(auth_token, stub_twilio):
    """create → list → update → delete works end-to-end."""
    from STT_server import db_twilio_credentials as db_twcreds

    saved = db_twcreds.create_credential(
        "user-admin-001",
        name="Sales",
        account_sid=VALID_SID,
        auth_token=VALID_TOKEN,
        status="connected",
        last_test_message="ok",
    )
    assert saved["id"].startswith("twcred-")
    assert saved["name"] == "Sales"
    assert saved["account_sid_last4"] == VALID_SID[-4:]
    assert saved["status"] == "connected"

    rows = db_twcreds.list_credentials("user-admin-001")
    assert len(rows) == 1
    assert rows[0]["id"] == saved["id"]

    # ponytail: never leak secrets through list_credentials.
    assert "account_sid" not in rows[0]
    assert "auth_token" not in rows[0]

    updated = db_twcreds.update_credential(
        saved["id"],
        "user-admin-001",
        name="Sales (renamed)",
    )
    assert updated["name"] == "Sales (renamed)"
    assert updated["id"] == saved["id"]

    assert db_twcreds.delete_credential(saved["id"], "user-admin-001") is True
    assert db_twcreds.list_credentials("user-admin-001") == []


def test_list_credentials_returns_only_user_rows(auth_token, stub_twilio):
    """tenant isolation — a user never sees another user's credentials."""
    from STT_server import db_twilio_credentials as db_twcreds

    db_twcreds.create_credential(
        "user-admin-001",
        name="Mine",
        account_sid=VALID_SID,
        auth_token=VALID_TOKEN,
    )
    db_twcreds.create_credential(
        "user-other-002",
        name="Theirs",
        account_sid="AC" + "b" * 32,
        auth_token=VALID_TOKEN,
    )

    mine = db_twcreds.list_credentials("user-admin-001")
    theirs = db_twcreds.list_credentials("user-other-002")
    assert len(mine) == 1 and mine[0]["name"] == "Mine"
    assert len(theirs) == 1 and theirs[0]["name"] == "Theirs"


def test_reveal_round_trips_secrets_through_fernet(auth_token, stub_twilio):
    """Encryption is real — the row on disk is not the plaintext SID/Token."""
    from STT_server import db_twilio_credentials as db_twcreds

    saved = db_twcreds.create_credential(
        "user-admin-001",
        name="Encrypted",
        account_sid=VALID_SID,
        auth_token=VALID_TOKEN,
    )

    revealed = db_twcreds.get_credential(saved["id"], "user-admin-001", include_secrets=True)
    assert revealed["account_sid"] == VALID_SID
    assert revealed["auth_token"] == VALID_TOKEN

    # ponytail: the on-disk file (JSON-fallback path) must NOT contain
    # the plaintext secret. Even though we pass include_secrets=True
    # above, the storage format encrypts the values before they touch
    # disk. A leaked twilio_credentials.json still doesn't leak keys.
    file_path = db_twcreds.CREDENTIALS_FILE
    raw = file_path.read_text(encoding="utf-8")
    assert VALID_TOKEN not in raw, "auth token leaked to JSON file"


def _build_app():
    from fastapi import FastAPI
    from STT_server.routes.api import api_router
    application = FastAPI()
    application.include_router(api_router)
    return application


def test_endpoint_create_persists_and_returns_last4(auth_token, stub_twilio):
    """POST /twilio-credentials: format-validates, calls Twilio, persists."""
    from fastapi.testclient import TestClient
    from STT_server import db_twilio_credentials as db_twcreds

    client = TestClient(_build_app())
    r = client.post(
        "/twilio-credentials",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Sales", "account_sid": VALID_SID, "auth_token": VALID_TOKEN},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["account_sid_last4"] == VALID_SID[-4:]
    assert body["status"] == "connected"
    assert "account_sid" not in body
    assert "auth_token" not in body

    rows = db_twcreds.list_credentials("user-admin-001")
    assert len(rows) == 1 and rows[0]["id"] == body["id"]


def test_endpoint_create_rejects_bad_sid(auth_token, stub_twilio):
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    r = client.post(
        "/twilio-credentials",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Bad", "account_sid": "not-a-sid", "auth_token": VALID_TOKEN},
    )
    assert r.status_code == 400
    assert "Account SID" in r.json()["detail"]


def test_endpoint_test_reflects_invalid_credentials(auth_token, stub_twilio, monkeypatch):
    """When Twilio rejects the SID/Token, /test flips status='invalid'."""
    from fastapi.testclient import TestClient
    from STT_server.adapters import twilio_api as _twilio_api_for_test

    # ponytail: stub_twilio accepts any SID+token. To exercise the
    # "test flips status to invalid" path we need a credential that
    # was successfully created, then we swap the stub to reject and
    # hit /test. Create through the default stub first (so the row
    # actually exists), then patch the SDK for the /test call only.
    async def _reject(*_args, **_kwargs):
        return {"valid": False, "message": "Authentication Error", "error": "auth token invalid"}

    client = TestClient(_build_app())
    r1 = client.post(
        "/twilio-credentials",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Will fail", "account_sid": VALID_SID, "auth_token": VALID_TOKEN},
    )
    assert r1.status_code == 200, r1.text
    cred_id = r1.json()["id"]

    monkeypatch.setattr(_twilio_api_for_test, "validate_twilio_credentials", _reject)
    r2 = client.post(
        f"/twilio-credentials/{cred_id}/test",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["valid"] is False
    assert body["status"] == "invalid"
    assert body["credential"]["status"] == "invalid"
    assert "Authentication Error" in body["credential"]["last_test_message"]


def test_endpoint_list_excludes_secrets(auth_token, stub_twilio):
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    client.post(
        "/twilio-credentials",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"name": "Listed", "account_sid": VALID_SID, "auth_token": VALID_TOKEN},
    )
    r = client.get("/twilio-credentials", headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200
    creds = r.json()["credentials"]
    assert len(creds) == 1
    assert "account_sid" not in creds[0]
    assert "auth_token" not in creds[0]
    assert creds[0]["account_sid_last4"] == VALID_SID[-4:]


def test_settings_api_keys_no_longer_lists_twilio(auth_token):
    """After the refactor, /settings/api-keys must NOT include Twilio —
    the BE dropped it from PROVIDER_CATALOG-related fields. Settings →
    API now exposes Twilio exclusively via /twilio-credentials."""
    from fastapi.testclient import TestClient

    client = TestClient(_build_app())
    r = client.get("/settings/api-keys", headers={"Authorization": f"Bearer {auth_token}"})
    assert r.status_code == 200
    services = r.json()["services"]
    assert all(s["id"] != "twilio" for s in services), "Twilio must not appear under /settings/api-keys"


def test_platform_env_keys_dropped_twilio():
    """Twilio is no longer in PLATFORM_ENV_KEYS — there is no Railway
    fallback for Twilio credentials. The product brief requires this."""
    from STT_server.services.credentials_resolver import PLATFORM_ENV_KEYS
    assert "twilio" not in PLATFORM_ENV_KEYS
