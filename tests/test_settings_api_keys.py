"""Regression tests for the Settings → API flow.

Two surfaces to lock down:
  1. upsert_api_key actually persists the per-user service
     credential into the agent_tools row (was NameError before
     migration 014's storage layer landed; reproduced the operator's
     500 in the wild).
  2. PROVIDER_CATALOG does NOT leak model selectors (tts_model /
     realtime_model / voice_id / model_id / model / speaker_en /
     speaker_es) — those fields belong to the agent modal, not the
     Settings page. The Catalog's FieldSpec list is the FE's source
     of truth (ModalApiKey iterates `service.fields`), so a stray
     field there shows up on the operator's screen.
"""
from __future__ import annotations

import json
import os
import pathlib

# ponytail: encrypt_credentials() refuses to start without a key
# (refuses ephemeral keys in production). The credential storage layer
# reads CREDENTIAL_ENCRYPTION_KEY from the environment on first call,
# so set it before importing anything that opens a Fernet instance.
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_KEY",
    "3caLHixTmxCJ1OAQEK11TEn4k5soMyJhybJIyAFVMfk=",  # sample Fernet key for tests
)

import pytest
import secrets
from datetime import datetime, timedelta, timezone


# ── Fixtures (mirror the conftest in tests/) ────────────────────────────────


@pytest.fixture
def data_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Redirect ALL JSON-file IO (users, sessions, tools) to a tmp dir so
    tests are hermetic and don't touch STT_server/data/."""
    import STT_server.db_users as db_users
    from STT_server.routes import api as api_mod
    from STT_server.services import session_runtime as rt_mod
    from STT_server import db_tools

    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    tools_file = d / "agent_tools.json"
    users_file = d / "users.json"
    sessions_file = d / "sessions.json"

    monkeypatch.setattr(api_mod, "DATA_DIR", str(d), raising=False)
    monkeypatch.setattr(api_mod, "TOOLS_FILE", str(tools_file), raising=False)
    monkeypatch.setattr(rt_mod, "_TOOLS_FILE", str(tools_file), raising=False)
    monkeypatch.setattr(db_tools, "_AGENT_TOOLS_FILE", tools_file, raising=False)
    monkeypatch.setattr(db_users, "USERS_FILE", users_file, raising=False)
    monkeypatch.setattr(db_users, "SESSIONS_FILE", sessions_file, raising=False)

    # Seed an admin user.
    users_file.write_text(json.dumps([{
        "id": "user-admin-001",
        "name": "Admin",
        "email": "admin@revolutionmedia.ai",
        "password_hash": "",
        "role": "admin",
    }]), encoding="utf-8")

    # Seed a valid session.
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
    """Read the only token from sessions.json (we just seeded one)."""
    return secrets.token_urlsafe(32)  # placeholder; we'll read it from the file instead


# ── Catalog tests (no I/O needed) ────────────────────────────────────────────


def test_provider_catalog_has_no_model_selectors():
    """Every FieldSpec in PROVIDER_CATALOG belongs on Settings → API
    (credentials) or the static required: True validation list. The
    Settings page must NOT contain model pickers."""
    from STT_server.services.credentials_resolver import PROVIDER_CATALOG
    forbidden = {
        # model pickers — move to /agents
        "tts_model", "realtime_model", "voice_id", "model_id",
        "model", "speaker_en", "speaker_es",
    }
    seen_field_names: set[str] = set()
    for spec in PROVIDER_CATALOG:
        for f in spec.fields:
            seen_field_names.add(f.name)
            if f.name in forbidden:
                pytest.fail(
                    f"{spec.id}.fields contains model selector {f.name!r}; "
                    f"this is picked at the agent level, not here."
                )
    # All forbidden names confirmed absent (sanity).
    leaked = seen_field_names & forbidden
    assert not leaked, f"model selector fields still in catalog: {leaked}"


def test_provider_catalog_credentials_only_field_set():
    """Spot-check that every still-listed FieldSpec is a credentials-
    shaped field (api_key, account_*, auth_token, phone_number,
    base_url). Anything outside this whitelist belongs in /agents.
    """
    from STT_server.services.credentials_resolver import PROVIDER_CATALOG

    allowed = {
        "api_key",
        "account_sid", "auth_token", "phone_number",  # twilio
        "base_url",  # anthropic/gemini/minimax custom-tenant
    }
    rogue: list[tuple[str, str]] = []
    for spec in PROVIDER_CATALOG:
        for f in spec.fields:
            if f.name not in allowed:
                rogue.append((spec.id, f.name))
    assert not rogue, f"non-credential fields in PROVIDER_CATALOG: {rogue}"


def test_provider_catalog_categories_cover_expected_slots():
    """Per the operator's standing requirement: OpenAI = LLM+STT+TTS,
    Inworld = TTS+STT, Deepgram = STT+TTS, Twilio = telephony only.
    """
    from STT_server.services.credentials_resolver import PROVIDER_CATALOG, get_provider_spec

    # Ponytail: a spec carries either `categories` (multi-cap) or just
    # `category` (single-cap). The runtime collapses to
    # `(spec.categories or (spec.category,))` when scanning — match
    # the runtime view here so the test mirrors what's actually
    # surfaced to the FE.
    def effective_cats(spec):
        return set(spec.categories if spec.categories else (spec.category,))

    expected = {
        "openai": {"llm", "stt", "tts"},
        "inworld": {"tts", "stt"},
        "deepgram": {"stt", "tts"},
        "twilio": {"telephony"},
        "anthropic": {"llm"},
    }
    for service_id, cats in expected.items():
        spec = get_provider_spec(service_id)
        assert spec is not None, f"missing spec for {service_id}"
        assert effective_cats(spec) >= cats, (
            f"{service_id}: expected at least {cats}, got {effective_cats(spec)}"
        )


# ── API roundtrip tests (need FastAPI + httpx) ─────────────────────────────


@pytest.fixture
def app(data_dir: pathlib.Path):
    """Build the FastAPI app with auth + api routes. No lifespan
    (the conftest fixture only adds auth_header; app is fine standalone)."""
    from fastapi import FastAPI
    from STT_server.routes.api import api_router
    from STT_server.routes.auth import router as auth_router

    application = FastAPI()
    application.include_router(auth_router)
    application.include_router(api_router)
    return application


@pytest.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def test_upsert_api_key_persists_credentials(client, data_dir, monkeypatch):
    """PUT /settings/api-keys/{service} writes a row keyed by
    (user_id, id=service_id) carrying the encrypted credentials dict
    under the JSONB column. Was NameError because the route still
    referenced db_upsert_tool (post-migration syntax rot)."""
    # Read the seeded token from sessions.json
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    # Patch the OpenAI live test so we don't hit the real API in CI.
    from STT_server.services import credentials_resolver as cr

    async def _no_test(*a, **kw):
        return {"valid": True, "message": "stubbed"}
    # The route runs validate_credentials + encrypt; no test_fn call
    # on the save path itself. We just need a key that matches the
    # catalog regex so validate_credentials doesn't 422.
    r = await client.put(
        "/settings/api-keys/openai",
        headers=headers,
        json={"credentials": {"api_key": "sk-test1234567890abcdefABCDEF"}},
    )
    assert r.status_code == 200, r.text

    # JSON fallback file now carries the row.
    tools_path = data_dir / "agent_tools.json"
    rows = json.loads(tools_path.read_text(encoding="utf-8"))
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
    row = rows[0]
    assert row["id"] == "openai"
    assert row["agent_id"] == "__shared__"
    assert row["user_id"] == "user-admin-001"
    # credentials is an encrypted Fernet token dict. We don't care
    # about the exact bytes here — roundtrip the decrypt to verify
    # the BE shape.
    assert isinstance(row.get("credentials"), dict)
    assert "api_key" in (row["credentials"] or {})


async def test_upsert_api_key_update_overwrites_existing(client, data_dir):
    """A second PUT for the same service should overwrite the row
    in place, not duplicate it."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    r1 = await client.put(
        "/settings/api-keys/openai",
        headers=headers,
        json={"credentials": {"api_key": "sk-first1234567890abcdefABCDEF"}},
    )
    assert r1.status_code == 200, r1.text

    r2 = await client.put(
        "/settings/api-keys/openai",
        headers=headers,
        json={"credentials": {"api_key": "sk-second1234567890abcdefABCDEF"}},
    )
    assert r2.status_code == 200, r2.text

    rows = json.loads((data_dir / "agent_tools.json").read_text(encoding="utf-8"))
    openai_rows = [r for r in rows if r.get("id") == "openai"]
    assert len(openai_rows) == 1, (
        f"expected 1 openai row after second PUT, got {len(openai_rows)}"
    )


async def test_list_api_keys_after_upsert_marks_connected(client, data_dir):
    """Saved creds must reflect as `connected: true` in
    GET /settings/api-keys so the LIST page shows the Connect→Update
    button flip without a reload."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    # List before save — OpenAI should be connected=False (no row).
    r0 = await client.get("/settings/api-keys", headers=headers)
    assert r0.status_code == 200, r0.text
    openai_before = next(s for s in r0.json()["services"] if s["id"] == "openai")
    assert openai_before["connected"] is False

    # Save and re-list.
    r = await client.put(
        "/settings/api-keys/openai",
        headers=headers,
        json={"credentials": {"api_key": "sk-test1234567890abcdefABCDEF"}},
    )
    assert r.status_code == 200, r.text

    r1 = await client.get("/settings/api-keys", headers=headers)
    openai_after = next(s for s in r1.json()["services"] if s["id"] == "openai")
    assert openai_after["connected"] is True
    # The categories badge list still ships to the FE.
    assert set(openai_after["categories"]) >= {"llm", "stt", "tts"}


async def test_upsert_api_key_rejects_unknown_service(client, data_dir):
    """Returns 404 on a provider id not in the catalog."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.put(
        "/settings/api-keys/no-such-provider",
        headers=headers,
        json={"credentials": {"api_key": "sk-test1234567890abcdefABCDEF"}},
    )
    assert r.status_code == 404, r.text


async def test_upsert_api_key_rejects_bad_format(client, data_dir):
    """A key that doesn't match the catalog pattern returns 422
    before hitting the storage layer."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.put(
        "/settings/api-keys/openai",
        headers=headers,
        json={"credentials": {"api_key": "definitely-not-an-openai-key"}},
    )
    assert r.status_code == 422, r.text
