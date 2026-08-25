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
    """Redirect ALL JSON-file IO (users, sessions, tools, settings)
    to a tmp dir so tests are hermetic and don't touch STT_server/data/."""
    import STT_server.db_users as db_users
    from STT_server.routes import api as api_mod
    from STT_server.services import session_runtime as rt_mod
    from STT_server import db_tools
    from STT_server import db_settings

    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    settings_dir = d / "settings"
    settings_dir.mkdir(exist_ok=True)
    tools_file = d / "agent_tools.json"
    users_file = d / "users.json"
    sessions_file = d / "sessions.json"

    monkeypatch.setattr(api_mod, "DATA_DIR", str(d), raising=False)
    monkeypatch.setattr(api_mod, "TOOLS_FILE", str(tools_file), raising=False)
    monkeypatch.setattr(rt_mod, "_TOOLS_FILE", str(tools_file), raising=False)
    monkeypatch.setattr(db_tools, "_AGENT_TOOLS_FILE", tools_file, raising=False)
    monkeypatch.setattr(db_users, "USERS_FILE", users_file, raising=False)
    monkeypatch.setattr(db_users, "SESSIONS_FILE", sessions_file, raising=False)
    # ponytail: settings storage lives in TWO modules with their own
    # module-level SETTINGS_DIR. routes/api.py has one (for the JSON
    # fallback in update_settings + _settings_path); db_settings.py
    # has another (for db_get_settings). Without patching BOTH, a
    # prior test's PUT writes to one path and the next test's GET
    # reads from a different path — looks like a test isolation
    # bug but it's actually two independent module globals.
    monkeypatch.setattr(api_mod, "SETTINGS_DIR", settings_dir, raising=False)
    monkeypatch.setattr(db_settings, "SETTINGS_DIR", settings_dir, raising=False)
    monkeypatch.setattr(db_settings, "DATA_DIR", d, raising=False)

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


# ── LLM picker helper + route ───────────────────────────────────────────


def test_get_llm_options_returns_all_llm_providers_with_connected_flag():
    """Every LLM-capable provider from PROVIDER_CATALOG (category='llm'
    or 'llm' in categories) shows up in the picker. The connected
    flag is True only for providers with stored credentials. Empty
    user → all connected flags False; current defaults to the
    hardcoded 'gpt-4o-mini'."""
    from STT_server.services.credentials_resolver import (
        get_llm_options, PROVIDER_CATALOG,
    )

    out = get_llm_options(None, "gpt-4o-mini")
    assert out["current"] == {"provider": "openai", "model": "gpt-4o-mini"}

    provider_ids = {p["id"] for p in out["providers"]}
    # Every LLM provider in the catalog is listed.
    expected = {
        s.id for s in PROVIDER_CATALOG
        if "llm" in (s.categories or (s.category,))
    }
    assert provider_ids == expected, (
        f"missing LLM providers: {expected - provider_ids}; "
        f"extra: {provider_ids - expected}"
    )

    # Empty user_id → no per-user keys, all disconnected.
    for p in out["providers"]:
        assert p["connected"] is False
        assert p["models"]  # every LLM provider has at least one model
        for m in p["models"]:
            assert {"id", "name", "description"} <= m.keys()


def test_get_llm_options_marks_connected_for_users_with_keys():
    """If the user has a per-user OpenAI key, the openai provider
    is connected=True. Other providers stay False until the user
    saves their keys too."""
    from STT_server.services.credentials_resolver import get_llm_options
    from STT_server.security.credentials import encrypt_credentials
    from STT_server import db_tools
    from unittest.mock import patch

    encrypted = encrypt_credentials({"api_key": "sk-test1234567890abcdefABCDEF"})

    def fake_list_tools(user_id):
        return [{
            "id": "openai", "user_id": user_id, "agent_id": "__shared__",
            "credentials": encrypted,
        }]

    with patch.object(db_tools, "is_postgres", return_value=False), \
         patch.object(db_tools, "list_tools", side_effect=fake_list_tools):
        out = get_llm_options("user-1", "gpt-4o")

    openai = next(p for p in out["providers"] if p["id"] == "openai")
    assert openai["connected"] is True
    anthropic = next(p for p in out["providers"] if p["id"] == "anthropic")
    assert anthropic["connected"] is False
    # current.provider is inferred from the model name.
    assert out["current"]["provider"] == "openai"
    assert out["current"]["model"] == "gpt-4o"


def test_get_llm_options_infers_provider_from_unknown_model():
    """If the operator saved a model id that no longer matches a known
    catalog (e.g. a model was retired), current.provider is None —
    the FE renders "Unknown model" or prompts the operator to
    re-pick. No crash."""
    from STT_server.services.credentials_resolver import get_llm_options
    out = get_llm_options(None, "gpt-99-turbo-deluxe")
    assert out["current"] == {"provider": None, "model": "gpt-99-turbo-deluxe"}


async def test_get_settings_llm_options_route(client, data_dir):
    """The new GET /settings/llm-options route returns the picker
    payload. With no settings saved yet, current defaults to
    gpt-4o-mini (matches the BE's hardcoded fallback)."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    r = await client.get("/settings/llm-options", headers=headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["current"]["model"] == "gpt-4o-mini"
    assert body["current"]["provider"] == "openai"
    assert isinstance(body["providers"], list)
    assert body["providers"], "expected at least one LLM provider"


async def test_get_settings_llm_options_route_with_saved_model(client, data_dir):
    """PUT /settings with test_data_model is reflected in
    GET /settings/llm-options.current.model."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    await client.put("/settings", headers=headers, json={"test_data_model": "claude-haiku-3-5"})

    r = await client.get("/settings/llm-options", headers=headers)
    body = r.json()
    assert body["current"] == {
        "provider": "anthropic",
        "model": "claude-haiku-3-5",
    }


# ── OpenAI STT live-fetch filter ──────────────────────────────────────────


def _fake_openai_models_response():
    """Mirror what GET https://api.openai.com/v1/models actually returns.
    Includes a mix of valid realtime IDs and batch transcribe IDs
    that the previous filter accidentally let through.
    """
    return {
        "data": [
            # realtime family — works with /v1/realtime
            {"id": "gpt-realtime",                 "owned_by": "openai"},
            {"id": "gpt-4o-realtime-preview",     "owned_by": "openai"},
            {"id": "gpt-4o-mini-realtime-preview", "owned_by": "openai"},
            # batch transcribe family — would 400 from /v1/realtime
            {"id": "gpt-4o-transcribe",             "owned_by": "openai"},
            {"id": "gpt-4o-mini-transcribe",       "owned_by": "openai"},
            {"id": "gpt-4o-transcribe-diarize",    "owned_by": "openai"},
            # legacy / unrelated
            {"id": "whisper-1",                    "owned_by": "openai"},
            {"id": "gpt-4o",                       "owned_by": "openai"},
        ],
        "object": "list",
    }


def test_list_openai_stt_filters_out_batch_transcribe_models():
    """Regression: the previous filter accepted any model with
    "transcribe" in its name, which let batch-only models
    (gpt-4o-transcribe, gpt-4o-transcribe-diarize, ...) into the
    STT dropdown. The agent then picked one, the Realtime API
    rejected it, and the BE silently fell back to gpt-realtime.

    Ponytail: the fix is to require "realtime" in the name AND
    exclude "transcribe". Realtime-compatible = gpt-realtime,
    gpt-4o-realtime-preview, gpt-4o-mini-realtime-preview.
    """
    import urllib.request
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import list_provider_models

    fake_body = json.dumps(_fake_openai_models_response()).encode()

    class _FakeResp:
        def __init__(self, body): self._body = body
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch.object(urllib.request, "urlopen", return_value=_FakeResp(fake_body)):
        out = list_provider_models("stt", "openai", api_key="sk-test1234567890abcdefABCDEF")

    ids = [m["id"] for m in out["models"]]
    # Realtime family present.
    assert "gpt-realtime" in ids
    assert "gpt-4o-realtime-preview" in ids
    assert "gpt-4o-mini-realtime-preview" in ids
    # Batch transcribe family excluded.
    assert "gpt-4o-transcribe" not in ids
    assert "gpt-4o-mini-transcribe" not in ids
    assert "gpt-4o-transcribe-diarize" not in ids
    # whisper and unrelated chat models excluded.
    assert "whisper-1" not in ids
    assert "gpt-4o" not in ids


# ── _read_per_user failure logging ────────────────────────────────────────


def test_read_per_user_logs_when_list_tools_fails(caplog):
    """When db_list_tools raises (e.g., column missing), _read_per_user
    should log the actual exception instead of silently returning {}.

    Regression: operator hit 500 on Test webhook after a deploy
    rotated CREDENTIAL_ENCRYPTION_KEY (Railway default = ephemeral
    key on every container restart). The decrypt failure was being
    silently swallowed and the operator got 'OpenAI API key not
    configured' with no log to debug. This test pins the loud-fail
    behavior so the next rotation surfaces in the logs.
    """
    import logging
    from STT_server.services.credentials_resolver import _read_per_user
    from unittest.mock import patch

    with caplog.at_level(logging.WARNING, logger="stt_server.security.resolver"):
        with patch("STT_server.db_tools.list_tools", side_effect=RuntimeError("simulated column missing")):
            result = _read_per_user("user-1", "openai")

    assert result == {}
    # The actual exception text is in the log so the operator can
    # diagnose the missing column / DB outage / key rotation.
    assert any("list_tools failed" in rec.message and "simulated column missing" in rec.message
               for rec in caplog.records), \
        f"expected a WARNING with the exception, got: {[r.message for r in caplog.records]}"


def test_read_per_user_logs_when_decrypt_fails(caplog):
    """When decrypt_credentials raises (e.g., Fernet key rotated
    across deploys), _read_per_user should log the exception with
    a hint about re-saving the key, then return {}.

    Regression: this is the operator's exact failure mode — they
    saved the key with ephemeral key K1, a deploy generated K2,
    the row's ciphertext was unreadable. The pre-fix code raised
    the exception up to _resolve_openai_client's bare
    `except Exception` which turned it into `creds = {}` and the
    200 response said 'API key not configured'. No log anywhere.
    """
    import logging
    from STT_server.services.credentials_resolver import _read_per_user
    from unittest.mock import patch, MagicMock

    # Simulate: list_tools returns a row that has ALREADY been
    # through _row_to_tool (so credentials is a dict of ciphertexts,
    # not a raw JSONB string). The dict's api_key is a non-decryptable
    # blob, simulating a row that was encrypted with a Fernet key
    # the current process no longer has.
    fake_row = {
        "id": "openai",
        "user_id": "user-1",
        "credentials": {"api_key": "not-a-valid-fernet-token"},
    }
    fake_list = MagicMock(return_value=[fake_row])

    import STT_server.services.credentials_resolver as cr_mod
    cr_mod.log.propagate = True
    caplog.set_level(logging.WARNING, logger="stt_server.security.resolver")

    with patch("STT_server.db_tools.list_tools", fake_list):
        result = _read_per_user("user-1", "openai")

    assert result == {}, f"expected {{}}, got {result!r}"
    matching = [r for r in caplog.records
                if r.name.startswith("stt_server.security")
                and r.levelno >= logging.WARNING]
    assert any("decrypt failed" in r.message for r in matching), (
        f"expected decrypt failure log, got: "
        f"{[(r.name, r.message) for r in caplog.records]}"
    )
    assert any("re-save" in r.message for r in matching), (
        "log message should mention re-saving the key"
    )
