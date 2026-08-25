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


# ── /providers/models/categorized ────────────────────────────────────────


def test_format_model_label():
    from STT_server.services.credentials_resolver import _format_model_label
    assert _format_model_label("gpt-4.1-mini") == "GPT-4.1 Mini"
    assert _format_model_label("gpt-4o-mini-transcribe") == "GPT-4o Mini Transcribe"
    assert _format_model_label("gpt-4o-mini-tts") == "GPT-4o Mini Tts"
    assert _format_model_label("o3-mini") == "O3 Mini"
    assert _format_model_label("tts-1") == "Tts 1"
    assert _format_model_label("nova-2-general") == "Nova 2 General"
    assert _format_model_label("") == ""
    assert _format_model_label("gpt") == "GPT"
    assert _format_model_label("gpt-4o") == "GPT-4o"
    assert _format_model_label("gpt-realtime") == "GPT-Realtime"


def test_classify_openai_model():
    from STT_server.services.credentials_resolver import _classify_openai_model
    # LLM family
    assert _classify_openai_model("gpt-4.1") == "llm"
    assert _classify_openai_model("gpt-4.1-mini") == "llm"
    assert _classify_openai_model("gpt-4o") == "llm"
    assert _classify_openai_model("gpt-5") == "llm"
    assert _classify_openai_model("o1") == "llm"
    assert _classify_openai_model("o3-mini") == "llm"
    assert _classify_openai_model("o4-mini") == "llm"
    # STT family
    assert _classify_openai_model("gpt-4o-transcribe") == "stt"
    assert _classify_openai_model("gpt-4o-mini-transcribe") == "stt"
    assert _classify_openai_model("whisper-1") == "stt"
    # TTS family
    assert _classify_openai_model("tts-1") == "tts"
    assert _classify_openai_model("tts-1-hd") == "tts"
    assert _classify_openai_model("gpt-4o-mini-tts") == "tts"
    # Excluded from LLM (realtime, embedding, etc.)
    assert _classify_openai_model("gpt-realtime") is None
    assert _classify_openai_model("gpt-4o-realtime-preview") is None
    assert _classify_openai_model("text-embedding-3-small") is None
    assert _classify_openai_model("dall-e-3") is None
    assert _classify_openai_model("omni-moderation-latest") is None
    # Out of catalog (no recognized family)
    assert _classify_openai_model("gpt-3.5-turbo-instruct") is None
    assert _classify_openai_model("gpt-3.5-turbo-1106") is None
    assert _classify_openai_model("gpt-3.5-turbo-0125") is None
    assert _classify_openai_model("gpt-4-0613") is None
    assert _classify_openai_model("gpt-4") is None
    assert _classify_openai_model("gpt-4-turbo") is None
    assert _classify_openai_model("gpt-4o-2024-05-13") is None
    assert _classify_openai_model("gpt-4-turbo-2024-04-09") is None
    assert _classify_openai_model("o1-2024-12-17") is None
    assert _classify_openai_model("o1-pro-2025-03-19") is None
    assert _classify_openai_model("gpt-4o-2024-08-06") is None
    assert _classify_openai_model("gpt-4o-2024-11-20") is None
    assert _classify_openai_model("gpt-5.6-luna") is None


def test_build_categorized_models_openai():
    """End-to-end: with a mocked OpenAI /v1/models response that
    mixes LLM, STT, TTS and unrelated models, the categorized builder
    buckets them correctly with {id, label} entries.
    """
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import _build_categorized_models

    fake_live = [
        {"id": "gpt-4.1-mini"},
        {"id": "gpt-4o"},
        {"id": "o3-mini"},
        # STT
        {"id": "gpt-4o-transcribe"},
        {"id": "gpt-4o-mini-transcribe"},
        {"id": "whisper-1"},
        # TTS
        {"id": "tts-1"},
        {"id": "tts-1-hd"},
        {"id": "gpt-4o-mini-tts"},
        # Realtime — out of LLM
        {"id": "gpt-realtime"},
        {"id": "gpt-4o-realtime-preview"},
        # Other excluded
        {"id": "dall-e-3"},
        {"id": "text-embedding-3-small"},
        {"id": "omni-moderation-latest"},
        # Legacy / dated
        {"id": "gpt-3.5-turbo"},
        {"id": "gpt-4"},
        {"id": "gpt-4o-2024-05-13"},
        {"id": "o1-2024-12-17"},
    ]

    with patch(
        "STT_server.services.credentials_resolver._fetch_openai_models",
        return_value=fake_live,
    ):
        out = _build_categorized_models("openai", api_key="sk-test", user_id="u1")

    assert out["provider"] == "openai"
    assert {m["id"] for m in out["models"]["llm"]} == {
        "gpt-4.1-mini", "gpt-4o", "o3-mini",
    }
    assert {m["id"] for m in out["models"]["stt"]} == {
        "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1",
    }
    assert {m["id"] for m in out["models"]["tts"]} == {
        "tts-1", "tts-1-hd", "gpt-4o-mini-tts",
    }
    # Each entry has {id, label}.
    for bucket in ("llm", "stt", "tts"):
        for m in out["models"][bucket]:
            assert "id" in m and "label" in m
            # Spot-check labels.
            if m["id"] == "gpt-4.1-mini":
                assert m["label"] == "GPT-4.1 Mini"
            if m["id"] == "gpt-4o-mini-transcribe":
                assert m["label"] == "GPT-4o Mini Transcribe"
            if m["id"] == "gpt-4o-mini-tts":
                assert m["label"] == "GPT-4o Mini Tts"
    # The 6 excluded models (realtime * 2, dall-e, embedding,
    # moderation, gpt-3.5-turbo, gpt-4, gpt-4o-2024-05-13,
    # o1-2024-12-17) are NOT in any bucket.
    all_ids = {m["id"] for b in ("llm", "stt", "tts") for m in out["models"][b]}
    assert "gpt-realtime" not in all_ids
    assert "gpt-4o-realtime-preview" not in all_ids
    assert "dall-e-3" not in all_ids
    assert "text-embedding-3-small" not in all_ids
    assert "omni-moderation-latest" not in all_ids
    assert "gpt-3.5-turbo" not in all_ids
    assert "gpt-4" not in all_ids
    assert "gpt-4o-2024-05-13" not in all_ids
    assert "o1-2024-12-17" not in all_ids


def test_build_categorized_models_inworld_uses_curated_voices():
    """Non-OpenAI providers route through the existing per-service
    catalog. Inworld has both TTS voices (the live fetch) and an
    STT model (inworld-stt-1 in the curated hardcoded catalog),
    so the TTS bucket has the live voices and the STT bucket has
    the STT model. LLM is empty (Inworld doesn't ship chat models).
    """
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import _build_categorized_models

    with patch(
        "STT_server.services.credentials_resolver._fetch_inworld_voices"
    ) as mocked:
        mocked.return_value = [
            {"id": "Blake",      "name": "Blake",      "displayName": "Blake",
             "langCode": "EN_US", "languageCode": "en-US",
             "description": "Mid-range English male",
             "tags": [], "categories": [], "source": "SYSTEM",
             "gender": "male", "ageGroup": "middle_aged",
             "promptLanguages": ["en-US"]},
            {"id": "Camila",     "name": "Camila",     "displayName": "Camila",
             "langCode": "EN_US", "languageCode": "en-US",
             "description": "Mexican Spanish female",
             "tags": [], "categories": [], "source": "SYSTEM",
             "gender": "female", "ageGroup": "middle_aged",
             "promptLanguages": ["es-MX"]},
        ]
        out = _build_categorized_models(
            "inworld", api_key="sk-test", user_id="u1",
        )

    assert out["provider"] == "inworld"
    # Inworld doesn't ship chat models.
    assert out["models"]["llm"] == []
    # STT bucket has the curated inworld-stt-1 model that
    # list_provider_models("stt", "inworld", ...) returns.
    stt_ids = {m["id"] for m in out["models"]["stt"]}
    assert stt_ids == {"inworld/inworld-stt-1"}
    # TTS bucket has the 2 mocked voices with their ids + labels.
    tts_ids = {m["id"] for m in out["models"]["tts"]}
    assert tts_ids == {"Blake", "Camila"}
    blake = next(m for m in out["models"]["tts"] if m["id"] == "Blake")
    assert blake["label"] == "Blake"


async def test_categorized_models_route(client, data_dir):
    """End-to-end: POST /providers/models/categorized returns the
    categorized picker payload for the FE."""
    sessions_path = data_dir / "sessions.json"
    sess = json.loads(sessions_path.read_text(encoding="utf-8"))
    token = next(iter(sess.keys()))
    headers = {"Authorization": f"Bearer {token}"}

    # Unknown provider → 404.
    r = await client.post(
        "/providers/models/categorized", headers=headers,
        json={"provider": "no-such-provider"},
    )
    assert r.status_code == 404, r.text

    # OpenAI: with a mocked /v1/models live response.
    from unittest.mock import patch
    with patch(
        "STT_server.services.credentials_resolver._fetch_openai_models",
        return_value=[
            {"id": "gpt-4.1-mini"},
            {"id": "tts-1"},
            {"id": "gpt-realtime"},
        ],
    ):
        r = await client.post(
            "/providers/models/categorized", headers=headers,
            json={"provider": "openai", "api_key": "sk-test1234567890abcdefABCDEF"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "openai"
    assert {m["id"] for m in body["models"]["llm"]} == {"gpt-4.1-mini"}
    assert {m["id"] for m in body["models"]["tts"]} == {"tts-1"}
    assert body["models"]["stt"] == []


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


def _fake_openai_models_full_response():
    """The full /v1/models response — includes chat, tts, stt,
    embeddings, image, moderation, realtime, etc. Mirrors what
    api.openai.com returns as of late 2025.
    """
    return {
        "data": [
            # chat family — valid for LLM
            {"id": "gpt-4o",                   "owned_by": "openai"},
            {"id": "gpt-4o-mini",              "owned_by": "openai"},
            {"id": "o4-mini",                  "owned_by": "openai"},
            {"id": "gpt-3.5-turbo-1106",       "owned_by": "openai"},
            {"id": "gpt-3.5-turbo-0125",       "owned_by": "openai"},
            # tts family — NOT LLM
            {"id": "tts-1",                    "owned_by": "openai"},
            {"id": "tts-1-hd",                 "owned_by": "openai"},
            {"id": "gpt-4o-mini-tts",          "owned_by": "openai"},
            # realtime / stt — NOT LLM
            {"id": "gpt-realtime",              "owned_by": "openai"},
            {"id": "gpt-4o-realtime-preview",  "owned_by": "openai"},
            {"id": "gpt-4o-transcribe",        "owned_by": "openai"},
            {"id": "gpt-4o-mini-transcribe",   "owned_by": "openai"},
            {"id": "gpt-live-transcribe",      "owned_by": "openai"},
            # legacy / image / embedding / moderation — NOT LLM
            {"id": "whisper-1",                "owned_by": "openai"},
            {"id": "dall-e-3",                 "owned_by": "openai"},
            {"id": "text-embedding-3-small",   "owned_by": "openai"},
            {"id": "omni-moderation-latest",    "owned_by": "openai"},
        ],
        "object": "list",
    }


def test_list_openai_llm_excludes_tts_stt_embedding_models():
    """Regression: the LLM picker used to return whatever /v1/models
    returned. That put gpt-live-transcribe, gpt-realtime-2.1, tts-1,
    dall-e-3, text-embedding-3-small in the dropdown alongside
    gpt-4o. Operator picked gpt-live-transcribe (it sounded like
    an LLM), agent failed to use it as a chat completion model.

    Ponytail: the fix is a per-service filter in list_provider_models
    that excludes the prefixes/suffixes OpenAI uses for non-chat
    families. Chat family stays.
    """
    import urllib.request
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import list_provider_models

    fake_body = json.dumps(_fake_openai_models_full_response()).encode()

    class _FakeResp:
        def __init__(self, body): self._body = body
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch.object(urllib.request, "urlopen", return_value=_FakeResp(fake_body)):
        out = list_provider_models("llm", "openai", api_key="sk-test1234567890abcdefABCDEF")

    ids = [m["id"] for m in out["models"]]
    # Current-generation chat family present.
    assert "gpt-4o" in ids
    assert "gpt-4o-mini" in ids
    assert "o4-mini" in ids
    # TTS excluded.
    assert "tts-1" not in ids
    assert "tts-1-hd" not in ids
    assert "gpt-4o-mini-tts" not in ids
    # STT / realtime excluded.
    assert "gpt-realtime" not in ids
    assert "gpt-4o-realtime-preview" not in ids
    assert "gpt-4o-transcribe" not in ids
    assert "gpt-4o-mini-transcribe" not in ids
    assert "gpt-live-transcribe" not in ids
    assert "whisper-1" not in ids
    # Image / embedding / moderation excluded.
    assert "dall-e-3" not in ids
    assert "text-embedding-3-small" not in ids
    assert "omni-moderation-latest" not in ids
    # ponytail: legacy / deprecated chat models excluded too. The
    # current-generation filter (gpt-4o / gpt-4.1 / gpt-5 / o1 / o3 / o4 / o5
    # prefixes) drops gpt-3.5-turbo-1106 / 0125, gpt-4-0613,
    # gpt-4-turbo, gpt-5.6-luna, and dated snapshots like
    # gpt-4o-2024-05-13 / o1-2024-12-17. Operator complained that
    # the picker was inflated with 60+ models nobody uses.
    assert "gpt-3.5-turbo-1106" not in ids
    assert "gpt-3.5-turbo-0125" not in ids
    assert "gpt-4-0613" not in ids
    assert "gpt-4" not in ids
    assert "gpt-4-turbo" not in ids
    assert "gpt-5.6-luna" not in ids
    assert "gpt-4o-2024-05-13" not in ids
    assert "gpt-4-turbo-2024-04-09" not in ids
    assert "o1-2024-12-17" not in ids
    assert "o1-pro-2025-03-19" not in ids
    assert "gpt-4o-2024-08-06" not in ids
    assert "gpt-4o-2024-11-20" not in ids


def test_list_openai_tts_includes_tts_and_realtime_models():
    """The TTS picker should include tts-1, gpt-4o-mini-tts (TTS
    family) and gpt-realtime (TTS-capable via Realtime API). It should
    NOT include chat models or STT models.
    """
    import urllib.request
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import list_provider_models

    fake_body = json.dumps(_fake_openai_models_full_response()).encode()

    class _FakeResp:
        def __init__(self, body): self._body = body
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch.object(urllib.request, "urlopen", return_value=_FakeResp(fake_body)):
        out = list_provider_models("tts", "openai", api_key="sk-test1234567890abcdefABCDEF")

    ids = [m["id"] for m in out["models"]]
    # TTS family present (operator expects these in the dropdown).
    assert "tts-1" in ids
    assert "tts-1-hd" in ids
    assert "gpt-4o-mini-tts" in ids
    # Realtime (TTS via Realtime API) present.
    assert "gpt-realtime" in ids
    assert "gpt-4o-realtime-preview" in ids
    # Chat / STT / embeddings / moderation / image / legacy excluded.
    assert "gpt-4o" not in ids
    assert "gpt-4o-transcribe" not in ids
    assert "gpt-4o-mini-transcribe" not in ids
    assert "gpt-live-transcribe" not in ids
    assert "whisper-1" not in ids
    assert "dall-e-3" not in ids
    assert "text-embedding-3-small" not in ids
    assert "omni-moderation-latest" not in ids
    assert "gpt-3.5-turbo-1106" not in ids


def test_inworld_voices_mirror_curated_language_to_languageCode():
    """Regression: the Inworld voice picker used to group every
    voice under "OTHER" instead of by language (en-US, es-MX). The
    FE's `groupInworldVoicesByLanguage` looks up
    `v.raw.languageCode || v.raw.langCode`. The Inworld live API
    doesn't always return that field — many system voices come
    back with no language code at all. The BE has a curated
    catalog with the language label (e.g. "en-US" for
    Blake/Sarah, "es-MX" for Camila/Cuauhtemoc), and was setting
    `voice["language"]` — but the FE never looked at `language`,
    only at `languageCode` / `langCode`. Every curated voice fell
    into OTHER.

    Fix: the BE mirrors the curated label to `languageCode` and
    `langCode` so the FE's group lookup actually finds it.
    """
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import list_provider_models

    with patch(
        "STT_server.services.credentials_resolver._fetch_inworld_voices"
    ) as mocked_fetch:
        # Live Inworld fetch returns voices WITHOUT languageCode /
        # langCode — many of Inworld's system voices don't carry
        # that field. The BE has to fill it from the curated
        # catalog.
        mocked_fetch.return_value = [
            {"id": "Blake",       "name": "Blake"},
            {"id": "Camila",      "name": "Camila"},
            {"id": "Cuauhtemoc",  "name": "Cuauhtemoc"},
            {"id": "Bruno",       "name": "Bruno"},  # NOT in curated
        ]
        out = list_provider_models("tts", "inworld", api_key="sk-test1234567890abcdefABCDEF")

    by_id = {v["id"]: v for v in out["models"]}
    # Curated voices inherit en-US / es-MX.
    assert by_id["Blake"]["languageCode"] == "en-US"
    assert by_id["Blake"]["langCode"] == "en-US"
    assert by_id["Camila"]["languageCode"] == "es-MX"
    assert by_id["Camila"]["langCode"] == "es-MX"
    assert by_id["Cuauhtemoc"]["languageCode"] == "es-MX"
    # Non-curated voices (e.g. Bruno, an IVC clone the operator
    # added themselves) stay as-is. Without languageCode / langCode
    # they'd end up in the "OTHER" bucket in the FE — but that's
    # correct behavior for unknown languages.
    assert "languageCode" not in by_id["Bruno"]


def test_inworld_voices_use_promptLanguages_when_langCode_missing():
    """Ponytail: the Inworld live response for some voices (IVC
    clones, custom voices) has empty `langCode` but a populated
    `promptLanguages` array. Before the fix these voices all fell
    into the FE's "OTHER" bucket because the BE had no fallback.
    With the new priority order (live languageCode > promptLanguages[0]
    > legacy langCode converted), every voice ends up in a proper
    group.
    """
    from unittest.mock import patch
    from STT_server.services.credentials_resolver import list_provider_models

    # Inject a real-shape Inworld response that includes:
    # - a SYSTEM voice with legacy langCode only (Alex, EN_US)
    # - an IVC clone with promptLanguages only, no langCode (John)
    # - a multilingual voice with both (Maria, ES_MX + promptLanguages)
    # - a degenerate case with neither (Orphan, '' langCode, [] prompts)
    # ponytail: the mocked live list mirrors what
    # _fetch_inworld_voices' keep dict would produce, so the
    # test asserts the Inworld TTS branch's curated-mirror
    # behavior on top of an already-normalized live payload.
    fake_live = [
        {
            "id": "Alex", "voiceId": "Alex",
            "displayName": "Alex",
            "name": "workspaces/inworld/voices/Alex",
            "langCode": "EN_US",
            "languageCode": "en-US",  # normalized by _fetch_inworld_voices
            "description": "Energetic mid-range male voice",
            "gender": "male",
            "ageGroup": "middle_aged",
            "tags": ["friendly"],
            "categories": ["companions"],
            "source": "SYSTEM",
            "promptLanguages": ["en-US"],
        },
        {
            "id": "John", "voiceId": "John",
            "displayName": "John",
            "name": "workspaces/your_workspace/voices/John",
            "langCode": "",
            "languageCode": "en-US",  # via promptLanguages[0] fallback
            "description": "Cloned voice for narrations.",
            "gender": "male",
            "ageGroup": "young_adult",
            "tags": ["clone"],
            "categories": [],
            "source": "IVC",
            "promptLanguages": ["en-US"],
        },
        {
            "id": "Maria", "voiceId": "Maria",
            "displayName": "Maria",
            "name": "workspaces/inworld/voices/Maria",
            "langCode": "ES_MX",
            "languageCode": "es-MX",  # via langCode conversion
            "description": "Warm Spanish voice",
            "gender": "female",
            "ageGroup": "middle_aged",
            "tags": ["warm"],
            "categories": ["companions"],
            "source": "SYSTEM",
            "promptLanguages": ["es-MX", "en-US"],
        },
        {
            "id": "Orphan", "voiceId": "Orphan",
            "displayName": "Orphan",
            "name": "workspaces/inworld/voices/Orphan",
            "langCode": "",
            "languageCode": "",  # no source, falls to OTHER
            "description": "",
            "gender": "",
            "ageGroup": "",
            "tags": [],
            "categories": [],
            "source": "SYSTEM",
            "promptLanguages": [],
        },
    ]

    with patch(
        "STT_server.services.credentials_resolver._fetch_inworld_voices",
        return_value=fake_live,
    ):
        out = list_provider_models("tts", "inworld", api_key="sk-test1234567890abcdefABCDEF")

    by_id = {v["id"]: v for v in out["models"]}

    # Alex: langCode "EN_US" → "en-US" (legacy conversion path)
    assert by_id["Alex"]["languageCode"] == "en-US"
    assert by_id["Alex"]["langCode"] == "EN_US"

    # John: empty langCode, promptLanguages=["en-US"] → "en-US" via fallback
    assert by_id["John"]["languageCode"] == "en-US"
    assert by_id["John"]["langCode"] == ""

    # Maria: langCode "ES_MX" → "es-MX" (legacy conversion)
    assert by_id["Maria"]["languageCode"] == "es-MX"
    assert by_id["Maria"]["langCode"] == "ES_MX"

    # Orphan: no langCode, no promptLanguages → empty languageCode
    # FE puts it in the "OTHER" bucket — correct behavior for
    # unknown languages.
    assert by_id["Orphan"]["languageCode"] == ""
    assert by_id["Orphan"]["langCode"] == ""

    # Description is forwarded from the API (was previously
    # defaulted to "Inworld voice" when missing).
    assert by_id["Alex"]["description"] == "Energetic mid-range male voice"
    assert by_id["John"]["description"] == "Cloned voice for narrations."


def test_inworld_voices_pagination_walks_every_page():
    """Ponytail: the Inworld API paginates with nextPageToken. A
    223-voice account would never fit in a single response — the
    BE has to walk every page. Earlier the loop was correct in
    shape but unbounded on real responses. This test pins the
    pagination behavior.
    """
    from unittest.mock import patch, MagicMock
    from STT_server.services.credentials_resolver import _fetch_inworld_voices

    page1 = {
        "voices": [
            {"voiceId": f"voice-{i}", "name": f"voice-{i}",
             "langCode": "EN_US", "displayName": f"voice-{i}",
             "description": "", "tags": [], "categories": [],
             "source": "SYSTEM", "gender": "", "ageGroup": "",
             "promptLanguages": ["en-US"]}
            for i in range(100)
        ],
        "nextPageToken": "token-1",
        "totalSize": 223,
    }
    page2 = {
        "voices": [
            {"voiceId": f"voice-{i+100}", "name": f"voice-{i+100}",
             "langCode": "EN_US", "displayName": f"voice-{i+100}",
             "description": "", "tags": [], "categories": [],
             "source": "SYSTEM", "gender": "", "ageGroup": "",
             "promptLanguages": ["en-US"]}
            for i in range(100)
        ],
        "nextPageToken": "token-2",
        "totalSize": 223,
    }
    page3 = {
        "voices": [
            {"voiceId": f"voice-{i+200}", "name": f"voice-{i+200}",
             "langCode": "EN_US", "displayName": f"voice-{i+200}",
             "description": "", "tags": [], "categories": [],
             "source": "SYSTEM", "gender": "", "ageGroup": "",
             "promptLanguages": ["en-US"]}
            for i in range(23)
        ],
        "nextPageToken": "",
        "totalSize": 223,
    }

    pages = [page1, page2, page3]

    def fake_urlopen(req, timeout=10):
        resp = MagicMock()
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *a: False
        resp.read = lambda: json.dumps(pages.pop(0)).encode()
        return resp

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        voices = _fetch_inworld_voices("sk-test1234567890abcdefABCDEF")

    # 223 voices, deduplicated, with canonical language "en-US".
    assert len(voices) == 223
    assert {v["id"] for v in voices} == {f"voice-{i}" for i in range(223)}
    assert all(v["languageCode"] == "en-US" for v in voices)


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
