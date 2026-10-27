"""Pytest fixtures for BE tests.

Builds a minimal FastAPI app around the routes we want to test
(`api_router` + `auth_router`), without importing the heavy call
adapters (deepgram/inworld/assemblyai/openai_realtime) that
STT_Server.py pulls in. Tests redirect the JSON-file backend to a
tmp dir so they never touch production data in STT_server/data/.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncIterator

import sys
import types

import pytest


def _stub_absent_optional_deps() -> list[str]:
    """Register minimal stand-ins for DECLARED runtime dependencies that
    are absent from this dev environment, and only for those.

    A real installation always wins: we import the genuine module first
    and fall back to a stub only on ImportError, so this can never shadow
    production behaviour where the dependency exists.

    Needed because `import STT_server.STT_Server` (the real app, required
    to test the real /voice routes) pulls these at module scope:

      python_multipart — Starlette ASSERTS it is importable before it will
        parse ANY form body, but only actually uses the parser when the
        content-type is multipart/form-data. Twilio sends
        application/x-www-form-urlencoded, which takes the stdlib
        unquote_plus path. This is not hypothetical: requirements.txt
        documents a production incident where a missing python-multipart
        made Request.form() raise, the route swallowed it into {}, and
        /voice answered every call with no agent_id.
      openai / webrtcvad — STT/LLM concerns; the call-routing paths never
        touch them.

    This must run BEFORE anything imports starlette/fastapi, because
    starlette resolves the multipart import at module import time.
    """
    stubbed: list[str] = []

    try:
        import python_multipart  # noqa: F401
    except ImportError:
        from urllib.parse import unquote_plus

        pkg = types.ModuleType("python_multipart")
        sub = types.ModuleType("python_multipart.multipart")

        def _parse_options_header(value):
            """Stand-in for python_multipart.parse_options_header.

            Returns the media type as LOWERCASE BYTES plus an options
            dict. The bytes part is load-bearing: Starlette compares the
            result against b"application/x-www-form-urlencoded", so a stub
            returning a str falls through to the empty-FormData branch
            and every form parses as {}.
            """
            if not value:
                return b"", {}
            parts = str(value).split(";")
            media_type = parts[0].strip().lower().encode("latin-1")
            options: dict = {}
            for raw in parts[1:]:
                if "=" in raw:
                    key, val = raw.split("=", 1)
                    options[key.strip().lower().encode("latin-1")] = (
                        val.strip().strip('"').encode("latin-1")
                    )
            return media_type, options

        class _QuerystringParser:
            """Stand-in for python_multipart.QuerystringParser.

            Starlette 0.52 does NOT use parse_qsl for urlencoded bodies; it
            drives this parser's write()/finalize() callbacks. So the stub
            has to actually parse, not just exist.

            Implements the documented application/x-www-form-urlencoded
            grammar: split on '&', split each pair on the first '=', emit
            the field callback sequence.

            CONTRACT THAT MATTERS: the bytes handed to the callbacks are
            RAW, still percent-encoded. Starlette does its own
            `unquote_plus(field.decode('latin-1'))` afterwards, so a
            parser that pre-decodes gets double-decoded and every '+'
            turns into a space — which silently corrupts the E.164
            destination in Twilio's `To` field.

            KNOWN LIMIT, stated rather than hidden: values are split on
            the latin-1 byte view, matching what Starlette decodes.
            Files are not part of this grammar, so there is nothing to
            drop.
            """

            def __init__(self, callbacks):
                self._cb = callbacks or {}

            def write(self, data):
                if not data:
                    return
                text = (
                    data.decode("latin-1")
                    if isinstance(data, (bytes, bytearray))
                    else str(data)
                )
                start = self._cb.get("on_field_start")
                on_name = self._cb.get("on_field_name")
                on_data = self._cb.get("on_field_data")
                on_end = self._cb.get("on_field_end")
                for pair in text.split("&"):
                    if not pair:
                        continue
                    raw_name, _, raw_value = pair.partition("=")
                    # raw, NOT unquoted - see the class docstring
                    name = raw_name.encode("latin-1")
                    value = raw_value.encode("latin-1")
                    if start:
                        start()
                    if on_name:
                        on_name(name, 0, len(name))
                    if on_data:
                        on_data(value, 0, len(value))
                    if on_end:
                        on_end()

            def finalize(self):
                done = self._cb.get("on_end")
                if done:
                    done()

        class _MultipartParser:
            """Unreachable for call routing.

            Twilio posts application/x-www-form-urlencoded. Failing loudly
            beats silently returning an empty form if a test ever does send
            multipart, which would otherwise look like a routing bug.
            """

            def __init__(self, *a, **k):
                raise RuntimeError(
                    "python_multipart is stubbed for tests; multipart bodies "
                    "are not supported. Use urlencoded, like Twilio does."
                )

            def write(self, *a, **k):
                raise RuntimeError("multipart not supported in test stub")

            def finalize(self):
                raise RuntimeError("multipart not supported in test stub")

        sub.parse_options_header = _parse_options_header
        sub.QuerystringParser = _QuerystringParser
        sub.MultipartParser = _MultipartParser
        pkg.QuerystringParser = _QuerystringParser
        pkg.MultipartParser = _MultipartParser
        pkg.multipart = sub
        sys.modules["python_multipart"] = pkg
        sys.modules["python_multipart.multipart"] = sub
        stubbed.append("python_multipart")

    try:
        import openai  # noqa: F401
    except ImportError:
        mod = types.ModuleType("openai")

        class OpenAI:  # routing never constructs a client
            def __init__(self, *a, **k):
                raise RuntimeError("openai SDK stubbed for tests")

        mod.OpenAI = OpenAI
        sys.modules["openai"] = mod
        stubbed.append("openai")

    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        mod = types.ModuleType("webrtcvad")

        class Vad:  # audio_ingest builds one at import time
            def __init__(self, *a, **k):
                pass

            def is_speech(self, *a, **k):
                return False

        mod.Vad = Vad
        sys.modules["webrtcvad"] = mod
        stubbed.append("webrtcvad")

    return stubbed


STUBBED_OPTIONAL_DEPS = _stub_absent_optional_deps()

from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

# Ensure config imports cleanly before any app code runs. PUBLIC_URL is
# required at module load; tests don't need telephony so any URL works.
os.environ.setdefault("PUBLIC_URL", "http://localhost:8080")
# ponytail: 016 — encryption key for the new `integrations` table's
# credentials_encrypted column. The encryption module refuses to
# start without it in production; tests run as dev so we set one
# eagerly. Tests don't care about the key value, just that it's
# stable across the run.
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_KEY",
    "oahqImB7aGYfEFxfWIJLZzJs27YSYAgr5rHUyc3gIRU=",
)
os.environ.setdefault("ENVIRONMENT", "test")

import STT_server.db_users as db_users  # noqa: E402
import STT_server.routes.api as api_mod  # noqa: E402
import STT_server.services.session_runtime as rt_mod  # noqa: E402
from STT_server.routes.api import api_router  # noqa: E402
from STT_server.routes.auth import router as auth_router  # noqa: E402


def _build_test_app() -> FastAPI:
    """Minimal app: only the routers we test. No lifespan (no DB
    backfill, no heartbeat task — keeps tests fast and hermetic)."""
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(api_router)
    return app


# Every JSON store the backend falls back to when DATABASE_URL is unset.
# Each entry is (module, attribute, filename): the attribute is the
# module-level path constant, and every one of those modules reads it at
# CALL time (not import time), so monkeypatching the module attribute
# redirects the store.
#
# ponytail: the list is deliberately EXHAUSTIVE rather than "the ones the
# current tests happen to touch". The previous fixture only redirected
# agent_tools.json, sessions.json and users.json, so any test that
# seeded an integration wrote to the REAL STT_server/data/integrations.json
# and replaced the operator's store with test rows. A store that is
# missing from this table is a store a test can silently overwrite.
#
# NOTE: api.py keeps its OWN copy of these paths (DATA_DIR / AGENTS_FILE
# / ...) in addition to the db_* modules'. Both layers MUST be redirected
# or the two disagree: the route writes to one file and the db layer
# reads from another, which shows up as a bogus "Agent not found" instead
# of the obvious "you forgot to patch a path".
_JSON_STORES = (
    ("STT_server.routes.api", "USERS_FILE", "users.json"),
    ("STT_server.routes.api", "SESSIONS_FILE", "sessions.json"),
    ("STT_server.routes.api", "AGENTS_FILE", "agents.json"),
    ("STT_server.routes.api", "NUMBERS_FILE", "phone_numbers.json"),
    ("STT_server.db_agents", "AGENTS_FILE", "agents.json"),
    ("STT_server.db_call_sessions", "CALL_SESSIONS_FILE", "call_sessions.json"),
    ("STT_server.db_call_usage", "LEGACY_CALLS_FILE", "calls.json"),
    ("STT_server.db_integrations", "_INTEGRATIONS_FILE", "integrations.json"),
    ("STT_server.db_phone_numbers", "NUMBERS_FILE", "phone_numbers.json"),
    ("STT_server.db_tenants", "TENANTS_FILE", "tenants.json"),
    ("STT_server.db_tools", "_AGENT_TOOLS_FILE", "agent_tools.json"),
    ("STT_server.db_twilio_credentials", "CREDENTIALS_FILE", "twilio_credentials.json"),
    ("STT_server.db_users", "USERS_FILE", "users.json"),
    ("STT_server.db_users", "SESSIONS_FILE", "sessions.json"),
)

# Directory-shaped constants. None means "the data dir itself"; a name
# means "a subdirectory of it".
_JSON_STORE_DIRS = (
    ("STT_server.routes.api", "DATA_DIR", None),
    ("STT_server.routes.api", "SETTINGS_DIR", "settings"),
    ("STT_server.db_agents", "DATA_DIR", None),
    ("STT_server.db_call_sessions", "DATA_DIR", None),
    ("STT_server.db_call_usage", "DATA_DIR", None),
    ("STT_server.db_campaigns", "data_dir", None),
    ("STT_server.db_integrations", "_DATA_DIR", None),
    ("STT_server.db_phone_numbers", "DATA_DIR", None),
    ("STT_server.db_settings", "DATA_DIR", None),
    ("STT_server.db_settings", "SETTINGS_DIR", "settings"),
    ("STT_server.db_tenants", "DATA_DIR", None),
    ("STT_server.db_twilio_credentials", "DATA_DIR", None),
    ("STT_server.db_users", "DATA_DIR", None),
)


def _same_type(current, value: Path):
    """api.py stores its paths as str, the db_* modules as Path. Keep each
    attribute's own type so nothing does an unexpected join."""
    if isinstance(current, str):
        return str(value)
    return Path(value)


# One Fernet key for the whole test session.
#
# ponytail: the suite used to INHERIT whatever CREDENTIAL_ENCRYPTION_KEY
# the developer's shell happened to export — conftest used setdefault, so
# a shell variable silently won — and other test files overwrote the
# variable mid-session. Every mismatch surfaced as
#
#     ValueError: Could not decrypt with any configured key (tried 1: ...)
#
# which reads like a data problem but is really a non-reproducible
# environment. The stores are isolated (see _isolate_json_stores), so the
# VALUE does not matter; what matters is that it is the SAME everywhere.
_SESSION_ENCRYPTION_KEY = "oahqImB7aGYfEFxfWIJLZzJs27YSYAgr5rHUyc3gIRU="


@pytest.fixture(autouse=True)
def _stable_encryption_key(monkeypatch: pytest.MonkeyPatch):
    """Pin the encryption key for every test and rebuild the cache.

    The credentials module caches its Fernet instance at first use, so
    the cache is reset here as well — otherwise a previous test that
    changed the key leaves the next one decrypting with the wrong key.
    """
    from STT_server.security import credentials as credentials_mod

    monkeypatch.setenv("CREDENTIAL_ENCRYPTION_KEY", _SESSION_ENCRYPTION_KEY)
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS", raising=False)
    credentials_mod._fernet_cache.update(
        {"key": None, "instance": None, "ring": None, "fingerprints": None}
    )


@pytest.fixture(autouse=True)
def _isolate_json_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point EVERY JSON store at a per-test tmp dir.

    Autouse on purpose: a test must not be able to reach the operator's
    real store by simply not requesting `data_dir`. The stores are
    redirected to a fresh tmp_path per test, so tests stay isolated from
    each other too.
    """
    d = tmp_path / "data"
    d.mkdir()
    (d / "settings").mkdir(exist_ok=True)

    for module_name, attr, filename in _JSON_STORES:
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            # The constant does not exist in this module — do not invent
            # it. A phantom attribute reads as "patched" in review but
            # redirects nothing.
            continue
        current = getattr(module, attr)
        monkeypatch.setattr(
            module, attr, _same_type(current, d / filename), raising=True
        )

    for module_name, attr, subdir in _JSON_STORE_DIRS:
        module = importlib.import_module(module_name)
        if not hasattr(module, attr):
            continue
        current = getattr(module, attr)
        target = d if subdir is None else d / subdir
        monkeypatch.setattr(
            module, attr, _same_type(current, target), raising=True
        )

    return d


@pytest.fixture
def data_dir(_isolate_json_stores: Path) -> Path:
    """The per-test JSON store dir. Every store points here."""
    return _isolate_json_stores


@pytest.fixture(scope="session")
def json_store_paths() -> tuple[tuple[str, str, str], ...]:
    """The (module, attribute, filename) table of redirected stores.

    Exposed as a fixture so the isolation test can assert the table is
    complete without importing conftest as a module.
    """
    return _JSON_STORES


def _snapshot_data_dir() -> dict[str, str]:
    """Hash every file in the REAL data dir. Used by the guard below."""
    real = Path(__file__).resolve().parent.parent / "STT_server" / "data"
    out: dict[str, str] = {}
    if not real.exists():
        return out
    for path in sorted(real.rglob("*")):
        if not path.is_file():
            continue
        try:
            out[str(path.relative_to(real))] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


@pytest.fixture(scope="session", autouse=True)
def _guard_real_data_dir():
    """Fail the run if ANY test wrote to the operator's real data dir.

    ponytail: the isolation above is the fix; this is the tripwire. The
    operator lost a store of live integrations because the suite wrote to
    STT_server/data/integrations.json. If a future test reaches a store
    that is missing from _JSON_STORES, this catches it at the END of the
    run instead of the operator discovering it in production.
    """
    before = _snapshot_data_dir()
    yield
    after = _snapshot_data_dir()
    changed = sorted(
        name for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )
    assert not changed, (
        "The test suite modified the operator's real data dir "
        "(STT_server/data/): " + ", ".join(changed) + ". "
        "Add the store to _JSON_STORES in tests/conftest.py so it is "
        "redirected to a tmp dir."
    )


@pytest.fixture
def auth_token(data_dir: Path) -> str:
    """Write a valid 7-day session to the tmp sessions.json and return
    the bearer token. Also seeds the matching user in users.json so
    future /me-style checks would work."""
    user_id = "user-test-001"
    email = "tester@example.com"
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    sessions = {token: {
        "user_id": user_id,
        "email": email,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires,
    }}
    (data_dir / "sessions.json").write_text(json.dumps(sessions), encoding="utf-8")
    (data_dir / "users.json").write_text(
        json.dumps([{
            "id": user_id,
            "name": "Tester",
            "email": email,
            "password": "",
            "role": "admin",
        }]),
        encoding="utf-8",
    )
    return token


@pytest.fixture
def other_user_token(data_dir: Path) -> str:
    """Second user's token — used for cross-user isolation tests."""
    user_id = "user-other-999"
    email = "other@example.com"
    token = secrets.token_urlsafe(32)
    sessions_path = data_dir / "sessions.json"
    sessions = json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else {}
    sessions[token] = {
        "user_id": user_id,
        "email": email,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
    }
    sessions_path.write_text(json.dumps(sessions), encoding="utf-8")
    return token


@pytest.fixture
def app(data_dir: Path) -> FastAPI:
    return _build_test_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """ASGI client. No network port — httpx runs the app in-process."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac