"""Pytest fixtures for BE tests.

Builds a minimal FastAPI app around the routes we want to test
(`api_router` + `auth_router`), without importing the heavy call
adapters (deepgram/inworld/assemblyai/openai_realtime) that
STT_Server.py pulls in. Tests redirect the JSON-file backend to a
tmp dir so they never touch production data in STT_server/data/.
"""
from __future__ import annotations

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


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect all JSON-file IO (tools + sessions + users) to a tmp dir."""
    d = tmp_path / "data"
    d.mkdir()

    # Tools file paths used by the API endpoints and by
    # _load_agent_tools in session_runtime.
    tools_file = d / "agent_tools.json"
    monkeypatch.setattr(api_mod, "DATA_DIR", str(d), raising=False)
    monkeypatch.setattr(api_mod, "TOOLS_FILE", str(tools_file), raising=False)
    monkeypatch.setattr(rt_mod, "_TOOLS_FILE", str(tools_file), raising=False)

    # Sessions/users files used by the auth shim. require_auth
    # re-imports load_sessions/save_sessions on every call, so we patch
    # the underlying module attribute (which the `from X import Y`
    # inside the function picks up fresh each invocation).
    sessions_file = d / "sessions.json"
    users_file = d / "users.json"
    monkeypatch.setattr(db_users, "SESSIONS_FILE", sessions_file, raising=False)
    monkeypatch.setattr(db_users, "USERS_FILE", users_file, raising=False)
    return d


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