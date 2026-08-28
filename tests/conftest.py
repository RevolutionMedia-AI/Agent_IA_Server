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

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

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