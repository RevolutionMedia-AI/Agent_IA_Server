"""Characterization test for GET /version.

Pins the deployed-commit / container-started-at contract so future audits can
verify repo == deployed. Locks in:

  * commit comes from the GIT_SHA env var (set by the Dockerfile ARG).
  * build_started_at comes from BUILD_STARTED_AT (set by start.sh at boot).
  * missing env vars degrade to "unknown" (don't crash local dev containers).

No auth: /version mirrors /health so the operator can curl it from any
context (browser, Railway CLI, CI) without a bearer token.
"""
from __future__ import annotations

import os

import pytest


@pytest.mark.asyncio
async def test_version_returns_injected_commit_and_started_at(client, monkeypatch):
    monkeypatch.setenv("GIT_SHA", "abcdef1234567890")
    monkeypatch.setenv("BUILD_STARTED_AT", "2026-09-25T10:00:00Z")

    resp = await client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "commit": "abcdef1234567890",
        "build_started_at": "2026-09-25T10:00:00Z",
    }


@pytest.mark.asyncio
async def test_version_falls_back_to_unknown_when_env_missing(client, monkeypatch):
    # ponytail: explicit unset so we override the conftest default (none of
    # the test fixtures set GIT_SHA / BUILD_STARTED_AT today, but a future
    # one might).
    monkeypatch.delenv("GIT_SHA", raising=False)
    monkeypatch.delenv("BUILD_STARTED_AT", raising=False)

    resp = await client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"commit": "unknown", "build_started_at": "unknown"}


@pytest.mark.asyncio
async def test_version_does_not_require_auth(client):
    # ponytail: /version is intentionally unauthenticated so the operator can
    # probe it from a Railway CLI shell without minting a session token.
    # Locking the no-auth contract here means a future refactor that adds
    # Depends(require_auth) trips the test.
    resp = await client.get("/version")
    assert resp.status_code == 200