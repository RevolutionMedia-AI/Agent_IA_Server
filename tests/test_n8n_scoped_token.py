"""Per-integration n8n token travels in the webhook body.

The platform-wide INTEGRATIONS_N8N_TOKEN resolves to user_id=None, which
means UNSCOPED: one shared n8n credential can read any tenant's decrypted
provider token by editing the integration_id in the URL. For a multi-tenant
product that is a cross-tenant read, so the tool executor now hands n8n the
integration's OWN token and /internal/.../credentials enforces the match.

What this pins:
  1. get_integration_n8n_token reads the cleartext token out of the
     encrypted blob (only the SHA-256 prefix is indexed, so this is the one
     place that can).
  2. execute_tool_call injects it under "n8n_token" in the POST body.
  3. The LLM cannot forge or override it — it rides `extra`, which is
     server-owned, and _strip_forbidden_args owns the argument dict.
  4. A missing/undecryptable token omits the key rather than failing the
     call, so an un-minted integration still works on the platform token.
"""
from __future__ import annotations

import pytest

from STT_server.db_integrations import get_integration_n8n_token
from STT_server.services import tool_executor

TOKEN = "tok-scoped-abc123"


@pytest.fixture
def integration_row(monkeypatch):
    """One integration whose encrypted blob decrypts to {n8n_token: TOKEN}."""
    import STT_server.db_integrations as dbi
    import STT_server.security.credentials as sec

    monkeypatch.setattr(
        sec, "decrypt_credentials", lambda blob: {"n8n_token": TOKEN},
    )
    monkeypatch.setattr(
        dbi, "get_integration",
        lambda iid, uid: {
            "id": iid, "user_id": uid,
            "credentials_cipher": "fernet-v1",
            "credentials_encrypted": b"opaque",
        },
    )
    return dbi


# ── the reader ──────────────────────────────────────────────────────────

def test_returns_cleartext_token(integration_row):
    assert get_integration_n8n_token("int-1", "u-1") == TOKEN


def test_missing_row_is_none(integration_row, monkeypatch):
    monkeypatch.setattr(integration_row, "get_integration", lambda i, u: None)
    assert get_integration_n8n_token("int-1", "u-1") is None


def test_undecryptable_blob_is_none(integration_row, monkeypatch):
    import STT_server.security.credentials as sec

    def boom(_blob):
        raise ValueError("bad key")

    monkeypatch.setattr(sec, "decrypt_credentials", boom)
    assert get_integration_n8n_token("int-1", "u-1") is None


def test_legacy_cipher_skips_decrypt(integration_row, monkeypatch):
    """A non-fernet row must not be decrypted at all."""
    called = []
    import STT_server.security.credentials as sec

    monkeypatch.setattr(
        sec, "decrypt_credentials", lambda b: called.append(b) or {},
    )
    monkeypatch.setattr(
        integration_row, "get_integration",
        lambda i, u: {"id": i, "credentials_cipher": "plain", "credentials_encrypted": b"x"},
    )
    assert get_integration_n8n_token("int-1", "u-1") is None
    assert called == [], "no decrypt on a legacy cipher row"


def test_absent_token_is_none(integration_row, monkeypatch):
    import STT_server.security.credentials as sec

    monkeypatch.setattr(sec, "decrypt_credentials", lambda b: {"access_token": "a"})
    assert get_integration_n8n_token("int-1", "u-1") is None


# ── the injection ───────────────────────────────────────────────────────

@pytest.fixture
def captured_body(monkeypatch):
    """Run execute_tool_call and return the body it would have POSTed."""
    import STT_server.db_integrations as dbi
    import STT_server.security.credentials as sec

    monkeypatch.setattr(sec, "decrypt_credentials", lambda blob: {"n8n_token": TOKEN})
    monkeypatch.setattr(
        dbi, "get_integration",
        lambda iid, uid: {
            "id": iid, "user_id": uid, "provider": "dynamics365",
            "credentials_cipher": "fernet-v1", "credentials_encrypted": b"opaque",
        },
    )
    seen = {}

    class FakeExec:
        async def execute(self, url, args, name, method="POST", extra=None):
            seen.update(extra or {})
            seen["_args"] = args
            return {"ok": True}

    monkeypatch.setattr(tool_executor, "get_tool_executor", lambda: FakeExec())
    monkeypatch.setattr(
        tool_executor, "_resolve_integration_webhook",
        lambda integ: "https://n8n.example/hook",
    )
    return seen


@pytest.mark.asyncio
async def test_body_carries_scoped_token(captured_body):
    await tool_executor.execute_tool_call(
        {"id": "t-1", "integration_id": "int-1", "action": "get_cases",
         "function_name": "get_cases", "webhook_url": "https://n8n.example/hook"},
        "u-1",
        {"limit": 5},
    )
    assert captured_body["n8n_token"] == TOKEN
    assert captured_body["integration_id"] == "int-1"
    assert captured_body["action"] == "get_cases"


@pytest.mark.asyncio
async def test_llm_cannot_forge_token(captured_body):
    """An n8n_token in the argument dict is stripped, not forwarded."""
    await tool_executor.execute_tool_call(
        {"id": "t-1", "integration_id": "int-1", "action": "get_cases",
         "function_name": "get_cases", "webhook_url": "https://n8n.example/hook"},
        "u-1",
        {"n8n_token": "attacker-supplied", "limit": 5},
    )
    assert captured_body["n8n_token"] == TOKEN, "server value must win"
    assert "n8n_token" not in captured_body["_args"]


@pytest.mark.asyncio
async def test_absent_token_omits_key_not_fails(captured_body, monkeypatch):
    """No minted token -> key absent, call still succeeds on platform token."""
    import STT_server.security.credentials as sec

    monkeypatch.setattr(sec, "decrypt_credentials", lambda b: {"access_token": "a"})
    await tool_executor.execute_tool_call(
        {"id": "t-1", "integration_id": "int-1", "action": "get_cases",
         "function_name": "get_cases", "webhook_url": "https://n8n.example/hook"},
        "u-1",
        {"limit": 5},
    )
    assert "n8n_token" not in captured_body
    assert captured_body["integration_id"] == "int-1"