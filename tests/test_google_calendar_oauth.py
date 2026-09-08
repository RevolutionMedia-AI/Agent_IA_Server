"""Regression tests for the Google Calendar OAuth wiring.

Mirrors the Salesforce round-trip tests so the second OAuth provider
in the catalog is locked down end-to-end:
  * oauth_providers builds a valid config from GOOGLE_* env vars
  * the catalog spec advertises auth_type=oauth + the calendar_event
    action (no LLM-controlled host_email field)
  * the tool executor body forwards calendar_id + timezone + the
    internal credentials endpoint to n8n
  * _test_google_calendar reports the right operator-facing strings
    when calendar_id / timezone / access_token are missing
"""
from __future__ import annotations

import os
import pathlib

import pytest


# ponytail: same Fernet key as the rest of the suite. Set before
# importing anything that opens a Fernet instance.
os.environ.setdefault(
    "CREDENTIAL_ENCRYPTION_KEY",
    "3caLHixTmxCJ1OAQEK11TEn4k5soMyJhybJIyAFVMfk=",  # sample Fernet key for tests
)


@pytest.fixture
def google_env(monkeypatch):
    """Set GOOGLE_* env vars so _build_google_calendar_config succeeds."""
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "google-client-id.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "google-client-secret")
    monkeypatch.setenv(
        "GOOGLE_REDIRECT_URI",
        "https://agentiaserver.example.com/integrations/google_calendar/oauth/callback",
    )
    return None


def test_google_calendar_is_a_registered_oauth_provider(google_env):
    from STT_server.services.oauth_providers import (
        known_oauth_providers,
        get_oauth_config,
        validate_oauth_env,
    )
    assert "google_calendar" in known_oauth_providers()
    cfg = get_oauth_config("google_calendar")
    assert cfg.provider_id == "google_calendar"
    assert cfg.authorize_url == "https://accounts.google.com/o/oauth2/v2/auth"
    assert cfg.token_url == "https://oauth2.googleapis.com/token"
    assert cfg.client_id == "google-client-id.apps.googleusercontent.com"
    assert cfg.client_secret == "google-client-secret"
    # ponytail: scopes include the delegated credential for creating
    # events on the user's calendar. The `openid + email + profile`
    # triplet covers userinfo hydration in the post-connect hook.
    assert "https://www.googleapis.com/auth/calendar.events" in cfg.default_scopes
    assert "openid" in cfg.default_scopes
    assert "email" in cfg.default_scopes
    # Env validator reports no missing vars.
    ok, missing = validate_oauth_env("google_calendar")
    assert ok is True
    assert missing == ()


def test_google_calendar_validate_env_reports_missing(monkeypatch):
    from STT_server.services.oauth_providers import validate_oauth_env
    # Clear everything so we can confirm the error path.
    for var in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REDIRECT_URI"):
        monkeypatch.delenv(var, raising=False)
    ok, missing = validate_oauth_env("google_calendar")
    assert ok is False
    assert set(missing) == {
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_REDIRECT_URI",
    }


def test_catalog_spec_advertises_google_calendar_oauth_without_host_email():
    """The catalog entry must use auth_type=oauth and the
    create_appointment action must NOT carry host_email (the LLM must
    not pick the host calendar — it's a configuration field).

    ponytail: 2026-09-04 — the action was renamed from
    ``calendar_event`` to ``create_appointment`` to match the
    dispatcher in services/integrations_executor.py. The schema
    gained ``title`` + ``description`` so the LLM can summarise the
    call before invoking the action. ``additionalProperties: false``
    stops the model from inventing extra fields the executor doesn't
    read.
    """
    from STT_server.services.integrations_catalog import get_integration_provider_spec

    spec = get_integration_provider_spec("google_calendar")
    assert spec is not None
    assert spec.auth_type == "oauth"
    assert spec.oauth_label == "Connect Google Calendar"

    action_ids = [a.id for a in spec.actions]
    assert "create_appointment" in action_ids
    for action in spec.actions:
        if action.id == "create_appointment":
            schema = action.parameters_schema or {}
            props = schema.get("properties", {})
            # No host_email — the host calendar is always resolved
            # server-side from configuration.calendar_id.
            assert "host_email" not in props
            # The required fields are what the n8n workflow + executor
            # consume. ``title`` and ``description`` are NEW
            # (2026-09-04): the LLM must summarise the call before
            # invoking the action so the operator sees a useful event.
            required = set(schema.get("required") or [])
            assert {"name", "email", "datetime", "duration_minutes",
                    "title", "description"} <= required
            # duration_minutes has explicit min/max so the LLM can't
            # book a 0-minute meeting.
            duration_schema = props.get("duration_minutes") or {}
            assert duration_schema.get("minimum") == 5
            assert duration_schema.get("maximum") == 240
            # additionalProperties:false stops the model from
            # inventing extra fields the executor doesn't read.
            assert schema.get("additionalProperties") is False


def test_test_google_calendar_rejects_missing_config():
    """The live-test function must surface clear operator messages when
    configuration.calendar_id or .timezone is missing — these are
    the post-OAuth-handshake fields the operator fills in /integrations
    before the integration can be used by tools."""
    from STT_server.services.integrations_tester import _test_google_calendar

    ok, msg = _test_google_calendar({}, {"access_token": "x"})
    assert ok is False
    assert "calendar_id" in msg

    ok, msg = _test_google_calendar(
        {"calendar_id": "ventas@clienteB.com"},
        {"access_token": "x"},
    )
    assert ok is False
    assert "timezone" in msg

    ok, msg = _test_google_calendar(
        {"calendar_id": "ventas@clienteB.com", "timezone": "America/Tijuana"},
        {},
    )
    assert ok is False
    assert "Conectar" in msg or "Google" in msg


def test_tool_executor_body_includes_calendar_id_and_timezone(monkeypatch):
    """The body the BE POSTs to n8n must include calendar_id + timezone
    (read from configuration) plus an internal endpoint pointer for
    credentials. Without these, the n8n workflow can't reach the
    operator's calendar or call the Google Calendar API on the
    operator's behalf."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from STT_server.services import tool_executor

    captured = {}

    class _FakeExecutor:
        async def execute(self, url, body, tool_name, method="POST"):
            captured["url"] = url
            captured["body"] = body
            return {"ok": True}

    tool_executor.get_tool_executor = lambda: _FakeExecutor()
    tool_executor._resolve_integration_webhook = lambda integ: "https://n8n.example.com/webhook/calendar"

    # ponytail: execute_tool_call resolves the integration row by
    # `from STT_server.db_integrations import get_integration` inside
    # the function body, so we patch the name on that module — same
    # effect because Python imports are by reference at call time.
    import STT_server.db_integrations as _db_int
    integration_row = {
        "id": "int-1",
        "provider": "google_calendar",
        "configuration": {
            "calendar_id": "ventas@clienteB.com",
            "timezone": "America/Tijuana",
        },
    }
    monkeypatch.setattr(_db_int, "get_integration", lambda *a, **kw: integration_row)

    from STT_server.services.tool_executor import execute_tool_call
    import asyncio

    tool = {
        "id": "tool-1",
        "function_name": "create_calendar_event",
        "name": "Create Calendar Appointment",
        "integration_id": "int-1",
        "action": "create_appointment",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    async def _run():
        return await execute_tool_call(
            tool, "user-admin-001",
            {
                "name": "Alice",
                "email": "a@b.com",
                "datetime": "2026-09-04T15:00:00-06:00",
                "duration_minutes": 30,
                "title": "Consulta sobre renovación",
                "description": "El cliente quiere revisar las opciones disponibles.",
                "notes": "Solicitó una reunión de 30 minutos.",
            },
        )

    out = asyncio.run(_run())
    body = captured["body"]
    assert body["integration_id"] == "int-1"
    assert body["provider"] == "google_calendar"
    assert body["action"] == "create_appointment"
    assert body["calendar_id"] == "ventas@clienteB.com"
    assert body["timezone"] == "America/Tijuana"
    assert body["credentials_endpoint"] == "/internal/integrations/int-1/credentials"
    # The LLM-supplied arguments travel verbatim — calendar_id + timezone
    # are NOT in the function schema, so the LLM can't send them.
    args = body["arguments"]
    assert "calendar_id" not in args
    assert "timezone" not in args
    # ponytail: 2026-09-04 schema includes title + description. They
    # pass through to n8n untouched so the Switch node can route by
    # verb + forward to the executor.
    assert args == {
        "name": "Alice",
        "email": "a@b.com",
        "datetime": "2026-09-04T15:00:00-06:00",
        "duration_minutes": 30,
        "title": "Consulta sobre renovación",
        "description": "El cliente quiere revisar las opciones disponibles.",
        "notes": "Solicitó una reunión de 30 minutos.",
    }
    assert out == {"ok": True}
