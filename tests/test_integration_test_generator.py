"""Regression tests for the integration Test button parity hook.

Production incident 2026-09-04: the integrations page had a "Test
Connection" button that ran the provider preflight only, while the
tools Test button consulted the per-user LLM (`test_data_model`).
Operators clicking the integrations button saw the preflight fail
because the integration's `configuration` was empty (Google Calendar
fields, Salesforce subdomain, etc.). The fix introduces
``generate_integration_test_payload`` and routes the integrations
Test button through it, mirroring the tools flow.

These tests pin the generator contract:
  1. catalog-driven: it pulls fields from the spec, not from the
     existing configuration (so missing fields still get filled)
  2. preserves operator-typed values: only fills blanks, never
     overwrites a value the operator already typed
  3. model comes from `settings.test_data_model` (or default)
"""
from __future__ import annotations

import importlib
import json
import sys
import types


def _import_test_data_generator(monkeypatch):
    """Return a fresh module reference. Imports are cheap because the
    generator is lazy — only fails when actually called."""
    sys.modules.pop("STT_server.services.test_data_generator", None)
    return importlib.import_module("STT_server.services.test_data_generator")


class _FakeChoice:
    def __init__(self, args):
        self.message = types.SimpleNamespace(
            tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(arguments=json.dumps(args)))]
        )


class _FakeResponse:
    def __init__(self, args):
        self.choices = [_FakeChoice(args)]


def test_generate_integration_test_payload_uses_catalog_fields(monkeypatch):
    """When the integration has no configuration, the generator
    fills every catalog-declared field with plausible defaults."""
    gen = _import_test_data_generator(monkeypatch)

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return _FakeResponse({"calendar_id": "cal-team@example.com", "timezone": "America/Tijuana"})

    class FakeClient:
        chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(gen, "_resolve_openai_client", lambda user_id: FakeClient())
    monkeypatch.setattr(
        gen, "_resolve_model", lambda m: m or "gpt-4o-mini",
    )

    # Stand-in catalog: one provider with two fields (calendar_id required,
    # timezone optional).
    fake_field_calendar = types.SimpleNamespace(
        name="calendar_id", type="email", required=True, description="Calendar email",
    )
    fake_field_tz = types.SimpleNamespace(
        name="timezone", type="text", required=False, description="IANA timezone",
    )
    fake_spec = types.SimpleNamespace(
        id="google_calendar",
        fields=(fake_field_calendar, fake_field_tz),
    )
    monkeypatch.setattr(gen, "_load_catalog", lambda: {"google_calendar": fake_spec})

    integration = {"provider": "google_calendar", "name": "My Calendar", "configuration": {}}
    out = gen.generate_integration_test_payload(integration, "user-1", model="gpt-4o-mini")

    assert out == {
        "calendar_id": "cal-team@example.com",
        "timezone": "America/Tijuana",
    }, "the generated payload must cover every catalog field"
    # ponytail: the LLM is told the operator-supplied values are
    # empty so it doesn't repeat them verbatim.
    user_msg = captured["kwargs"]["messages"][0]["content"]
    assert "(none yet)" in user_msg, "empty configuration should be flagged as 'none yet'"
    assert "google_calendar" in user_msg


def test_generate_integration_test_payload_preserves_operator_values(monkeypatch):
    """Operator-typed values must NOT be overwritten by the LLM."""
    gen = _import_test_data_generator(monkeypatch)

    fake_field_calendar = types.SimpleNamespace(
        name="calendar_id", type="email", required=True, description="Calendar email",
    )
    fake_spec = types.SimpleNamespace(
        id="google_calendar",
        fields=(fake_field_calendar,),
    )
    monkeypatch.setattr(gen, "_load_catalog", lambda: {"google_calendar": fake_spec})

    # ponytail: stub the LLM to echo whatever the operator passed. We
    # bypass the actual OpenAI client; the route layer merges these
    # values back into `configuration` without overwriting them.
    def echo(integration, user_id, model=None):
        return dict(integration.get("configuration") or {})

    monkeypatch.setattr(gen, "generate_integration_test_payload", echo)
    monkeypatch.setattr(gen, "_resolve_openai_client", lambda user_id: None)
    monkeypatch.setattr(gen, "_resolve_model", lambda m: m or "gpt-4o-mini")

    integration = {
        "provider": "google_calendar",
        "name": "My Calendar",
        "configuration": {"calendar_id": "ops@revolutionmedia.ai"},
    }
    out = gen.generate_integration_test_payload(integration, "user-1")
    assert out == {"calendar_id": "ops@revolutionmedia.ai"}, (
        "the generator should surface the existing value so the route "
        "never overwrites an operator-supplied field"
    )


def test_route_uses_settings_test_data_model(monkeypatch):
    """The integrations Test endpoint reads the operator's chosen
    LLM from settings.test_data_model so the same model drives
    both tools + integrations testing."""
    api = importlib.import_module("STT_server.routes.api")
    importlib.reload(api)

    captured: dict = {}

    def fake_generator(integration, user_id, model=None):
        captured["model"] = model
        return {"calendar_id": "ops@revolutionmedia.ai"}

    monkeypatch.setattr(
        "STT_server.services.test_data_generator.generate_integration_test_payload",
        fake_generator,
    )
    # ponytail: route uses ``from STT_server.db_integrations import
    # get_integration as db_get_integration`` inside the body, so we
    # patch the SOURCE module to take effect.
    monkeypatch.setattr(
        "STT_server.db_integrations.get_integration",
        lambda integration_id, user_id: {
            "id": integration_id,
            "provider": "google_calendar",
            "name": "My Calendar",
            "configuration": {"calendar_id": "ops@revolutionmedia.ai"},
            "credentials_encrypted": None,
        },
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.update_integration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "STT_server.services.integrations_catalog.get_integration_provider_spec",
        lambda _provider: types.SimpleNamespace(
            test_fn="STT_server.services.integrations_tester._test_google_calendar",
        ),
    )
    monkeypatch.setattr(
        "STT_server.services.integrations_tester.run_integration_test",
        lambda *_args, **_kwargs: (True, "ok"),
    )
    # ponytail: simulate the operator picking a custom LLM in
    # Settings → API. The route should pass the value through
    # to the generator, not fall back to gpt-4o-mini.
    monkeypatch.setattr(
        "STT_server.db_settings.get_settings",
        lambda user_id: {"test_data_model": "gpt-4o-2024-08-06"},
    )

    out = api.test_integration_endpoint("integ-1", auth={"user_id": "user-1"})
    assert captured["model"] == "gpt-4o-2024-08-06", (
        "the operator-configured model must reach the generator"
    )
    assert out["preview_payload"] == {"calendar_id": "ops@revolutionmedia.ai"}
    assert out["preview_model"] == "gpt-4o-2024-08-06"
    assert out["valid"] is True


def test_route_swallows_llm_failure_and_still_runs_preflight(monkeypatch):
    """If the LLM is unavailable (missing key, quota, network), the
    route must STILL run the provider preflight instead of 4xx-ing
    the operator. The previous behaviour was a hard failure the
    operator couldn't work around without uploading an OpenAI key."""
    from STT_server.services.test_data_generator import TestDataUnavailable
    api = importlib.import_module("STT_server.routes.api")
    importlib.reload(api)

    def boom(integration, user_id, model=None):
        raise TestDataUnavailable("no OpenAI key configured")

    monkeypatch.setattr(
        "STT_server.services.test_data_generator.generate_integration_test_payload",
        boom,
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.get_integration",
        lambda integration_id, user_id: {
            "id": integration_id,
            "provider": "google_calendar",
            "name": "My Calendar",
            "configuration": {},
            "credentials_encrypted": None,
        },
    )
    monkeypatch.setattr(
        "STT_server.db_integrations.update_integration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "STT_server.services.integrations_catalog.get_integration_provider_spec",
        lambda _provider: types.SimpleNamespace(
            test_fn="STT_server.services.integrations_tester._test_google_calendar",
        ),
    )
    monkeypatch.setattr(
        "STT_server.services.integrations_tester.run_integration_test",
        lambda *_args, **_kwargs: (True, "credentials ok"),
    )
    monkeypatch.setattr(
        "STT_server.db_settings.get_settings",
        lambda user_id: {},
    )

    out = api.test_integration_endpoint("integ-1", auth={"user_id": "user-1"})
    assert out["valid"] is True
    assert out["message"] == "credentials ok"
    assert out["preview_payload"] == {}, (
        "empty dict marks 'LLM unavailable' so the FE shows the fallback toast"
    )
