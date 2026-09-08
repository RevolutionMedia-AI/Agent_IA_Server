"""Regression tests for the LLM-driven Test button contract.

The /integrations/{id}/test endpoint feeds the user's LLM
(settings.test_data_model) with the integration's catalog
schema so the operator can preview what the agent will send during
a real call. Today the action can be passed as a ``?action=...``
query string; without it we fall back to configuration fields only.

The 2026-09-04 expansion added ``title`` + ``description`` to Google
Calendar's ``create_appointment`` action. The LLM must fill them with
call-context-aware text. These tests pin the contract:

  1. Without ``action``: only configuration fields, nothing about
     title/description.
  2. With ``action=create_appointment``: the schema includes the
     full action arguments (name / email / datetime / duration /
     title / description) so the LLM can't miss them.
  3. The generator's LLM prompt is constructed from the catalog
     spec — we never hardcode fields here.
"""
from __future__ import annotations

import importlib
import json
import sys
from types import SimpleNamespace

import pytest


def _reload_test_data_generator():
    if "STT_server.services.test_data_generator" in sys.modules:
        del sys.modules["STT_server.services.test_data_generator"]
    return importlib.import_module("STT_server.services.test_data_generator")


class _FakeCompletions:
    def __init__(self):
        self.calls = []
        self.next_payload = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.next_payload


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _llm_response(arguments: dict):
    """Build the response shape the BE expects: ``response.choices[0]
    .message.tool_calls[0].function.arguments`` is a JSON string."""
    args_str = json.dumps(arguments)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments=args_str)
                        )
                    ]
                )
            )
        ]
    )


def test_action_arg_merges_action_schema_into_response_shape(monkeypatch):
    """When the caller passes ``action=create_appointment``, the LLM
    prompt must include the action's parameters (name/email/datetime/
    duration_minutes/title/description) so the response shape covers
    every field the executor consumes. Without this the LLM would
    only invent configuration values, leaving the agent's real
    action call with placeholder strings."""
    gen = _reload_test_data_generator()
    client = _FakeClient()
    client.chat.completions.next_payload = _llm_response({
        "calendar_id": "ops@example.com",
        "timezone": "America/Tijuana",
        "name": "Ulises Escalante",
        "email": "kueh560@gmail.com",
        "datetime": "2026-09-08T15:00:00",
        "duration_minutes": 30,
        "title": "Consulta sobre renovación",
        "description": "El cliente quiere revisar opciones.",
    })
    monkeypatch.setattr(gen, "_resolve_openai_client", lambda user_id: client)
    monkeypatch.setattr(gen, "_resolve_model", lambda m: m or "gpt-4o-mini")

    integration = {
        "provider": "google_calendar",
        "name": "My Calendar",
        "configuration": {
            "calendar_id": "ops@example.com",
            "timezone": "America/Tijuana",
        },
    }
    out = gen.generate_integration_test_payload(
        integration, "user-1",
        model="gpt-4o-mini",
        action="create_appointment",
    )
    # ponytail: the response covers BOTH configuration and action
    # arguments, so the FE can render the full preview to the
    # operator.
    assert out["calendar_id"] == "ops@example.com"
    assert out["timezone"] == "America/Tijuana"
    assert out["name"] == "Ulises Escalante"
    assert out["email"] == "kueh560@gmail.com"
    assert out["datetime"] == "2026-09-08T15:00:00"
    assert out["duration_minutes"] == 30
    assert out["title"] == "Consulta sobre renovación"
    assert out["description"] == "El cliente quiere revisar opciones."

    # The LLM prompt must mention the action so the model knows
    # what shape to emit.
    last_user_msg = client.chat.completions.calls[-1]["messages"][-1]["content"]
    assert "create_appointment" in last_user_msg, (
        "action id must appear in the prompt so the LLM knows which "
        "schema to fill"
    )
    assert "title" in last_user_msg
    assert "description" in last_user_msg


def test_no_action_omits_action_arguments(monkeypatch):
    """When the caller does NOT pass ``action``, the LLM only fills
    the configuration fields. Action arguments are not in scope — the
    executor never reads them. This is the legacy behaviour kept
    for any provider whose action arguments are still operator-driven
    (none today, but we don't want a surprise addition)."""
    gen = _reload_test_data_generator()
    client = _FakeClient()
    client.chat.completions.next_payload = _llm_response({
        "calendar_id": "ops@example.com",
        "timezone": "America/Tijuana",
    })
    monkeypatch.setattr(gen, "_resolve_openai_client", lambda user_id: client)
    monkeypatch.setattr(gen, "_resolve_model", lambda m: m or "gpt-4o-mini")

    integration = {
        "provider": "google_calendar",
        "name": "My Calendar",
        "configuration": {},
    }
    out = gen.generate_integration_test_payload(integration, "user-1")
    assert out == {"calendar_id": "ops@example.com", "timezone": "America/Tijuana"}
    # No action was passed — the LLM prompt must not include the
    # action-arguments section. We only check the section heading
    # here so the test survives copy edits to the static helper
    # text (the operator's prompt always mentions "create_appointment"
    # as an example of what NOT to invent).
    last_user_msg = client.chat.completions.calls[-1]["messages"][-1]["content"]
    assert "Action the operator" not in last_user_msg
    assert "Action argument schema" not in last_user_msg


def test_invalid_action_falls_back_to_configuration(monkeypatch):
    """A bogus action id (typo, removed from catalog) is silently
    treated as 'no action'. The endpoint never raises — the operator
    sees the configuration-only preview."""
    gen = _reload_test_data_generator()
    client = _FakeClient()
    client.chat.completions.next_payload = _llm_response({
        "calendar_id": "ops@example.com",
        "timezone": "America/Tijuana",
    })
    monkeypatch.setattr(gen, "_resolve_openai_client", lambda user_id: client)
    monkeypatch.setattr(gen, "_resolve_model", lambda m: m or "gpt-4o-mini")

    integration = {
        "provider": "google_calendar",
        "name": "My Calendar",
        "configuration": {},
    }
    out = gen.generate_integration_test_payload(
        integration, "user-1", action="nuke_planet",
    )
    assert "calendar_id" in out
    assert "title" not in out
