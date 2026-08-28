"""Verify the IntegrationProviderSpec catalog matches the wire shape
GET /integrations/providers returns, and that the validation helpers
gate on the catalog correctly.

This catches accidental drift between the dataclass used by the BE
internally and what the FE receives."""
from __future__ import annotations

import pytest


async def test_providers_endpoint_returns_well_formed_catalog(client, auth_token):
    resp = await client.get(
        "/integrations/providers",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    providers = {p["id"]: p for p in body["providers"]}
    # All V1 providers present
    assert {"zendesk", "salesforce", "dynamics365",
            "genesys_cloud", "nice_cxone", "generic_webhook"}.issubset(providers.keys())
    # Each provider has the expected wire shape
    for pid, p in providers.items():
        assert p["id"] == pid
        assert isinstance(p["name"], str) and p["name"]
        assert p["category"] in ("crm", "contact_center", "custom")
        assert isinstance(p["fields"], list)
        assert isinstance(p["actions"], list)
        assert isinstance(p["has_test"], bool)
        for f in p["fields"]:
            assert {"name", "label", "type", "required"}.issubset(f.keys())
            assert f["type"] in ("text", "password", "url", "email")


async def test_zendesk_has_real_test_and_actions(client, auth_token):
    body = (await client.get(
        "/integrations/providers",
        headers={"Authorization": f"Bearer {auth_token}"},
    )).json()
    zendesk = next(p for p in body["providers"] if p["id"] == "zendesk")
    assert zendesk["has_test"] is True
    action_ids = {a["id"] for a in zendesk["actions"]}
    assert {"find_customer", "get_tickets", "create_ticket",
            "add_comment", "update_ticket"}.issubset(action_ids)


async def test_non_zendesk_providers_marked_as_no_test(client, auth_token):
    """Salesforce / Dynamics / Genesys / NICE are stubs in V1 — the
    FE renders Configure but Test Connection returns 'not yet
    implemented'. This test pins that contract so a future dev
    doesn't accidentally flip has_test=True without wiring the
    test_fn."""
    body = (await client.get(
        "/integrations/providers",
        headers={"Authorization": f"Bearer {auth_token}"},
    )).json()
    for pid in ("salesforce", "dynamics365", "genesys_cloud", "nice_cxone"):
        spec = next(p for p in body["providers"] if p["id"] == pid)
        assert spec["has_test"] is False, f"{pid} should be has_test=False in V1"


async def test_generic_webhook_accepts_free_form_action():
    """The catalog validator + is_valid_action must agree that
    generic_webhook's empty actions tuple means 'any well-formed
    id is OK'."""
    from STT_server.services.integrations_catalog import is_valid_action
    assert is_valid_action("generic_webhook", "my_custom_action") is True
    assert is_valid_action("generic_webhook", "Find Customer") is False  # uppercase / space


def test_is_valid_action_rejects_unknown_provider():
    from STT_server.services.integrations_catalog import is_valid_action
    assert is_valid_action("made_up", "anything") is False


def test_validate_integration_fields_clean_zendesk():
    """The catalog-level validator returns (cleaned_config, cleaned_creds, errors)."""
    from STT_server.services.integrations_catalog import validate_integration_fields
    config, creds, errors = validate_integration_fields(
        "zendesk",
        {"subdomain": "acme"},
        {"email": "admin@acme.com", "api_token": "a" * 25},
    )
    assert config == {"subdomain": "acme"}
    assert creds == {"email": "admin@acme.com", "api_token": "a" * 25}
    assert errors == []


def test_validate_integration_fields_drops_empty_fields():
    """Empty strings are silently dropped (so the FE can clear a field
    on update without re-typing it)."""
    from STT_server.services.integrations_catalog import validate_integration_fields
    config, creds, errors = validate_integration_fields(
        "zendesk",
        {"subdomain": "acme"},
        {"email": "", "api_token": "a" * 25},
    )
    # Empty email dropped
    assert "email" not in creds
    assert creds["api_token"] == "a" * 25
    assert errors == []


def test_validate_integration_fields_short_subdomain_errors():
    from STT_server.services.integrations_catalog import validate_integration_fields
    _, _, errors = validate_integration_fields(
        "zendesk",
        {"subdomain": "a"},
        {"email": "x@y.com", "api_token": "a" * 25},
    )
    assert errors
    assert errors[0]["field"].startswith("config.")


def test_validate_integration_fields_unknown_provider():
    from STT_server.services.integrations_catalog import validate_integration_fields
    _, _, errors = validate_integration_fields("made_up", {}, {})
    assert errors and errors[0]["field"] == "provider"