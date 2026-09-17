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
            assert f["type"] in ("text", "password", "url", "email", "select")


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


async def test_provider_test_flags_match_implemented_probes(client, auth_token):
    body = (await client.get(
        "/integrations/providers",
        headers={"Authorization": f"Bearer {auth_token}"},
    )).json()
    for pid in ("salesforce", "dynamics365"):
        spec = next(p for p in body["providers"] if p["id"] == pid)
        assert spec["has_test"] is True
    for pid in ("genesys_cloud", "nice_cxone"):
        spec = next(p for p in body["providers"] if p["id"] == pid)
        assert spec["has_test"] is False


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


def test_dynamics365_customer_insights_capability():
    """Customer Insights – Data lives under the same dynamics365 provider
    as capability='customer_insights', reusing the same OAuth/client."""
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
        actions_for_capability,
        is_valid_action_for_capability,
    )

    spec = get_integration_provider_spec("dynamics365")
    assert spec is not None
    assert spec.id == "dynamics365"  # no new provider
    # no new provider id
    from STT_server.services.integrations_catalog import list_integration_providers
    assert "dynamics365_customer_insights" not in {p.id for p in list_integration_providers()}

    ci_actions = actions_for_capability("dynamics365", "customer_insights")
    assert len(ci_actions) == 4
    assert {a.id for a in ci_actions} == {
        "ci_get_profile",
        "ci_search_profiles",
        "ci_get_segments",
        "ci_get_measures",
    }
    for a in ci_actions:
        assert getattr(a, "capability", None) == "customer_insights"

    # gate: core actions are not valid under customer_insights and vice-versa
    assert is_valid_action_for_capability("dynamics365", "ci_get_profile", "customer_insights") is True
    assert is_valid_action_for_capability("dynamics365", "find_customer", "customer_insights") is False
    assert is_valid_action_for_capability("dynamics365", "find_customer", "core") is True

    # provider still single OAuth / single client
    assert "customer_insights" in spec.capabilities
    assert spec.auth_type == "oauth"
