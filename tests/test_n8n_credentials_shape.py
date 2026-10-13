"""Fail-closed credential shaping for /internal/integrations/{id}/credentials.

This route is the ONE place a decrypted credential crosses to a caller
outside the backend. It decides by allowlist: what n8n may see is derived
from the provider's own catalog spec, and anything not declared is denied.

The bug it pins: the shape used to be an OAuth allowlist plus a catch-all
that returned the whole credentials dict minus n8n_token. Every provider
missing from that list leaked its refresh_token — dynamics365 was one of
them, so n8n was holding the token that keeps the Dynamics connection
alive. A provider nobody remembered to add to the list was a silent leak.
An allowlist cannot leak by omission, which is the property asserted
throughout.
"""
from __future__ import annotations

import pytest

from STT_server.routes import api
from STT_server.services.integrations_catalog import list_integration_providers

# The two credentials that must never leave the backend, for ANY provider.
NEVER_LEAVES = ("refresh_token", "n8n_token")


@pytest.fixture
def shape(monkeypatch):
    """Reach the nested _credentials_for_n8n without booting the route.

    The function is nested inside the endpoint, so it is lifted out of the
    source once here rather than duplicated per test — a copy would drift
    from the thing it is meant to police.
    """
    import inspect
    import re
    import textwrap

    src = inspect.getsource(api)
    start = src.index("def _credentials_for_n8n")
    body = textwrap.dedent(src[start:])
    # stop at the next def at the same indent as the helper
    m = re.search(r"\n    return \{", body)
    ns: dict = {}
    exec(compile(body[: m.start()], "<n8n-shape>", "exec"), ns)  # noqa: S102
    return ns["_credentials_for_n8n"]


FULL_ENVELOPE = {
    "access_token": "at-secret",
    "refresh_token": "rt-secret",
    "expires_at": 1700000000,
    "n8n_token": "our-service-token",
    # Zendesk's two SECRET_FIELD_TYPES members
    "email": "admin@acme.com",
    "api_token": "zendesk-secret",
    "oauth_client_secret": "genesys-secret",
    "environment_url": "https://org.crm.dynamics.com",
    "something_unexpected": "surprise",
}


# ── the two that must never leave ──────────────────────────────────────────

@pytest.mark.parametrize("provider", [p.id for p in list_integration_providers()])
def test_no_provider_ever_receives_the_refresh_token(shape, provider):
    out = shape(provider, FULL_ENVELOPE)
    assert "refresh_token" not in out, f"{provider} leaked its refresh_token"


@pytest.mark.parametrize("provider", [p.id for p in list_integration_providers()])
def test_no_provider_ever_receives_the_n8n_token(shape, provider):
    out = shape(provider, FULL_ENVELOPE)
    assert "n8n_token" not in out, f"{provider} leaked our service token"


@pytest.mark.parametrize("provider", [p.id for p in list_integration_providers()])
def test_no_provider_receives_an_undeclared_field(shape, provider):
    """The leak was always "something the catalog never promised"."""
    out = shape(provider, FULL_ENVELOPE)
    assert "something_unexpected" not in out, f"{provider} leaked an undeclared field"
    assert "environment_url" not in out, f"{provider} leaked configuration"


# ── what each provider SHOULD get ──────────────────────────────────────────

def test_oauth_providers_get_only_the_bearer(shape):
    for provider in ("salesforce", "dynamics365", "google_calendar",
                     "intercom", "hubspot", "nice_cxone"):
        assert shape(provider, FULL_ENVELOPE) == {"access_token": "at-secret"}


def test_zendesk_gets_both_secret_fields(shape):
    # subdomain is a text field -> configuration, not credentials.
    # email IS a secret field type (HTTP Basic), so it must be here.
    assert shape("zendesk", FULL_ENVELOPE) == {
        "api_token": "zendesk-secret", "email": "admin@acme.com",
    }


def test_genesys_gets_only_its_client_secret(shape):
    assert shape("genesys_cloud", FULL_ENVELOPE) == {"oauth_client_secret": "genesys-secret"}


def test_generic_webhook_has_no_secret_fields_so_gets_nothing(shape):
    # its webhook_url is a "url" field -> configuration, already returned
    assert shape("generic_webhook", FULL_ENVELOPE) == {}


def test_unknown_provider_is_denied_not_passed_through(shape):
    """Fail-closed: a provider that is not in the catalog must not get
    the raw dict. That was the whole leak."""
    assert shape("some_provider_we_never_wrote", FULL_ENVELOPE) == {}


def test_the_allowlist_follows_the_catalog(shape, monkeypatch):
    """Direction 1: declare a password field, it is allowed out.

    Without this the allowlist could silently rot into "nothing ever
    goes out" and nobody would notice until an integration broke.

    The patch works because _credentials_for_n8n resolves
    get_integration_provider_spec INSIDE its own body, so it picks up the
    monkeypatched attribute at call time.
    """
    from STT_server.services.integrations_catalog import (
        IntegrationFieldSpec, IntegrationProviderSpec,
        get_integration_provider_spec as real,
    )

    with_secret = IntegrationProviderSpec(
        id="zendesk", name="Zendesk", category="crm",
        fields=(IntegrationFieldSpec(name="brand_new_secret", label="S", type="password"),),
        actions=(),
    )
    monkeypatch.setattr(
        "STT_server.services.integrations_catalog.get_integration_provider_spec",
        lambda pid: with_secret if pid == "zendesk" else real(pid),
    )
    assert shape("zendesk", {"brand_new_secret": "ok"}) == {"brand_new_secret": "ok"}


def test_email_is_a_secret_field_type():
    """`email` is the field that broke. It is not obviously a secret --
    it looks like PII/config -- but it is the other half of HTTP Basic,
    so the BE encrypts it. Both the writer (validate_integration_fields)
    and the reader (_credentials_for_n8n) must agree, which is why they
    now share SECRET_FIELD_TYPES instead of hardcoding the tuple."""
    from STT_server.services.integrations_catalog import (
        SECRET_FIELD_TYPES, get_integration_provider_spec,
    )

    assert "email" in SECRET_FIELD_TYPES, (
        "email must stay a secret field type; if you are changing this, "
        "Zendesk HTTP Basic auth breaks for n8n"
    )
    spec = get_integration_provider_spec("zendesk")
    names = {f.name for f in spec.fields if f.type in SECRET_FIELD_TYPES}
    assert names == {"email", "api_token"}
    # and the non-secret sibling is NOT in that set
    assert "subdomain" not in names


def test_text_fields_are_not_credentials(shape, monkeypatch):
    """A text field must not become a credential just because it is in the
    catalog — text fields are already returned under configuration, and
    duplicating them here would widen the surface for no reason."""
    from STT_server.services.integrations_catalog import (
        IntegrationFieldSpec, IntegrationProviderSpec,
        get_integration_provider_spec as real,
    )

    with_text = IntegrationProviderSpec(
        id="zendesk", name="Zendesk", category="crm",
        fields=(IntegrationFieldSpec(name="subdomain", label="S", type="text"),),
        actions=(),
    )
    monkeypatch.setattr(
        "STT_server.services.integrations_catalog.get_integration_provider_spec",
        lambda pid: with_text if pid == "zendesk" else real(pid),
    )
    assert shape("zendesk", {"subdomain": "acme"}) == {}


def test_empty_credentials_is_safe(shape):
    for provider in ("salesforce", "zendesk", "unknown"):
        assert shape(provider, {}) == ({"access_token": None} if provider == "salesforce" else {})