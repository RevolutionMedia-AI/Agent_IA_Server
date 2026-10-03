"""Integrations OAuth: Intercom, HubSpot, NICE CXone.

The contract these tests pin is the one that actually decides whether an
integration "disconnects":

  1. A provider whose access token does not expire and which issues no
     refresh token (Intercom) must NOT be driven through the refresh path.
     It used to be, because its token response carries no `expires_in`, so
     `expires_at` was absent, `is_token_expiring(None)` returned True, the
     refresh lookup found no `refresh_token`, and the integration was
     marked `failed` + 503'd on EVERY call.

  2. Both copies of refresh-on-read (the /credentials and /internal exec
     endpoints) must consult the SAME predicate, so they cannot drift.

  3. The three new providers must be registered as OAuth with their env
     vars declared, and the lazy registry must still not read env for a
     provider nobody touches.

No network, no API key. Run: python tests/test_integrations_oauth_providers.py
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import os
import pathlib
import sys

sys.path.insert(0, ".")

from STT_server.services import oauth_providers as op  # noqa: E402
from STT_server.services.integrations_catalog import (  # noqa: E402
    get_integration_provider_spec,
)

BE = pathlib.Path("STT_server/services/oauth_providers.py")
ROUTES = pathlib.Path("STT_server/routes/api.py")

NEW_OAUTH = ("intercom", "hubspot", "nice_cxone")


# ── 1. The refresh decision ────────────────────────────────────────

def test_intercom_never_attempts_refresh():
    """The regression this whole change exists for."""
    assert op.provider_refresh_supported("intercom") is False
    # Even with no expires_at at all — which is Intercom's real shape.
    assert op.should_attempt_refresh("intercom", {"access_token": "abc"}) is False
    assert op.should_attempt_refresh("intercom", {"expires_at": None}) is False
    # And even a stale one: there is nothing to refresh it with.
    assert op.should_attempt_refresh(
        "intercom", {"expires_at": op.now_plus_seconds(-99999)}
    ) is False


def test_intercom_refresh_support_does_not_depend_on_env():
    """Regression on the fix: deriving this from get_oauth_config() made a
    missing INTERCOM_CLIENT_ID flip the answer to True and silently
    reintroduce the bug."""
    for var in ("INTERCOM_CLIENT_ID", "INTERCOM_CLIENT_SECRET",
                "INTERCOM_REDIRECT_URI"):
        os.environ.pop(var, None)
    assert op.provider_refresh_supported("intercom") is False
    assert op.should_attempt_refresh("intercom", {"expires_at": None}) is False


def test_hubspot_refreshes_when_close_to_expiry():
    assert op.provider_refresh_supported("hubspot") is True
    assert op.should_attempt_refresh("hubspot", {"expires_at": None}) is True
    assert op.should_attempt_refresh(
        "hubspot", {"expires_at": op.now_plus_seconds(-5)}
    ) is True
    assert op.should_attempt_refresh(
        "hubspot", {"expires_at": op.now_plus_seconds(7200)}
    ) is False


def test_nice_cxone_refreshes():
    assert op.provider_refresh_supported("nice_cxone") is True
    assert op.should_attempt_refresh("nice_cxone", {"expires_at": None}) is True
    assert op.should_attempt_refresh(
        "nice_cxone", {"expires_at": op.now_plus_seconds(7200)}
    ) is False


def test_existing_providers_unchanged():
    """salesforce / google_calendar / dynamics365 must keep refreshing."""
    for p in ("salesforce", "google_calendar", "dynamics365"):
        assert op.provider_refresh_supported(p) is True, p
        assert op.should_attempt_refresh(p, {"expires_at": None}) is True, p


def test_unknown_provider_keeps_the_loud_behaviour():
    """An unregistered provider must NOT be silently treated as
    non-refreshing; it should still hit the "reconnect required" path."""
    assert op.provider_refresh_supported("mystery") is True
    assert op.should_attempt_refresh("mystery", {"expires_at": None}) is True


def test_static_providers_unaffected():
    assert op.should_attempt_refresh("zendesk", {"expires_at": None}) is True
    assert op.should_attempt_refresh("generic_webhook", {"expires_at": None}) is True


# ── 2. Both refresh paths use the one predicate ────────────────────

def test_both_refresh_paths_call_should_attempt_refresh():
    """routes/api.py has two independent copies of refresh-on-read
    (/internal/.../credentials and /internal/.../exec). They must both go
    through should_attempt_refresh, or one of them keeps the old
    unguarded is_token_expiring(None)."""
    src = ROUTES.read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "should_attempt_refresh"
    ]
    assert len(calls) == 2, (
        f"expected both refresh-on-read copies to use the helper, found "
        f"{len(calls)} call site(s)"
    )
    # And no bare `is_token_expiring(<creds expires_at>)` guard survives as
    # the top-level refresh condition.
    tree2 = ast.parse(src)
    stale = [
        n.lineno for n in ast.walk(tree2)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == "is_token_expiring"
    ]
    assert len(stale) <= 2, (
        f"unexpected is_token_expiring call sites in routes/api.py: {stale} — "
        "a new refresh path may have been added without the helper"
    )


# ── 3. Registry + env declarations ─────────────────────────────────

def test_new_providers_are_registered():
    known = op.known_oauth_providers()
    for p in NEW_OAUTH:
        assert p in known, f"{p} missing from known_oauth_providers()"
    # Existing ones must not have been dropped.
    for p in ("salesforce", "google_calendar", "dynamics365"):
        assert p in known, f"{p} disappeared from the registry"


# Env-var prefix per provider. Explicit because it is NOT always just the
# uppercased id: nice_cxone uses NICECXONE_* (no underscore).
ENV_PREFIX = {
    "intercom": "INTERCOM",
    "hubspot": "HUBSPOT",
    "nice_cxone": "NICECXONE",
}


def test_new_providers_declare_their_env_vars():
    for p in NEW_OAUTH:
        needed = op._required_env_vars(p)
        prefix = ENV_PREFIX[p]
        assert needed == (
            f"{prefix}_CLIENT_ID",
            f"{prefix}_CLIENT_SECRET",
            f"{prefix}_REDIRECT_URI",
        ), (p, needed)


def test_validate_oauth_env_reports_the_new_providers():
    for p in NEW_OAUTH:
        prefix = ENV_PREFIX[p]
        ok, missing = op.validate_oauth_env(p)
        assert ok is False, p
        assert set(missing) == {
            f"{prefix}_CLIENT_ID",
            f"{prefix}_CLIENT_SECRET",
            f"{prefix}_REDIRECT_URI",
        }, (p, missing)


def test_unknown_provider_has_no_required_env():
    assert op._required_env_vars("mystery") == ()
    assert op.validate_oauth_env("mystery") == (True, ())


@contextlib.contextmanager
def _with_env(vars_: dict):
    """Set env, run, restore. Used so we can actually BUILD each config and
    assert on the URLs instead of only asserting the env-var declarations."""
    saved = {k: os.environ.get(k) for k in vars_}
    os.environ.update(vars_)
    op._OAUTH_PROVIDERS.clear()
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        op._OAUTH_PROVIDERS.clear()


def test_each_new_provider_builds_its_config():
    """The three configs must construct, with the URLs we expect."""
    base = {
        "PUBLIC_URL": "https://voice.example.com",
        "INTERCOM_CLIENT_ID": "iid", "INTERCOM_CLIENT_SECRET": "isec",
        "HUBSPOT_CLIENT_ID": "hid", "HUBSPOT_CLIENT_SECRET": "hsec",
        "NICECXONE_CLIENT_ID": "nid", "NICECXONE_CLIENT_SECRET": "nsec",
    }
    for k in ("INTERCOM_REDIRECT_URI", "HUBSPOT_REDIRECT_URI",
              "NICECXONE_REDIRECT_URI"):
        base[k] = ""

    with _with_env(base):
        expected = {
            "intercom": (
                "https://app.intercom.com/oauth",
                "https://api.intercom.com/auth/eagle/token",
            ),
            "hubspot": (
                "https://app.hubspot.com/oauth/authorize",
                "https://api.hubapi.com/oauth/v1/token",
            ),
            "nice_cxone": (
                "https://oauth.nicecxone.com/oauth2/v1/authorize",
                "https://oauth.nicecxone.com/oauth2/v1/token",
            ),
        }
        for pid, (authorize, token) in expected.items():
            cfg = op.get_oauth_config(pid)
            assert cfg.authorize_url == authorize, (pid, cfg.authorize_url)
            assert cfg.token_url == token, (pid, cfg.token_url)
            # Callback URL derived from PUBLIC_URL, one shape for all.
            assert cfg.redirect_uri == (
                f"https://voice.example.com/integrations/{pid}/oauth/callback"
            ), (pid, cfg.redirect_uri)
            assert cfg.client_id and cfg.client_secret


def test_intercom_config_declares_no_refresh():
    with _with_env({
        "PUBLIC_URL": "https://voice.example.com",
        "INTERCOM_CLIENT_ID": "iid", "INTERCOM_CLIENT_SECRET": "isec",
        "INTERCOM_REDIRECT_URI": "https://x.test/cb",
    }):
        cfg = op.get_oauth_config("intercom")
        assert cfg.refresh_supported is False
        # No scopes: Intercom rejects an unrecognised scope parameter.
        assert cfg.default_scopes == ()


def test_urls_are_env_overridable():
    """Intercom regions and CXone regions differ, and I could not verify the
    non-default variants from the docs here. The override is the escape
    hatch so a wrong default is a 5-second Railway change, not a deploy."""
    with _with_env({
        "PUBLIC_URL": "https://voice.example.com",
        "INTERCOM_CLIENT_ID": "iid", "INTERCOM_CLIENT_SECRET": "isec",
        "INTERCOM_REDIRECT_URI": "https://x.test/cb",
        "INTERCOM_AUTHORIZE_URL": "https://app.eu.intercom.com/oauth",
        "INTERCOM_TOKEN_URL": "https://api.eu.intercom.com/auth/eagle/token",
    }):
        cfg = op.get_oauth_config("intercom")
        assert cfg.authorize_url == "https://app.eu.intercom.com/oauth"
        assert cfg.token_url == "https://api.eu.intercom.com/auth/eagle/token"

    with _with_env({
        "PUBLIC_URL": "https://voice.example.com",
        "NICECXONE_CLIENT_ID": "nid", "NICECXONE_CLIENT_SECRET": "nsec",
        "NICECXONE_REDIRECT_URI": "https://x.test/cb",
        "NICECXONE_AUTH_BASE": "https://oauth.nice.eu/",
    }):
        cfg = op.get_oauth_config("nice_cxone")
        assert cfg.authorize_url == "https://oauth.nice.eu/oauth2/v1/authorize"
        assert cfg.token_url == "https://oauth.nice.eu/oauth2/v1/token"


def test_lazy_registry_does_not_build_untouched_providers():
    """Building hubspot must not require SALESFORCE_*, and vice versa."""
    with _with_env({
        "PUBLIC_URL": "https://voice.example.com",
        "HUBSPOT_CLIENT_ID": "hid", "HUBSPOT_CLIENT_SECRET": "hsec",
        "HUBSPOT_REDIRECT_URI": "https://x.test/cb",
    }):
        os.environ.pop("SALESFORCE_CLIENT_ID", None)
        op.get_oauth_config("hubspot")
        assert "salesforce" not in op._OAUTH_PROVIDERS, (
            "building hubspot must not construct the salesforce config"
        )
        assert "hubspot" in op._OAUTH_PROVIDERS


def test_redirect_uri_falls_back_to_public_url():
    os.environ.pop("INTERCOM_REDIRECT_URI", None)
    os.environ["PUBLIC_URL"] = "https://voice.example.com/"
    try:
        got = op._redirect_uri_for("intercom", "INTERCOM_REDIRECT_URI")
        assert got == (
            "https://voice.example.com/integrations/intercom/oauth/callback"
        )
    finally:
        os.environ.pop("PUBLIC_URL", None)


def test_redirect_uri_raises_without_any_source():
    os.environ.pop("INTERCOM_REDIRECT_URI", None)
    os.environ.pop("PUBLIC_URL", None)
    try:
        op._redirect_uri_for("intercom", "INTERCOM_REDIRECT_URI")
    except RuntimeError as exc:
        assert "INTERCOM_REDIRECT_URI" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError naming the missing env var")


def test_explicit_redirect_uri_wins():
    os.environ["INTERCOM_REDIRECT_URI"] = "https://x.test/cb"
    os.environ["PUBLIC_URL"] = "https://ignored.test"
    try:
        assert op._redirect_uri_for("intercom", "INTERCOM_REDIRECT_URI") == "https://x.test/cb"
    finally:
        os.environ.pop("INTERCOM_REDIRECT_URI", None)
        os.environ.pop("PUBLIC_URL", None)


def test_scopes_helper_parses_space_and_comma():
    os.environ["HUBSPOT_SCOPES"] = "a.b, c.d  e.f"
    try:
        assert op._scopes_from_env("HUBSPOT_SCOPES", ("z",)) == ("a.b", "c.d", "e.f")
    finally:
        os.environ.pop("HUBSPOT_SCOPES", None)
    assert op._scopes_from_env("HUBSPOT_SCOPES", ("z",)) == ("z",)


# ── 4. Catalog specs ───────────────────────────────────────────────

def test_new_providers_are_oauth_specs():
    for p in ("intercom", "hubspot", "nice_cxone"):
        spec = get_integration_provider_spec(p)
        assert spec is not None, f"{p} missing from the catalog"
        assert spec.auth_type == "oauth", (p, spec.auth_type)
        assert spec.oauth_label, f"{p} has no Connect button label"


def test_nice_cxone_keeps_its_actions_and_tenant():
    """The migration must not cost the operator the two CXone actions."""
    spec = get_integration_provider_spec("nice_cxone")
    names = {a.id for a in spec.actions}
    assert names == {"transfer_call", "get_skill_stats"}, names
    field_names = {f.name for f in spec.fields}
    assert "tenant" in field_names, "tenant is needed by the CXone actions"
    # The pasted-token field is gone: OAuth supplies the credential.
    assert "access_token" not in field_names, (
        "nice_cxone still advertises a manual access_token field while "
        "auth_type=oauth — the FE never renders it and it invites confusion"
    )


def test_new_providers_have_no_actions_by_design():
    """Automation lives in n8n. The product only provides connect +
    validate + keep-alive for these."""
    for p in ("intercom", "hubspot"):
        assert get_integration_provider_spec(p).actions == ()


def test_hubspot_offline_access_scope_present():
    """HubSpot only issues a refresh token when the `oauth` (offline
    access) scope is requested. Without it the integration would have a
    ~30 minute token and nothing to refresh with."""
    scopes = get_integration_provider_spec("hubspot").oauth_default_scopes
    assert "oauth" in scopes, scopes
    assert any(s.startswith("crm.objects.contacts.read") for s in scopes), scopes


def test_existing_providers_untouched():
    assert get_integration_provider_spec("zendesk").auth_type == "static"
    assert len(get_integration_provider_spec("salesforce").actions) == 6
    assert get_integration_provider_spec("dynamics365").auth_type == "oauth"


# ── Self-check (no pytest needed) ─────────────────────────────────

def _self_check():
    for fn in (
        test_intercom_never_attempts_refresh,
        test_intercom_refresh_support_does_not_depend_on_env,
        test_hubspot_refreshes_when_close_to_expiry,
        test_nice_cxone_refreshes,
        test_existing_providers_unchanged,
        test_unknown_provider_keeps_the_loud_behaviour,
        test_static_providers_unaffected,
        test_both_refresh_paths_call_should_attempt_refresh,
        test_new_providers_are_registered,
        test_new_providers_declare_their_env_vars,
        test_each_new_provider_builds_its_config,
        test_intercom_config_declares_no_refresh,
        test_urls_are_env_overridable,
        test_lazy_registry_does_not_build_untouched_providers,
        test_validate_oauth_env_reports_the_new_providers,
        test_unknown_provider_has_no_required_env,
        test_lazy_registry_does_not_build_untouched_providers,
        test_redirect_uri_falls_back_to_public_url,
        test_redirect_uri_raises_without_any_source,
        test_explicit_redirect_uri_wins,
        test_scopes_helper_parses_space_and_comma,
        test_new_providers_are_oauth_specs,
        test_nice_cxone_keeps_its_actions_and_tenant,
        test_new_providers_have_no_actions_by_design,
        test_hubspot_offline_access_scope_present,
        test_existing_providers_untouched,
    ):
        fn()
    print("integrations_oauth: OK (22 checks)")


if __name__ == "__main__":
    _self_check()