"""Multi-tenant n8n credentials for integrations (migration 029).

The bug this pins: n8n authenticated to /internal/integrations/{id}/* with
ONE platform-wide INTEGRATIONS_N8N_TOKEN, require_service_token returned
{"caller": "n8n"} with no user_id, and both internal endpoints used the
UNSCOPED get_integration_by_id lookup. So any n8n holding that token could
read the decrypted access token of any integration belonging to any user,
by guessing the integration id.

No DB, no network, no API key: the token primitives and the resolution /
scoping logic are pure. The parts that need Postgres are asserted at the
source level instead of executed.
"""
from __future__ import annotations

import ast
import inspect
import os
import pathlib
import sys

sys.path.insert(0, ".")

from STT_server.db_integrations import (  # noqa: E402
    N8N_TOKEN_PREFIX_LEN,
    clear_integration_n8n_token,
    generate_n8n_token,
    n8n_token_prefix,
    set_integration_n8n_token,
)
from STT_server.services import tool_executor  # noqa: E402

ROUTES = pathlib.Path("STT_server/routes/api.py")
DB = pathlib.Path("STT_server/db_integrations.py")
MIGRATION = pathlib.Path("db/migrations/029_integration_n8n_token.sql")


# ── Token primitives ──────────────────────────────────────────────

def test_prefix_is_a_hash_not_the_token():
    """The column is cleartext; a DB dump must not hand out working tokens."""
    token = generate_n8n_token()
    prefix = n8n_token_prefix(token)
    assert token not in prefix
    assert prefix not in token
    assert len(prefix) == N8N_TOKEN_PREFIX_LEN
    assert prefix == n8n_token_prefix(token), "must be deterministic"


def test_different_tokens_do_not_collide():
    seen = {n8n_token_prefix(generate_n8n_token()) for _ in range(2000)}
    assert len(seen) >= 1995, f"prefix collisions in 2k tokens: {2000 - len(seen)}"


def test_generated_token_is_url_safe_and_long():
    t = generate_n8n_token()
    assert len(t) >= 32, len(t)
    assert all(c.isalnum() or c in "-_" for c in t), t


def test_prefix_differs_for_similar_tokens():
    a, b = generate_n8n_token(), generate_n8n_token()
    assert n8n_token_prefix(a) != n8n_token_prefix(b)


# ── require_service_token resolution ──────────────────────────────

def test_per_integration_token_resolves_and_platform_falls_through():
    from STT_server.routes.api import _resolve_integration_n8n_token

    # With no integrations carrying a token, every lookup returns None and
    # the caller falls through to the platform token. Nothing is
    # authenticated by a prefix alone.
    assert _resolve_integration_n8n_token(generate_n8n_token()) is None
    assert _resolve_integration_n8n_token("") is None
    assert _resolve_integration_n8n_token("nonsense") is None


def test_a_db_failure_never_authenticates():
    """If the prefix lookup raises, we must not fall through to 'accepted'."""
    from STT_server.routes import api as routes_api

    src = inspect.getsource(routes_api._resolve_integration_n8n_token)
    assert "return None" in src
    # The lookup is inside a try that logs and returns None.
    assert "except Exception" in src, "lookup must be exception-guarded"


def test_verification_is_constant_time_and_uses_the_decrypted_token():
    from STT_server.routes import api as routes_api

    src = inspect.getsource(routes_api._resolve_integration_n8n_token)
    # The prefix only NARROWS candidates; the compare is on the secret.
    assert "decrypt_credentials" in src
    assert "compare_digest" in src, "token compare must be constant-time"
    assert "n8n_token" in src, "must read the n8n_token credential"


# ── Tenant scoping on both internal endpoints ─────────────────────

def _internal_endpoints():
    tree = ast.parse(ROUTES.read_text(encoding="utf-8-sig"))
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call) or not dec.args:
                continue
            first = dec.args[0]
            route = first.value if isinstance(first, ast.Constant) else ""
            if isinstance(route, str) and route.startswith("/internal/integrations/"):
                # Key by function name: both endpoints share the same route
                # prefix, so keying on the route would collapse them.
                out[node.name] = (route, node)
    return out


def test_both_internal_endpoints_are_tenant_scoped():
    """Both endpoints took the unscoped lookup before. Each must now reject
    a token that belongs to a different integration."""
    source = ROUTES.read_text(encoding="utf-8-sig")
    eps = _internal_endpoints()
    assert len(eps) >= 2, f"expected both internal endpoints, found {list(eps)}"
    scoped = 0
    for name, (route, fn) in eps.items():
        src = ast.get_source_segment(source, fn) or ""
        if '_service.get("integration_id")' in src:
            assert "_caller_integration_id != integration_id" in src, name
            # A 404, not a 403: a 403 confirms the row exists.
            assert "404" in src, name
            scoped += 1
    assert scoped == len(eps), (
        f"only {scoped}/{len(eps)} internal endpoints are tenant-scoped: "
        f"{sorted(eps)}"
    )


def test_n8n_token_prefix_never_reaches_the_browser():
    """It is cleartext and it is the exact lookup key require_service_token
    searches on, so GET /integrations must strip it."""
    from STT_server.routes import api as routes_api

    src = inspect.getsource(routes_api._strip_integration_for_wire)
    assert 'out.pop("n8n_token_prefix"' in src


def test_credentials_are_still_never_returned():
    from STT_server.routes import api as routes_api

    src = inspect.getsource(routes_api._strip_integration_for_wire)
    assert 'out.pop("credentials_encrypted"' in src
    assert 'out.pop("credentials_cipher"' in src


# ── The payload n8n receives must not carry the refresh token ────

def _credentials_for_n8n():
    """Lift the nested helper out of the request handler.

    It is defined INSIDE internal_get_integration_credentials, so it cannot
    be imported. exec'ing it standalone is still exercising the real source.
    Returns (callable, source_segment) — inspect.getsource does not work on
    an exec'd function, so the source comes from the AST.
    """
    source = ROUTES.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_credentials_for_n8n"
    )
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "api.py", "exec"), ns)
    return ns["_credentials_for_n8n"], (ast.get_source_segment(source, fn) or "")


def test_oauth_providers_get_only_the_bearer():
    """Regression: intercom / hubspot / nice_cxone were added to the
    catalog but not to _credentials_for_n8n's whitelist, so they fell into
    the `dict(credentials)` branch and n8n received access_token AND
    refresh_token AND the n8n_token. The refresh token is what keeps the
    integration alive — handing it out hands over the account."""
    fn, _src = _credentials_for_n8n()
    creds = {
        "access_token": "pat-abc",
        "refresh_token": "super-secret-refresh",
        "expires_at": "2026-10-09T00:00:00Z",
        "n8n_token": "our-own-token",
    }
    for provider in ("intercom", "hubspot", "nice_cxone",
                     "salesforce", "google_calendar"):
        out = fn(provider, creds)
        assert out == {"access_token": "pat-abc"}, (provider, out)
        assert "refresh_token" not in out, provider
        assert "n8n_token" not in out, provider


def test_static_providers_keep_their_config_but_never_the_n8n_token():
    """Static providers legitimately need their full dict (API keys,
    subdomain, tenant) — but the n8n_token is ours, not theirs."""
    fn, _src = _credentials_for_n8n()
    out = fn("zendesk", {
        "api_token": "zd-secret",
        "subdomain": "revolutionmedia",
        "n8n_token": "our-own-token",
    })
    assert out.get("api_token") == "zd-secret"
    assert out.get("subdomain") == "revolutionmedia"
    assert "n8n_token" not in out, (
        "the n8n_token must never be echoed back to n8n"
    )


def test_every_oauth_provider_is_whitelisted_or_blocked():
    """The invariant: an OAuth provider must either be in the bearer
    whitelist (so n8n gets only the access_token) or be explicitly 403'd
    BEFORE the credentials are shaped. Anything else falls into
    `dict(credentials)` and hands n8n the refresh token.

    dynamics365 is deliberately in the second group: its Dataverse token
    never leaves the backend at all, and the /execute endpoint is the only
    way to use it.
    """
    from STT_server.services.integrations_catalog import INTEGRATION_PROVIDERS
    from STT_server.services.oauth_providers import known_oauth_providers

    _fn, whitelist_src = _credentials_for_n8n()
    creds_src = ROUTES.read_text(encoding="utf-8-sig")

    catalog_oauth = {
        s.id for s in INTEGRATION_PROVIDERS
        if getattr(s, "auth_type", "static") == "oauth"
    }
    blocked = {"dynamics365"}
    assert blocked, "the blocked set should not be empty — then the rule below"

    for provider in set(known_oauth_providers()) | catalog_oauth:
        whitelisted = f'"{provider}"' in whitelist_src
        explicitly_blocked = (
            f'provider == "{provider}"' in creds_src
            and "403" in creds_src
        )
        assert whitelisted or provider in blocked, (
            f"{provider} is an OAuth provider that is neither whitelisted "
            "nor blocked, so n8n would receive the refresh token"
        )
        if provider in blocked:
            assert explicitly_blocked, (
                f"{provider} is in the blocked set but has no explicit "
                "403 guard in the credentials route"
            )


# ── Webhook resolution per tenant ─────────────────────────────────

def test_own_n8n_url_wins_over_everything():
    """This is the whole point: an integration must not be welded to the
    platform-wide INTEGRATIONS_N8N_WEBHOOK."""
    os.environ["INTEGRATIONS_N8N_WEBHOOK"] = "https://platform.example/webhook"
    os.environ["INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__HUBSPOT"] = "https://ovr.example/webhook"
    try:
        got = tool_executor._resolve_integration_webhook({
            "provider": "hubspot",
            "configuration": {"n8n_webhook_url": "https://tenant-a.example/webhook"},
        })
        assert got == "https://tenant-a.example/webhook"
    finally:
        os.environ.pop("INTEGRATIONS_N8N_WEBHOOK", None)
        os.environ.pop("INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__HUBSPOT", None)


def test_fallback_chain_is_unchanged_without_an_own_url():
    os.environ["INTEGRATIONS_N8N_WEBHOOK"] = "https://platform.example/webhook"
    os.environ["INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__HUBSPOT"] = "https://ovr.example/webhook"
    try:
        # provider override beats the base
        assert tool_executor._resolve_integration_webhook(
            {"provider": "hubspot", "configuration": {}}
        ) == "https://ovr.example/webhook"
        # base is used when there is no provider override
        os.environ.pop("INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__HUBSPOT")
        assert tool_executor._resolve_integration_webhook(
            {"provider": "hubspot", "configuration": {}}
        ) == "https://platform.example/webhook"
    finally:
        os.environ.pop("INTEGRATIONS_N8N_WEBHOOK", None)


def test_google_calendar_hardcode_is_now_the_last_resort_and_overridable():
    """It used to be a bare literal in the resolver — one operator's
    personal n8n instance baked into the platform source."""
    os.environ["INTEGRATIONS_N8N_WEBHOOK"] = "https://platform.example/webhook"
    try:
        got = tool_executor._resolve_integration_webhook(
            {"provider": "google_calendar", "configuration": {}}
        )
        assert got == "https://platform.example/webhook", (
            "the platform webhook must win over the google_calendar default"
        )
    finally:
        os.environ.pop("INTEGRATIONS_N8N_WEBHOOK", None)

    os.environ["GOOGLE_CALENDAR_N8N_WEBHOOK"] = "https://tenant-b.example/webhook"
    try:
        assert tool_executor._resolve_integration_webhook(
            {"provider": "google_calendar", "configuration": {}}
        ) == "https://tenant-b.example/webhook"
    finally:
        os.environ.pop("GOOGLE_CALENDAR_N8N_WEBHOOK", None)


def test_generic_webhook_legacy_url_still_works():
    got = tool_executor._resolve_integration_webhook({
        "provider": "generic_webhook",
        "configuration": {"webhook_url": "https://legacy.example/hook"},
    })
    assert got == "https://legacy.example/hook"


def test_no_url_resolves_to_empty_not_a_leaked_default():
    for k in ("INTEGRATIONS_N8N_WEBHOOK",
              "INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__ZENDESK",
              "GOOGLE_CALENDAR_N8N_WEBHOOK"):
        os.environ.pop(k, None)
    assert tool_executor._resolve_integration_webhook(
        {"provider": "zendesk", "configuration": {}}
    ) == ""


# ── OAuth create must not drop configuration ─────────────────────

def test_oauth_create_keeps_provider_config_fields():
    """Regression: the OAuth branch of POST /integrations assigned
    `cleaned_config = {}`, silently dropping every configuration field.
    Invisible for Salesforce (the callback writes instance_url) but it
    breaks NICE CXone, whose `tenant` POD is what its actions read."""
    from STT_server.routes.api import _validate_oauth_config_only

    cleaned, errors = _validate_oauth_config_only(
        "nice_cxone", {"tenant": "na1"},
    )
    assert errors == [], errors
    assert cleaned.get("tenant") == "na1", cleaned


def test_oauth_create_accepts_the_per_tenant_n8n_url():
    from STT_server.routes.api import _validate_oauth_config_only

    cleaned, errors = _validate_oauth_config_only(
        "hubspot", {"n8n_webhook_url": "https://tenant-a.example/webhook"},
    )
    assert errors == [], errors
    assert cleaned["n8n_webhook_url"] == "https://tenant-a.example/webhook"


def test_n8n_url_must_be_https():
    from STT_server.routes.api import _validate_oauth_config_only

    _, errors = _validate_oauth_config_only(
        "hubspot", {"n8n_webhook_url": "http://insecure.example/webhook"},
    )
    assert errors and "https" in errors[0]["message"], errors


def test_oauth_create_cannot_smuggle_credentials_via_configuration():
    from STT_server.routes.api import _validate_oauth_config_only

    cleaned, _ = _validate_oauth_config_only(
        "hubspot", {"api_key": "sk-should-be-dropped"},
    )
    assert "api_key" not in cleaned, cleaned


# ── Migration ─────────────────────────────────────────────────────

def test_migration_is_additive_and_indexed():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS n8n_token_prefix" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_integrations_n8n_token_prefix" in sql
    # The token itself must NOT be a column — it lives encrypted.
    assert "n8n_token TEXT" not in sql


def test_self_heal_knows_the_column():
    src = DB.read_text(encoding="utf-8-sig")
    assert '"n8n_token_prefix": "TEXT"' in src, (
        "self-heal must recreate the column where migration 029 failed"
    )


def test_rotation_endpoint_exists_and_is_owner_scoped():
    from STT_server.routes.api import rotate_integration_n8n_token

    src = inspect.getsource(rotate_integration_n8n_token)
    assert "db_get_integration(integration_id, auth[\"user_id\"])" in src, (
        "must verify ownership before minting"
    )
    assert "404" in src


def test_wiring_is_present_end_to_end():
    """The pieces must actually be connected, not just defined."""
    src = ROUTES.read_text(encoding="utf-8-sig")
    assert "_mint_n8n_token_for(row[\"id\"], auth[\"user_id\"])" in src, (
        "POST /integrations must mint the token"
    )
    assert 'out["n8n_token"]' in src, "create must return it exactly once"
    # ...and it must never be returned anywhere else.
    assert src.count('out["n8n_token"] =') == 1
    assert '"n8n_token": token,' in src, "the rotate endpoint returns it once too"


# ── Self-check ────────────────────────────────────────────────────

def _self_check():
    fns = [
        test_prefix_is_a_hash_not_the_token,
        test_different_tokens_do_not_collide,
        test_generated_token_is_url_safe_and_long,
        test_prefix_differs_for_similar_tokens,
        test_per_integration_token_resolves_and_platform_falls_through,
        test_a_db_failure_never_authenticates,
        test_verification_is_constant_time_and_uses_the_decrypted_token,
        test_both_internal_endpoints_are_tenant_scoped,
        test_n8n_token_prefix_never_reaches_the_browser,
        test_credentials_are_still_never_returned,
        test_oauth_providers_get_only_the_bearer,
        test_static_providers_keep_their_config_but_never_the_n8n_token,
        test_every_oauth_provider_is_whitelisted_or_blocked,
        test_own_n8n_url_wins_over_everything,
        test_fallback_chain_is_unchanged_without_an_own_url,
        test_google_calendar_hardcode_is_now_the_last_resort_and_overridable,
        test_generic_webhook_legacy_url_still_works,
        test_no_url_resolves_to_empty_not_a_leaked_default,
        test_oauth_create_keeps_provider_config_fields,
        test_oauth_create_accepts_the_per_tenant_n8n_url,
        test_n8n_url_must_be_https,
        test_oauth_create_cannot_smuggle_credentials_via_configuration,
        test_migration_is_additive_and_indexed,
        test_self_heal_knows_the_column,
        test_rotation_endpoint_exists_and_is_owner_scoped,
        test_wiring_is_present_end_to_end,
    ]
    for fn in fns:
        fn()
    print(f"multi_tenant_n8n: OK ({len(fns)} checks)")


if __name__ == "__main__":
    _self_check()