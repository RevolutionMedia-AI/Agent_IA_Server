"""OAuth provider registry + helpers for the Integration entity.

Authorization Code flow only (V1). Resource-Owner-Password and Client-
Credentials live behind a different entry point that doesn't apply to the
Integrations feature — operators consent via their own provider account,
the BE never holds a username/password.

Salesforce is the only OAuth provider in V1. The registry is shaped so
the next provider (Dynamics 365, Google, HubSpot) drops in by adding an
entry to `_OAUTH_PROVIDERS` and reading its own env vars; the route layer
+ the catalog entry stay untouched.

Boot contract (fail-closed):
  Each OAuth provider declares the env vars it REQUIRES. The boot
  validator runs at module import and raises RuntimeError on the first
  missing var, naming the provider so the deploy logs are actionable.
  We don't fall back to env-less mode — a missing SALESFORCE_CLIENT_ID
  means the OAuth flow literally cannot complete; better to crash the
  container than to serve a half-built integration that 500s on the
  first click.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

import urllib.error
import urllib.parse
import urllib.request
import json as _json

log = logging.getLogger("stt_server.services.oauth_providers")


# ── Wire types ──────────────────────────────────────────────────────────────


class OAuthError(Exception):
    """Raised when a provider interaction (exchange / refresh / revoke)
    returns a non-2xx or a payload we can't parse. The route layer maps
    this to a 502 (exchange) / 200 with reason='refresh_failed' (refresh
    on read) / log+continue (revoke is best-effort)."""


class RefreshTokenRevoked(OAuthError):
    """Salesforce rejected the refresh token — operator must reconnect."""


@dataclass(frozen=True)
class OAuthConfig:
    provider_id: str                       # "salesforce"
    authorize_url: str                     # full https://...
    token_url: str                         # full https://...
    revoke_url: str = ""                   # optional; some providers don't support revoke
    default_scopes: tuple[str, ...] = ()
    # ponytail: client_id / client_secret / redirect_uri come from env at
    # boot. We snapshot them onto the config so the runtime doesn't have
    # to read os.environ on every call.
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str = ""


@dataclass(frozen=True)
class OAuthTokenResponse:
    access_token: str
    refresh_token: Optional[str] = None    # Salesforce only emits on initial exchange
    expires_in: Optional[int] = None        # seconds; None = unknown
    scope: Optional[str] = None
    instance_url: Optional[str] = None      # provider-specific (Salesforce returns this)
    raw: dict = field(default_factory=dict)


# ── Registry ────────────────────────────────────────────────────────────────


_REQUIRED_ENV = ("SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET", "SALESFORCE_REDIRECT_URI")
_OPTIONAL_ENV = {"SALESFORCE_SCOPES": "api refresh_token"}


def _build_salesforce_config() -> OAuthConfig:
    """Reads SALESFORCE_* env vars. The boot validator runs first; this
    assumes they're present. The redirect_uri defaults to
    `${PUBLIC_URL}/integrations/salesforce/oauth/callback` when the
    explicit env var is missing — only relevant in dev where PUBLIC_URL
    is set. Production should set REDIRECT_URI explicitly to the public
    callback URL."""
    redirect_uri = os.environ.get("SALESFORCE_REDIRECT_URI", "").strip()
    if not redirect_uri:
        public_url = os.environ.get("PUBLIC_URL", "").rstrip("/")
        if public_url:
            redirect_uri = f"{public_url}/integrations/salesforce/oauth/callback"
    if not redirect_uri:
        raise RuntimeError(
            "SALESFORCE_REDIRECT_URI is not set and PUBLIC_URL is not "
            "configured; cannot build the redirect target. Set "
            "SALESFORCE_REDIRECT_URI=https://<backend>/integrations/salesforce/oauth/callback"
        )
    scopes_env = os.environ.get("SALESFORCE_SCOPES", "").strip()
    scopes = tuple(s.strip() for s in scopes_env.split() if s.strip()) if scopes_env else ("api", "refresh_token")
    return OAuthConfig(
        provider_id="salesforce",
        authorize_url="https://login.salesforce.com/services/oauth2/authorize",
        token_url="https://login.salesforce.com/services/oauth2/token",
        revoke_url="https://login.salesforce.com/services/oauth2/revoke",
        default_scopes=scopes,
        client_id=os.environ["SALESFORCE_CLIENT_ID"],
        client_secret=os.environ["SALESFORCE_CLIENT_SECRET"],
        redirect_uri=redirect_uri,
    )


_OAUTH_PROVIDERS: dict[str, OAuthConfig] = {
    "salesforce": _build_salesforce_config(),
}


def get_oauth_config(provider_id: str) -> OAuthConfig:
    cfg = _OAUTH_PROVIDERS.get(provider_id)
    if cfg is None:
        raise KeyError(f"Provider '{provider_id}' is not registered as OAuth")
    return cfg


def known_oauth_providers() -> list[str]:
    return list(_OAUTH_PROVIDERS.keys())


# ── Boot validator ───────────────────────────────────────────────────────────


def validate_oauth_env() -> None:
    """Run at module import. Fails closed if any registered OAuth
    provider is missing its required env vars.

    Called once during STT_server startup (the BE imports this module
    early enough). Crash with a clear message rather than serve a
    broken OAuth flow.
    """
    missing: list[tuple[str, tuple[str, ...]]] = []
    for provider_id in _OAUTH_PROVIDERS.keys():
        if provider_id == "salesforce":
            req = _REQUIRED_ENV
        else:
            # Future providers: each registers its own list. Default to
            # env vars named `<PROVIDER>_CLIENT_ID` + `_CLIENT_SECRET`
            # + `_REDIRECT_URI` if we add a registry helper.
            req = ()
        absent = [v for v in req if not os.environ.get(v, "").strip()]
        if absent:
            missing.append((provider_id, tuple(absent)))
    if missing:
        lines = "\n".join(
            f"  {pid}: missing {', '.join(vars_)}"
            for pid, vars_ in missing
        )
        raise RuntimeError(
            "OAuth provider env vars missing — refusing to start.\n"
            + lines
            + "\nSet these env vars on the backend service and redeploy."
        )


# ── State token ─────────────────────────────────────────────────────────────


def generate_state() -> tuple[str, str]:
    """Generate a fresh OAuth state.

    Returns (state, state_hash). The state is what we send to the
    provider; the state_hash is what we persist on the integration row.
    The callback hashes the incoming state and compares to the stored
    hash. Storing only the hash means a DB row leak can't be used to
    ride an active OAuth flow.
    """
    state = secrets.token_urlsafe(32)
    state_hash = hashlib.sha256(state.encode("ascii")).hexdigest()
    return state, state_hash


def hash_state(state: str) -> str:
    return hashlib.sha256(state.encode("ascii")).hexdigest()


def constant_time_eq(a: str, b: str) -> bool:
    """Constant-time compare for the state hash. SHA-256 comparison
    isn't a timing oracle in practice (the function is constant-time),
    but the helper keeps the intent explicit at the call site."""
    return hmac.compare_digest(a.encode("ascii"), b.encode("ascii"))


# ── Authorize URL ───────────────────────────────────────────────────────────


def build_authorize_url(
    config: OAuthConfig,
    state: str,
    scopes: Optional[tuple[str, ...]] = None,
    extra_params: Optional[dict] = None,
) -> str:
    """Build the provider's authorize URL. `state` is the raw token
    (not the hash) — the provider echoes it back, we hash on the way
    in. `scopes` defaults to the config's default_scopes."""
    effective_scopes = scopes if scopes is not None else config.default_scopes
    params = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(effective_scopes),
        "state": state,
    }
    if extra_params:
        params.update(extra_params)
    # ponytail: Salesforce uses 'scope' (space-separated). Some
    # providers use 'scopes' (array). We standardize on 'scope' here;
    # add a provider-specific override if a future OAuth provider
    # needs a different parameter name.
    sep = "&" if "?" in config.authorize_url else "?"
    return f"{config.authorize_url}{sep}{urllib.parse.urlencode(params)}"


# ── HTTP helper ─────────────────────────────────────────────────────────────


def _http_post_form(url: str, data: dict, *, timeout: float = 15.0) -> dict:
    """POST application/x-www-form-urlencoded and parse JSON.

    Salesforce (and most OAuth providers) want form-encoded bodies on
    their token endpoint, NOT JSON. We use urllib to keep the
    dependency surface small — the only OAuth calls are start,
    callback, refresh, and revoke, none of them hot enough to need
    aiohttp.
    """
    body = urllib.parse.urlencode(data).encode("ascii")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            try:
                return _json.loads(payload)
            except Exception:
                # ponytail: some providers return form-encoded bodies
                # on error (Salesforce's token endpoint does when the
                # client secret is wrong). Parse that into a dict so
                # the caller can read `error` / `error_description`.
                try:
                    parsed = dict(urllib.parse.parse_qsl(payload.decode("utf-8")))
                    return parsed
                except Exception:
                    return {"_raw": payload.decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as exc:
        # Read the error body — Salesforce puts the reason in there.
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
            err_payload = _json.loads(err_body) if err_body.strip().startswith("{") else dict(urllib.parse.parse_qsl(err_body))
        except Exception:
            err_payload = {"error": "http_error", "http_status": exc.code}
        raise OAuthError(
            f"OAuth HTTP {exc.code} from {url}: {err_payload.get('error') or err_payload.get('error_description') or 'unknown'}"
        ) from exc
    except Exception as exc:
        raise OAuthError(f"OAuth transport error against {url}: {exc}") from exc


def _oauth_error_kind(payload: dict) -> str:
    """Map a Salesforce error payload to one of our internal kinds so
    the route layer can react (revoked vs transient)."""
    err = (payload.get("error") or "").lower()
    desc = (payload.get("error_description") or "").lower()
    if err == "invalid_grant" or "refresh_token" in desc and "expired" in desc:
        return "refresh_revoked"
    if err in {"invalid_client", "invalid_client_id"}:
        return "config_error"
    return "transient"


# ── Token exchange ──────────────────────────────────────────────────────────


def exchange_code_for_tokens(config: OAuthConfig, code: str) -> OAuthTokenResponse:
    """POST to the provider's token endpoint with `grant_type=authorization_code`.
    Returns the parsed token response. Raises OAuthError on transport
    failure; raises RefreshTokenRevoked if the response explicitly says
    the code was rejected."""
    payload = _http_post_form(
        config.token_url,
        {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "redirect_uri": config.redirect_uri,
        },
    )
    if "error" in payload:
        kind = _oauth_error_kind(payload)
        msg = payload.get("error_description") or payload.get("error")
        if kind == "refresh_revoked":
            raise RefreshTokenRevoked(f"Salesforce rejected the code: {msg}")
        raise OAuthError(f"OAuth exchange failed: {payload.get('error')}: {msg}")
    return _parse_token_response(payload)


def refresh_access_token(config: OAuthConfig, refresh_token: str) -> OAuthTokenResponse:
    """POST grant_type=refresh_token. Salesforce only re-emits
    refresh_token when the operator's session policy rotates them; we
    keep the existing one in the caller."""
    payload = _http_post_form(
        config.token_url,
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        },
    )
    if "error" in payload:
        kind = _oauth_error_kind(payload)
        msg = payload.get("error_description") or payload.get("error")
        if kind == "refresh_revoked":
            raise RefreshTokenRevoked(f"Refresh token revoked: {msg}")
        raise OAuthError(f"OAuth refresh failed: {payload.get('error')}: {msg}")
    return _parse_token_response(payload)


def _parse_token_response(payload: dict) -> OAuthTokenResponse:
    """Salesforce-specific parsing. id URL + instance_url are
    provider-specific — keep this in one place so other providers can
    add their own branch."""
    instance_url = payload.get("instance_url")
    return OAuthTokenResponse(
        access_token=payload.get("access_token", ""),
        refresh_token=payload.get("refresh_token"),
        expires_in=_safe_int(payload.get("expires_in")),
        scope=payload.get("scope"),
        instance_url=instance_url,
        raw=payload,
    )


def _safe_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── Revoke (best-effort) ─────────────────────────────────────────────────────


def revoke_token(config: OAuthConfig, access_token: str) -> None:
    """Best-effort revoke. Salesforce returns 200 on success and 400 if
    the token is already gone. Either is fine from our perspective —
    we wipe the encrypted credentials locally regardless."""
    if not config.revoke_url:
        log.info("[oauth] %s has no revoke_url — skipping remote revoke", config.provider_id)
        return
    payload = _http_post_form(
        config.revoke_url,
        {"token": access_token},
    )
    if payload.get("error"):
        log.warning(
            "[oauth] revoke returned %s: %s — proceeding with local wipe",
            payload.get("error"), payload.get("error_description"),
        )


# ── Token expiry helper (used by route layer) ──────────────────────────────


def is_token_expiring(expires_at_iso: Optional[str], buffer_seconds: int = 60) -> bool:
    """True if the token expires within buffer_seconds from now (or has
    already expired, or is missing). None = unknown → treat as
    expiring so the next refresh establishes ground truth."""
    if not expires_at_iso:
        return True
    try:
        from datetime import datetime, timezone
        # Stored as ISO8601 with Z suffix (see db_integrations).
        s = expires_at_iso.rstrip("Z")
        exp = datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return True
    return exp <= (time.time() + buffer_seconds)


def now_plus_seconds(seconds: int) -> str:
    """ISO8601 UTC string for 'now + N seconds'. The route layer stores
    this as the new expires_at after a refresh."""
    from datetime import datetime, timezone, timedelta
    t = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return t.isoformat().replace("+00:00", "Z")