"""OAuth provider registry + helpers for the Integration entity.

Authorization Code flow only (V1). Resource-Owner-Password and Client-
Credentials live behind a different entry point that doesn't apply to the
Integrations feature — operators consent via their own provider account,
the BE never holds a username/password.

Salesforce is the only OAuth provider in V1. The registry is shaped so
the next provider (Dynamics 365, Google, HubSpot) drops in by adding an
entry to `_OAUTH_PROVIDERS` and reading its own env vars; the route
layer + the catalog entry stay untouched.

Boot contract (LAZY):
  Providers are NOT validated at boot. A deployment that doesn't use
  Salesforce shouldn't have to set SALESFORCE_* env vars just to
  start. Instead:
    * /integrations/providers lists every provider as available
      (the catalog is static — what you build with is what ships).
    * /oauth/start checks env vars at call time. If they're missing,
      it returns 503 with a clear message naming the missing variable.
    * /internal/integrations/{id}/credentials returns 503 if the
      provider is misconfigured (so n8n never gets a half-broken
      credential blob).
  Trade-off: a misconfigured deploy doesn't crash, but the operator
  discovers the misconfiguration the first time they click Connect
  on Salesforce instead of at deploy time. The error message names
  the missing variable, so the fix is one env var set + restart.
"""
from __future__ import annotations

import base64
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
    """Reads SALESFORCE_* env vars. The redirect_uri defaults to
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


# ponyy: registry is empty at boot. Providers are built lazily on
# first call to get_oauth_config — that means a deployment that
# doesn't use Salesforce can start without setting SALESFORCE_*
# env vars at all. The first /oauth/start for salesforce surfaces
# the misconfiguration (via 503), not a container crash.
_OAUTH_PROVIDERS: dict[str, OAuthConfig] = {}


def _ensure_provider_built(provider_id: str) -> None:
    """Lazy provider registry: build the OAuthConfig from env on first
    call. A deployment that doesn't touch a given OAuth provider
    never reads its env vars, never crashes, and never builds the
    registry for it. The first /oauth/start for the provider is
    what surfaces the misconfiguration (with a 503 + actionable
    message) instead of crashing the container at boot.
    """
    if provider_id in _OAUTH_PROVIDERS:
        return
    if provider_id == "salesforce":
        _OAUTH_PROVIDERS["salesforce"] = _build_salesforce_config()
    else:
        raise KeyError(f"Provider '{provider_id}' is not registered as OAuth")


def get_oauth_config(provider_id: str) -> OAuthConfig:
    _ensure_provider_built(provider_id)
    return _OAUTH_PROVIDERS[provider_id]


def known_oauth_providers() -> list[str]:
    return ["salesforce"]


def _required_env_vars(provider_id: str) -> tuple[str, ...]:
    """Env vars each provider needs. The route layer uses this to
    build the 503 message at call time. Keeping the lookup in this
    module means adding a provider only touches the registry + this
    helper — no FE or route changes."""
    if provider_id == "salesforce":
        return ("SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET", "SALESFORCE_REDIRECT_URI")
    return ()


def validate_oauth_env(provider_id: str) -> tuple[bool, tuple[str, ...]]:
    """Returns (ok, missing_vars_tuple). ok=False means the OAuth flow
    for this provider can't complete and the route should 503.

    Called at request time (not at boot). The boot no longer crashes
    on missing env vars — a deployment that doesn't use Salesforce
    can start clean.
    """
    needed = _required_env_vars(provider_id)
    missing = tuple(v for v in needed if not os.environ.get(v, "").strip())
    return (not missing, missing)


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


# ── PKCE (RFC 7636) ──────────────────────────────────────────────────────────
# ponyy: Salesforce Connected Apps default to "Require Proof Key for
# Code Exchange (PKCE) Extension for the Authorization Code Flow"
# on External Client Apps. Without `code_challenge` + `code_challenge
# _method=S256` on the authorize URL, the call 400s with
# "missing required code challenge". The verifier itself must be
# sent in the token-exchange POST.

def _code_challenge_from_verifier(code_verifier: str) -> str:
    """RFC 7636 §4.2: base64url(SHA256(verifier)) with padding stripped."""
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Generate a PKCE pair. Returns (code_verifier, code_challenge).

    Per RFC 7636 §4.1, the verifier is 43-128 chars of
    unreserved base64url alphabet. secrets.token_urlsafe(64) gives
    ~86 base64url chars — well within the spec.

    The challenge is base64url(SHA256(verifier)) with padding
    stripped (§4.2). Salesforce's PKCE verifier requires SHA-256;
    plain (no challenge) is rejected.
    """
    code_verifier = secrets.token_urlsafe(64)
    return code_verifier, _code_challenge_from_verifier(code_verifier)


# ── Authorize URL ───────────────────────────────────────────────────────────


def build_authorize_url(
    config: OAuthConfig,
    state: str,
    scopes: Optional[tuple[str, ...]] = None,
    extra_params: Optional[dict] = None,
    code_verifier: Optional[str] = None,
) -> str:
    """Build the provider's authorize URL. `state` is the raw token
    (not the hash) — the provider echoes it back, we hash on the way
    in. `scopes` defaults to the config's default_scopes.

    ponyy: PKCE. If `code_verifier` is provided, the URL also
    includes `code_challenge` + `code_challenge_method=S256`. The
    verifier itself is NOT sent in the authorize URL — it's stored
    on the integration row and sent in the token-exchange call on
    the callback. Salesforce's Connected App defaults to "Require
    PKCE for the Authorization Code flow" — without these params,
    the call 400s with "missing required code challenge".
    """
    effective_scopes = scopes if scopes is not None else config.default_scopes
    params = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(effective_scopes),
        "state": state,
    }
    if code_verifier:
        # RFC 7636 §4.2: the challenge is the base64url-encoded
        # SHA-256 of the verifier (no padding). The verifier is the
        # random 32-43 char string; the challenge is derived.
        challenge = _code_challenge_from_verifier(code_verifier)
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"
    if extra_params:
        params.update(extra_params)
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


def exchange_code_for_tokens(
    config: OAuthConfig, code: str, code_verifier: Optional[str] = None,
) -> OAuthTokenResponse:
    """POST to the provider's token endpoint with `grant_type=authorization_code`.
    Returns the parsed token response. Raises OAuthError on transport
    failure; raises RefreshTokenRevoked if the response explicitly says
    the code was rejected.

    ponyy: PKCE. If `code_verifier` is passed, we include it in the
    POST body — that's the second half of RFC 7636. The challenge
    was sent in the authorize URL; the verifier is sent here. If
    the challenge was sent but the verifier is missing (or wrong),
    Salesforce 400s with "invalid grant" / "invalid code_verifier".
    """
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": config.client_id,
        "client_secret": config.client_secret,
        "redirect_uri": config.redirect_uri,
    }
    if code_verifier:
        body["code_verifier"] = code_verifier
    payload = _http_post_form(config.token_url, body)
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