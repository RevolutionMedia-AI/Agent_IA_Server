"""Microsoft Dynamics 365 / Dataverse OAuth discovery and action executor."""
from __future__ import annotations

from dataclasses import replace
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


_DISCOVERY_URL = "https://globaldisco.crm.dynamics.com/api/discovery/v2.0/Instances"
_GUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_ENV_HOST_RE = re.compile(
    r"^[a-z0-9-]+(?:\.api)?\.crm\d*\."
    r"(?:dynamics\.com|microsoftdynamics\.us|appsplatform\.us|dynamics\.cn)$"
)


class DataverseError(RuntimeError):
    def __init__(self, message: str, status: int = 0, code: str = "dataverse_error"):
        super().__init__(message)
        self.status = status
        self.code = code


def normalize_environment_url(value: str) -> str:
    raw = (value or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or not _ENV_HOST_RE.fullmatch(host)
    ):
        raise ValueError("Invalid Dynamics 365 environment URL")
    marker = "/api/data/"
    path = parsed.path or ""
    if marker in path.lower():
        path = path[: path.lower().index(marker)]
    return urllib.parse.urlunparse(("https", parsed.netloc, path.rstrip("/"), "", "", ""))


def _json_request(method: str, url: str, access_token: str, body: dict | None = None):
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {access_token}")
    req.add_header("Accept", "application/json")
    req.add_header("OData-MaxVersion", "4.0")
    req.add_header("OData-Version", "4.0")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    if method in ("POST", "PATCH"):
        req.add_header("Prefer", "return=representation")
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8")
            return (json.loads(raw) if raw else {}), dict(response.headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")[:1000]
        try:
            payload_error = json.loads(raw).get("error") or {}
            code = payload_error.get("code") or "dataverse_http_error"
            message = payload_error.get("message") or raw
        except Exception:
            code, message = "dataverse_http_error", raw
        raise DataverseError(message or f"Dataverse HTTP {exc.code}", exc.code, code) from exc
    except urllib.error.URLError as exc:
        raise DataverseError(f"Dataverse network error: {exc.reason}") from exc


def discover_environments(access_token: str) -> list[dict]:
    query = urllib.parse.urlencode({
        "$select": "ApiUrl,FriendlyName,Id,EnvironmentId,TenantId,UniqueName,State"
    })
    payload, _ = _json_request("GET", f"{_DISCOVERY_URL}?{query}", access_token)
    environments = []
    for item in payload.get("value") or []:
        if item.get("State") not in (None, 0):
            continue
        try:
            api_url = normalize_environment_url(item.get("ApiUrl") or item.get("Url") or "")
        except ValueError:
            continue
        environments.append({
            "environment_url": api_url,
            "name": item.get("FriendlyName") or item.get("UniqueName") or api_url,
            "organization_id": item.get("Id"),
            "environment_id": item.get("EnvironmentId"),
            "tenant_id": item.get("TenantId"),
        })
    return environments


def prepare_oauth_connection(credentials: dict) -> dict:
    """Discover accessible Dataverse environments and hydrate OAuth metadata."""
    environments = discover_environments(credentials.get("access_token") or "")
    if not environments:
        raise DataverseError("No accessible Dynamics 365 environments were found")
    config = {
        "capabilities": ["sales", "customer_service"],
        "environments": environments,
        "tenant_id": environments[0].get("tenant_id"),
    }
    if len(environments) == 1:
        selected = environments[0]
        config.update({
            "environment_url": selected["environment_url"],
            "environment_name": selected["name"],
            "organization_id": selected.get("organization_id"),
            "environment_id": selected.get("environment_id"),
        })
        _replace_with_environment_token(
            credentials, selected["environment_url"], selected.get("tenant_id")
        )
    return config


def oauth_config_for_tenant(tenant_id: str | None):
    from STT_server.services.oauth_providers import get_oauth_config
    config = get_oauth_config("dynamics365")
    if tenant_id and _GUID_RE.fullmatch(tenant_id):
        return replace(
            config,
            token_url=f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
        )
    return config


def _replace_with_environment_token(
    credentials: dict,
    environment_url: str,
    tenant_id: str | None = None,
) -> None:
    refresh_token = credentials.get("refresh_token")
    if not refresh_token:
        raise DataverseError("Microsoft did not return a refresh token; reconnect and grant offline_access")
    from STT_server.services.oauth_providers import (
        now_plus_seconds,
        refresh_access_token,
    )
    tokens = refresh_access_token(
        oauth_config_for_tenant(tenant_id),
        refresh_token,
        scopes=(f"{normalize_environment_url(environment_url)}/.default",),
    )
    credentials["access_token"] = tokens.access_token
    credentials["refresh_token"] = tokens.refresh_token or refresh_token
    if tokens.expires_in is not None:
        credentials["expires_at"] = now_plus_seconds(tokens.expires_in)
    if tokens.scope:
        credentials["scope"] = tokens.scope


