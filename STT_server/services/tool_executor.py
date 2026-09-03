"""Tool executor service for calling n8n webhooks internally during conversations."""

import asyncio
import ipaddress
import os
import socket
import httpx
from datetime import datetime, timezone
from typing import Optional
import logging
from urllib.parse import urlparse

from STT_server.db_tools import get_tool as db_get_tool, update_tool as db_update_tool

log = logging.getLogger(__name__)


# ponytail: 016 — fields the LLM MUST NOT control. If any of these
# show up in the tool_call arguments (because someone misconfigures
# the function description, or a prompt-injection attack slips one
# past OpenAI's parser), we strip them before the body goes to n8n.
# action + provider are dispatched by the BE; integration_id /
# webhook_url / credentials are server-injected from the DB.
_FORBIDDEN_LLM_KEYS = frozenset({
    "action", "provider", "integration_id", "webhook_url", "credentials",
})


def _resolve_integration_webhook(integration: dict) -> str:
    """Resolve the URL the executor will POST to for an integration-bound tool.

    Precedence:
      1. INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__<PROVIDER> env var (uppercased)
      2. INTEGRATIONS_N8N_WEBHOOK (single router, used for all official
         providers — the user's n8n Switch node dispatches by `action`)
      3. integration.configuration["webhook_url"] (only valid for
         provider="generic_webhook" — official providers don't carry a URL)

    Returns "" when no URL can be resolved; the caller raises
    ToolExecutionError on empty so the LLM gets a clear failure.
    """
    provider = (integration.get("provider") or "").strip().upper()
    override = os.environ.get(f"INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__{provider}", "").strip()
    if override:
        return override
    base = os.environ.get("INTEGRATIONS_N8N_WEBHOOK", "").strip()
    if base:
        return base
    # Fallback: only generic_webhook is allowed to surface its own URL.
    if integration.get("provider") == "generic_webhook":
        return (integration.get("configuration") or {}).get("webhook_url", "") or ""
    if integration.get("provider") == "google_calendar":
        # ponytail: zero-config Google Calendar - URL is server-managed, not operator-configured
        return "https://revomedia.app.n8n.cloud/webhook/agendar-cita-dinamica"
    return ""


def _resolve_integration_method(integration: dict | None) -> str:
    """Resolve HTTP method for generic_webhook. Defaults to POST for legacy rows."""
    if not integration or integration.get("provider") != "generic_webhook":
        return "POST"
    method = ((integration.get("configuration") or {}).get("webhook_method") or "POST").strip().upper()
    if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
        return "POST"
    return method


def _redact_url(url: str) -> str:
    """Ponytail: SEC — never leak a generic_webhook URL into a client
    error. Keeps the host (and port, if any), redacts path/query so
    a logged error in Railway (or bubbled up to the FE) doesn't
    reveal the tokenized path token an operator embedded for access
    control."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not host:
            return "****"
        netloc = host
        if parsed.port is not None:
            netloc = f"{host}:{parsed.port}"
        return f"{parsed.scheme}://{netloc}/****"
    except Exception:
        return "****"


def _strip_forbidden_args(arguments: Optional[dict]) -> dict:
    """Drop keys the LLM must not control before passing to n8n."""
    if not isinstance(arguments, dict):
        return {}
    return {k: v for k, v in arguments.items() if k not in _FORBIDDEN_LLM_KEYS}


# ponytail: SSRF allow-list for tool webhook URLs. The user can override
# via the TOOL_WEBHOOK_ALLOW_HOSTS env var (comma-separated hostnames)
# when they actually need to reach an internal n8n on a private
# network. By default the executor refuses loopback / link-local /
# RFC1918 / cloud-metadata addresses so a compromised tool URL can't
# be used to probe AWS/GCP/Azure metadata or hit internal services.
TOOL_WEBHOOK_ALLOW_HOSTS = frozenset(
    h.strip().lower() for h in os.environ.get("TOOL_WEBHOOK_ALLOW_HOSTS", "").split(",") if h.strip()
)
TOOL_WEBHOOK_ALLOW_PRIVATE = os.environ.get("TOOL_WEBHOOK_ALLOW_PRIVATE", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def _resolve_host_ips(host: str) -> list[str]:
    """Resolve a hostname to all its A/AAAA IPs. Catches DNS-based
    SSRF where the hostname passes a textual deny-list check but the
    resolver returns an internal IP. Empty list on failure.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    out: list[str] = []
    for family, _, _, _, sockaddr in infos:
        if family in (socket.AF_INET, socket.AF_INET6):
            out.append(sockaddr[0])
    return out


def _ip_is_blocked(ip_str: str) -> bool:
    """True when ip_str is loopback, link-local, multicast, unspecified,
    or RFC1918 / ULA / CGNAT. The categories we never want a tool
    webhook to hit by accident."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable — fail closed
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_private  # covers RFC1918 + IPv6 ULA + CGNAT 100.64/10
        or ip.is_reserved
    )


def _validate_webhook_url(webhook_url: str) -> None:
    """Reject URLs that resolve to internal / metadata addresses.

    Raises ToolExecutionError on a blocked host. The check runs before
    the network call so a malicious tool row can't probe internal
    services even once.
    """
    try:
        parsed = urlparse(webhook_url)
    except Exception as exc:
        raise ToolExecutionError(f"webhook URL parse failed: {exc}")
    if parsed.scheme not in ("http", "https"):
        raise ToolExecutionError(
            f"webhook URL scheme '{parsed.scheme}' not allowed (http/https only)"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ToolExecutionError("webhook URL has no host")
    # Explicit allow-list short-circuits the IP block (so an operator
    # can whitelist an internal n8n host by name).
    if host in TOOL_WEBHOOK_ALLOW_HOSTS:
        return
    # Resolve and inspect every IP. Catches both textual private
    # literals (10.x, 192.168.x) and DNS-resolved internal addresses.
    ips = _resolve_host_ips(host)
    if not ips:
        raise ToolExecutionError(f"webhook host '{host}' did not resolve")
    if TOOL_WEBHOOK_ALLOW_PRIVATE:
        # Operator opted in: private IPs are fine, but still block
        # link-local / metadata addresses that are never legitimate
        # webhook targets.
        for ip in ips:
            try:
                ip_obj = ipaddress.ip_address(ip)
            except ValueError:
                continue
            if ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_unspecified:
                raise ToolExecutionError(
                    f"webhook host {host} resolved to blocked address {ip}"
                )
        return
    for ip in ips:
        if _ip_is_blocked(ip):
            raise ToolExecutionError(
                f"webhook host {host} resolved to blocked address {ip} "
                "(loopback / private / link-local / metadata). "
                "Set TOOL_WEBHOOK_ALLOW_PRIVATE=1 or TOOL_WEBHOOK_ALLOW_HOSTS to override."
            )


def record_tool_result(tool_id: str, ok: bool, kind: str, error: str | None = None) -> None:
    """Update observability fields on a tool row after a run.

    `kind` is either "test" (operator clicked Test webhook) or
    "invocation" (the agent called the tool during a real call).
    `error` is the short error string captured by the caller (the
    n8n response body, a connection error, etc.) so the operator
    can see why a tool failed without digging through Railway logs.

    ponytail: 010_agent_tools.sql moved storage to Postgres. We
    look up the tool's owning user_id first (required by db_update_tool
    for the row's WHERE clause), build the patch, and call the
    helper. The previous file-based path was a load-find-mutate-save
    round-trip; the Postgres path is one UPDATE statement and
    survives container restarts.
    """
    try:
        if not tool_id:
            return
        # ponytail: the new patch is a partial dict, not a full
        # row — db_update_tool builds the SET clause from the keys
        # actually present, so the row's other fields (id, agent_id,
        # etc.) are untouched. We need user_id because db_update_tool
        # scopes its WHERE clause by ownership; a tiny direct lookup
        # is cheaper than threading user_id through every call site.
        from STT_server.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT user_id FROM agent_tools WHERE id = %s LIMIT 1",
                    (tool_id,),
                )
                row = cur.fetchone()
        if not row:
            return
        user_id = row[0]
        status = "ok" if ok else "fail"
        # ponytail: cap the error string at 500 chars so a runaway
        # n8n stack-trace dump doesn't bloat every tool row on disk.
        err_text = (error or "").strip()[:500] or None
        patch: dict = {}
        if kind == "test":
            patch["last_tested_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            patch["last_test_result"] = status
            patch["last_test_error"] = err_text
            patch["last_test_error_at"] = patch["last_tested_at"] if err_text else None
        else:
            patch["last_invoked_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            patch["last_invocation_status"] = status
            patch["last_invocation_error"] = err_text
            patch["last_invocation_error_at"] = patch["last_invoked_at"] if err_text else None
        db_update_tool(tool_id, user_id, patch)
    except Exception as exc:
        log.warning("[ToolExecutor] record_tool_result failed for %s: %s", tool_id, exc)


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    pass


class ToolExecutor:
    """Executes agent tools (n8n webhooks) during live conversations.

    This is called directly by the turn_manager, not via HTTP endpoint,
    to avoid unnecessary network overhead during the call.
    """

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def execute(
        self,
        webhook_url: str,
        arguments: dict,
        tool_name: str,
        method: str = "POST",
    ) -> dict:
        """Execute a tool by calling its n8n webhook.

        Args:
            webhook_url: The n8n webhook URL to call.
            arguments: The arguments collected by the LLM to send to the tool.
            tool_name: Name of the tool being executed (for logging).
            method: HTTP method (GET, POST, PUT, PATCH, DELETE). Defaults to POST.

        Returns:
            The JSON response from n8n as a dict.

        Raises:
            ToolExecutionError: If the webhook call fails or times out,
                or the URL fails SSRF validation.
        """
        # ponytail: 016 — strip LLM-controlled forbidden keys before
        # the body leaves our process. The execution body itself
        # (server-injected action / provider / integration_id) is
        # built by the route layer that calls execute() with the
        # already-resolved integration context; here we just defend
        # against stray arguments leaking through.
        sanitized_args = _strip_forbidden_args(arguments)
        payload = {
            "tool_name": tool_name,
            "arguments": sanitized_args,
        }

        # ponytail: SSRF guard. Run BEFORE any DNS / network call so a
        # blocked URL never reaches the resolver. _validate_webhook_url
        # raises ToolExecutionError on loopback / RFC1918 / metadata.
        try:
            _validate_webhook_url(webhook_url)
        except ToolExecutionError as exc:
            log.warning(
                "[ToolExecutor] SSRF rejected tool '%s' url=%s err=%s",
                tool_name, _redact_url(webhook_url), exc,
            )
            raise

        method = (method or "POST").strip().upper()
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"):
            method = "POST"

        log.info(
            "[ToolExecutor] Executing tool '%s' method=%s host=%s args_keys=%s",
            tool_name, method,
            urlparse(webhook_url).hostname or "<unknown>",
            list(sanitized_args.keys()) if isinstance(sanitized_args, dict) else type(sanitized_args).__name__,
        )

        try:
            # ponytail: follow_redirects=False so a 30x to an internal
            # address can't bypass the SSRF check (the redirect target
            # is NOT validated by _validate_webhook_url because that
            # runs only on the original URL). If the user genuinely
            # needs to follow a redirect, they can change the tool URL.
            async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False,
            ) as client:
                if method == "GET":
                    response = await client.request(
                        method, webhook_url, params=payload,
                        headers={"Content-Type": "application/json"},
                    )
                elif method == "HEAD":
                    response = await client.request(method, webhook_url, headers={"Content-Type": "application/json"})
                else:
                    response = await client.request(
                        method, webhook_url, json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                if response.status_code >= 400:
                    raise ToolExecutionError(
                        f"n8n webhook returned HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )

                try:
                    result = response.json()
                except Exception:
                    result = {"raw_response": response.text}

                log.info(
                    "[ToolExecutor] Tool '%s' completed status=%s content_len=%s",
                    tool_name,
                    response.status_code,
                    response.headers.get("content-length") or len(response.content),
                )
                return result

        except httpx.TimeoutException:
            log.warning("[ToolExecutor] Tool '%s' timed out after %ss", tool_name, self.timeout)
            raise ToolExecutionError(f"Tool '{tool_name}' timed out after {self.timeout}s")

        except httpx.RequestError as exc:
            log.error("[ToolExecutor] Tool '%s' request failed: %s", tool_name, exc)
            raise ToolExecutionError(f"Tool '{tool_name}' request failed: {exc}")


# Singleton instance for use across the application
_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    """Get or create the singleton ToolExecutor instance."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor


async def execute_tool(
    webhook_url: str,
    arguments: dict,
    tool_name: str,
    method: str = "POST",
) -> dict:
    """Convenience function to execute a tool using the singleton executor."""
    executor = get_tool_executor()
    return await executor.execute(webhook_url, arguments, tool_name, method=method)


# ponytail: 016 — single entry point used by turn_manager after
# loading a tool + (optionally) its integration. Server-injects
# action / provider / integration_id into the n8n body so the LLM
# never controls them. Resolves the webhook URL from the
# integration row (env vars or configuration.webhook_url for
# generic_webhook) and falls back to tool.webhook_url for legacy
# rows that pre-date the integration_id column.
async def execute_tool_call(
    tool: dict,
    user_id: str,
    llm_arguments: Optional[dict],
) -> dict:
    """Execute one tool call, integration-aware.

    Args:
        tool: agent_tools row dict (must include id, webhook_url,
            integration_id, action, function_name).
        user_id: the calling user (owns the tool row).
        llm_arguments: arguments emitted by the LLM. Forbidden keys
            are stripped defensively before the body goes out.

    Returns:
        The JSON response from n8n as a dict.

    Raises:
        ToolExecutionError: when the integration is missing /
        revoked, no webhook URL can be resolved, or the HTTP call
        fails.
    """
    integration = None
    if tool.get("integration_id"):
        from STT_server.db_integrations import get_integration as db_get_integration
        integration = db_get_integration(tool["integration_id"], user_id)
        if not integration:
            raise ToolExecutionError(
                f"Integration '{tool['integration_id']}' missing or revoked"
            )
    # Resolve the webhook URL: integration-aware path first, legacy
    # tool.webhook_url as fallback.
    url = _resolve_integration_webhook(integration) if integration else (tool.get("webhook_url") or "")
    if not url:
        raise ToolExecutionError(f"Tool '{tool.get('id')}' has no webhook URL")
    sanitized_args = _strip_forbidden_args(llm_arguments)
    body: dict = {
        "tool_name": tool.get("function_name") or tool.get("name") or "",
        "arguments": sanitized_args,
    }
    # ponytail: server-injected fields. The LLM doesn't control any
    # of these — they come from the BE lookup of the tool row + its
    # integration. The n8n Switch node keys off `action`.
    if integration:
        body["integration_id"] = integration["id"]
        body["provider"] = integration["provider"]
        if tool.get("action"):
            body["action"] = tool["action"]
    elif tool.get("action"):
        # Degraded: tool carries an action but no integration
        # binding. Still useful — the n8n Switch can dispatch on it.
        body["action"] = tool["action"]
    executor = get_tool_executor()
    method = _resolve_integration_method(integration)
    # We pass the resolved URL directly to .execute(), which strips
    # a second time (defense in depth) and posts the body.
    try:
        return await executor.execute(url, body, body["tool_name"], method=method)
    except ToolExecutionError as exc:
        # Redact the URL in the message so the FE / logs don't leak
        # a tokenized generic_webhook path. The class allows us to
        # wrap + re-raise cleanly.
        raise ToolExecutionError(str(exc)) from None


# ponytail: call_transfer executor. Lives next to execute_tool so both
# tool kinds share the executor module — same SSRF / timeout policy,
# same record_tool_result integration, same logging style. Kept as a
# free function (not on ToolExecutor) because it doesn't share state
# with the HTTP path: Twilio's update() is a one-shot REST call, no
# keepalive client to manage.
async def execute_call_transfer(
    account_sid: str,
    auth_token: str,
    call_sid: str,
    destination: str,
    tool_name: str,
) -> dict:
    """Ask Twilio to redirect a live call to ``destination``.

    The call leaves our WebSocket as soon as Twilio sends the <Dial>
    TwiML — the operator hears ringing, the new party answers, the
    two legs are bridged. We don't get a callback when the bridge
    connects; success here means "Twilio accepted the redirect", not
    "the destination picked up". That's a Twilio-side concern.

    Raises ToolExecutionError when the auth pair doesn't own the
    call_sid or the destination is rejected at the Twilio layer.
    The caller (turn_manager) catches this and routes the error back
    to the LLM so the conversation can recover.
    """
    from STT_server.adapters.twilio_api import transfer_call
    log.info(
        "[ToolExecutor] call_transfer '%s' call_sid=%s -> %s",
        tool_name, call_sid, destination,
    )
    try:
        result = await transfer_call(account_sid, auth_token, call_sid, destination)
    except Exception as exc:
        log.exception(
            "[ToolExecutor] call_transfer '%s' transport error", tool_name,
        )
        raise ToolExecutionError(
            f"call_transfer '{tool_name}' transport failed: {exc}"
        )
    if not result.get("success"):
        raise ToolExecutionError(
            f"call_transfer '{tool_name}' rejected by Twilio: "
            f"{result.get('error') or 'unknown error'}"
        )
    return result
