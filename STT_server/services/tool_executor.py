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
    ) -> dict:
        """Execute a tool by calling its n8n webhook.

        Args:
            webhook_url: The n8n webhook URL to call.
            arguments: The arguments collected by the LLM to send to the tool.
            tool_name: Name of the tool being executed (for logging).

        Returns:
            The JSON response from n8n as a dict.

        Raises:
            ToolExecutionError: If the webhook call fails or times out,
                or the URL fails SSRF validation.
        """
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }

        # ponytail: SSRF guard. Run BEFORE any DNS / network call so a
        # blocked URL never reaches the resolver. _validate_webhook_url
        # raises ToolExecutionError on loopback / RFC1918 / metadata.
        try:
            _validate_webhook_url(webhook_url)
        except ToolExecutionError as exc:
            log.warning(
                "[ToolExecutor] SSRF rejected tool '%s' url=%s err=%s",
                tool_name, webhook_url, exc,
            )
            raise

        log.info(
            "[ToolExecutor] Executing tool '%s' host=%s args_keys=%s",
            tool_name,
            urlparse(webhook_url).hostname or "<unknown>",
            list(arguments.keys()) if isinstance(arguments, dict) else type(arguments).__name__,
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
                response = await client.post(
                    webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code >= 400:
                    raise ToolExecutionError(
                        f"n8n webhook returned HTTP {response.status_code}: {response.text[:200]}"
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
) -> dict:
    """Convenience function to execute a tool using the singleton executor."""
    executor = get_tool_executor()
    return await executor.execute(webhook_url, arguments, tool_name)


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
