"""Tool executor service for calling n8n webhooks internally during conversations."""

import asyncio
import json
import os
import httpx
from datetime import datetime, timezone
from typing import Optional
import logging

log = logging.getLogger(__name__)


# ponytail: per-tool observability. Same storage as api.py's
# _load_tools / _save_tools (STT_server/data/agent_tools.json) —
# computed from __file__ so the executor module doesn't need to
# import the routes module (which would create a circular import).
_TOOLS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "agent_tools.json",
)


def record_tool_result(tool_id: str, ok: bool, kind: str) -> None:
    """Update observability fields on a tool row after a run.

    `kind` is either "test" (operator clicked Test webhook) or
    "invocation" (the agent called the tool during a real call).
    Best-effort: any I/O or parse error is swallowed so a write
    failure never breaks the tool call. The caller treats this as
    fire-and-forget.
    """
    try:
        if not tool_id:
            return
        try:
            with open(_TOOLS_FILE, "r", encoding="utf-8") as f:
                tools = json.load(f) or []
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, IOError, OSError) as exc:
            log.warning("[ToolExecutor] stats load failed (%s): %s", type(exc).__name__, exc)
            return
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        status = "ok" if ok else "fail"
        updated = False
        for t in tools:
            if not isinstance(t, dict) or t.get("id") != tool_id:
                continue
            if kind == "test":
                t["last_tested_at"] = now
                t["last_test_result"] = status
            else:
                t["last_invoked_at"] = now
                t["last_invocation_status"] = status
                t["invocation_count"] = int(t.get("invocation_count", 0) or 0) + 1
            updated = True
            break
        if not updated:
            return
        with open(_TOOLS_FILE, "w", encoding="utf-8") as f:
            json.dump(tools, f, indent=2, ensure_ascii=False)
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
            ToolExecutionError: If the webhook call fails or times out.
        """
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }

        log.info(
            "[ToolExecutor] Executing tool '%s' -> %s with args: %s",
            tool_name,
            webhook_url,
            arguments,
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
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
                    "[ToolExecutor] Tool '%s' completed successfully: %s",
                    tool_name,
                    str(result)[:200],
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
