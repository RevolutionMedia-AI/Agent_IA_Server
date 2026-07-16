"""Tool executor service for calling n8n webhooks internally during conversations."""

import asyncio
import httpx
from typing import Optional
import logging

log = logging.getLogger(__name__)


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
