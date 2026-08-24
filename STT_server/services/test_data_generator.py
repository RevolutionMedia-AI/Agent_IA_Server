"""LLM-driven test data generator for the tool Test button.

The Test button (POST /agents/{id}/tools/{toolId}/test or
/tools/{toolId}/test) asks the user's configured LLM to generate
realistic data matching the tool's parameters schema before
POSTing to the n8n webhook. Without this layer the BE used
"sample_<paramname>" placeholders that n8n often rejects as
unprocessable input — every Test button click was a 4xx.

The generator uses OpenAI's function-calling API with the tool's
own parameters schema as the function definition, so the LLM is
forced to return a JSON object that matches the schema. The model
is configured per-user via Settings (settings.test_data_model,
default gpt-4o-mini) — the operator picks the model in the FE
and the BE passes the choice through on every Test click.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("stt_server.test_data_generator")

# ponytail: default model when the user hasn't picked one in
# Settings. Cheapest in the OpenAI function-calling family and
# handles JSON Schema function definitions reliably. The BE
# Settings → API page lets the operator pick a different model
# (gpt-4o, gpt-4-turbo) for higher-quality test data on complex
# schemas; the value is persisted in settings.test_data_model and
# read here on every click.
DEFAULT_TEST_DATA_MODEL = "gpt-4o-mini"


class TestDataUnavailable(RuntimeError):
    """Raised when the generator can't reach the LLM (no key, quota,
    network, missing model). The caller surfaces this as a 4xx with
    a clear message instead of a generic 500."""


def _resolve_model(_model_override: str | None) -> str:
    """Pick the LLM the generator should use.

    Per-provider override (passed in from the route layer via
    test_data_model on the agent_tools row, set by the FE Connect
    modal) wins. Falls back to gpt-4o-mini for cost when unset.
    """
    m = (_model_override or "").strip()
    return m or DEFAULT_TEST_DATA_MODEL


def _resolve_openai_client(user_id: str | None):
    """Lazy-import OpenAI + resolve the user's key. Separated from
    the main flow so a missing openai package / unset key doesn't
    import-time-crash the whole app — we only pay the cost when an
    operator actually hits the Test button."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise TestDataUnavailable(
            "openai SDK not installed on the BE — cannot generate test data"
        ) from exc
    from STT_server.services.credentials_resolver import resolve_provider
    if not user_id:
        raise TestDataUnavailable("no authenticated user — cannot resolve API key")
    creds = resolve_provider(user_id, "openai")
    api_key = creds.get("api_key")
    if not api_key:
        raise TestDataUnavailable(
            "OpenAI API key not configured. Save your OpenAI key in "
            "Settings → API before running the Test button."
        )
    return OpenAI(api_key=api_key)


def generate_test_payload(tool: dict, user_id: str, model: str | None = None) -> dict[str, Any]:
    """Ask the configured LLM for realistic test data matching the
    tool's parameters schema. Returns the parsed args dict that
    the Test button POSTs to the n8n webhook.

    The LLM gets just the tool's name + description + parameters
    schema — no per-tool prompt needed. It uses the schema's
    field descriptions and required markers to decide what makes
    sense. Operators who want more specific context can extend
    the tool's description; we don't expose a separate prompt
    field any more (the previous `test_prompt` column stays in
    the schema for back-compat but is ignored by this generator).

    Raises TestDataUnavailable on missing config / SDK / quota /
    invalid model so the route layer can return a clear 4xx
    instead of a 500.
    """
    schema = tool.get("parameters") or {"type": "object", "properties": {}}
    tool_name = (tool.get("name") or "tool").strip()
    tool_description = (tool.get("description") or "").strip()

    model = _resolve_model(model)
    client = _resolve_openai_client(user_id)
    user_msg = (
        f"Generate realistic test data for the n8n webhook of the "
        f"\"{tool_name}\" tool.\n"
        f"Description: {tool_description or '(none)'}\n"
        f"Return a JSON object matching the parameters schema below. "
        f"Use plausible, real-world values for the specific domain "
        f"implied by the tool's name and description. Field "
        f"descriptions in the schema are the operator's own "
        f"hints — honour them.\n"
        f"Parameters schema:\n{json.dumps(schema, indent=2)}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            tools=[{
                "type": "function",
                "function": {
                    # ponytail: emit_test_data is a sentinel function
                    # name that exists only to force structured
                    # output. The LLM never actually "calls" it as a
                    # real tool — it just has to return arguments
                    # that match the parameters schema. Same trick
                    # the test_data generation pattern uses elsewhere
                    # (foreman, anthropic cookbook, etc.).
                    "name": "emit_test_data",
                    "description": "Return the generated test data exactly as specified.",
                    "parameters": schema,
                },
            }],
            tool_choice={
                "type": "function",
                "function": {"name": "emit_test_data"},
            },
        )
    except Exception as exc:
        # ponytail: wrap any LLM-side failure (rate limit, network,
        # bad key, invalid model) as TestDataUnavailable so the
        # route layer can return a clean 4xx with a useful message
        # instead of a 500. The full trace still goes to log.warning
        # for the deploy log.
        log.warning("[test_data_generator] LLM call failed: %s", exc)
        raise TestDataUnavailable(f"LLM call failed: {exc}") from exc

    if not response.choices or not response.choices[0].message.tool_calls:
        raise TestDataUnavailable(
            f"LLM ({model}) did not return a structured tool call. "
            "The model may be temporarily unavailable — try again."
        )
    args_str = response.choices[0].message.tool_calls[0].function.arguments
    try:
        return json.loads(args_str)
    except json.JSONDecodeError as exc:
        raise TestDataUnavailable(
            f"LLM returned malformed JSON for test data: {exc}"
        ) from exc