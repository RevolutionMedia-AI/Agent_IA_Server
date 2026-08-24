"""LLM-driven test data generator for the tool Test button.

When the operator has curated a `test_prompt` on a tool, the
Test button (POST /agents/{id}/tools/{toolId}/test or
/tools/{toolId}/test) asks the user's configured LLM to generate
realistic data matching the tool's parameters schema before
POSTing to the n8n webhook. Without this layer the BE used
"sample_<paramname>" placeholders that n8n often rejects as
unprocessable input — every Test button click was a 4xx.

The generator uses OpenAI's function-calling API with the tool's
own parameters schema as the function definition, so the LLM is
forced to return a JSON object that matches the schema. We pick
the cheapest model that supports function calling (gpt-4o-mini)
and resolve the user's OpenAI key via the existing
credentials_resolver — no new env vars, no new billing.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("stt_server.test_data_generator")

# ponytail: gpt-4o-mini is the cheapest model in the OpenAI
# function-calling family and handles JSON Schema function
# definitions reliably. Switching to the user's preferred model
# would need a small model-resolution layer that we don't have yet
# — for now hard-code it and add a follow-up if the operator asks
# for Anthropic / Gemini support for the test generator.
_OPENAI_MODEL = "gpt-4o-mini"


class TestDataUnavailable(RuntimeError):
    """Raised when the generator can't reach the LLM (no key, quota,
    network). The caller surfaces this as a 4xx with a clear message
    instead of the generic 500 the previous implementation emitted
    when the JSON-file fallback path broke on Railway."""


def _resolve_openai_client(user_id: str | None):
    """Lazy-import OpenAI + resolve the user's key. Separated from
    the main flow so a missing openai package / unset key doesn't
    import-time-crash the whole app — we only pay the cost when an
    operator actually hits the Test button with a test_prompt."""
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
            "Settings → API (or leave test_prompt blank to use the old "
            "placeholder behavior)."
        )
    return OpenAI(api_key=api_key)


def generate_test_payload(tool: dict, user_id: str) -> dict[str, Any]:
    """Ask the configured LLM for realistic test data matching the
    tool's parameters schema. Returns the parsed args dict that
    the Test button POSTs to the n8n webhook.

    Raises TestDataUnavailable on missing config / SDK / quota so
    the route layer can return a clear 4xx instead of a 500.

    Edge case: a tool with no `properties` (or `parameters` missing
    entirely) still gets an empty JSON object back. n8n's Webhook
    node accepts an empty body as long as the workflow downstream
    doesn't require fields.
    """
    schema = tool.get("parameters") or {"type": "object", "properties": {}}
    tool_name = tool.get("name", "tool")
    tool_description = tool.get("description", "")
    test_prompt = (tool.get("test_prompt") or "").strip()
    if not test_prompt:
        # ponytail: caller should not invoke us with an empty
        # test_prompt — it falls back to the legacy placeholder
        # branch instead. This guard is just defensive in case a
        # future caller forgets the check.
        return {}

    client = _resolve_openai_client(user_id)
    user_msg = (
        f"Generate realistic test data for the n8n webhook of the "
        f"\"{tool_name}\" tool.\n"
        f"Description: {tool_description or '(none)'}\n"
        f"Context from the operator: {test_prompt}\n"
        f"Return a JSON object matching the parameters schema below. "
        f"Use plausible, real-world values for the specific domain "
        f"described in the context.\n"
        f"Parameters schema:\n{json.dumps(schema, indent=2)}"
    )
    try:
        response = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "user", "content": user_msg}],
            tools=[{
                "type": "function",
                "function": {
                    # ponytail: emit_test_data is a sentinel function
                    # name that exists only to force structured
                    # output. The LLM never actually "calls" it as a
                    # real tool — it just has to return arguments that
                    # match the parameters schema. We use
                    # tool_choice=function to make sure no other tool
                    # call slips in. Same trick the test_data
                    # generation pattern uses elsewhere (foreman,
                    # anthropic cookbook, etc.).
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
        # bad key) as TestDataUnavailable so the route layer can
        # return a clean 4xx with a useful message instead of a 500.
        log.warning("[test_data_generator] LLM call failed: %s", exc)
        raise TestDataUnavailable(f"LLM call failed: {exc}") from exc

    if not response.choices or not response.choices[0].message.tool_calls:
        raise TestDataUnavailable(
            "LLM did not return a structured tool call. The model may "
            "be temporarily unavailable — try again."
        )
    args_str = response.choices[0].message.tool_calls[0].function.arguments
    try:
        return json.loads(args_str)
    except json.JSONDecodeError as exc:
        raise TestDataUnavailable(
            f"LLM returned malformed JSON for test data: {exc}"
        ) from exc