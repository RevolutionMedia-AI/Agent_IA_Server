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
    tool's parameters schema. Returns the parsed args dict that the
    Test button POSTs to the n8n webhook.

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
    return _call_structured(
        client, model, user_msg, schema,
        kind_label="test data for a tool",
    )


def generate_integration_test_payload(
    integration: dict,
    user_id: str,
    model: str | None = None,
    *,
    action: str | None = None,
) -> dict[str, Any]:
    """Same idea as ``generate_test_payload`` but for integrations.

    Integrations don't have a per-tool JSON Schema — they have a
    flat `configuration` dict (provider-specific, e.g. Google
    Calendar's ``calendar_id`` + ``timezone`` or Salesforce's
    ``instance_url`` + ``username`` + ``password``). The Test button
    feeds the LLM the catalog-declared configuration fields plus
    `provider` + `name`, and asks for realistic values the operator
    could preview against the provider's preflight.

    ponytail: 2026-09-04 — this is the parity hook the user asked
    for. Both Test buttons (tools + integrations) now drive the same
    generator, configured with the same per-user LLM, so the
    operator sees consistent "Test Connection" behaviour wherever
    they click it. The integration row's `connection_status` is
    still driven by the provider's preflight after generation —
    the LLM only fills the payload, it doesn't ping n8n.

    ponytail: 2026-09-04 (v2) — when the caller passes ``action`` we
    also seed the LLM with the action's ``parameters_schema`` so the
    preview reflects the shape the LLM will eventually hand to
    n8n. Google Calendar's ``create_appointment`` action is the
    motivating case: the LLM has to invent ``name``, ``email``,
    ``datetime``, ``duration_minutes``, ``title`` and
    ``description`` from nothing, so we ask for the same shape.
    """
    provider = (integration.get("provider") or "").strip() or "integration"
    name = (integration.get("name") or "").strip() or provider
    existing_cfg = (
        integration.get("configuration")
        if isinstance(integration.get("configuration"), dict)
        else {}
    )

    catalog = _load_catalog()
    spec = catalog.get(provider)

    # ── configuration fields (always) ──
    cfg_fields: list[dict] = []
    if spec is not None and getattr(spec, "configuration_fields", None):
        for f in spec.configuration_fields:
            entry = {
                "name": f.name,
                "type": getattr(f, "type", "string"),
                "required": getattr(f, "required", False),
                "description": getattr(f, "description", "") or "",
            }
            if f.name in existing_cfg:
                entry["example"] = existing_cfg[f.name]
            cfg_fields.append(entry)

    # ── action-level parameters (optional) ──
    action_props: list[dict] = []
    if action and spec is not None:
        for a in spec.actions:
            if a.id != action:
                continue
            for pname, pdef in (a.parameters_schema or {}).get("properties", {}).items():
                prop = {
                    "name": pname,
                    "type": pdef.get("type", "string"),
                    "required": pname in (a.parameters_schema or {}).get("required", []),
                    "description": pdef.get("description", ""),
                }
                if pdef.get("minimum") is not None:
                    prop["minimum"] = pdef["minimum"]
                if pdef.get("maximum") is not None:
                    prop["maximum"] = pdef["maximum"]
                action_props.append(prop)
            break

    # Build the response schema. The LLM returns one flat object; the
    # caller splits it by field group downstream.
    properties: dict[str, dict] = {}
    required: list[str] = []
    for f in cfg_fields:
        properties[f["name"]] = f
        if f.get("required"):
            required.append(f["name"])
    for p in action_props:
        properties[p["name"]] = p
        if p.get("required"):
            required.append(p["name"])
    if not properties:
        properties = {"value": {"type": "string"}}
        required = ["value"]

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }

    cfg_blob = (
        json.dumps(existing_cfg, indent=2) if existing_cfg else "(none yet)"
    )
    cfg_fields_blob = (
        json.dumps(cfg_fields, indent=2) if cfg_fields else "(none)"
    )
    if action_props:
        action_blob = (
            f"Action the operator is testing: ``{action}``\n"
            f"Action argument schema:\n{json.dumps(action_props, indent=2)}\n\n"
        )
    else:
        action_blob = ""

    user_msg = (
        f"Generate realistic test data for the \"{name}\" {provider} "
        "integration so the operator can preview what the agent will "
        "send during a real call.\n\n"
        f"Provider: {provider}\n"
        f"Integration name: {name}\n\n"
        f"Configuration fields the operator must fill in:\n{cfg_fields_blob}\n"
        f"Operator-supplied configuration so far:\n{cfg_blob}\n\n"
        f"{action_blob}"
        "Return a single JSON object with one key per declared field "
        "(configuration + action arguments). Reuse operator-supplied "
        "values when they exist; only invent blanks. Field descriptions "
        "are the operator's own hints — honour them. For Google "
        "Calendar's ``create_appointment`` action, fill the calendar "
        "event's ``summary`` and ``description`` based purely on what "
        "the operator would have said on a call. Do NOT invent dates or "
        "times — the LLM only fills these on a real call when the "
        "caller has agreed on a specific moment."
    )
    resolved_model = _resolve_model(model)
    client = _resolve_openai_client(user_id)
    return _call_structured(
        client, resolved_model, user_msg, schema,
        kind_label="integration test data",
    )


def _load_catalog() -> dict:
    """Return the integrations catalog as a ``{provider: spec}`` dict.

    Lazy import so the rest of the BE doesn't pay the cost when it
    only exercises the tool Test button.
    """
    try:
        from STT_server.services.integrations_catalog import INTEGRATION_PROVIDERS
        return {spec.id: spec for spec in INTEGRATION_PROVIDERS}
    except Exception:
        return {}


def _call_structured(
    client, model: str, user_msg: str, schema: dict,
    *,
    kind_label: str,
) -> dict[str, Any]:
    """Single-use helper that calls the LLM with a sentinel function
    so the response comes back as a JSON object matching ``schema``.

    Same trick as the original ``generate_test_payload``: the
    ``emit_test_data`` tool never actually runs, it just forces the
    LLM to return structured arguments. Centralised here so the
    integrations path stays a one-line wrapper.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_msg}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "emit_test_data",
                    "description": (
                        f"Return the generated {kind_label} exactly as "
                        "specified in the schema."
                    ),
                    "parameters": schema,
                },
            }],
            tool_choice={
                "type": "function",
                "function": {"name": "emit_test_data"},
            },
        )
    except Exception as exc:
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
            f"LLM returned malformed JSON for {kind_label}: {exc}"
        ) from exc