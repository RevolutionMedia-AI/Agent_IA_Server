"""Helpers for persisting tool/integration instructions inside agents.prompt.

ponytail: design notes
=====================
The agent's `prompt` column is now the single source of truth for "what
the LLM needs to know about each tool/integration". Before this module
the runtime injected `IntegrationProviderSpec.prompt_snippet` on every
call start (see `STT_server/STT_Server.py` legacy block around
`_build_instructions`). That path is gone — the snippet now lives
physically inside `agents.prompt`, marked with stable delimiters so we
can patch / replace / remove a section without disturbing the operator's
free-form copy.

Delimiters
----------
Two kinds, namespaced so an integration and a tool can never collide:

    <!-- AGENT_TOOL:<tool_id> -->
    ... body ...
    <!-- END_AGENT_TOOL:<tool_id> -->

    <!-- INTEGRATION:<integration_id> -->
    ... body ...
    <!-- END_INTEGRATION:<integration_id> -->

The end-tag prefix is `END_<KIND>` to make `<!-- END_*` a visually
distinct block when the operator scrolls the raw text. The regex
matcher always pairs a begin-tag with its end-tag by entity_id, so two
sections with the same id but different kinds can never overlap.

Reconciler
----------
`reconcile_agent_prompt(agent_id, user_id, current_prompt)` iterates the
agent's currently-assigned tools + integrations, regenerates each
section, and patches the prompt in-place. It returns
`(new_prompt, change_log)` so callers can surface a human-readable log
to the FE ("Google Calendar section regenerated", "Transfer to Sales
section removed because the tool was unassigned", etc.). The
reconciler is the ONLY authority that decides which sections are
required; the operator can edit the prompt freely otherwise — manual
copy outside the delimiters is preserved verbatim.

Round-trip contract
-------------------
The reconciler treats `current_prompt` as the input — it never refuses
to merge. Sections that the operator deleted by hand are re-added on
the next save. Sections that are no longer assigned (orphan blocks
left over from an old assign-then-unassign) are cleaned up too, but
only when the reconciler runs after an explicit assignment change. The
PUT /agents/{id} endpoint always reconciles; assign/unassign only
patch the single section they touched (faster path, no full sweep).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable, Optional

log = logging.getLogger("stt_server.services.agent_prompt_tools")


# ── Section kind + delimiters ────────────────────────────────────────────

KIND_AGENT_TOOL = "AGENT_TOOL"
KIND_INTEGRATION = "INTEGRATION"
VALID_KINDS = frozenset({KIND_AGENT_TOOL, KIND_INTEGRATION})

_BEGIN_FMT = "<!-- {kind}:{entity_id} -->"
_END_FMT = "<!-- END_{kind}:{entity_id} -->"


def begin_tag(kind: str, entity_id: str) -> str:
    return _BEGIN_FMT.format(kind=kind, entity_id=entity_id)


def end_tag(kind: str, entity_id: str) -> str:
    return _END_FMT.format(kind=kind, entity_id=entity_id)


# ── Section find / replace / remove ──────────────────────────────────────


def _section_pattern(kind: str, entity_id: str) -> re.Pattern[str]:
    """Match a single complete section, including its begin/end tags.

    Captures the body in group(1) so callers can read or replace just
    the inside. The body match is non-greedy and ends at the matching
    end-tag, never at a later begin-tag for the same id (which can only
    happen on a malformed prompt — we trust the input).
    """
    return re.compile(
        re.escape(begin_tag(kind, entity_id))
        + r"\s*(.*?)\s*"
        + re.escape(end_tag(kind, entity_id)),
        re.DOTALL,
    )


def add_or_update_section(prompt: str, kind: str, entity_id: str, body: str) -> str:
    """Insert or replace the section for `(kind, entity_id)`.

    If the section already exists, replaces its body in place. If not,
    appends a fresh section at the end of the prompt (separated by a
    blank line so it sits cleanly after the operator's free-form copy).

    `body` is the multi-line block to place between the begin/end tags.
    Callers should NOT include the begin/end tags themselves — those are
    added here so the on-disk format stays consistent regardless of who
    calls the helper.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown section kind: {kind!r}")
    prompt = prompt or ""
    pattern = _section_pattern(kind, entity_id)
    block = f"{begin_tag(kind, entity_id)}\n{body.rstrip()}\n{end_tag(kind, entity_id)}"
    if pattern.search(prompt):
        return pattern.sub(lambda m: block, prompt, count=1)
    # Append at end. Two blank lines between the operator's copy and
    # the section keeps the System Prompt scannable when the operator
    # inspects it in Edit Agent.
    if prompt.strip() == "":
        return block
    return prompt.rstrip() + "\n\n" + block + "\n"


def remove_section(prompt: str, kind: str, entity_id: str) -> str:
    """Strip the section for `(kind, entity_id)` from the prompt.

    Idempotent — if the section isn't present, the prompt is returned
    unchanged. Whitespace around the removed section is collapsed so we
    don't leave dangling blank lines.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"Unknown section kind: {kind!r}")
    prompt = prompt or ""
    pattern = _section_pattern(kind, entity_id)
    if not pattern.search(prompt):
        return prompt
    # Strip the section, then collapse trailing blank lines.
    cleaned = pattern.sub("", prompt, count=1)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.rstrip() + "\n" if cleaned.strip() else ""


def list_sections(prompt: str) -> list[tuple[str, str]]:
    """Enumerate every section in the prompt as `(kind, entity_id)`.

    Used by the reconciler to detect orphan sections (blocks for tools
    the agent no longer has assigned). Returned in document order so
    the change_log reads in the same order as the prompt.
    """
    if not prompt:
        return []
    out: list[tuple[str, str]] = []
    for m in re.finditer(
        r"<!-- (AGENT_TOOL|INTEGRATION):([^\s]+) -->",
        prompt,
    ):
        out.append((m.group(1), m.group(2)))
    return out


# ── Section builders ─────────────────────────────────────────────────────


def _bilingual(when_en: str, when_es: str) -> str:
    """Render the two-language block. `when_*` strings come from the
    caller (the catalog for integrations, the tool row for tools)."""
    return (
        f"ENGLISH:\n{when_en.strip()}\n\n"
        f"ESPAÑOL:\n{when_es.strip()}"
    )


def _neutral_when_en(when_en: str, when_es: str) -> tuple[str, str]:
    """Fallback when only one language is provided.

    The bilingual block uses the same string for both languages with a
    per-language wrapper sentence ("Use this tool according to its
    purpose: ..." / "Usa esta herramienta de acuerdo con su
    propósito: ..."). This avoids a silent translation and keeps the
    output deterministic — the operator can edit either side directly.
    """
    return when_en.strip(), when_es.strip()


def _format_schema_example(schema: dict) -> str:
    """Render the JSON-Schema's `properties` as an inline example object.

    We do NOT include descriptions in the example (would inflate the
    block); the property name is what the LLM needs to send back. The
    schema's `required` list is rendered separately in the
    "Required / Optional" bullets below.
    """
    properties = (schema or {}).get("properties") or {}
    if not isinstance(properties, dict) or not properties:
        return "{ }"
    lines = ["{"]
    items = list(properties.items())
    for i, (name, spec) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        typ = (spec or {}).get("type", "string")
        placeholder = {
            "string": f'"{name.replace("_", " ")}"',
            "integer": "0",
            "number": "0",
            "boolean": "true",
        }.get(typ, '"value"')
        lines.append(f'  "{name}": {placeholder}{comma}')
    lines.append("}")
    return "\n".join(lines)


def _required_optional(schema: dict) -> tuple[list[str], list[str]]:
    """Return (required_props, optional_props) for the schema."""
    properties = (schema or {}).get("properties") or {}
    if not isinstance(properties, dict):
        return [], []
    required_set = set((schema or {}).get("required") or [])
    required = [p for p in properties if p in required_set]
    optional = [p for p in properties if p not in required_set]
    return required, optional


def build_agent_tool_section(
    tool_id: str,
    name: str,
    description: str,
    parameters_schema: dict,
    *,
    when_to_use_en: Optional[str] = None,
    when_to_use_es: Optional[str] = None,
) -> str:
    """Render the bilingual prompt block for one agent_tool.

    ponytail: the operator-facing `name` (e.g. "Google Calendar
    Schedule") is rendered as the section header. The LLM-facing
    `function_name` (OpenAI-safe id) is NOT included — the FE keeps
    those separated, and the LLM learns the function name from the
    OpenAI `tools[]` payload, not from the prompt.
    """
    schema = parameters_schema or {"type": "object", "properties": {}}
    required, optional = _required_optional(schema)

    # when_to_use_* defaults: bilingual wrappers around the tool's
    # description. Operators who want richer copy can pass explicit
    # when_to_use_en/es — those win.
    if when_to_use_en is None or when_to_use_es is None:
        wrap_en = when_to_use_en or (
            f"Use this tool according to its purpose: {description.strip()}"
            if description else "Use this tool when its description applies."
        )
        wrap_es = when_to_use_es or (
            f"Usa esta herramienta de acuerdo con su propósito: {description.strip()}"
            if description else "Usa esta herramienta cuando su descripción aplique."
        )
        when_en, when_es = _neutral_when_en(wrap_en, wrap_es)
    else:
        when_en, when_es = when_to_use_en.strip(), when_to_use_es.strip()

    example = _format_schema_example(schema)
    en_required = ", ".join(required) if required else "(none)"
    en_optional = ", ".join(optional) if optional else "(none)"
    es_required = ", ".join(required) if required else "(ninguno)"
    es_optional = ", ".join(optional) if optional else "(ninguno)"

    parts: list[str] = []
    parts.append(f"## Tool: {name}")
    parts.append("")
    parts.append("ENGLISH:")
    parts.append(when_en)
    parts.append("")
    parts.append("When calling this tool, send the following JSON data:")
    parts.append("")
    parts.append("```json")
    parts.append(example)
    parts.append("```")
    parts.append("")
    parts.append(f"Required: {en_required}")
    parts.append(f"Optional: {en_optional}")
    parts.append("")
    parts.append("Do not rename the JSON properties.")
    parts.append("")
    parts.append("ESPAÑOL:")
    parts.append(when_es)
    parts.append("")
    parts.append("Al utilizar la herramienta, envía los siguientes datos en formato JSON:")
    parts.append("")
    parts.append("```json")
    parts.append(example)
    parts.append("```")
    parts.append("")
    parts.append(f"Obligatorio: {es_required}")
    parts.append(f"Opcional: {es_optional}")
    parts.append("")
    parts.append("No cambies los nombres de las propiedades JSON.")
    return "\n".join(parts)


def build_integration_section(
    integration_id: str,
    provider_name: str,
    actions: Iterable[dict],
) -> str:
    """Render the bilingual prompt block for one integration.

    `actions` is an iterable of dicts with shape::

        {
            "id":              "create_event",     # required
            "name":            "Create Event",     # required
            "description":     "...",              # optional
            "when_to_use_en":  "...",              # optional, falls back to wrap
            "when_to_use_es":  "...",              # optional, falls back to wrap
            "parameters_schema": { ... },          # optional
        }

    Rendered as `## Integration: <provider_name>` followed by one
    `### Action:` sub-section per action, so a multi-action integration
    stays readable as the catalog grows.
    """
    parts: list[str] = [f"## Integration: {provider_name}", ""]

    # Edge case: integration with zero actions (rare but legal for the
    # `generic_webhook` template where the operator picks the action
    # per tool). Render an explicit "no actions configured" note in
    # both languages so the LLM knows the integration is wired but
    # currently has no callable verbs.
    actions_list = list(actions)
    if not actions_list:
        parts.append("ENGLISH:")
        parts.append("This integration has no actions configured yet.")
        parts.append("")
        parts.append("ESPAÑOL:")
        parts.append("Esta integración aún no tiene acciones configuradas.")
        return "\n".join(parts)

    for action in actions_list:
        act_id = action.get("id") or action.get("name") or "action"
        act_name = action.get("name") or act_id
        act_desc = (action.get("description") or "").strip()
        schema = action.get("parameters_schema") or {"type": "object", "properties": {}}

        when_en = (action.get("when_to_use_en") or "").strip()
        when_es = (action.get("when_to_use_es") or "").strip()
        if not when_en:
            when_en = (
                f"Use {act_name} when: {act_desc}"
                if act_desc else f"Use {act_name} according to its purpose."
            )
        if not when_es:
            when_es = (
                f"Usa {act_name} cuando: {act_desc}"
                if act_desc else f"Usa {act_name} de acuerdo con su propósito."
            )

        required, optional = _required_optional(schema)
        example = _format_schema_example(schema)
        en_required = ", ".join(required) if required else "(none)"
        en_optional = ", ".join(optional) if optional else "(none)"
        es_required = ", ".join(required) if required else "(ninguno)"
        es_optional = ", ".join(optional) if optional else "(ninguno)"

        parts.append(f"### Action: {act_name}")
        parts.append("")
        parts.append("ENGLISH:")
        parts.append(when_en)
        parts.append("")
        parts.append("When calling this action, send the following JSON data:")
        parts.append("")
        parts.append("```json")
        parts.append(example)
        parts.append("```")
        parts.append("")
        parts.append(f"Required: {en_required}")
        parts.append(f"Optional: {en_optional}")
        parts.append("")
        parts.append("Do not rename the JSON properties.")
        parts.append("")
        parts.append("ESPAÑOL:")
        parts.append(when_es)
        parts.append("")
        parts.append("Al utilizar esta acción, envía los siguientes datos en formato JSON:")
        parts.append("")
        parts.append("```json")
        parts.append(example)
        parts.append("```")
        parts.append("")
        parts.append(f"Obligatorio: {es_required}")
        parts.append(f"Opcional: {es_optional}")
        parts.append("")
        parts.append("No cambies los nombres de las propiedades JSON.")
        parts.append("")

    # Trailing blank so the last action's text doesn't run into whatever
    # follows in the prompt.
    return "\n".join(parts).rstrip() + "\n"


# ── High-level helpers used by routes/api.py ────────────────────────────


def patch_agent_tool_in_prompt(prompt: str, tool_row: dict) -> str:
    """Single-tool patch — used by assign/unassign/update_agent_tool.

    `tool_row` is the canonical row from db_list_tools / db_get_tool
    (must include `id`, `name`, `description`, `parameters`,
    `kind`). We skip call_transfer tools because they don't take
    parameters from the LLM — the destination is operator-set.
    """
    if tool_row.get("kind") == "call_transfer":
        return prompt
    body = build_agent_tool_section(
        tool_id=tool_row["id"],
        name=tool_row.get("name") or tool_row["id"],
        description=tool_row.get("description") or "",
        parameters_schema=tool_row.get("parameters") or {},
    )
    return add_or_update_section(prompt, KIND_AGENT_TOOL, tool_row["id"], body)


def patch_integration_in_prompt(prompt: str, integration_id: str, provider_name: str, actions: list[dict]) -> str:
    """Single-integration patch — used by assign/unassign integration."""
    body = build_integration_section(integration_id, provider_name, actions)
    return add_or_update_section(prompt, KIND_INTEGRATION, integration_id, body)


# ── Full reconciler ───────────────────────────────────────────────────────


def _section_diff(
    old_prompt: str,
    new_prompt: str,
    missing_sections: list[tuple[str, str]],
    stale_sections: list[tuple[str, str]],
    regenerated_sections: list[tuple[str, str, str]],
) -> list[str]:
    """Compose a human-readable change_log list.

    `missing_sections` = (kind, id) tuples that should be present but
    are not. `stale_sections` = (kind, id) tuples that ARE present but
    no longer correspond to an assignment. `regenerated_sections` =
    (kind, id, op) tuples where op is "added" or "updated" depending on
    whether the section was already in the prompt.
    """
    log_lines: list[str] = []
    missing_set = set(missing_sections)
    for entry in regenerated_sections:
        # ponytail: regenerated_sections carries 3-tuples
        # (kind, id, op) so we don't have to re-derive "added" vs
        # "updated" from the diff.
        kind, eid, op = entry
        label = _section_label(kind, eid)
        log_lines.append(f"{label} section {op}")
    for kind, eid in stale_sections:
        label = _section_label(kind, eid)
        log_lines.append(f"{label} section removed (no longer assigned)")
    if old_prompt.strip() == "" and new_prompt.strip():
        log_lines.append("System prompt now contains tool instructions")
    return log_lines


def _section_label(kind: str, entity_id: str) -> str:
    return f"{kind}:{entity_id}"


def reconcile_agent_prompt(
    agent_id: str,
    user_id: str,
    current_prompt: str,
    *,
    # Optional injected dependencies so the helper stays decoupled from
    # db_tools / db_integrations / integrations_catalog (which would
    # create a circular import when those modules import the catalog).
    list_agent_tools_fn=None,
    list_agent_integrations_fn=None,
    get_integration_provider_spec_fn=None,
    integration_label_fn=None,
) -> tuple[str, list[str]]:
    """Reconcile an agent's prompt against its current assignments.

    Returns `(new_prompt, change_log)`. Run from PUT /agents/{id} and
    from any place that wants to assert "every assigned tool/integration
    has a section". Idempotent — running twice on the same prompt yields
    `change_log == []`.

    `list_agent_tools_fn(agent_id, user_id) -> list[dict]` returns the
    per-agent view (already filtered to owned + assigned shared rows,
    with credential/provider rows excluded — same filter the FE uses).
    `list_agent_integrations_fn(agent_id, user_id) -> list[dict]`
    returns every integration the agent can call (private to it +
    shared whose `assignments` contains the agent id).
    `get_integration_provider_spec_fn(provider_id) -> IntegrationProviderSpec | None`
    provides the action catalog used to render sections.
    `integration_label_fn(integration_row) -> str` returns the
    provider-name string used as the section header (defaults to
    `integration_row["name"]`).
    """
    if list_agent_tools_fn is None or list_agent_integrations_fn is None:
        # Lazy default — import inside the function to avoid the
        # top-of-module circular dependency the routes layer would
        # otherwise hit.
        from STT_server.db_tools import list_tools as _list_tools
        from STT_server.db_integrations import list_integrations as _list_integ

        def list_agent_tools_fn(_agent_id, _user_id):
            return [t for t in _list_tools(_user_id, agent_id=_agent_id) or []]

        def list_agent_integrations_fn(_agent_id, _user_id):
            return _list_integ(_user_id, agent_id=_agent_id) or []

    if get_integration_provider_spec_fn is None:
        from STT_server.services.integrations_catalog import (
            get_integration_provider_spec,
        )
        get_integration_provider_spec_fn = get_integration_provider_spec

    if integration_label_fn is None:
        def integration_label_fn(row: dict) -> str:
            return row.get("name") or row.get("id")

    new_prompt = current_prompt or ""
    change_log: list[str] = []

    # Collect the assignments the agent SHOULD have a section for.
    expected_sections: dict[tuple[str, str], dict] = {}

    for tool in list_agent_tools_fn(agent_id, user_id) or []:
        if tool.get("kind") == "call_transfer":
            continue
        if tool.get("credentials") and not (tool.get("webhook_url") or tool.get("destination")):
            # Provider-credential rows live in agent_tools with the
            # same table shape; skip them — they aren't callable.
            continue
        expected_sections[(KIND_AGENT_TOOL, tool["id"])] = {
            "kind": KIND_AGENT_TOOL,
            "tool": tool,
        }

    for integ in list_agent_integrations_fn(agent_id, user_id) or []:
        spec = get_integration_provider_spec_fn(integ.get("provider"))
        if not spec:
            continue
        # ponytail: accept both dataclass specs (production — the
        # catalog returns ActionSpec) and dict specs (tests can
        # inject a simpler stub). `_get_attr` normalizes both.
        def _get_attr(obj, name, default=""):
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        actions = [
            {
                "id": _get_attr(a, "id"),
                "name": _get_attr(a, "name"),
                "description": _get_attr(a, "description"),
                # Catalog is the source of truth for the bilingual
                # copy; the integration row never overrides it.
                "when_to_use_en": _get_attr(a, "when_to_use_en", "") or "",
                "when_to_use_es": _get_attr(a, "when_to_use_es", "") or "",
                "parameters_schema": _get_attr(
                    a, "parameters_schema",
                    {"type": "object", "properties": {}, "required": []},
                ),
            }
            for a in spec.actions
        ]
        expected_sections[(KIND_INTEGRATION, integ["id"])] = {
            "kind": KIND_INTEGRATION,
            "integration": integ,
            "actions": actions,
            "label": integration_label_fn(integ),
        }

    # Phase 1 — patch every expected section. Track which ones were
    # added (newly present) vs updated (already present).
    existing = set(list_sections(new_prompt))
    regenerated: list[tuple[str, str, str]] = []  # (kind, id, op)
    for (kind, eid), payload in expected_sections.items():
        if kind == KIND_AGENT_TOOL:
            body = build_agent_tool_section(
                tool_id=eid,
                name=payload["tool"].get("name") or eid,
                description=payload["tool"].get("description") or "",
                parameters_schema=payload["tool"].get("parameters") or {},
            )
        else:
            body = build_integration_section(
                eid,
                payload["label"],
                payload["actions"],
            )
        was_present = (kind, eid) in existing
        new_prompt = add_or_update_section(new_prompt, kind, eid, body)
        regenerated.append((kind, eid, "added" if not was_present else "updated"))

    # Phase 2 — clean up orphan sections (blocks for things no longer
    # assigned). Same listing logic as before; anything we didn't
    # regenerate is stale.
    new_existing = set(list_sections(new_prompt))
    stale = sorted(
        (kind, eid)
        for (kind, eid) in new_existing
        if (kind, eid) not in expected_sections
    )
    for kind, eid in stale:
        new_prompt = remove_section(new_prompt, kind, eid)

    # Compose the change_log. We only report changes that actually
    # moved the prompt — a no-op reconcile (prompt already matches
    # every assignment) yields `change_log == []`. This is the
    # contract the FE relies on to decide whether to surface a toast.
    if (regenerated or stale) and new_prompt != (current_prompt or ""):
        change_log = _section_diff(
            current_prompt or "",
            new_prompt,
            missing_sections=[
                (k, e) for (k, e), _ in expected_sections.items()
                if (k, e) not in existing
            ],
            stale_sections=stale,
            regenerated_sections=regenerated,
        )
        log.info(
            "[agent_prompt_tools] reconciled agent=%s user=%s changes=%s",
            agent_id, user_id, change_log,
        )

    return new_prompt, change_log


__all__ = [
    "KIND_AGENT_TOOL",
    "KIND_INTEGRATION",
    "VALID_KINDS",
    "begin_tag",
    "end_tag",
    "add_or_update_section",
    "remove_section",
    "list_sections",
    "build_agent_tool_section",
    "build_integration_section",
    "patch_agent_tool_in_prompt",
    "patch_integration_in_prompt",
    "reconcile_agent_prompt",
]
