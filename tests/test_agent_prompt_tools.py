"""Round-trip tests for STT_server.services.agent_prompt_tools.

Locking down the prompt-section contract:
  * delimiters don't collide between AGENT_TOOL and INTEGRATION ids
  * add_or_update_section / remove_section are idempotent
  * build_agent_tool_section produces the bilingual block the brief
    shows for Google Calendar (EN+ES, JSON example, required/optional)
  * build_integration_section renders every action under `### Action:`
  * reconcile_agent_prompt regenerates missing sections and removes
    orphan ones, with a non-empty change_log only when something moved
"""
from __future__ import annotations

import pytest

from STT_server.services.agent_prompt_tools import (
    KIND_AGENT_TOOL,
    KIND_INTEGRATION,
    add_or_update_section,
    begin_tag,
    build_agent_tool_section,
    build_integration_section,
    end_tag,
    list_sections,
    reconcile_agent_prompt,
    remove_section,
)


# ── Delimiter / section helpers ──────────────────────────────────────────


def test_add_or_update_section_appends_when_missing():
    """New tool gets a fresh section at the end, separated by blank lines."""
    out = add_or_update_section(
        "You are a receptionist.", KIND_AGENT_TOOL, "tool_1",
        "## Tool: Find Customer\nUse when ...",
    )
    assert "You are a receptionist." in out
    assert begin_tag(KIND_AGENT_TOOL, "tool_1") in out
    assert end_tag(KIND_AGENT_TOOL, "tool_1") in out
    # Two blank lines separate the operator's free-form copy from the
    # first injected section (readability rule).
    assert "\n\n<!-- AGENT_TOOL:tool_1 -->" in out


def test_add_or_update_section_replaces_in_place():
    """Editing an existing tool updates its body without duplicating."""
    body_v1 = "## Tool: Old Body\nv1"
    body_v2 = "## Tool: New Body\nv2"
    out = add_or_update_section(
        "intro", KIND_AGENT_TOOL, "tool_1", body_v1,
    )
    out2 = add_or_update_section(out, KIND_AGENT_TOOL, "tool_1", body_v2)
    # No duplicates — the section should appear exactly once.
    assert out2.count(begin_tag(KIND_AGENT_TOOL, "tool_1")) == 1
    assert out2.count(end_tag(KIND_AGENT_TOOL, "tool_1")) == 1
    assert "New Body" in out2
    assert "Old Body" not in out2
    # The operator's intro copy is preserved verbatim.
    assert out2.startswith("intro")


def test_remove_section_is_idempotent():
    """remove_section on a missing section is a no-op."""
    assert remove_section("plain prompt", KIND_AGENT_TOOL, "absent") == "plain prompt"
    section = add_or_update_section(
        "p", KIND_AGENT_TOOL, "tool_1", "body",
    )
    cleaned = remove_section(section, KIND_AGENT_TOOL, "tool_1")
    assert KIND_AGENT_TOOL not in cleaned
    assert "p" in cleaned
    # Second call is a no-op too.
    assert remove_section(cleaned, KIND_AGENT_TOOL, "tool_1") == cleaned


def test_sections_with_same_id_but_different_kinds_dont_collide():
    """Two sections with the same id but different kinds coexist."""
    a = add_or_update_section("", KIND_AGENT_TOOL, "shared-id", "tool body")
    b = add_or_update_section(a, KIND_INTEGRATION, "shared-id", "integration body")
    assert b.count(begin_tag(KIND_AGENT_TOOL, "shared-id")) == 1
    assert b.count(begin_tag(KIND_INTEGRATION, "shared-id")) == 1
    assert "tool body" in b
    assert "integration body" in b
    # list_sections reports both, in document order.
    sections = list_sections(b)
    assert (KIND_AGENT_TOOL, "shared-id") in sections
    assert (KIND_INTEGRATION, "shared-id") in sections


# ── Section builders ──────────────────────────────────────────────────────


def test_build_agent_tool_section_renders_bilingual_block():
    """The section the brief asks for: EN+ES, JSON example, required/optional."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"},
            "datetime": {"type": "string"},
            "duration_minutes": {"type": "integer"},
            "host_email": {"type": "string"},
        },
        "required": ["datetime"],
    }
    body = build_agent_tool_section(
        tool_id="google-calendar",
        name="Google Calendar",
        description="Schedule a calendar appointment",
        parameters_schema=schema,
    )
    assert "## Tool: Google Calendar" in body
    assert "ENGLISH:" in body
    assert "ESPAÑOL:" in body
    assert "Required: datetime" in body
    assert "Optional: name, email, duration_minutes, host_email" in body
    assert "Do not rename the JSON properties." in body
    assert "No cambies los nombres de las propiedades JSON." in body
    assert '"datetime"' in body


def test_build_agent_tool_section_uses_explicit_bilingual_when_provided():
    """when_to_use_en / when_to_use_es override the default description wrap."""
    body = build_agent_tool_section(
        tool_id="t",
        name="t",
        description="ignored",
        parameters_schema={"type": "object", "properties": {}, "required": []},
        when_to_use_en="Use when scheduling.",
        when_to_use_es="Úsala al agendar.",
    )
    assert "Use when scheduling." in body
    assert "Úsala al agendar." in body


def test_build_integration_section_renders_every_action():
    """Multiple actions -> multiple `### Action:` sub-blocks in order."""
    actions = [
        {
            "id": "create_event",
            "name": "Create Event",
            "description": "Creates a calendar event.",
            "when_to_use_en": "Use when the caller wants to schedule.",
            "when_to_use_es": "Usa cuando el cliente quiera agendar.",
            "parameters_schema": {
                "type": "object",
                "properties": {"start": {"type": "string"}},
                "required": ["start"],
            },
        },
        {
            "id": "cancel_event",
            "name": "Cancel Event",
            "description": "Cancels a calendar event.",
            "when_to_use_en": "Use when the caller wants to cancel.",
            "when_to_use_es": "Usa cuando el cliente quiera cancelar.",
            "parameters_schema": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
        },
    ]
    body = build_integration_section("int_1", "Google Calendar", actions)
    assert body.count("### Action:") == 2
    assert "### Action: Create Event" in body
    assert "### Action: Cancel Event" in body
    # create_event appears before cancel_event (document order).
    assert body.index("Create Event") < body.index("Cancel Event")
    assert "Required: start" in body
    assert "Required: event_id" in body


def test_build_integration_section_handles_zero_actions():
    """Empty actions list renders a deterministic bilingual fallback."""
    body = build_integration_section("int_x", "Empty Provider", [])
    assert "## Integration: Empty Provider" in body
    assert "no actions configured" in body.lower()


# ── Reconciler ────────────────────────────────────────────────────────────


def _fake_tools(*rows):
    """Stub `list_agent_tools_fn` for the reconciler.

    Each row must have id, name, description, parameters, kind.
    `credentials`/webhook_url filter happens upstream — tests only
    include real tools here.
    """
    def _fn(_agent_id, _user_id):
        return list(rows)
    return _fn


def _fake_integrations(*rows):
    def _fn(_agent_id, _user_id):
        return list(rows)
    return _fn


def _fake_spec(actions):
    """Stub `get_integration_provider_spec_fn` returning a fixed action set."""
    from types import SimpleNamespace
    spec = SimpleNamespace(actions=actions)
    def _fn(_provider_id):
        return spec
    return _fn


def _action(id_, name, when_en="", when_es="", required=(), properties=None):
    return {
        "id": id_,
        "name": name,
        "description": f"action {id_}",
        "when_to_use_en": when_en,
        "when_to_use_es": when_es,
        "parameters_schema": {
            "type": "object",
            "properties": properties or {id_: {"type": "string"}},
            "required": list(required),
        },
    }


def test_reconcile_adds_missing_sections():
    """Operator saved a prompt without sections; reconciler injects them."""
    prompt, log = reconcile_agent_prompt(
        "agent_1", "user_1", "free-form copy",
        list_agent_tools_fn=_fake_tools({
            "id": "tool_1", "name": "Tool One",
            "description": "desc", "parameters": {"type": "object", "properties": {}, "required": []},
            "kind": "webhook",
        }),
        list_agent_integrations_fn=_fake_integrations(),
    )
    assert "## Tool: Tool One" in prompt
    assert prompt.startswith("free-form copy")
    assert any("AGENT_TOOL:tool_1" in line for line in log)


def test_reconcile_removes_orphan_sections():
    """Stale sections (tool no longer assigned) are cleaned up."""
    initial = (
        "free-form\n\n"
        "<!-- AGENT_TOOL:tool_old -->\nbody\n<!-- END_AGENT_TOOL:tool_old -->\n\n"
        "<!-- INTEGRATION:int_old -->\nbody\n<!-- END_INTEGRATION:int_old -->\n"
    )
    prompt, log = reconcile_agent_prompt(
        "agent_1", "user_1", initial,
        list_agent_tools_fn=_fake_tools(),
        list_agent_integrations_fn=_fake_integrations(),
    )
    assert "tool_old" not in prompt
    assert "int_old" not in prompt
    assert "free-form" in prompt
    assert any("removed" in line for line in log)


def test_reconcile_is_idempotent_on_clean_prompt():
    """Running reconcile on a prompt that already has every section is a no-op."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": []}
    once, _ = reconcile_agent_prompt(
        "agent_1", "user_1", "",
        list_agent_tools_fn=_fake_tools({
            "id": "tool_1", "name": "T", "description": "d",
            "parameters": schema, "kind": "webhook",
        }),
        list_agent_integrations_fn=_fake_integrations(),
    )
    twice, log = reconcile_agent_prompt(
        "agent_1", "user_1", once,
        list_agent_tools_fn=_fake_tools({
            "id": "tool_1", "name": "T", "description": "d",
            "parameters": schema, "kind": "webhook",
        }),
        list_agent_integrations_fn=_fake_integrations(),
    )
    assert once == twice
    # Second pass produced no change_log — the prompt is stable.
    assert log == []


def test_reconcile_skips_call_transfer_and_credential_rows():
    """call_transfer has no LLM parameters; provider-credential rows aren't tools."""
    prompt, log = reconcile_agent_prompt(
        "agent_1", "user_1", "",
        list_agent_tools_fn=_fake_tools(
            {
                "id": "tool_ct", "name": "Transfer",
                "description": "x", "parameters": {},
                "kind": "call_transfer", "destination": "+15071234567",
            },
            {
                "id": "openai_creds", "name": "OpenAI",
                "description": "creds", "parameters": {},
                "kind": "webhook",
                "credentials": "ciphertext_blob",
                "webhook_url": "",
            },
        ),
        list_agent_integrations_fn=_fake_integrations(),
    )
    # Neither should generate a section.
    assert "tool_ct" not in prompt
    assert "openai_creds" not in prompt
    assert log == []


def test_reconcile_propagates_integration_changes():
    """The reconciler is what PUT /integrations/{id} calls per-agent."""
    initial = "free-form"
    after, log = reconcile_agent_prompt(
        "agent_1", "user_1", initial,
        list_agent_tools_fn=_fake_tools(),
        list_agent_integrations_fn=_fake_integrations({
            "id": "int_1",
            "provider": "google_calendar",
            "name": "Google Calendar",
        }),
        get_integration_provider_spec_fn=_fake_spec([
            _action("agendar_cita_dinamica", "Agendar Cita",
                    when_en="Use when the caller wants to schedule.",
                    when_es="Usa cuando el cliente quiera agendar.",
                    required=["datetime"],
                    properties={
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "datetime": {"type": "string"},
                    }),
        ]),
    )
    assert "## Integration: Google Calendar" in after
    assert "### Action: Agendar Cita" in after
    assert any("INTEGRATION:int_1" in line for line in log)
