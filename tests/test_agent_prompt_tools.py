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
            _action("calendar_event", "Create Calendar Event",
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
    assert "### Action: Create Calendar Event" in after
    assert any("INTEGRATION:int_1" in line for line in log)


def test_salesforce_real_catalog_exposes_six_actions_with_brief_specs():
    """Lock down the live Salesforce spec against the brief:

      find_customer · create_lead · create_case · update_customer
        · get_cases · log_call

    ponytail: this test reads the real `INTEGRATION_PROVIDERS` from
    `integrations_catalog` — no `_fake_spec`, no stub. We assert:

      * exactly six action ids in the canonical order
      * each action has a non-empty bilingual when_to_use_* (rendered
        into the agent's System Prompt by build_integration_section)
      * required lists match the brief:
          - find_customer  → [query]
          - create_lead    → [last_name, company]
          - create_case    → [customer_id, subject, description, priority, origin]
          - update_customer → [customer_id]
          - get_cases      → [customer_id]
          - log_call       → [customer_id, subject, description]
      * every parameters_schema carries `additionalProperties: false`
        so the executor rejects fields the LLM wasn't supposed to
        invent (e.g. `integration_id` inside `arguments`)
      * the create_case enums are closed: priority in
        {Low, Medium, High}, origin in {Phone}
      * log_call.outcome stays a free string (per the brief)
      * Salesforce remains OAuth-scoped (the executor pulls
        calendar_id/timezone/credentials from the integration row,
        not from the schema)

    The test fails immediately if anyone tries to swap the order,
    rename an action, drop a field, or remove the bilingual copy.
    """
    from STT_server.services.integrations_catalog import (
        get_integration_provider_spec,
    )

    spec = get_integration_provider_spec("salesforce")
    assert spec is not None, "salesforce must be a registered provider"
    assert spec.auth_type == "oauth", "salesforce stays OAuth-scoped"

    actions_by_id = {a.id: a for a in spec.actions}
    expected_ids = [
        "find_customer",
        "create_lead",
        "create_case",
        "update_customer",
        "get_cases",
        "log_call",
    ]
    assert list(actions_by_id) == expected_ids, (
        f"salesforce must expose exactly these six actions in this order: "
        f"{expected_ids}, got {list(actions_by_id)}"
    )

    # Bilingual when_to_use_* present on every action.
    for action in spec.actions:
        assert action.when_to_use_en, f"{action.id} missing English copy"
        assert action.when_to_use_es, f"{action.id} missing Spanish copy"
        assert action.parameters_schema.get("additionalProperties") is False, (
            f"{action.id} parameters_schema must set additionalProperties: false"
        )

    # Required-field assertions per action.
    def required(action_id):
        return set(actions_by_id[action_id].parameters_schema.get("required") or [])

    assert required("find_customer") == {"query"}
    assert required("create_lead") == {"last_name", "company"}
    assert required("create_case") == {
        "customer_id", "subject", "description", "priority", "origin",
    }
    assert required("update_customer") == {"customer_id"}
    assert required("get_cases") == {"customer_id"}
    assert required("log_call") == {"customer_id", "subject", "description"}

    # Enums where the brief asks for them.
    priority_enum = (
        actions_by_id["create_case"].parameters_schema["properties"]["priority"].get("enum")
    )
    origin_enum = (
        actions_by_id["create_case"].parameters_schema["properties"]["origin"].get("enum")
    )
    assert priority_enum == ["Low", "Medium", "High"]
    assert origin_enum == ["Phone"]

    # update_customer keeps every non-customer_id field optional so
    # the LLM only sends what changed.
    update_required = required("update_customer")
    assert "first_name" not in update_required
    assert "last_name" not in update_required
    assert "email" not in update_required
    assert "phone" not in update_required

    # log_call.outcome stays a free string (no enum) so the LLM can
    # write custom outcomes like "Rescheduled callback".
    outcome_spec = actions_by_id["log_call"].parameters_schema["properties"]["outcome"]
    assert outcome_spec["type"] == "string"
    assert "enum" not in outcome_spec


# ponytail: end-to-end reconciliation test against the LIVE catalog.
# This is the contract the brief asked for: an agent with the
# Salesforce integration assigned has a System Prompt that contains
# all six actions. We use the HTTP layer (assign_shared_integration)
# so the path includes everything: create_agent → create_integration
# (JSON path) → assign → reconciler → db_update_agent → read back.
async def test_reconciling_real_salesforce_assignment_emits_six_action_sections(
    client, auth_token,
):
    """After an agent assigns the real Salesforce integration, its
    System Prompt contains exactly the six action sections the brief
    requires — find_customer, create_lead, create_case, update_customer,
    get_cases, log_call — with their JSON Schemas inline."""
    from STT_server.db_integrations import create_integration as db_create_integration
    from STT_server.db_agents import create_agent as db_create_agent

    headers = {"Authorization": f"Bearer {auth_token}"}

    # Seed an agent via the HTTP layer so the row matches what the
    # FE would produce (POST /agents → agent_id).
    agent_resp = await client.post(
        "/agents", headers=headers,
        json={"name": "Reconciliation test agent"},
    )
    assert agent_resp.status_code == 200, agent_resp.text
    agent_id = agent_resp.json()["id"]

    # Seed a real Salesforce integration via the DB helper (JSON-file
    # path; the test backend uses JSON per the conftest). We skip the
    # HTTP create + OAuth dance — what matters is the catalog-driven
    # section, and the catalog is the same in both paths.
    integ = db_create_integration(
        "user-test-001",
        {
            "provider": "salesforce",
            "name": "Test Salesforce",
            "agent_id": "__shared__",
            "configuration": {},
        },
    )
    iid = integ["id"]

    # Assign → reconciler runs server-side. The reconciler reads the
    # live catalog (NOT a stub) and patches agents.prompt with one
    # `### Action:` per salesforce action.
    r = await client.post(
        f"/agents/{agent_id}/integrations/{iid}/assign",
        headers=headers,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["agent"]["id"] == agent_id
    # The reconciliation log includes an INTEGRATION section entry —
    # we don't pin the exact wording, only that something landed.
    assert any("INTEGRATION:" in line for line in body["change_log"])

    # Read the agent back. The System Prompt must contain every one
    # of the six actions in the canonical order.
    agent = body["agent"]
    prompt = agent["prompt"]
    expected_actions = [
        "Find Customer",
        "Create Lead",
        "Create Case",
        "Update Customer",
        "Get Cases",
        "Log Call",
    ]
    for name in expected_actions:
        marker = f"### Action: {name}"
        assert marker in prompt, (
            f"agent System Prompt is missing the `{marker}` section after "
            f"Salesforce assign. Full prompt:\n{prompt}"
        )
    # Order check: the actions appear in the same order we list them
    # in INTEGRATION_PROVIDERS. If anyone shuffles the catalog, the
    # operator will see the new order, but the LLM still gets the
    # right schema.
    indices = [prompt.index(f"### Action: {n}") for n in expected_actions]
    assert indices == sorted(indices), (
        f"action sections out of order: {list(zip(expected_actions, indices))}"
    )

    # Verify each section carries the bilingual copy + the JSON
    # schema inline. We don't pin the exact shape — the schema is
    # already covered by the unit test against the real catalog —
    # but we do confirm both ENGLISH/ESPAÑOL headers appear once per
    # action so the LLM gets the bilingual instructions.
    for name in expected_actions:
        section_start = prompt.index(f"### Action: {name}")
        section_end = prompt.index("<!-- END_INTEGRATION", section_start)
        section = prompt[section_start:section_end]
        assert "ENGLISH:" in section
        assert "ESPAÑOL:" in section
        assert "Do not rename the JSON properties." in section
        assert "No cambies los nombres de las propiedades JSON." in section
