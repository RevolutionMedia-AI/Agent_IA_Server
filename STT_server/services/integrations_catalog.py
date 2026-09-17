"""Catalog of third-party integrations the platform can connect to.

Mirrors STT_server/services/credentials_resolver.py's PROVIDER_CATALOG
shape (FieldSpec/ProviderSpec) but for the new "Integration" entity —
one third-party connection the agent talks to through n8n. The
existing provider catalog stays where it is for STT/TTS/LLM/Twilio
credentials managed via /settings/api-keys; integrations are a
separate, parallel entity with its own CRUD surface.

Key differences vs. credentials_resolver:
  * ProviderSpec has no `category` (the new entity is provider-shaped,
    not slot-shaped like LLM/STT/TTS). Categories live on
    integrations_catalog.ProviderSpec.category as a coarse FE grouping
    ("CRM & Customer Service" / "Contact Center" / "Custom").
  * ProviderSpec has `actions` — the documented verbs each integration
    exposes. The FE uses these to populate the "Action" dropdown when
    creating a Tool that points at the integration. Server-injected
    into the n8n POST body (LLM never controls action).
  * No `webhook_url` field for official providers — the URL lives in
    env config (INTEGRATIONS_N8N_WEBHOOK) and is resolved at
    executor time. Only `generic_webhook` carries a URL because the
    whole point is "let me point at any endpoint".

Add a new integration provider here and the FE picks it up via
GET /integrations/providers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("stt_server.services.integrations_catalog")


# ── Spec dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntegrationFieldSpec:
    """One form field the operator fills in to configure an integration.

    `type` is "text" or "password". Password fields are encrypted at
    rest in integrations.credentials_encrypted; text fields live
    plaintext in integrations.configuration. Subdomain, email, etc.
    are not secrets — only api_token-style values are.
    """
    name: str
    label: str
    type: str = "text"            # "text" | "password" | "url" | "email" | "select"
    placeholder: str = ""
    required: bool = False
    pattern: str | None = None    # regex (re.search)
    min_length: int = 0
    max_length: int = 0
    help: str = ""
    options: tuple[str, ...] = ()  # for type="select" - allowed values


@dataclass(frozen=True)
class ActionSpec:
    """One verb an integration exposes (find_customer, get_tickets, ...).

    `parameters_schema` is a JSON Schema the FE renders as the Tool's
    parameters field when the operator picks this action. Server-
    injected into the n8n POST body (LLM never controls action).

    `when_to_use_en` / `when_to_use_es` are the explicit bilingual
    copy rendered into the agent's System Prompt when this action is
    assigned. They live at the action level (NOT inside
    parameters_schema) so they never reach OpenAI's `tools[]` payload
    — that schema stays clean for function calling. Empty strings fall
    back to a deterministic wrapper around `description` in
    agent_prompt_tools.build_integration_section.
    """
    id: str                       # "find_customer" (matches ^[a-z0-9_]+$)
    name: str                     # "Find Customer"
    description: str = ""
    when_to_use_en: str = ""
    when_to_use_es: str = ""
    parameters_schema: dict = field(default_factory=lambda: {
            "type": "object", "properties": {}, "required": [],
        })
    capability: str = "core"  # ponytail: gate for Dynamics 365 (core | customer_insights)


@dataclass(frozen=True)
class IntegrationProviderSpec:
    """One third-party integration (Zendesk, Salesforce, ...)."""
    id: str                       # "zendesk"
    name: str                     # "Zendesk"
    category: str # "crm" | "contact_center" | "custom"
    fields: tuple[IntegrationFieldSpec, ...]
    actions: tuple[ActionSpec, ...]
    # Optional dotted path to a sync test function:
    #   test(creds: dict, config: dict) -> tuple[bool, str]
    # None means "Test Connection not yet implemented" — preflight
    # returns {valid: false, message: "Test not yet implemented for ..."}.
    test_fn: Optional[str] = None
    description: str = ""
    # ponytail: auth_type drives the FE form + BE create flow.
    #   "static"  → operator types credentials (api_token, etc.) into
    #               a regular form. preflight + encrypt + save.
    #   "oauth"   → Authorization Code flow. The BE handles the OAuth
    #               dance via /integrations/{id}/oauth/start +
    #               /integrations/{provider}/oauth/callback. The FE
    #               form is just Name + a [Connect with <provider>]
    #               button. preflight is skipped (no test_fn for
    #               OAuth — the OAuth dance IS the test).
    auth_type: str = "static"
    # OAuth-only fields (ignored when auth_type="static"). authorise
    # / token URLs come from oauth_providers.py; we only need the
    # button label + default scopes here for the FE.
    oauth_label: str = ""          # "Connect Salesforce"
    oauth_default_scopes: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    # ponytail: prompt_snippet was removed in the prompt-persistence
    # refactor. Instructions for assigned integrations now live
    # inside `agents.prompt`, generated by agent_prompt_tools from
    # each Action's description + when_to_use_en/es +
    # parameters_schema. The runtime no longer concatenates anything
    # at call time — see STT_Server.py for the legacy block that's
    # also gone.


# ── Validation helpers (mirror credentials_resolver but for integrations) ─

import re


def validate_integration_fields(
    provider_id: str,
    configuration: dict,
    credentials: dict,
) -> tuple[dict, dict, list[dict]]:
    """Run regex / length checks against the IntegrationProviderSpec.

    Returns (cleaned_configuration, cleaned_credentials, errors).
    Empty / missing fields are dropped (so the FE can clear a field
    by submitting ""). Errors are a list of {field, message} objects ready
    for a 400/422 response.

    Configuration errors are tagged {field: "config.<name>"} so the FE
    can split the error banner by section. Credential errors use
    {field: "credential.<name"}.

    ponytail: bucket rule — fields are routed to credentials when
    their type is "password" or "email" (auth data the operator
    shouldn't see in the FE log or error banners), and to
    configuration otherwise (subdomain, instance_url, oauth_client_id,
    webhook_url — all things that aren't secrets but the operator
    fills in on the same form). The decision lives here, in one
    place, so every caller agrees on which bucket a value lands in.
    """
    spec = get_integration_provider_spec(provider_id)
    if spec is None:
        return {}, [], [{"field": "provider", "message": f"Unknown provider '{provider_id}'"}]
    errors: list[dict] = []
    cleaned_config: dict = {}
    cleaned_creds: dict = {}
    if not isinstance(configuration, dict):
        configuration = {}
    if not isinstance(credentials, dict):
        credentials = {}
    for f in spec.fields:
        is_secret = f.type in ("password", "email")
        raw = (credentials.get(f.name) if is_secret else configuration.get(f.name))
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            # Field absent or empty: only an error if it's required AND
            # the user submitted at least one value somewhere. They
            # might be clearing a previously-set value (PUT path).
            continue
        if not isinstance(raw, str):
            errors.append({
                "field": f"credential.{f.name}" if is_secret else f"config.{f.name}",
                "message": f"{f.label} must be a string",
            })
            continue
        value = raw.strip()
        if f.max_length and len(value) > f.max_length:
            errors.append({
                "field": f"credential.{f.name}" if is_secret else f"config.{f.name}",
                "message": f"{f.label} is too long (max {f.max_length})",
            })
            continue
        if f.min_length and len(value) < f.min_length:
            errors.append({
                "field": f"credential.{f.name}" if is_secret else f"config.{f.name}",
                "message": f"{f.label} is too short (min {f.min_length})",
            })
            continue
        if f.pattern and not re.search(f.pattern, value):
            errors.append({
                "field": f"credential.{f.name}" if is_secret else f"config.{f.name}",
                "message": f"{f.label} doesn't match the expected format. {f.help}".strip(),
            })
            continue
        if f.options and value not in f.options:
            errors.append({
                "field": f"credential.{f.name}" if is_secret else f"config.{f.name}",
                "message": f"{f.label} must be one of: {', '.join(f.options)}",
            })
            continue
        if is_secret:
            cleaned_creds[f.name] = value
        else:
            cleaned_config[f.name] = value
    return cleaned_config, cleaned_creds, errors


# ── Catalog ─────────────────────────────────────────────────────────────────

# ponytail: action ids must match ^[a-z0-9_]+$ because the n8n Switch
# node keys off them and the BE executor uses them as a string key in
# the dispatch body. Hyphens / spaces would force us to URL-encode the
# whole dispatch contract.
_ACTION_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


def _a(
    aid: str,
    name: str,
    desc: str = "",
    schema: dict | None = None,
    *,
    when_to_use_en: str = "",
    when_to_use_es: str = "",
    capability: str = "core",
) -> ActionSpec:
    """Shorthand constructor with id-format guard."""
    if not _ACTION_ID_PATTERN.match(aid):
        raise ValueError(
            f"action id '{aid}' must match ^[a-z0-9_]+$ (got disallowed chars)"
        )
    return ActionSpec(
        id=aid,
        name=name,
        description=desc,
        when_to_use_en=when_to_use_en,
        when_to_use_es=when_to_use_es,
        parameters_schema=schema or {
            "type": "object", "properties": {}, "required": [],
        },
        capability=capability,
    )


def _dyn_schema(required=(), **properties) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _dyn_text(description: str, **extra) -> dict:
    return {"type": "string", "description": description, **extra}


_DYN_GUID = _dyn_text("Dataverse GUID returned by a previous Dynamics action.", pattern=r"^[0-9a-fA-F-]{36}$")
_DYN_CUSTOMER_TYPE = _dyn_text("Dataverse customer type.", enum=["contact", "account"])
_DYN_REGARDING_TYPE = _dyn_text(
    "Dataverse record type.", enum=["contact", "account", "lead", "incident", "opportunity", "workorder", "booking", "asset"]
)
_DYN_WO_STATUS = _dyn_text("Work order scope.", enum=["open", "all"])


def _dynamics365_actions() -> tuple[ActionSpec, ...]:
    def text(label: str) -> dict:
        return _dyn_text(label)
    return (
        _a("find_customer", "Find Customer", "Search contacts and accounts.", _dyn_schema(("query",), query=text("Name, email, or phone."))),
        _a("get_customer", "Get Customer", "Get a contact or account by GUID.", _dyn_schema(("customer_id",), customer_id=_DYN_GUID, customer_type=_DYN_CUSTOMER_TYPE)),
        _a("create_contact", "Create Contact", "Create a Dataverse contact.", _dyn_schema(("last_name",), first_name=text("First name."), last_name=text("Last name."), email=text("Email address."), phone=text("Telephone number."), account_id=_DYN_GUID)),
        _a("update_customer", "Update Customer", "Update a contact or account.", _dyn_schema(("customer_id",), customer_id=_DYN_GUID, customer_type=_DYN_CUSTOMER_TYPE, first_name=text("First name."), last_name=text("Last name."), name=text("Account name."), email=text("Email address."), phone=text("Telephone number."))),
        _a("find_account", "Find Account", "Search Dataverse accounts.", _dyn_schema(("query",), query=text("Account name, email, or phone."))),
        _a("create_account", "Create Account", "Create a Dataverse account.", _dyn_schema(("name",), name=text("Account name."), email=text("Email address."), phone=text("Telephone number."), website=text("Website URL."))),
        _a("update_account", "Update Account", "Update a Dataverse account.", _dyn_schema(("account_id",), account_id=_DYN_GUID, name=text("Account name."), email=text("Email address."), phone=text("Telephone number."), website=text("Website URL."))),
        _a("find_lead", "Find Lead", "Search Dataverse leads.", _dyn_schema(("query",), query=text("Lead name, email, or company."))),
        _a("create_lead", "Create Lead", "Create a Dataverse lead.", _dyn_schema(("subject", "last_name"), subject=text("Lead topic."), first_name=text("First name."), last_name=text("Last name."), company=text("Company name."), email=text("Email address."), phone=text("Telephone number."))),
        _a("update_lead", "Update Lead", "Update a Dataverse lead.", _dyn_schema(("lead_id",), lead_id=_DYN_GUID, subject=text("Lead topic."), first_name=text("First name."), last_name=text("Last name."), company=text("Company name."), email=text("Email address."), phone=text("Telephone number."))),
        _a("qualify_lead", "Qualify Lead", "Run the Dataverse QualifyLead action.", _dyn_schema(("lead_id",), lead_id=_DYN_GUID, create_account={"type": "boolean"}, create_contact={"type": "boolean"}, create_opportunity={"type": "boolean"}, status={"type": "integer", "description": "Qualified lead status code; defaults to 3."})),
        _a("create_case", "Create Case", "Create a Customer Service case.", _dyn_schema(("title",), title=text("Case title."), description=text("Case description."), customer_id=_DYN_GUID, customer_type=_DYN_CUSTOMER_TYPE)),
        _a("get_cases", "Get Cases", "List Customer Service cases.", _dyn_schema(customer_id=_DYN_GUID, active_only={"type": "boolean"}, limit={"type": "integer", "minimum": 1, "maximum": 50})),
        _a("get_case", "Get Case", "Get a case by GUID.", _dyn_schema(("case_id",), case_id=_DYN_GUID)),
        _a("update_case", "Update Case", "Update a Customer Service case.", _dyn_schema(("case_id",), case_id=_DYN_GUID, title=text("Case title."), description=text("Case description."), priority_code={"type": "integer"})),
        _a("resolve_case", "Resolve Case", "Run the Dataverse CloseIncident action.", _dyn_schema(("case_id",), case_id=_DYN_GUID, resolution=text("Resolution subject."), status={"type": "integer", "description": "Resolved status code; defaults to 5."})),
        _a("reopen_case", "Reopen Case", "Set a resolved case back to active.", _dyn_schema(("case_id",), case_id=_DYN_GUID)),
        _a("create_opportunity", "Create Opportunity", "Create a sales opportunity.", _dyn_schema(("name",), name=text("Opportunity topic."), customer_id=_DYN_GUID, customer_type=_DYN_CUSTOMER_TYPE, estimated_value={"type": "number"}, estimated_close_date=text("ISO 8601 date."))),
        _a("get_opportunities", "Get Opportunities", "List sales opportunities.", _dyn_schema(customer_id=_DYN_GUID, active_only={"type": "boolean"}, limit={"type": "integer", "minimum": 1, "maximum": 50})),
        _a("get_opportunity", "Get Opportunity", "Get an opportunity by GUID.", _dyn_schema(("opportunity_id",), opportunity_id=_DYN_GUID)),
        _a("update_opportunity", "Update Opportunity", "Update an opportunity.", _dyn_schema(("opportunity_id",), opportunity_id=_DYN_GUID, name=text("Opportunity topic."), estimated_value={"type": "number"}, estimated_close_date=text("ISO 8601 date."), description=text("Description."))),
        _a("close_opportunity_won", "Close Opportunity Won", "Run the Dataverse WinOpportunity action.", _dyn_schema(("opportunity_id",), opportunity_id=_DYN_GUID, subject=text("Close activity subject."), status={"type": "integer", "description": "Won status code; defaults to 3."})),
        _a("close_opportunity_lost", "Close Opportunity Lost", "Run the Dataverse LoseOpportunity action.", _dyn_schema(("opportunity_id",), opportunity_id=_DYN_GUID, subject=text("Close activity subject."), description=text("Loss reason."), status={"type": "integer", "description": "Lost status code; defaults to 4."})),
        _a("log_call", "Log Call", "Create a Dataverse phone call activity.", _dyn_schema(("subject",), subject=text("Call subject."), description=text("Call notes."), phone=text("Telephone number."), direction={"type": "string", "enum": ["incoming", "outgoing"]}, regarding_id=_DYN_GUID, regarding_type=_DYN_REGARDING_TYPE)),
        _a("add_note", "Add Note", "Attach a note to a Dataverse record.", _dyn_schema(("regarding_id", "regarding_type", "text"), regarding_id=_DYN_GUID, regarding_type=_DYN_REGARDING_TYPE, text=text("Note text."), subject=text("Note subject."))),
        _a("create_task", "Create Task", "Create a Dataverse task activity.", _dyn_schema(("subject",), subject=text("Task subject."), description=text("Task description."), due_at=text("ISO 8601 due date/time."), regarding_id=_DYN_GUID, regarding_type=_DYN_REGARDING_TYPE)),
        _a("complete_task", "Complete Task", "Mark a Dataverse task completed.", _dyn_schema(("task_id",), task_id=_DYN_GUID)),
        _a("find_asset", "Find Asset", "Search Field Service customer assets by name, asset tag, or account.",
           _dyn_schema(("query",), query=text("Asset name or asset tag."), account_id=_DYN_GUID),
           when_to_use_en="Use this action to identify the caller's equipment before creating a work order. The asset_id returned links the work order to the asset.",
           when_to_use_es="Usa esta acción para identificar el equipo del cliente antes de crear una orden de trabajo. El asset_id devuelto vincula la orden al activo."),
        _a("get_asset", "Get Asset", "Get a Field Service customer asset by GUID.", _dyn_schema(("asset_id",), asset_id=_DYN_GUID)),
        _a("create_asset", "Create Asset", "Create a Field Service customer asset. Only name is required.",
           _dyn_schema(("name",), name=text("Asset name."), asset_tag=text("Asset tag or serial identifier."), account_id=_DYN_GUID, product_id=_DYN_GUID, parent_asset_id=_DYN_GUID)),
        _a("update_asset", "Update Asset", "Update a Field Service customer asset.", _dyn_schema(("asset_id",), asset_id=_DYN_GUID, name=text("Asset name."), asset_tag=text("Asset tag or serial identifier."), account_id=_DYN_GUID, product_id=_DYN_GUID)),
        _a("create_work_order", "Create Work Order", "Create an unscheduled Field Service work order. The work order number is auto-assigned.",
           _dyn_schema(service_account_id=_DYN_GUID, description=text("Problem description; stored as work instructions."), asset_id=_DYN_GUID, priority=text("Priority name, e.g. High. Resolved against environment priorities."), priority_id=_DYN_GUID, billing_account_id=_DYN_GUID, work_order_type_id=_DYN_GUID, price_list_id=_DYN_GUID, incident_type_id=_DYN_GUID, scheduled_start=text("Promised window start, ISO 8601."), scheduled_end=text("Promised window end, ISO 8601.")),
           when_to_use_en="Use this action after find_customer and find_asset when the caller reports an issue needing an on-site visit. Creating a work order is not scheduling — use get_available_resources + create_booking next.",
           when_to_use_es="Usa esta acción después de find_customer y find_asset cuando el cliente reporta un problema que requiere visita. Crear la orden no es agendar — luego usa get_available_resources y create_booking."),
        _a("get_work_order", "Get Work Order", "Get a Field Service work order by GUID.", _dyn_schema(("work_order_id",), work_order_id=_DYN_GUID)),
        _a("get_work_orders", "Get Work Orders", "List Field Service work orders. Defaults to open ones.",
           _dyn_schema(account_id=_DYN_GUID, asset_id=_DYN_GUID, status=_DYN_WO_STATUS, limit={"type": "integer", "minimum": 1, "maximum": 50})),
        _a("update_work_order", "Update Work Order", "Update a Field Service work order.", _dyn_schema(("work_order_id",), work_order_id=_DYN_GUID, description=text("Updated work instructions."), service_account_id=_DYN_GUID, asset_id=_DYN_GUID, priority=text("Priority name."), priority_id=_DYN_GUID, scheduled_start=text("Promised window start, ISO 8601."), scheduled_end=text("Promised window end, ISO 8601."))),
        _a("cancel_work_order", "Cancel Work Order", "Cancel a Field Service work order with an optional reason saved to its timeline.", _dyn_schema(("work_order_id",), work_order_id=_DYN_GUID, reason=text("Cancellation reason."))),
        _a("complete_work_order", "Complete Work Order", "Mark a Field Service work order completed with optional technician notes.", _dyn_schema(("work_order_id",), work_order_id=_DYN_GUID, completion_notes=text("Completion notes saved to the timeline."))),
        _a("get_available_resources", "Get Available Resources", "Find eligible technicians and time slots via the official schedule assistant.",
           _dyn_schema(("start", "end"), start=text("Window start, ISO 8601."), end=text("Window end, ISO 8601."), duration_minutes={"type": "integer", "minimum": 1, "description": "Job duration in minutes; defaults to 60 or the work order estimate."}, work_order_id=_DYN_GUID),
           when_to_use_en="Use this action after create_work_order to find who can take the job. Never invent availability — only offer what this action returns.",
           when_to_use_es="Usa esta acción después de create_work_order para saber quién puede tomar el trabajo. Nunca inventes disponibilidad — solo ofrece lo que devuelva esta acción."),
        _a("create_booking", "Create Booking", "Book a technician for a work order in a time window.", _dyn_schema(("work_order_id", "resource_id", "start", "end"), work_order_id=_DYN_GUID, resource_id=_DYN_GUID, start=text("Booking start, ISO 8601."), end=text("Booking end, ISO 8601."))),
        _a("update_booking", "Update Booking", "Reschedule a Field Service booking.", _dyn_schema(("booking_id",), booking_id=_DYN_GUID, start=text("New start, ISO 8601."), end=text("New end, ISO 8601."))),
        _a("cancel_booking", "Cancel Booking", "Cancel a Field Service booking with an optional reason saved to its timeline.", _dyn_schema(("booking_id",), booking_id=_DYN_GUID, reason=text("Cancellation reason."))),
        _a("find_service_agreement", "Find Service Agreement", "Search Field Service agreements by name, optionally for one account.", _dyn_schema(("query",), query=text("Agreement name."), account_id=_DYN_GUID)),
        _a("get_service_agreements", "Get Service Agreements", "List Field Service agreements. Defaults to active ones.", _dyn_schema(account_id=_DYN_GUID, active_only={"type": "boolean"}, limit={"type": "integer", "minimum": 1, "maximum": 50})),
        # ponytail: Customer Insights – Data (capability=customer_insights) — same provider/OAuth/client
        _a("ci_get_profile", "Get Customer Profile (CI)", "Get a Customer Insights profile by ID.", _dyn_schema(("profile_id",), profile_id=_dyn_text("CI profile ID.")), capability="customer_insights"),
        _a("ci_search_profiles", "Search Profiles (CI)", "Search Customer Insights profiles.", _dyn_schema(("query",), query=_dyn_text("Search query.")), capability="customer_insights"),
        _a("ci_get_segments", "Get Segments (CI)", "List Customer Insights segments.", _dyn_schema(limit={"type": "integer", "minimum": 1, "maximum": 50}), capability="customer_insights"),
        _a("ci_get_measures", "Get Measures (CI)", "List Customer Insights measures.", _dyn_schema(limit={"type": "integer", "minimum": 1, "maximum": 50}), capability="customer_insights"),
    )


INTEGRATION_PROVIDERS: tuple[IntegrationProviderSpec, ...] = (
    IntegrationProviderSpec(
        id="zendesk",
        name="Zendesk",
        category="crm",
        description="Zendesk Support API — tickets, customers, comments.",
        fields=(
            IntegrationFieldSpec(
                name="subdomain", label="Subdomain", type="text",
                required=True,
                pattern=r"^[a-z0-9][a-z0-9-]{1,30}[a-z0-9]$",
                min_length=3, max_length=32,
                placeholder="revolutionmedia",
                help="The 'company' in https://company.zendesk.com — no protocol, no .zendesk.com suffix.",
            ),
            IntegrationFieldSpec(
                name="email", label="Account email", type="email",
                required=True,
                pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                min_length=5, max_length=254,
                placeholder="admin@revolutionmedia.ai",
                help="Email of the Zendesk admin or agent account used for API auth.",
            ),
            IntegrationFieldSpec(
                name="api_token", label="API Token", type="password",
                required=True,
                min_length=20, max_length=512,
                help="Generate in Zendesk Admin Center → Apps and integrations → APIs → Zendesk API → Settings.",
            ),
        ),
        actions=(
            _a("find_customer", "Find Customer",
               "Look up a Zendesk end-user by email.",
               {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]}),
            _a("get_tickets", "Get Tickets",
               "List the caller's tickets.",
               {"type": "object", "properties": {"email": {"type": "string"}, "status": {"type": "string"}}, "required": ["email"]}),
            _a("create_ticket", "Create Ticket",
               "Open a new support ticket.",
               {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}, "requester_email": {"type": "string"}}, "required": ["subject", "description", "requester_email"]}),
            _a("add_comment", "Add Comment",
               "Append a comment to an existing ticket.",
               {"type": "object", "properties": {"ticket_id": {"type": "string"}, "body": {"type": "string"}}, "required": ["ticket_id", "body"]}),
            _a("update_ticket", "Update Ticket",
               "Update status / priority / assignee of a ticket.",
               {"type": "object", "properties": {"ticket_id": {"type": "string"}, "status": {"type": "string"}}, "required": ["ticket_id"]}),
        ),
        # ponytail: only Zendesk ships a real test_fn in V1. The
        # Salesforce / Dynamics / Genesys / NICE specs are in place so
        # the FE renders them, but the test_fn returns "not yet
        # implemented" until someone validates each provider's API.
        test_fn="STT_server.services.integrations_tester._test_zendesk",
    ),
    IntegrationProviderSpec(
        id="salesforce",
        name="Salesforce",
        category="crm",
        description="Salesforce REST API — leads, contacts, cases. Connects via OAuth 2.0 Authorization Code.",
        fields=(),  # ponytail: OAuth — no operator-typed fields. The
                    # token exchange writes configuration.instance_url
                    # + credentials.{access,refresh}_token. The FE
                    # only asks for Integration Name + clicks [Connect].
        #
        # ponytail: 6 actions per the brief (find_customer, create_lead,
        # create_case, update_customer, get_cases, log_call). Every
        # schema carries `additionalProperties: false` so the executor
        # rejects fields the LLM wasn't supposed to invent (e.g. an
        # `integration_id` or `provider` in `arguments`). The runtime
        # also enforces a non-empty PATCH on `update_customer` — the
        # schema alone can't express "at least one of these must be
        # set besides customer_id", so the n8n workflow runs the
        # minProperties check on its side before issuing the PATCH.
        actions=(
            _a(
                "find_customer",
                "Find Customer",
                desc="Search an existing Salesforce Contact by name, email, or phone.",
                schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Nombre, correo electrónico o número telefónico del cliente que se desea buscar en Salesforce.",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action to look up an existing Salesforce Contact when "
                    "the caller identifies themselves (by name, email, or phone). "
                    "The returned customer.id is the link to every other Salesforce "
                    "action — never invent it."
                ),
                when_to_use_es=(
                    "Usa esta acción para buscar un Contact existente cuando el "
                    "cliente se identifica (por nombre, email o teléfono). El "
                    "customer.id devuelto es el vínculo con el resto de las acciones "
                    "de Salesforce — nunca lo inventes."
                ),
            ),
            _a(
                "create_lead",
                "Create Lead",
                desc="Create a new prospect in Salesforce.",
                schema={
                    "type": "object",
                    "properties": {
                        "first_name": {
                            "type": "string",
                            "description": "Nombre del prospecto.",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "Apellido del prospecto.",
                        },
                        "company": {
                            "type": "string",
                            "description": "Nombre de la empresa del prospecto.",
                        },
                        "email": {
                            "type": "string",
                            "description": "Correo electrónico del prospecto.",
                        },
                        "phone": {
                            "type": "string",
                            "description": "Número telefónico del prospecto.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Resumen de la necessidade, interés o contexto mencionado durante la llamada.",
                        },
                    },
                    "required": ["last_name", "company"],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action when the caller is a new prospect who is not "
                    "yet a Contact in Salesforce — typically a cold outreach or "
                    "a new business lead. After create_lead succeeds the operator "
                    "can promote the lead to a Contact later through Salesforce UI."
                ),
                when_to_use_es=(
                    "Usa esta acción cuando el cliente es un prospecto nuevo que aún "
                    "no es un Contact en Salesforce — generalmente una llamada en frío "
                    "o un lead nuevo. Después de que create_lead tenga éxito, el "
                    "operador puede promover el lead a Contact desde la UI de Salesforce."
                ),
            ),
            _a(
                "create_case",
                "Create Case",
                desc="Open a new Salesforce Case for follow-up.",
                schema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "ID del Contact de Salesforce obtenido previamente con find_customer. Nunca debe inventarse.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Título corto y específico que explique el motivo del caso.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Descripción detallada del problema, solicitud y contexto relevante explicado por el cliente.",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["Low", "Medium", "High"],
                            "description": "Nivel de prioridad del caso.",
                        },
                        "origin": {
                            "type": "string",
                            "enum": ["Phone"],
                            "description": "Origen del caso. Para el agente telefónico debe utilizar Phone.",
                        },
                    },
                    "required": [
                        "customer_id",
                        "subject",
                        "description",
                        "priority",
                        "origin",
                    ],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action when the caller has an issue or request that "
                    "needs human follow-up after the call. Always run get_cases "
                    "first to check for an existing open case with the same subject "
                    "before opening a duplicate."
                ),
                when_to_use_es=(
                    "Usa esta acción cuando el cliente tiene un problema o solicitud "
                    "que necesita seguimiento humano después de la llamada. Siempre "
                    "ejecuta get_cases primero para verificar si ya existe un caso "
                    "abierto con el mismo asunto antes de abrir uno duplicado."
                ),
            ),
            _a(
                "update_customer",
                "Update Customer",
                desc="Patch an existing Salesforce Contact with one or more changed fields.",
                schema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "ID del Contact de Salesforce obtenido previamente con find_customer. Nunca debe inventarse.",
                        },
                        "first_name": {
                            "type": "string",
                            "description": "Nuevo nombre del cliente. Enviar únicamente si el cliente solicita actualizarlo.",
                        },
                        "last_name": {
                            "type": "string",
                            "description": "Nuevo apellido del cliente. Enviar únicamente si el cliente solicita actualizarlo.",
                        },
                        "email": {
                            "type": "string",
                            "description": "Nuevo correo electrónico del cliente. Enviar únicamente si necesita actualizarse.",
                        },
                        "phone": {
                            "type": "string",
                            "description": "Nuevo número telefónico del cliente. Enviar únicamente si necesita actualizarse.",
                        },
                    },
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action when the caller wants to change their own "
                    "contact details (new email, new phone, name correction, etc). "
                    "Send ONLY the fields that actually changed — never re-send "
                    "fields whose value is unchanged. The runtime rejects an "
                    "empty PATCH (customer_id alone is not enough) so at least "
                    "one of first_name, last_name, email, phone must accompany it."
                ),
                when_to_use_es=(
                    "Usa esta acción cuando el cliente quiere cambiar sus propios "
                    "datos de contacto (nuevo email, nuevo teléfono, corrección de "
                    "nombre, etc). Envía SOLO los campos que realmente cambiaron "
                    "— nunca reenvíes campos cuyo valor no cambió. El runtime "
                    "rechaza un PATCH vacío (customer_id solo no es suficiente), "
                    "así que al menos uno de first_name, last_name, email, phone "
                    "debe acompañarlo."
                ),
            ),
            _a(
                "get_cases",
                "Get Cases",
                desc="List Salesforce Cases linked to a Contact.",
                schema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "ID del Contact de Salesforce obtenido previamente mediante find_customer.",
                        }
                    },
                    "required": ["customer_id"],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action to check whether the caller already has an "
                    "open Salesforce Case for their request. Run it BEFORE "
                    "create_case to avoid opening duplicates."
                ),
                when_to_use_es=(
                    "Usa esta acción para verificar si el cliente ya tiene un "
                    "caso abierto en Salesforce para su solicitud. Ejecútala "
                    "ANTES de create_case para evitar abrir duplicados."
                ),
            ),
            _a(
                "log_call",
                "Log Call",
                desc="Record the call summary on the customer's Salesforce record.",
                schema={
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "ID del Contact de Salesforce asociado con la llamada.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "Título corto que describa el propósito principal de la llamada.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Resumen profesional de la llamada, incluyendo motivo, información relevante, acciones realizadas y resultado.",
                        },
                        "duration_seconds": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Duración total de la llamada en segundos.",
                        },
                        "outcome": {
                            "type": "string",
                            "description": "Resultado final de la llamada, por ejemplo Resolved, Case created, Follow-up required, Information provided o Customer not interested.",
                        },
                    },
                    "required": ["customer_id", "subject", "description"],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action at the end of the call to persist a "
                    "professional summary on the customer's record. duration_seconds "
                    "and outcome are optional but recommended — they let the "
                    "operator audit call length and outcome later from Salesforce."
                ),
                when_to_use_es=(
                    "Usa esta acción al final de la llamada para guardar un "
                    "resumen profesional en el registro del cliente. duration_seconds "
                    "y outcome son opcionales pero recomendados — permiten al "
                    "operador auditar la duración y el resultado después desde Salesforce."
                ),
            ),
        ),
        test_fn="STT_server.services.integrations_tester._test_salesforce",
        auth_type="oauth",
        oauth_label="Connect Salesforce",
        oauth_default_scopes=("api", "refresh_token"),
    ),
    IntegrationProviderSpec(
        id="dynamics365",
        name="Microsoft Dynamics 365",
        category="crm",
        description="Microsoft Dynamics 365 Sales and Customer Service through Dataverse.",
        fields=(),
        actions=_dynamics365_actions(),
        test_fn="STT_server.services.integrations_tester._test_dynamics365",
        auth_type="oauth",
        oauth_label="Connect Microsoft Dynamics 365",
        oauth_default_scopes=("openid", "profile", "email", "offline_access", "https://globaldisco.crm.dynamics.com/.default"),
        capabilities=("sales", "customer_service", "field_service", "customer_insights"),
    ),
    IntegrationProviderSpec(
        id="genesys_cloud",
        name="Genesys Cloud CX",
        category="contact_center",
        description="Genesys Cloud — call routing, agent state, queues.",
        fields=(
            IntegrationFieldSpec(
                name="region", label="Region", type="text",
                required=True,
                pattern=r"^[a-z]+\.[a-z]+\.purecloud\.com$",
                placeholder="mypurecloud.com",
                help="Region host (e.g. mypurecloud.com, usw2.purecloud.com).",
            ),
            IntegrationFieldSpec(
                name="oauth_client_id", label="OAuth Client ID", type="text",
                required=True, min_length=8, max_length=128,
            ),
            IntegrationFieldSpec(
                name="oauth_client_secret", label="OAuth Client Secret", type="password",
                required=True, min_length=20, max_length=256,
            ),
        ),
        actions=(
            _a("transfer_call", "Transfer Call"),
            _a("set_agent_status", "Set Agent Status"),
            _a("get_queue_stats", "Get Queue Stats"),
        ),
        test_fn=None,
    ),
    IntegrationProviderSpec(
        id="nice_cxone",
        name="NICE CXone",
        category="contact_center",
        description="NICE CXone — contact center routing and reporting.",
        fields=(
            IntegrationFieldSpec(
                name="tenant", label="Tenant / POD", type="text",
                required=True, min_length=2, max_length=64,
                placeholder="na1",
            ),
            IntegrationFieldSpec(
                name="access_token", label="Access Token", type="password",
                required=True, min_length=20,
            ),
        ),
        actions=(
            _a("transfer_call", "Transfer Call"),
            _a("get_skill_stats", "Get Skill Stats"),
        ),
        test_fn=None,
    ),
    # ponytail: generic_webhook IS the provider for "I just want to
    # call any URL". The n8n router resolves the URL from this row's
    # configuration.webhook_url. For all the official providers
    # above, configuration has no webhook_url — the URL lives in
    # env (INTEGRATIONS_N8N_WEBHOOK).
    IntegrationProviderSpec(
        id="generic_webhook",
        name="Generic Webhook",
        category="custom",
        description="Send JSON to any HTTPS endpoint. Choose the HTTP method your webhook expects.",
        fields=(
            IntegrationFieldSpec(
                name="webhook_url", label="Webhook URL", type="url",
                required=True,
                pattern=r"^https?://\S+$",
                placeholder="https://example.com/webhook",
                help="The endpoint we will call when this tool is invoked.",
            ),
            IntegrationFieldSpec(
                name="webhook_method", label="HTTP Method", type="select",
                required=False,
                placeholder="POST",
                help="Method used to call the webhook. Most n8n webhooks use POST; use GET for fetch-only endpoints. Supported: GET, POST, PUT, PATCH, DELETE.",
                options=("GET", "POST", "PUT", "PATCH", "DELETE"),
            ),
        ),
        actions=(),  # empty: the operator picks the action free-form per tool
        test_fn="STT_server.services.integrations_tester._test_webhook_reachable",
    ),
    # ponytail: Google Calendar via OAuth. Each operator connects
    # THEIR Google account in /integrations → Connect with Google.
    # The OAuth handshake writes the user's access_token + refresh_token
    # to the integration row (encrypted) plus whatever scope metadata
    # Google returns. The flow reuses the OAuth plumbing that
    # Salesforce already established — same registry, same callback
    # shape, same refresh-on-read logic in
    # /internal/integrations/{id}/credentials. n8n then receives
    # the access_token (and refresh_token) inside its body alongside
    # `calendar_id` + `timezone` from the integration's configuration.
    #
    # ponytail: action parameters are the LLM's contract — they go
    # verbatim into the function-call body and into the `## Tool:`
    # section that the LLM reads. We deliberately exclude
    # `host_email` from the JSON Schema because the host calendar is
    # always resolved server-side from configuration.calendar_id —
    # the LLM cannot pick a calendar, so the field shouldn't appear
    # in the function definition.
    IntegrationProviderSpec(
        id="google_calendar",
        name="Google Calendar",
        category="custom",
        # ponytail: configuration keys. The operator sets these
        # AFTER the OAuth handshake (the connection-status page
        # surfaces an extra "calendar_id" / "timezone" picker once
        # the integration is `connected`). The LLM never sees
        # these — the n8n workflow picks them up off the body the
        # tool_executor sends.
        description=(
            "OAuth — connect a Google account to create + delete calendar "
            "events through n8n. The host calendar (configuration.calendar_id) "
            "and timezone (configuration.timezone) live on the integration row, "
            "not in the LLM's arguments."
        ),
        fields=(),  # OAuth — no operator-typed fields at create time.
        actions=(
            _a(
                "create_appointment",
                "Create Calendar Appointment",
                "Book a new event on the host calendar with a Google Meet link.",
                # ponytail: 2026-09-04 schema expansion. The LLM now
                # produces a short, descriptive ``title`` and a
                # ``description`` based on the call context, so anyone
                # opening the calendar event later sees the purpose
                # without replaying the call. ``datetime`` carries the
                # wall-clock moment in the integration's timezone (no
                # offset needed in the LLM prompt; the BE attaches
                # the offset at request time). ``additionalProperties: false``
                # stops the model from inventing extra fields the
                # executor doesn't read.
                {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Nombre completo de la persona que solicita la cita."
                            ),
                        },
                        "email": {
                            "type": "string",
                            "description": (
                                "Correo electrónico de la persona que asistirá a la reunión."
                            ),
                        },
                        "datetime": {
                            "type": "string",
                            "description": (
                                "Fecha y hora de inicio de la reunión en formato "
                                "ISO 8601 local, por ejemplo: 2026-09-08T15:00:00. "
                                "No inventar la fecha ni la hora; debe haber sido "
                                "acordada con el cliente."
                            ),
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Duración de la reunión en minutos.",
                            "minimum": 5,
                            "maximum": 240,
                        },
                        "title": {
                            "type": "string",
                            "description": (
                                "Título breve y descriptivo de la reunión. Debe "
                                "indicar claramente el propósito de la cita, por "
                                "ejemplo: 'Consulta sobre renovación de servicio' o "
                                "'Seguimiento de problema de facturación'. No usar "
                                "únicamente el nombre del cliente."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Resumen claro del motivo de la reunión basado "
                                "exclusivamente en lo hablado durante la llamada. "
                                "Debe explicar qué necesita el cliente, qué quiere "
                                "revisar o resolver y cualquier contexto relevante "
                                "para la persona que atenderá la cita."
                            ),
                        },
                        "notes": {
                            "type": "string",
                            "description": (
                                "Notas adicionales útiles para la reunión que no "
                                "formen parte del motivo principal, por ejemplo "
                                "preferencias, condiciones especiales o información "
                                "adicional mencionada por el cliente."
                            ),
                        },
                    },
                    "required": [
                        "name",
                        "email",
                        "datetime",
                        "duration_minutes",
                        "title",
                        "description",
                    ],
                    "additionalProperties": False,
                },
                when_to_use_en=(
                    "Use this action when the caller wants to schedule, book, or "
                    "create a calendar appointment or meeting."
                ),
                when_to_use_es=(
                    "Usa esta acción cuando el cliente quiera agendar, reservar "
                    "o crear una cita o reunión en el calendario."
                ),
            ),
        ),
        test_fn="STT_server.services.integrations_tester._test_google_calendar",
        auth_type="oauth",
        oauth_label="Connect Google Calendar",
        # Default scope list intentionally left empty here — the
        # registry in oauth_providers._build_google_calendar_config
        # supplies the canonical set (openid + email + profile +
        # calendar.events). This tuple is only used as a fallback
        # when the OAuthConfig.default_scopes field is missing.
        oauth_default_scopes=(
            "openid",
            "email",
            "profile",
            "https://www.googleapis.com/auth/calendar.events",
        ),
    ),
)


def get_integration_provider_spec(provider_id: str) -> IntegrationProviderSpec | None:
    for spec in INTEGRATION_PROVIDERS:
        if spec.id == provider_id:
            return spec
    return None


def list_integration_providers() -> tuple[IntegrationProviderSpec, ...]:
    return INTEGRATION_PROVIDERS


def action_ids_for_provider(provider_id: str) -> tuple[str, ...]:
    """Returns the set of action ids valid for this provider. Empty
    tuple = provider has no fixed actions (generic_webhook); the
    operator can use any matching action id."""
    spec = get_integration_provider_spec(provider_id)
    if spec is None:
        return ()
    return tuple(a.id for a in spec.actions)


def is_valid_action(provider_id: str, action: str) -> bool:
    """True if `action` is a registered action for `provider_id`.

    For generic_webhook (empty action list) we accept any id matching
    ^[a-z0-9_]+$ — the operator picks the verb per tool and the n8n
    Switch node decides what to do with it.
    """
    if not _ACTION_ID_PATTERN.match(action or ""):
        return False
    spec = get_integration_provider_spec(provider_id)
    if spec is None:
        return False
    if not spec.actions:
        # generic_webhook: any well-formed id is fine
        return True
    return action in action_ids_for_provider(provider_id)


# ponytail: Customer Insights — capability gate, same Dynamics365Client path
def actions_for_capability(provider_id: str, capability: str) -> tuple["ActionSpec", ...]:
    spec = get_integration_provider_spec(provider_id)
    if spec is None:
        return ()
    if not capability:
        return spec.actions
    return tuple(a for a in spec.actions if getattr(a, "capability", "core") == capability)


def is_valid_action_for_capability(provider_id: str, action: str, capability: str) -> bool:
    if not is_valid_action(provider_id, action):
        return False
    spec = get_integration_provider_spec(provider_id)
    if spec is None:
        return False
    for a in spec.actions:
        if a.id == action:
            return getattr(a, "capability", "core") == (capability or "core")
    return False
