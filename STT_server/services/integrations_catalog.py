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
    """
    id: str                       # "find_customer" (matches ^[a-z0-9_]+$)
    name: str                     # "Find Customer"
    description: str = ""
    parameters_schema: dict = field(default_factory=lambda: {
            "type": "object", "properties": {}, "required": [],
        })


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
    # ponytail: prompt_snippet auto-injected into agent's system prompt when this
    # integration's tool is assigned. Empty = no injection.
    prompt_snippet: str = ""


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


def _a(aid: str, name: str, desc: str = "", schema: dict | None = None) -> ActionSpec:
    """Shorthand constructor with id-format guard."""
    if not _ACTION_ID_PATTERN.match(aid):
        raise ValueError(
            f"action id '{aid}' must match ^[a-z0-9_]+$ (got disallowed chars)"
        )
    return ActionSpec(
        id=aid, name=name, description=desc,
        parameters_schema=schema or {
            "type": "object", "properties": {}, "required": [],
        },
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
        description="Salesforce REST API — leads, contacts, opportunities, cases. Connects via OAuth 2.0 Authorization Code.",
        fields=(),  # ponytail: OAuth — no operator-typed fields. The
                    # token exchange writes configuration.instance_url
                    # + credentials.{access,refresh}_token. The FE
                    # only asks for Integration Name + clicks [Connect].
        actions=(
            _a("find_contact", "Find Contact"),
            _a("create_lead", "Create Lead"),
            _a("update_opportunity", "Update Opportunity"),
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
        description="Dynamics 365 Customer Service — cases, contacts, accounts.",
        fields=(
            IntegrationFieldSpec(
                name="tenant_id", label="Tenant ID (Azure AD)", type="text",
                required=True, pattern=r"^[0-9a-f-]{36}$",
                placeholder="00000000-0000-0000-0000-000000000000",
            ),
            IntegrationFieldSpec(
                name="resource_url", label="Environment URL", type="url",
                required=True,
                pattern=r"^https://[a-zA-Z0-9-]+\.crm\d?\.dynamics\.com/?$",
            ),
            IntegrationFieldSpec(
                name="access_token", label="Access Token", type="password",
                required=True, min_length=20,
            ),
        ),
        actions=(
            _a("find_contact", "Find Contact"),
            _a("create_case", "Create Case"),
            _a("update_case", "Update Case"),
        ),
        test_fn=None,
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
    # ponytail: native Google Calendar via n8n - zero-config for the operator.
    # Webhook URL is server-managed (INTEGRATIONS_N8N_WEBHOOK_OVERRIDES__GOOGLE_CALENDAR or hardcoded fallback),
    # so the operator just clicks Add and the tool + prompt injection work automatically.
    IntegrationProviderSpec(
        id="google_calendar",
        name="Google Calendar",
        category="custom",
        description="Agenda citas en Google Calendar y envía correos vía n8n. URL fija: https://revomedia.app.n8n.cloud/webhook/agendar-cita-dinamica",
        fields=(),
        actions=(
            _a(
                "agendar_cita_dinamica",
                "Agendar Cita Dinámica",
                "Agenda una cita en Google Calendar y envía correo de confirmación.",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Nombre completo del asistente"},
                        "email": {"type": "string", "description": "Email del asistente"},
                        "datetime": {"type": "string", "description": "Fecha y hora de la cita en ISO 8601, ej: 2026-09-04T15:00:00-06:00"},
                        "duration_minutes": {"type": "integer", "description": "Duración en minutos, por defecto 30"},
                        "host_email": {"type": "string", "description": "Email del calendario destino (opcional)"},
                    },
                    "required": ["name", "email", "datetime"],
                },
            ),
        ),
        test_fn="STT_server.services.integrations_tester._test_google_calendar",
        prompt_snippet=(
            "[Google Calendar] Cuando el usuario quiera agendar, reservar o programar una cita, "
            "debes recopilar: name (nombre completo), email (correo), datetime (fecha y hora en formato ISO 8601, ej: 2026-09-04T15:00:00-06:00), "
            "duration_minutes (opcional, por defecto 30), host_email (opcional, por defecto kevin.escalante@revolutionmedia.ai). "
            "Convierte cualquier expresión como 'mañana a las 3pm' a ISO 8601. No inventes datetime: si falta, pregunta. "
            "Luego llama a la herramienta agendar_cita_dinamica con esos campos."
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