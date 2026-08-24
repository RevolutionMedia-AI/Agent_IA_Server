"""Agent Tool - Dynamic function calling integration with n8n."""

import re
from datetime import datetime, timezone
from typing import Optional
import json
import uuid


# ponytail: tool kinds. "webhook" is the legacy n8n integration; "call_transfer"
# is a platform-side action that redirects the live Twilio call to a configured
# destination (no external HTTP). The CRUD layer accepts either; the executor
# branches on this field. Defaults to "webhook" so legacy rows (no `kind`
# field) keep working unchanged.
TOOL_KIND_WEBHOOK = "webhook"
TOOL_KIND_CALL_TRANSFER = "call_transfer"
VALID_TOOL_KINDS = frozenset({TOOL_KIND_WEBHOOK, TOOL_KIND_CALL_TRANSFER})

# ponytail: cheap E.164 check for the transfer destination. Reuses the
# pattern that PROVIDER_CATALOG uses for the Twilio phone_number field so
# the operator gets the same validation in both places.
E164_PATTERN = re.compile(r"^\+[1-9]\d{6,14}$")


class AgentTool:
    """Represents a callable tool/function that an agent can invoke during a conversation.

    Tools are defined per-agent and allow the LLM to dynamically call external
    services (like n8n webhooks) when it determines the conversation requires it.

    The parameters field follows JSON Schema to be compatible with OpenAI's
    function calling specification.

    kind="call_transfer" replaces the webhook with a Twilio call redirect —
    the executor asks Twilio to <Dial> the configured destination and the
    call leaves our WebSocket. webhook_url is irrelevant for that kind;
    destination (E.164) is the only required field.
    """

    # ponytail: OpenAI's function-calling spec restricts the function
    # name to `^[a-zA-Z0-9_-]+$` — any space, slash, dot, or unicode
    # character makes the entire chat.completions request fail with a
    # 400. Operators are free to type human-readable names like
    # "Google Calendar Schedule" in the FE (the field is just a label
    # for the agent list), so we keep `name` as the display string and
    # store a separate `function_name` that is always OpenAI-compliant.
    # `function_name` is auto-derived from `name` on save if the
    # operator doesn't set one explicitly.
    @staticmethod
    def _sanitize_function_name(raw: str) -> str:
        """Map any string to `^[a-zA-Z0-9_-]+$`.

        Replaces every invalid run with a single underscore, collapses
        adjacent underscores, trims leading/trailing ones, and caps
        at 64 chars (OpenAI's documented maximum). Falls back to
        "tool" when the result would be empty (e.g. operator typed
        only spaces / punctuation).
        """
        if not raw:
            return "tool"
        # Replace invalid runs with a single underscore. The class
        # notation is the negation of the OpenAI whitelist.
        cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw))
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "tool"
        return cleaned[:64]

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        webhook_url: str = "",
        filler_phrase: str = "Let me check the system...",
        parameters: Optional[dict] = None,
        # ponytail: call_transfer support. Both default to None so legacy
        # rows (no `kind` set) deserialize cleanly as webhook tools.
        kind: Optional[str] = None,
        destination: Optional[str] = None,
        # ponytail: explicit per-agent tool assignment. Only meaningful
        # for shared tools (agent_id="__shared__"): the list of agent
        # ids that may invoke this tool. Per-agent tools are
        # implicitly available to their own agent and ignore this field.
        # Empty list = not assigned to anyone; the operator must
        # explicitly assign shared tools to give agents access.
        assignments: Optional[list] = None,
        # ponytail: OpenAI-safe function name. Auto-derived from the
        # operator-facing `name` on save so the LLM caller can use
        # the human label in the UI without breaking the chat
        # completions API. If the operator somehow produces a
        # collision (two different display names sanitising to the
        # same function_name), the second save logs a warning and
        # overwrites — the operator sees the agent list to detect.
        function_name: Optional[str] = None,
        id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        last_tested_at: Optional[str] = None,
        last_test_result: Optional[str] = None,
        last_invoked_at: Optional[str] = None,
        last_invocation_status: Optional[str] = None,
        invocation_count: int = 0,
    ):
        self.id = id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.webhook_url = webhook_url
        self.filler_phrase = filler_phrase
        self.parameters = parameters or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        # ponytail: coerce unknown kinds to "webhook" rather than
        # letting them silently route to the wrong branch. The CRUD
        # layer is the gatekeeper that should reject bad values; the
        # constructor only runs for tools we already saved.
        self.kind = kind if kind in VALID_TOOL_KINDS else TOOL_KIND_WEBHOOK
        self.destination = destination
        # ponytail: normalise assignments once at construction so the
        # runtime comparison doesn't have to defend against None /
        # non-list values on every read. De-dupe + sort isn't worth
        # the cost; agent ids are unique strings, set membership is
        # O(len) and the lists are tiny.
        if isinstance(assignments, list):
            self.assignments = [a for a in assignments if isinstance(a, str) and a]
        else:
            self.assignments = []
        # ponytail: see `_sanitize_function_name` above. The from_dict
        # path stores `function_name` on disk; this constructor is
        # also called directly from the route layer (create tool) which
        # doesn't pass the field, so we always derive when missing.
        self.function_name = (
            function_name
            if isinstance(function_name, str) and function_name
            else self._sanitize_function_name(self.name)
        )
        self.created_at = created_at or self._now_iso()
        self.updated_at = updated_at or self._now_iso()
        # ponytail: per-tool observability. None = never recorded;
        # counter defaults to 0 so legacy rows from before the field
        # existed behave like a fresh tool.
        self.last_tested_at = last_tested_at
        self.last_test_result = last_test_result
        self.last_invoked_at = last_invoked_at
        self.last_invocation_status = last_invocation_status
        self.invocation_count = int(invocation_count or 0)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "webhook_url": self.webhook_url,
            "filler_phrase": self.filler_phrase,
            "parameters": self.parameters,
            # ponytail: serialize the new fields too so a round-trip
            # through agent_tools.json (or a future Postgres tools
            # table) preserves the operator's intent.
            "kind": self.kind,
            "destination": self.destination,
            "assignments": list(self.assignments),
            "function_name": self.function_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_tested_at": self.last_tested_at,
            "last_test_result": self.last_test_result,
            "last_invoked_at": self.last_invoked_at,
            "last_invocation_status": self.last_invocation_status,
            "invocation_count": self.invocation_count,
        }

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function calling format."""
        # ponytail: a call_transfer tool has no parameters — the
        # destination is fixed on the tool row, so the LLM can't
        # influence where we route. Empty JSON Schema is the OpenAI
        # contract for "no arguments"; the model will still emit a
        # tool_call message, just with arguments={}.
        params = (
            {"type": "object", "properties": {}, "required": []}
            if self.kind == TOOL_KIND_CALL_TRANSFER
            else self.parameters
        )
        return {
            "type": "function",
            "function": {
                # ponytail: use the OpenAI-safe function_name (sanitised
                # on save), NOT the operator-facing display name. The
                # chat.completions API rejects names with spaces or
                # punctuation (`^[a-zA-Z0-9_-]+$`) — we sanitise once
                # in the constructor and round-trip via to_dict, so the
                # value is stable across the LLM/executor boundary.
                "name": self.function_name,
                "description": self.description,
                "parameters": params,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTool":
        return cls(
            id=data.get("id"),
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            webhook_url=data.get("webhook_url", ""),
            filler_phrase=data.get("filler_phrase", "Let me check the system..."),
            parameters=data.get("parameters"),
            kind=data.get("kind"),
            destination=data.get("destination"),
            # ponytail: assignments defaults to None so a legacy row
            # without the field deserialises as "no explicit assignment".
            # The runtime backfill in _load_agent_tools converts that
            # into "available to every agent the owner owns" so an
            # upgrade from the old auto-include behaviour keeps
            # working until the operator switches to explicit
            # assignment via the Assign/Unassign buttons.
            assignments=data.get("assignments"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            last_tested_at=data.get("last_tested_at"),
            last_test_result=data.get("last_test_result"),
            last_invoked_at=data.get("last_invoked_at"),
            last_invocation_status=data.get("last_invocation_status"),
            invocation_count=data.get("invocation_count", 0),
        )

    def update(self, **kwargs) -> None:
        """Update fields and set updated_at."""
        for key, value in kwargs.items():
            if hasattr(self, key) and key not in ("id", "created_at"):
                setattr(self, key, value)
        self.updated_at = self._now_iso()

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        if not self.name or not self.name.strip():
            errors.append("name is required")
        # ponytail: kind-aware validation. webhooks need a URL,
        # call_transfers need an E.164 destination. An old row
        # accidentally carrying neither URL nor destination is the
        # sign someone ran a partial migration — flag it loudly
        # rather than silently saving a tool that can never fire.
        if self.kind == TOOL_KIND_CALL_TRANSFER:
            if not self.destination or not E164_PATTERN.match(self.destination.strip()):
                errors.append("destination must be E.164 (e.g. +15071234567)")
        else:
            if not self.webhook_url or not self.webhook_url.strip():
                errors.append("webhook_url is required for webhook tools")
            else:
                try:
                    from urllib.parse import urlparse
                    result = urlparse(self.webhook_url)
                    if not all([result.scheme, result.netloc]):
                        errors.append("webhook_url must be a valid URL")
                except Exception:
                    errors.append("webhook_url must be a valid URL")
        if not isinstance(self.parameters, dict):
            errors.append("parameters must be a JSON Schema object")
        return errors


def validate_json_schema(schema: dict) -> tuple[bool, Optional[str]]:
    """Validate that a dict is a valid JSON Schema for function parameters."""
    if not isinstance(schema, dict):
        return False, "Schema must be an object"
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return False, "properties must be an object"
        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                return False, f"property '{prop_name}' must be an object"
            allowed_fields = {"type", "description", "enum", "properties", "items", "required"}
            for field in prop_def:
                if field not in allowed_fields:
                    return False, f"property '{prop_name}' has unknown field '{field}'"
    return True, None
