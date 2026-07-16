"""Agent Tool - Dynamic function calling integration with n8n."""

from datetime import datetime, timezone
from typing import Optional
import json
import uuid


class AgentTool:
    """Represents a callable tool/function that an agent can invoke during a conversation.

    Tools are defined per-agent and allow the LLM to dynamically call external
    services (like n8n webhooks) when it determines the conversation requires it.

    The parameters field follows JSON Schema to be compatible with OpenAI's
    function calling specification.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        webhook_url: str,
        filler_phrase: str = "Let me check the system...",
        parameters: Optional[dict] = None,
        id: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
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
        self.created_at = created_at or self._now_iso()
        self.updated_at = updated_at or self._now_iso()

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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_openai_function(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentTool":
        return cls(
            id=data.get("id"),
            agent_id=data["agent_id"],
            name=data["name"],
            description=data["description"],
            webhook_url=data["webhook_url"],
            filler_phrase=data.get("filler_phrase", "Let me check the system..."),
            parameters=data.get("parameters"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
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
        if not self.webhook_url or not self.webhook_url.strip():
            errors.append("webhook_url is required")
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
