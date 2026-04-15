"""Multi-tenant configuration for Twilio and agent settings.

Each tenant represents a client with their own Twilio account, phone number,
and agent configuration (prompt, TTS provider, language, etc.).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from threading import Lock

log = logging.getLogger("stt_server")


@dataclass
class TenantConfig:
    """Per-tenant configuration stored on the server."""

    tenant_id: str
    name: str = ""

    # ── Twilio credentials ──
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""  # E.164 format, e.g. "+15071234567"

    # ── Agent configuration ──
    custom_prompt: str | None = None
    tts_provider: str = "elevenlabs"  # "elevenlabs" or "rime"
    preferred_language: str = "es"  # "en" or "es"

    # ── API keys (optional overrides) ──
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    deepgram_api_key: str | None = None

    # ── Metadata ──
    webhook_configured: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def has_twilio_credentials(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone_number)

    def to_dict(self, include_secrets: bool = False) -> dict:
        """Serialize tenant config. Secrets are masked by default."""
        d = {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "twilio_phone_number": self.twilio_phone_number,
            "custom_prompt": self.custom_prompt,
            "tts_provider": self.tts_provider,
            "preferred_language": self.preferred_language,
            "webhook_configured": self.webhook_configured,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_secrets:
            d.update({
                "twilio_account_sid": self.twilio_account_sid,
                "twilio_auth_token": self.twilio_auth_token,
                "openai_api_key": self.openai_api_key,
                "elevenlabs_api_key": self.elevenlabs_api_key,
                "elevenlabs_voice_id": self.elevenlabs_voice_id,
                "deepgram_api_key": self.deepgram_api_key,
            })
        else:
            d.update({
                "twilio_account_sid": (self.twilio_account_sid[:8] + "...") if self.twilio_account_sid else "",
                "twilio_auth_token": "***" if self.twilio_auth_token else "",
                "openai_api_key": (self.openai_api_key[:8] + "...") if self.openai_api_key else None,
                "elevenlabs_api_key": (self.elevenlabs_api_key[:8] + "...") if self.elevenlabs_api_key else None,
                "elevenlabs_voice_id": self.elevenlabs_voice_id,
                "deepgram_api_key": (self.deepgram_api_key[:8] + "...") if self.deepgram_api_key else None,
            })
        return d


class TenantStore:
    """Thread-safe in-memory store for tenant configurations.

    In production, this should be backed by a database. For now, it's
    an in-memory dict that persists for the lifetime of the server.
    """

    def __init__(self) -> None:
        self._tenants: dict[str, TenantConfig] = {}
        self._phone_index: dict[str, str] = {}  # phone_number -> tenant_id
        self._lock = Lock()

    def upsert(self, tenant: TenantConfig) -> None:
        with self._lock:
            self._tenants[tenant.tenant_id] = tenant
            if tenant.twilio_phone_number:
                self._phone_index[tenant.twilio_phone_number] = tenant.tenant_id

    def get(self, tenant_id: str) -> TenantConfig | None:
        with self._lock:
            return self._tenants.get(tenant_id)

    def get_by_phone(self, phone_number: str) -> TenantConfig | None:
        """Look up a tenant by their Twilio phone number."""
        with self._lock:
            tid = self._phone_index.get(phone_number)
            if tid:
                return self._tenants.get(tid)
            return None

    def delete(self, tenant_id: str) -> bool:
        with self._lock:
            tenant = self._tenants.pop(tenant_id, None)
            if tenant and tenant.twilio_phone_number:
                self._phone_index.pop(tenant.twilio_phone_number, None)
            return tenant is not None

    def list_all(self) -> list[TenantConfig]:
        with self._lock:
            return list(self._tenants.values())

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._tenants.keys())


# Global singleton
tenant_store = TenantStore()