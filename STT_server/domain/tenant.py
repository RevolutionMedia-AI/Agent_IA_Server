"""Multi-tenant configuration for Twilio and agent settings.

Each tenant represents a client with their own Twilio sub-account, phone
number, and call-time configuration (prompt, TTS provider, language, etc.).

Persistence is delegated to STT_server.db_tenants. The TenantStore in this
file is a thin adapter that converts between TenantConfig (the dataclass
the route layer uses) and the dict shape the persistence layer speaks.
On Postgres the DB is the source of truth; on a JSON-only deployment
db_tenants itself falls back to STT_server/data/tenants.json.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from STT_server import db_tenants

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
    tts_provider: str = "elevenlabs"  # "elevenlabs" | "deepgram" | "rime" | "inworld"
    preferred_language: str = "es"  # "en" | "es"

    # ── Provider API keys (legacy fields, NOT persisted) ──
    # ponytail: per Opcion 2, these live on tools_integrations and are
    # resolved at call time by credentials_resolver. The fields stay on
    # the dataclass so the route layer's TenantCreateRequest and
    # to_dict() keep working unchanged. db_tenants.upsert_tenant() drops
    # them on write and _from_db_row() resets them to None on read.
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    elevenlabs_voice_id: str | None = None
    deepgram_api_key: str | None = None

    # ── Metadata ──
    webhook_configured: bool = False
    # ponytail: user_id is the FK to users.id. The FE never sees it
    # (omitted from to_dict); it's server-internal so a tenant can be
    # scoped to the admin who created it.
    user_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def has_twilio_credentials(self) -> bool:
        return bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone_number)

    def to_dict(self, include_secrets: bool = False) -> dict:
        """Serialize tenant config. Secrets are masked by default.

        user_id is intentionally NOT exposed — the owning-user relationship
        is server-internal. The FE contract stays identical to the
        pre-DB version."""
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


# ── Conversion helpers (TenantConfig <-> dict) ────────────────────────

def _to_db_row(tenant: TenantConfig) -> dict:
    return {
        "tenant_id": tenant.tenant_id,
        "user_id": tenant.user_id,
        "name": tenant.name,
        "twilio_account_sid": tenant.twilio_account_sid,
        "twilio_auth_token": tenant.twilio_auth_token,
        "twilio_phone_number": tenant.twilio_phone_number,
        "custom_prompt": tenant.custom_prompt,
        "tts_provider": tenant.tts_provider,
        "preferred_language": tenant.preferred_language,
        "webhook_configured": tenant.webhook_configured,
        "created_at": _float_to_iso(tenant.created_at),
        "updated_at": _float_to_iso(tenant.updated_at),
    }


def _from_db_row(row: dict) -> TenantConfig:
    return TenantConfig(
        tenant_id=row["tenant_id"],
        user_id=row.get("user_id"),
        name=row.get("name") or "",
        twilio_account_sid=row.get("twilio_account_sid") or "",
        twilio_auth_token=row.get("twilio_auth_token") or "",
        twilio_phone_number=row.get("twilio_phone_number") or "",
        custom_prompt=row.get("custom_prompt"),
        tts_provider=row.get("tts_provider") or "elevenlabs",
        preferred_language=row.get("preferred_language") or "es",
        # Opcion 2: provider keys are not on tenants — leave None so
        # to_dict() returns the same masked shape the FE already renders.
        openai_api_key=None,
        elevenlabs_api_key=None,
        elevenlabs_voice_id=None,
        deepgram_api_key=None,
        webhook_configured=bool(row.get("webhook_configured", False)),
        created_at=_iso_to_float(row.get("created_at"), default=time.time()),
        updated_at=_iso_to_float(row.get("updated_at"), default=time.time()),
    )


def _float_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _iso_to_float(s, default: float = 0.0) -> float:
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    if isinstance(s, datetime):
        return s.timestamp()
    if isinstance(s, str):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except (ValueError, TypeError):
            return default
    return default


# ── TenantStore facade ──────────────────────────────────────────────

class TenantStore:
    """Adapter in front of db_tenants. Every method is a thin pass-through
    that converts TenantConfig <-> dict. The facade is stateless on
    Postgres (the DB is the source of truth) and on the JSON backend
    (db_tenants itself persists to data/tenants.json)."""

    def upsert(self, tenant: TenantConfig) -> None:
        db_tenants.upsert_tenant(_to_db_row(tenant))

    def get(self, tenant_id: str) -> TenantConfig | None:
        row = db_tenants.get_tenant(tenant_id)
        return _from_db_row(row) if row else None

    def get_by_phone(self, phone_number: str) -> TenantConfig | None:
        row = db_tenants.get_tenant_by_phone(phone_number)
        return _from_db_row(row) if row else None

    def delete(self, tenant_id: str) -> bool:
        return db_tenants.delete_tenant(tenant_id)

    def list_all(self) -> list[TenantConfig]:
        return [_from_db_row(r) for r in db_tenants.list_tenants()]

    def list_ids(self) -> list[str]:
        return db_tenants.list_tenant_ids()


# Global singleton
tenant_store = TenantStore()
