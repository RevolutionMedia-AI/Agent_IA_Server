"""Call-usage billing ledger. Backed by Postgres (db_call_usage).

The legacy calls.json ledger was the storage for the per-call
records that aggregate into the /api/usage endpoint. As of the
audit fix that moved everything to Postgres, calls.json is
orphaned: db_call_usage.backfill_from_json() reads it once at
startup to migrate any existing data, then new code never touches
it. Delete the file after one release.

Cost tiers (config.py):
  PRICE_OWN_KEY_PER_MIN     - user brought their own provider key
  PRICE_PLATFORM_KEY_PER_MIN - user used our provisioned key

A call is classified as `used_platform_keys=True` when at least one
of the providers it actually used fell back to the env-var key
(because the user hadn't stored a credential for that provider).
Otherwise the call is `used_platform_keys=False`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from STT_server import db_call_usage

log = logging.getLogger("stt_server.usage_store")


def record_call(
    *,
    user_id: Optional[str],
    agent_id: Optional[str],
    tenant_id: Optional[str],
    call_sid: Optional[str],
    started_at: Optional[float],
    ended_at: Optional[float],
    providers: dict[str, Optional[str]],
    used_platform_keys: bool,
) -> None:
    """Append one billing row. Delegates to db_call_usage.
    Best-effort: a DB failure logs and returns; the call path
    must never crash on billing.
    """
    db_call_usage.record_call(
        user_id=user_id,
        agent_id=agent_id,
        tenant_id=tenant_id,
        call_sid=call_sid,
        started_at=started_at,
        ended_at=ended_at,
        providers=providers,
        used_platform_keys=used_platform_keys,
    )


def aggregate_usage(
    user_id: str,
    *,
    agent_name_lookup: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Return totals + per-agent breakdown + recent calls for one user.
    SQL aggregation in db_call_usage (single round-trip per metric)."""
    return db_call_usage.aggregate_usage(
        user_id, agent_name_lookup=agent_name_lookup
    )


def has_user_stored_key(user_id: str, provider: str | None) -> bool:
    """True if the user has a connected api-key record for this
    provider. Reads from tools_integrations (Postgres) via db_tools."""
    if not provider or not user_id:
        return False
    from STT_server.db_tools import list_tools
    return any(
        t.get("id") == provider and t.get("connected")
        for t in list_tools(user_id)
    )
