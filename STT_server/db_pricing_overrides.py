"""Per-agent per-model pricing overrides (db/migration 007).

Used when the public MODEL_PRICING doesn't have a row for the agent's
(service, provider, model, tier) tuple — typically Enterprise / custom
contracts. The runtime cost summary merges these with the catalog at
resolve time.

For now this is a thin CRUD layer. The runtime resolver that merges
overrides + catalog lives in services/pricing_resolver.py once we
plumb it (today the FE merges locally with VOICE_TO_MODEL + plan).
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from STT_server.db import get_conn

log = logging.getLogger("stt_server.db_pricing_overrides")


def _row_to_override(row: dict) -> dict:
    if row is None:
        return None
    out = dict(row)
    if hasattr(out.get("updated_at"), "isoformat"):
        out["updated_at"] = out["updated_at"].isoformat() + "Z"
    return out


def list_overrides(agent_id: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_id, service, provider, model_id, unit,
                       price, input_price, output_price, source, updated_at
                FROM agent_pricing_overrides
                WHERE agent_id = %s
                ORDER BY service, provider, model_id
                """,
                (agent_id,),
            )
            return [_row_to_override(r) for r in cur.fetchall()]


def upsert_override(agent_id: str, service: str, provider: str, model_id: str, payload: dict) -> dict:
    unit = payload.get("unit") or "minute"
    price = payload.get("price")
    input_price = payload.get("input_price")
    output_price = payload.get("output_price")
    source = payload.get("source") or "manual"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_pricing_overrides
                  (agent_id, service, provider, model_id, unit,
                   price, input_price, output_price, source, updated_at)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (agent_id, service, provider, model_id) DO UPDATE SET
                  unit         = EXCLUDED.unit,
                  price        = EXCLUDED.price,
                  input_price  = EXCLUDED.input_price,
                  output_price = EXCLUDED.output_price,
                  source       = EXCLUDED.source,
                  updated_at   = NOW()
                RETURNING agent_id, service, provider, model_id, unit,
                          price, input_price, output_price, source, updated_at
                """,
                (agent_id, service, provider, model_id, unit,
                 price, input_price, output_price, source),
            )
            row = cur.fetchone()
    return _row_to_override(row)


def delete_override(agent_id: str, service: str, provider: str, model_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_pricing_overrides "
                "WHERE agent_id = %s AND service = %s AND provider = %s AND model_id = %s",
                (agent_id, service, provider, model_id),
            )
            return cur.rowcount > 0
