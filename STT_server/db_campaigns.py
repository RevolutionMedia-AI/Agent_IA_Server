"""Postgres-backed implementations for the global campaign catalog.

A campaign is just a string tag — Life Insurance, After-Sales,
Customer Care, etc. The agent modal and the phone-number modal both
let the user pick from the curated list OR type a new one. Anything
they type gets registered here so the next time the modal opens, the
new value shows up in the suggestions list.
"""
from __future__ import annotations

from STT_server.db import get_conn, is_postgres


# ponytail: mirror of ModalNewAgent/ModalAgents CAMPAIGN_OPTIONS so
# the local-dev fallback returns the same curated list the FE shows.
# The FE merges this with /campaigns response, so the two stay in
# sync visually but the catalog is owned by the BE once Postgres is
# reachable.
_CURATED_CAMPAIGNS = (
    'Life Insurance', 'After-Sales', 'Customer Care', 'Collections',
    'Auto Insurance', 'Inheritance', 'Dental Plan',
)


def list_campaigns(limit: int = 200) -> list[str]:
    """Returns all campaign names, alphabetically sorted. Includes the
    curated options plus anything the user has typed in any modal.

    JSON-fallback for local dev returns the curated list + whatever
    campaign strings are stored in agents.json / phone_numbers.json.
    Production (Postgres) reads the values already stored on agents and
    phone numbers.
    """
    if not is_postgres():
        from pathlib import Path
        names = set(_CURATED_CAMPAIGNS)
        data_dir = Path(__file__).resolve().parent / "data"
        for fname in ("agents.json", "phone_numbers.json"):
            p = data_dir / fname
            if p.exists():
                try:
                    import json
                    items = json.loads(p.read_text() or "[]") or []
                except Exception:
                    continue
                for x in items:
                    c = (x.get("campaign") or "").strip()
                    if c:
                        names.add(c)
        return sorted(names)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name FROM ("
                "SELECT campaign AS name FROM agents "
                "UNION SELECT campaign AS name FROM phone_numbers"
                ") campaign_names "
                "WHERE name IS NOT NULL AND btrim(name) <> '' "
                "ORDER BY lower(name) LIMIT %s",
                (limit,),
            )
            return [r["name"] for r in cur.fetchall()]
