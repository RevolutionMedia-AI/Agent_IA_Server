#!/usr/bin/env python3
"""Re-encrypt credential ciphertexts onto the PRIMARY encryption key.

Part of the CREDENTIAL_ENCRYPTION_KEY rotation procedure. See the module
docstring in STT_server/security/credentials.py for the three-step
rotation; this is step 2.

    1. Set CREDENTIAL_ENCRYPTION_KEY=<new>
       and CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS=<old>   -> deploy
    2. python scripts/rotate_credential_key.py           -> this, dry-run
       python scripts/rotate_credential_key.py --apply
    3. drop CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS          -> deploy

Why a sweep instead of decrypt-on-read-and-rewrite: a row only gets
rewritten when something touches it, so a credential nobody has used
since the rotation would sit on the retired key forever and keep that key
load-bearing. The sweep is what lets step 3 actually be safe.

Safety properties:
  * Dry-run by DEFAULT. Nothing is written without --apply.
  * A row whose ciphertext no key can open is REPORTED and SKIPPED, never
    overwritten. Losing a credential silently is worse than leaving it.
  * A row already on the primary is not rewritten (no pointless churn,
    and no needless re-encryption of live tokens).
  * Only key FINGERPRINTS are ever printed, never key material and never
    a decrypted secret.

Covers every Fernet store in the project:
  agent_tools.credentials                     (per-user provider keys)
  integrations.credentials_encrypted          (OAuth tokens + n8n token)
  integrations.oauth_code_verifier_encrypted  (PKCE verifier, 600 s TTL)
  twilio_credentials.*_encrypted              (account SID / auth token)
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from STT_server.security.credentials import (  # noqa: E402
    decrypt_value,
    encrypt_value,
    key_fingerprint,
    key_ring_fingerprints,
    primary_key_fingerprint,
)


def _fp_of(value: str) -> str | None:
    """Fingerprint of the key that opens *value*, or None if unreadable."""
    try:
        from STT_server.security.credentials import decrypt_with_fingerprint
        _plain, fp = decrypt_with_fingerprint(value)
        return fp
    except Exception:
        return None


def _reencode_pair(row: dict, columns: list[str]) -> tuple[dict, list[str]]:
    """Re-encrypt each named column's value in place.

    Returns (changed_fields, unreadable_fields). Columns that are NULL,
    empty, not a Fernet token, or already on the primary are left alone.
    """
    changed: list[str] = []
    unreadable: list[str] = []
    primary = primary_key_fingerprint()
    for col in columns:
        val = row.get(col)
        if not val or not isinstance(val, str):
            continue
        fp = _fp_of(val)
        if fp is None:
            unreadable.append(col)
            continue
        if fp == primary:
            continue
        try:
            row[col] = encrypt_value(decrypt_value(val))
            changed.append(col)
        except Exception:
            unreadable.append(col)
    return changed, unreadable


def sweep_agent_tools(conn, apply: bool) -> tuple[int, int, int]:
    """Per-user provider credentials (OpenAI / Twilio / TTS keys)."""
    changed = unreadable = scanned = 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, credentials FROM agent_tools "
            "WHERE credentials IS NOT NULL"
        )
        rows = cur.fetchall()
    for id_, user_id, creds in rows:
        scanned += 1
        if not isinstance(creds, dict):
            continue
        out = {}
        row_changed = False
        for k, v in creds.items():
            if not isinstance(v, str) or not v:
                out[k] = v
                continue
            fp = _fp_of(v)
            if fp is None:
                unreadable += 1
                out[k] = v
                continue
            if fp == primary_key_fingerprint():
                out[k] = v
                continue
            out[k] = encrypt_value(decrypt_value(v))
            row_changed = True
            changed += 1
        if row_changed and apply:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE agent_tools SET credentials = %s::jsonb "
                    "WHERE id = %s AND user_id = %s",
                    (json.dumps(out), id_, user_id),
                )
    return scanned, changed, unreadable


def sweep_integrations(conn, apply: bool) -> tuple[int, int, int]:
    """OAuth access/refresh tokens, the per-integration n8n token, and the
    PKCE verifier.

    oauth_code_verifier_encrypted has a 600 s TTL and is normally NULL;
    it is included so a rotation that lands mid-flow does not strand an
    in-flight /oauth/start.
    """
    changed = unreadable = scanned = 0
    cols = ["credentials_encrypted", "oauth_code_verifier_encrypted"]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, user_id, credentials_encrypted, "
            "oauth_code_verifier_encrypted FROM integrations"
        )
        rows = cur.fetchall()

    for id_, user_id, blob, verifier in rows:
        if blob is None and verifier is None:
            continue
        scanned += 1
        updates: dict[str, object] = {}

        if isinstance(blob, memoryview):
            blob = blob.tobytes()
        if isinstance(blob, (bytes, bytearray)):
            try:
                creds = json.loads(bytes(blob).decode("utf-8"))
            except Exception:
                creds = None
            if isinstance(creds, dict):
                out = {}
                row_changed = False
                for k, v in creds.items():
                    if not isinstance(v, str) or not v:
                        out[k] = v
                        continue
                    fp = _fp_of(v)
                    if fp is None:
                        unreadable += 1
                        out[k] = v
                        continue
                    if fp == primary_key_fingerprint():
                        out[k] = v
                        continue
                    out[k] = encrypt_value(decrypt_value(v))
                    row_changed = True
                    changed += 1
                if row_changed:
                    from cryptography.fernet import Fernet  # noqa: F401
                    updates["credentials_encrypted"] = json.dumps(out).encode("utf-8")
            else:
                unreadable += 1

        if isinstance(verifier, (bytes, bytearray)):
            text = bytes(verifier).decode("ascii", "ignore")
            fp = _fp_of(text)
            if fp is None:
                unreadable += 1
            elif fp != primary_key_fingerprint():
                updates["oauth_code_verifier_encrypted"] = encrypt_value(
                    decrypt_value(text)
                ).encode("ascii")
                changed += 1

        if updates and apply:
            sets = ", ".join(f"{c} = %s" for c in updates)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE integrations SET {sets} WHERE id = %s AND user_id = %s",
                    (*updates.values(), id_, user_id),
                )
    return scanned, changed, unreadable


def sweep_twilio_credentials(conn, apply: bool) -> tuple[int, int, int]:
    """Per-value Fernet ciphertexts (no envelope), so handled per column."""
    changed = unreadable = scanned = 0
    cols = ["account_sid_encrypted", "auth_token_encrypted"]
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, {', '.join(cols)} FROM twilio_credentials"
        )
        rows = cur.fetchall()
    for row in rows:
        row = dict(zip(["id"] + cols, row))
        scanned += 1
        ch, un = _reencode_pair(row, cols)
        changed += len(ch)
        unreadable += len(un)
        if ch and apply:
            sets = ", ".join(f"{c} = %s" for c in ch)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE twilio_credentials SET {sets} WHERE id = %s",
                    (*[row[c] for c in ch], row["id"]),
                )
    return scanned, changed, unreadable


SWEEPS = (
    ("agent_tools.credentials", sweep_agent_tools),
    ("integrations", sweep_integrations),
    ("twilio_credentials", sweep_twilio_credentials),
)


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    from STT_server.db import get_conn, is_postgres

    if not is_postgres():
        print("DATABASE_URL is not set; this sweep only runs against Postgres.")
        return 2

    ring = key_ring_fingerprints()
    primary = ring[0] if ring else None
    print("key ring (primary first):", ", ".join(ring) or "<none>")
    if len(ring) < 2:
        print(
            "\nWARNING: only one key is configured, so nothing can be "
            "re-encrypted.\n"
            "Set CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS=<old key> alongside the\n"
            "new CREDENTIAL_ENCRYPTION_KEY, otherwise this script has no\n"
            "work to do."
        )
    mode = "APPLY" if apply else "DRY-RUN (nothing will be written)"
    print(f"mode: {mode}\n")

    total_changed = total_unreadable = 0
    with get_conn() as conn:
        for label, fn in SWEEPS:
            try:
                scanned, changed, unreadable = fn(conn, apply)
            except Exception as exc:
                print(f"  {label:32} SKIPPED ({type(exc).__name__}: {exc})")
                continue
            total_changed += changed
            total_unreadable += unreadable
            print(
                f"  {label:32} rows={scanned:<6} re-encrypted={changed:<6} "
                f"unreadable={unreadable}"
            )
        if apply:
            conn.commit()

    print(f"\ntotal re-encrypted: {total_changed}")
    if total_unreadable:
        print(
            f"WARNING: {total_unreadable} value(s) could not be opened by ANY "
            "key. They were left untouched.\n"
            "Either a key is missing from CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS "
            "or the row is not Fernet ciphertext.\n"
            "Do NOT drop CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS until this is 0."
        )
    if apply and total_changed == 0:
        print("\nnothing to do — every row is already on the primary key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))