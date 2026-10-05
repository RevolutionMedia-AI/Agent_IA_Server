"""
Fernet-based encryption for per-user provider credentials stored at rest.

Each user enters their own OpenAI / Twilio / ElevenLabs / etc. keys
through the Settings → API UI. Those values live in
`STT_server/data/tools_integrations.json` and must be encrypted
on disk so a leaked JSON file doesn't leak every user's keys.

The master key comes from the `CREDENTIAL_ENCRYPTION_KEY` env var —
base64-encoded 32-byte URL-safe key as returned by
`Fernet.generate_key()`. In dev, if the env var is missing, an
ephemeral key is generated (data won't survive a restart).

KEY ROTATION (2026-10-02)
-------------------------
`CREDENTIAL_ENCRYPTION_KEY` is the PRIMARY: every new encryption uses it.
`CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS` is an optional comma-separated list
of retired keys that are still accepted for DECRYPTION only. Decryption
tries the primary first, then each previous key in order, and raises only
when none of them can open the token.

That makes rotation a three-step operation with no migration and no
downtime, because a Fernet token carries no key id and decrypting with
the wrong key fails fast on the HMAC check:

    1. Set CREDENTIAL_ENCRYPTION_KEY=<new>
       and CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS=<old>
       Deploy. Reads keep working (old key is in the ring) and every new
       write uses the new key.
    2. Run `python scripts/rotate_credential_key.py` to re-encrypt the
       rows still on the old key. Dry-run by default; pass --apply.
    3. Once the sweep reports zero remaining, drop
       CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS on the next deploy.

Skipping step 2 is safe but leaves data on the retired key, so it cannot
be deleted while rows still reference it.

Backward compatible: with CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS unset the
behaviour is byte-for-byte what it was before this change.

Key FINGERPRINTS (not the keys) are what the logs and the sweep report,
so an operator can tell which key a row is on without ever seeing one.
"""
import os
import logging
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger("stt_server.security.credentials")

# "key" stays in the cache dict for backward compatibility with anything
# that introspects it; "ring" holds [Fernet, ...] with the primary first.
_fernet_cache: dict = {"key": None, "instance": None, "ring": None,
                       "fingerprints": None}


def _get_fernet() -> Fernet:
    """Returns a cached Fernet instance. Fails closed in production when the
    master key env var is missing unless the operator explicitly opts into
    dev mode via ENVIRONMENT in {development, dev, local, test} or the
    ALLOW_EPHEMERAL_ENCRYPTION_KEY opt-in."""
    if _fernet_cache["instance"] is not None:
        return _fernet_cache["instance"]

    raw = os.environ.get("CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if not raw:
        # ponytail: SEC-014 — fail closed in production; only allow
        # ephemeral keys when the operator explicitly opts into dev mode.
        # This prevents silent data loss after a restart and forces
        # the operator to set up persistent encryption before going live.
        allow_ephemeral = os.environ.get("ALLOW_EPHEMERAL_ENCRYPTION_KEY", "").strip().lower() in {"1", "true", "yes", "on"}
        env_label = os.environ.get("ENVIRONMENT", "production").strip().lower()
        is_dev = env_label in {"development", "dev", "local", "test"} or allow_ephemeral
        if not is_dev:
            raise RuntimeError(
                "CREDENTIAL_ENCRYPTION_KEY is not set. Refusing to start with an ephemeral key "
                "because encrypted credentials would be unreadable after a restart. Generate one with: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
                "and set it as CREDENTIAL_ENCRYPTION_KEY in the deployment environment. "
                "To override for local development only, set ENVIRONMENT=development or ALLOW_EPHEMERAL_ENCRYPTION_KEY=true."
            )
        log.warning(
            "CREDENTIAL_ENCRYPTION_KEY is not set — generating an ephemeral key for this process. "
            "All encrypted credentials saved during this session will be unreadable after a "
            "restart. Set CREDENTIAL_ENCRYPTION_KEY in Railway to a Fernet-generated key."
        )
        raw = Fernet.generate_key().decode("ascii")

    ring: list[tuple[Fernet, str]] = [
        (_build_fernet(raw, "CREDENTIAL_ENCRYPTION_KEY"),
         key_fingerprint(raw)),
    ]

    # Retired keys, accepted for DECRYPTION ONLY. A malformed entry is
    # fatal rather than skipped: silently ignoring a key the operator
    # believes is available would turn a rotation into data loss.
    previous_raw = os.environ.get("CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS", "").strip()
    if previous_raw:
        for idx, candidate in enumerate(previous_raw.split(",")):
            candidate = candidate.strip()
            if not candidate:
                continue
            ring.append((
                _build_fernet(
                    candidate, f"CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS[{idx}]",
                ),
                key_fingerprint(candidate),
            ))

    _fernet_cache["ring"] = ring
    _fernet_cache["fingerprints"] = [fp for _f, fp in ring]
    _fernet_cache["key"] = raw
    _fernet_cache["instance"] = ring[0][0]
    if len(ring) > 1:
        log.info(
            "credential key ring loaded: %d key(s) [%s] — decryption accepts "
            "retired keys, all writes use the primary",
            len(ring), ",".join(_fernet_cache["fingerprints"]),
        )
    return ring[0][0]


def _build_fernet(raw, env_name: str) -> Fernet:
    try:
        key_bytes = raw.encode("ascii") if isinstance(raw, str) else raw
        return Fernet(key_bytes)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{env_name} is not a valid Fernet key. "
            f"Generate one with:  python -c \"from cryptography.fernet import Fernet; "
            f"print(Fernet.generate_key().decode())\"  "
            f"Original error: {exc}"
        ) from exc


def key_fingerprint(key_material: str | bytes) -> str:
    """Short, non-reversible id for a key. Safe to log.

    Lets an operator see WHICH key a row is encrypted under during a
    rotation sweep without the log ever containing key material.
    """
    import hashlib
    raw = key_material.encode("ascii") if isinstance(key_material, str) else key_material
    return hashlib.sha256(raw).hexdigest()[:12]


def primary_key_fingerprint() -> str | None:
    _get_fernet()
    return (_fernet_cache.get("fingerprints") or [None])[0]


def key_ring_fingerprints() -> list[str]:
    """Every key currently accepted for decryption, primary first."""
    _get_fernet()
    return list(_fernet_cache.get("fingerprints") or [])


def encrypt_value(plaintext):
    """Encrypts a single string value. Returns the Fernet token (URL-safe base64)."""
    if plaintext is None or plaintext == "":
        return plaintext
    # Always the PRIMARY. A value must be written under the key we intend
    # to retire last, never under whichever key happened to decrypt it.
    return _get_fernet().encrypt(str(plaintext).encode("utf-8")).decode("ascii")


def decrypt_with_fingerprint(token: str) -> tuple[str, str]:
    """Decrypt *token* trying each key in the ring.

    Returns (plaintext, fingerprint_of_the_key_that_opened_it). Raises when
    no key can, so callers can fail closed AND report which key is
    missing instead of guessing.

    A Fernet token has no key id, so this tries each candidate. That is
    cheap: Fernet verifies the HMAC before touching the ciphertext, so a
    wrong key fails in microseconds and leaks nothing about the payload.
    """
    ring = _get_key_ring()
    last_error: Exception | None = None
    for fernet, fp in ring:
        try:
            data = fernet.decrypt(
                token.encode("utf-8") if isinstance(token, str) else token
            )
            return data.decode("utf-8"), fp
        except (InvalidToken, ValueError, TypeError) as exc:
            last_error = exc
            continue
    raise ValueError(
        "Could not decrypt with any configured key "
        f"(tried {len(ring)}: {','.join(fp for _f, fp in ring)}). If the "
        "encryption key was rotated, add the retired one to "
        "CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS. Original error: %s" % last_error
    )


def _get_key_ring() -> list[tuple[Fernet, str]]:
    _get_fernet()
    return list(_fernet_cache.get("ring") or [])


def decrypt_value(token):
    """Decrypts a Fernet token back to the original string.

    ponytail: SEC-014 — if decrypt fails (wrong key, corrupted token,
    cipher text from an old key), this is a real security/operational
    problem. We log the error and raise so the caller can fail closed
    rather than silently falling back to the raw token as if it were a
    credential.

    ponyy: BYTEA columns (oauth_code_verifier_encrypted,
    credentials_encrypted) come back as `memoryview`/`bytes` from
    psycopg2, not `str`. The old `str(token).encode("ascii")` path
    turned `b'gAAAA...'` into `b"b'gAAAA...'"` (note the extra
    `b'` prefix), which always fails with `Incorrect padding`.
    Handle `bytes`/`memoryview` by decoding directly.
    """
    if token is None or token == "":
        return token
    try:
        if isinstance(token, memoryview):
            token = token.tobytes()
        if isinstance(token, bytes):
            token_bytes = token
        else:
            token_bytes = str(token).encode("ascii")
        # ponytail: 2026-10-02 — try the whole key ring, not just the
        # primary. This is what makes CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS
        # work: rows still encrypted under a retired key keep decrypting
        # until the sweep re-encrypts them. Fail-closed is unchanged —
        # this still raises rather than passing the raw value through.
        plaintext, fp = decrypt_with_fingerprint(token_bytes)
        if fp != primary_key_fingerprint():
            log.info(
                "decrypt_value used a RETIRED key (%s); run "
                "scripts/rotate_credential_key.py to move this row onto the "
                "primary key",
                fp,
            )
        return plaintext
    except Exception as exc:
        log.exception(
            "decrypt_value failed — no key in the ring could open the token. "
            "CREDENTIAL_ENCRYPTION_KEY may have been rotated without adding "
            "the retired key to CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS, the "
            "token may be corrupted, or it may be plaintext. Refusing to "
            "pass the raw value as a credential. err=%s", exc,
        )
        raise


def encrypt_credentials(creds):
    """Encrypts every string value in a credentials dict. Returns a new dict."""
    if not isinstance(creds, dict):
        return {}
    return {k: encrypt_value(v) for k, v in creds.items()}


def decrypt_credentials(creds):
    """Decrypts every string value in a credentials dict. Returns plaintext values.

    ponytail: SEC-014 — decrypt_value now raises on failure, so this
    propagates the error instead of silently passing the raw stored
    value through as a credential (fail-open).

    ponytail: BYTEA columns come back as `memoryview`/`bytes` from
    psycopg2, not `dict`. The write path stores
    `Binary(json.dumps(encrypted_dict).encode('utf-8'))` for BYTEA,
    so on read we get the JSON string bytes. Handle that by
    decoding + json.loads before decrypting each value. The JSON-file
    fallback already stores the dict directly, so both shapes are
    handled.
    """
    # Handle BYTEA read: bytes/memoryview containing JSON string of the encrypted dict.
    if isinstance(creds, (bytes, bytearray, memoryview)):
        try:
            if isinstance(creds, memoryview):
                creds = creds.tobytes()
            s = creds.decode("utf-8") if isinstance(creds, (bytes, bytearray)) else str(creds)
            creds = __import__("json").loads(s)
        except Exception:
            return {}
    if not isinstance(creds, dict):
        return {}
    return {k: decrypt_value(v) for k, v in creds.items()}
