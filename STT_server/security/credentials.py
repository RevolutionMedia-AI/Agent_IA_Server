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

ONE KEY, NO FALLBACKS (2026-10-27)
---------------------------------
There is exactly ONE key. `CREDENTIAL_ENCRYPTION_KEY` is the only
thing that can open a stored credential, and there is no second key to
fall back to.

Why: a Fernet token carries no key id. When the key changes, a stored
token cannot tell you WHICH key wrote it — it just fails the HMAC check
and raises "InvalidToken". A key-ring (primary + retired keys) could
tolerate that, but it only helps if the retired key is still
recoverable, and in practice it is not: the operator rotates the key,
loses the old one, and every credential encrypted with it is
unreadable. The ring added a config surface and a rotation procedure
that could not actually be executed once the old key was gone.

So the rule is explicit and honest: **changing CREDENTIAL_ENCRYPTION_KEY
makes every credential encrypted with the old key unreadable, and there
is no recovery.** The fix is always the same: re-enter the credential
(Settings → API) or re-run the OAuth connect flow. The code below
reports that in the error message instead of pointing at a fallback
that does not exist.

Key FINGERPRINTS (not the keys) are what the logs print, so an
operator can confirm two environments hold the same key without either
ever showing the key itself.
"""
import os
import logging
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger("stt_server.security.credentials")

# "key" stays in the cache dict for backward compatibility with anything
# that introspects it.
_fernet_cache: dict = {"key": None, "instance": None}


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

    instance = _build_fernet(raw, "CREDENTIAL_ENCRYPTION_KEY")

    _fernet_cache["key"] = raw
    _fernet_cache["instance"] = instance
    return instance


def _describe_bad_key(raw) -> str:
    """Operator-facing diagnosis of WHY *raw* is not a Fernet key.

    "not a valid Fernet key" on its own sends the operator back to the
    docs without telling them what is actually wrong with the value they
    pasted. This names the shape of the mistake instead: a truncated
    key, a hex digest, or a placeholder copied straight out of the
    deployment docs.

    Only reports things that ACTUALLY break the key. Note that
    ``base64.urlsafe_b64decode`` silently discards characters outside the
    alphabet, so stray quotes or spaces do NOT break a key — mentioning
    them here would send the operator chasing a non-problem.

    Never echoes the key material itself.
    """
    text = raw.decode("ascii", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    hints: list[str] = []
    if len(text) != 44:
        hints.append(f"expected 44 characters, got {len(text)}")
    if len(text) == 64 and all(c in "0123456789abcdefABCDEF" for c in text):
        hints.append("this looks like a hex SHA-256 digest, not a Fernet key")
    if text.strip().lower() in {
        "your_fernet_key_here", "changeme", "change_me", "secret", "your_key_here",
    }:
        hints.append("this is a placeholder copied from the docs, not a generated key")
    return "; ".join(hints) or "the value does not decode to 32 bytes"


def _build_fernet(raw, env_name: str) -> Fernet:
    try:
        key_bytes = raw.encode("ascii") if isinstance(raw, str) else raw
        return Fernet(key_bytes)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{env_name} is not a valid Fernet key ({_describe_bad_key(raw)}). "
            f"Generate one with:  python -c \"from cryptography.fernet import Fernet; "
            f"print(Fernet.generate_key().decode())\"  "
            f"Original error: {exc}"
        ) from exc


def key_fingerprint(key_material: str | bytes) -> str:
    """Short, non-reversible id for a key. Safe to log.

    Lets an operator confirm that two environments hold the SAME key
    ("is the key in Railway the one I think it is?") without either
    ever showing the key.
    """
    import hashlib
    raw = key_material.encode("ascii") if isinstance(key_material, str) else key_material
    return hashlib.sha256(raw).hexdigest()[:12]


def primary_key_fingerprint() -> str | None:
    """Fingerprint of the key currently in use. Safe to log.

    None when CREDENTIAL_ENCRYPTION_KEY is unset — in dev that means an
    ephemeral in-process key nobody can reproduce, so there is nothing
    worth fingerprinting.
    """
    raw = os.environ.get("CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if not raw:
        return None
    return key_fingerprint(raw)


def encrypt_value(plaintext):
    """Encrypts a single string value. Returns the Fernet token (URL-safe base64)."""
    if plaintext is None or plaintext == "":
        return plaintext
    return _get_fernet().encrypt(str(plaintext).encode("utf-8")).decode("ascii")


def decrypt_value(token):
    """Decrypt a Fernet token back to the original string.

    ponytail: SEC-014 — if decrypt fails (the stored value was written
    with a DIFFERENT key than the one now configured, the token is
    corrupted, or it is plaintext) this is a real security/operational
    problem. We log the error and raise so the caller can fail closed
    rather than silently falling back to the raw token as if it were a
    credential.

    There is no second key to try: the only fix is to replace the
    stored value (re-enter the credential, or re-run the OAuth connect
    flow). The error says exactly that.

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
        plaintext = _get_fernet().decrypt(token_bytes)
        return plaintext.decode("utf-8")
    except Exception as exc:
        log.exception(
            "decrypt_value failed — the stored value could not be opened with the "
            "configured CREDENTIAL_ENCRYPTION_KEY. It was most likely encrypted "
            "with a DIFFERENT key (the key was changed), or the token is corrupted. "
            "Refusing to pass the raw value as a credential. err=%s", exc,
        )
        raise ValueError(
            "Could not decrypt this credential with the configured "
            "CREDENTIAL_ENCRYPTION_KEY. It was encrypted with a different key "
            "(the key was changed) or the stored value is corrupted. There is no "
            "fallback key, so the stored value must be replaced: re-enter the "
            "credential in Settings → API, or re-run the provider's connect flow. "
            "Original error: %s" % exc
        ) from exc


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
