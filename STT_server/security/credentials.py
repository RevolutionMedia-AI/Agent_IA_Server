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

Generate a key with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
import os
import logging
from cryptography.fernet import Fernet

log = logging.getLogger("stt_server.security.credentials")

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

    try:
        key_bytes = raw.encode("ascii") if isinstance(raw, str) else raw
        fernet = Fernet(key_bytes)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"CREDENTIAL_ENCRYPTION_KEY is not a valid Fernet key. "
            f"Generate one with:  python -c \"from cryptography.fernet import Fernet; "
            f"print(Fernet.generate_key().decode())\"  "
            f"Original error: {exc}"
        ) from exc

    _fernet_cache["instance"] = fernet
    return fernet


def encrypt_value(plaintext):
    """Encrypts a single string value. Returns the Fernet token (URL-safe base64)."""
    if plaintext is None or plaintext == "":
        return plaintext
    return _get_fernet().encrypt(str(plaintext).encode("utf-8")).decode("ascii")


def decrypt_value(token):
    """Decrypts a Fernet token back to the original string.

    ponytail: SEC-014 — if decrypt fails (wrong key, corrupted token,
    cipher text from an old key), this is a real security/operational
    problem. We log the error and raise so the caller can fail closed
    rather than silently falling back to the raw token as if it were a
    credential.
    """
    if token is None or token == "":
        return token
    try:
        return _get_fernet().decrypt(str(token).encode("ascii")).decode("utf-8")
    except Exception as exc:
        log.exception(
            "decrypt_value failed — CREDENTIAL_ENCRYPTION_KEY may be rotated, "
            "the token may be corrupted, or the token may be plaintext. "
            "Refusing to pass the raw value as a credential. err=%s", exc,
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
    """
    if not isinstance(creds, dict):
        return {}
    return {k: decrypt_value(v) for k, v in creds.items()}
