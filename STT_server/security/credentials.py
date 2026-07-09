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
    """Returns a cached Fernet instance, generating an ephemeral key in dev."""
    if _fernet_cache["instance"] is not None:
        return _fernet_cache["instance"]

    raw = os.environ.get("CREDENTIAL_ENCRYPTION_KEY", "").strip()
    if not raw:
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
    """Decrypts a Fernet token back to the original string. Returns None on failure."""
    if token is None or token == "":
        return token
    try:
        return _get_fernet().decrypt(str(token).encode("ascii")).decode("utf-8")
    except Exception:
        # If the token is from a different master key, or is corrupted,
        # we silently return None so the caller can fall back to the env var.
        return None


def encrypt_credentials(creds):
    """Encrypts every string value in a credentials dict. Returns a new dict."""
    if not isinstance(creds, dict):
        return {}
    return {k: encrypt_value(v) for k, v in creds.items()}


def decrypt_credentials(creds):
    """Decrypts every string value in a credentials dict. Returns plaintext values.

    Values that fail to decrypt (e.g. the master key was rotated, or the
    stored token is plaintext from a pre-encryption deploy) are returned
    as-is so the caller can fall back gracefully.
    """
    if not isinstance(creds, dict):
        return {}
    out = {}
    for k, v in creds.items():
        plain = decrypt_value(v)
        out[k] = plain if plain is not None else v
    return out
