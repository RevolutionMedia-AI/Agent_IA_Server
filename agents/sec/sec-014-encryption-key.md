# sec-014-encryption-key

**Severity:** MEDIUM — Secret exposure.

> Status: baseline fix landed in commit `8c833ae` (fail closed when
> `CREDENTIAL_ENCRYPTION_KEY` missing in prod; raise on decrypt failure).

## Scope (files)

- `STT_server/security/credentials.py` — envelope, key-id, rotation.
- `scripts/rotate_encryption_key.sh` (new) — dual-read / new-write runbook.

## Approach (NEEDS_DESIGN_FIRST)

1. Define `EncryptedSecret = {version: int, key_id: str, ciphertext: bytes}`.
2. On encrypt, prefix with `version||key_id`. On decrypt, look up the right
   key by `key_id`; distinguish "legacy plaintext" explicitly from "failed
   authenticated decryption".
3. Production fails closed without a persistent master key. Only
   `ENVIRONMENT=development` or `ALLOW_EPHEMERAL_ENCRYPTION_KEY=1` permits
   the in-memory key.
4. Alert and fail closed on corrupt ciphertext or unknown `key_id`.
5. Rotation runbook: dual-read with old key, new-write with new key, replay
   + rollback plan.

## Sub-agents

- `sec-014a-envelope-versioning` — extend `credentials.py` envelope.
- `sec-014b-distinguish-legacy-vs-failed` — explicit exception types:
  `LegacyPlaintextSecret`, `DecryptionFailed`.
- `sec-014c-rotation-runbook` — `scripts/rotate_encryption_key.sh` plus
  `docs/security/key-rotation.md`.

## Dependencies

- `sec-006-credential-storage` reuses the same envelope.

## Verification

```python
def test_missing_master_key_in_production_fails_closed(monkeypatch):
    monkeypatch.delenv("CREDENTIAL_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ALLOW_EPHEMERAL_ENCRYPTION_KEY", raising=False)
    with pytest.raises(RuntimeError):
        init_credentials()

def test_decrypt_failure_does_not_return_ciphertext(monkeypatch):
    monkeypatch.setattr(cipher, "decrypt", lambda c: None)
    with pytest.raises(DecryptionFailed):
        decrypt_credentials("any")
    # never returns the original ciphertext
```

## Acceptance

- Production boot without a persistent key raises.
- Decrypt failure raises a typed error and never returns ciphertext.
- Rotation runbook dry-run works against a test envelope.
