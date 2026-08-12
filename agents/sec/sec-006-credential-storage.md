# sec-006-credential-storage

**Severity:** HIGH — Secret exposure (plaintext telephony credentials).

## Scope (files)

- `STT_server/db_phone_numbers.py` — `twilio_auth_token`, `sip_password`,
  `whatsapp_access_token` (lines 55-86, 89-123, 126-240).
- `STT_server/db_tenants.py` — tenant Twilio secrets (lines 194-220).
- `STT_server/domain/tenant.py` — secret mapping (105-118).
- `STT_server/routes/api.py` — list/create/update responses (797-925).
- `STT_server/security/credentials.py` — reuse the Fernet envelope.

## Approach (NEEDS_DESIGN_FIRST)

1. Introduce `EncryptedSecret` type with `version`, `ciphertext`, `key_id`.
2. Encrypt every stored telephony secret through the same Fernet envelope used
   by provider API keys; tag with `key_id` for rotation.
3. Update `_row_to_number` / tenant row mapper to never return plaintext —
   return only `present: bool` + masked preview.
4. Provide an explicit reauthenticated reveal endpoint with audit log; default
   denied in production.
5. Migrate existing plaintext rows transactionally; keep both columns for one
   release and remove plaintext after dual-read window.

## Sub-agents

- `sec-006a-encrypted-secret-type` — dataclass + (de)serialize helpers.
- `sec-006b-row-mapper-update` — return masked presence, never plaintext.
- `sec-006c-plaintext-migration` — transactional migration script.
- `sec-006d-reveal-endpoint` — reauth-gated reveal with audit log.

## Dependencies

- `sec-014-encryption-key` — needs versioned envelope + key-id.

## Verification

```python
def test_list_phone_numbers_returns_no_plaintext():
    create_phone_number(twilio_auth_token="sk-test-XXXX")
    r = client.get("/phone-numbers")
    for row in r.json():
        assert "twilio_auth_token" not in row
        assert row["has_twilio_auth_token"] is True
        assert row["twilio_auth_token_preview"].startswith("sk-t")

def test_reveal_requires_recent_auth():
    ...
```

## Acceptance

- No code path returns plaintext telephony credentials in list/create/update
  responses.
- Existing rows are migrated; ciphertext + key_id stored.
- Reveal endpoint requires reauth + audit; otherwise absent.
