# sec-010-credential-reveal

**Severity:** MEDIUM — Secret exposure / Auth gap.

## Scope (files)

- `STT_server/routes/api.py` — `GET /settings/api-keys/{service_id}/value`
  around 1231-1249.
- `AgentsAi_Frontend/src/services/api.js` — token storage 32-54.

## Approach (NEEDS_DESIGN_FIRST)

1. Stop returning stored secrets in read responses. Return masked presence
   fields only.
2. Replace-only API: write the secret once, never re-read plaintext.
3. If a reveal endpoint is operationally necessary, gate it with:
   - recent reauthentication (within last 5 minutes),
   - short-lived purpose-bound authorization token,
   - audit log entry,
   - `Cache-Control: no-store` and `Pragma: no-cache`.
4. Move frontend token storage from `localStorage` to `HttpOnly Secure
   SameSite=Lax` cookie with CSRF protection; or keep memory-only with a
   service worker round-trip.

## Sub-agents

- `sec-010a-masked-fields-only` — list/create/update responses carry presence
  + preview only.
- `sec-010b-reveal-gate` — reauth + audit + short-lived authz.
- `sec-010c-cookie-auth-frontend` — switch storage + add CSRF token.

## Dependencies

- `sec-006-credential-storage` — masked fields already enforced there.

## Verification

```python
def test_settings_list_does_not_include_plaintext_secret():
    r = client.get("/settings/api-keys")
    for row in r.json():
        assert "value" not in row
        assert row["has_value"] is True

def test_reveal_requires_recent_auth(monkeypatch):
    monkeypatch.setattr("...", last_reauth_at=now() - timedelta(minutes=10))
    r = client.get("/settings/api-keys/openai/value")
    assert r.status_code in (401, 403)
```

## Acceptance

- No read endpoint returns plaintext stored secrets by default.
- Reveal endpoint (if kept) requires recent reauth and writes an audit event.
- Frontend does not rely on `localStorage` for the bearer token.
