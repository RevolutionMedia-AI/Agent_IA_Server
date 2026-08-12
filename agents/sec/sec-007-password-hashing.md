# sec-007-password-hashing

**Severity:** HIGH — Auth gap / Sensitive logging.

## Scope (files)

- `STT_server/routes/auth.py` — login + register (58-64, 102-184).
- `STT_server/routes/api.py` — login + change (1028-1033).

## Approach (NEEDS_TESTS_FIRST)

1. Replace unsalted SHA-256 with Argon2id (preferred) or bcrypt + per-password
   salt.
2. On login, verify with the chosen KDF; if stored hash uses legacy SHA-256,
   rehash under the new KDF transparently.
3. Remove every log line that emits the incoming password hash, stored hash,
   email-match comparison, or password length; keep only the high-water
   counter `auth.attempts{outcome}`.
4. Enforce a single password policy: minimum length (≥12), no trivial
   patterns; reject empty passwords at registration.
5. Add per-account and per-IP throttling: in-app token bucket + trusted edge
   rate-limit when available.

## Sub-agents

- `sec-007a-password-kdf` — `hash_password(plain) -> str`, `verify_password(plain, stored) -> bool`.
- `sec-007b-rehash-on-login` — upgrade legacy hashes transparently.
- `sec-007c-login-redaction` — audit + delete log lines emitting secrets.
- `sec-007d-throttle` — token bucket keyed on user_id and IP, with backoff.

## Dependencies

- `passlib[argon2]` or `bcrypt` in `requirements.txt`.

## Verification

```python
def test_login_does_not_log_password_material(caplog):
    caplog.set_level("WARNING")
    bad_login()
    assert "password" not in caplog.text.lower()
    assert "hash" not in caplog.text.lower()

def test_register_rejects_weak_passwords():
    for pw in ["", "12345678", "password", "short"]:
        r = client.post("/auth/register", json={"email": "x@y", "password": pw})
        assert r.status_code in (400, 422)

def test_legacy_hash_upgrades_on_login():
    seed_user_with_sha256(...)
    login_ok(...)
    stored = get_user_hash(...)
    assert stored.startswith("argon2$")
```

## Acceptance

- No log line emits password material under any code path.
- New KDF is the only verification mechanism.
- Legacy hashes are upgraded on successful login.
- Throttle reduces brute-force success rate in load test.
