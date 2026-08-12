# sec-002-tenant-authorization

**Severity:** CRITICAL — Auth gap (auth != authz).

## Scope (files)

- `STT_server/STT_Server.py` — tenant routes around lines 1157-1354.
- `STT_server/domain/tenant.py` — add owner field handling.
- `STT_server/db_tenants.py` — enforce `user_id` filtering.
- `STT_server/routes/auth.py` — gate public registration behind verification.

## Approach (NEEDS_TESTS_FIRST)

1. Pass authenticated principal into every tenant handler (CurrentUser
   dependency).
2. Set `TenantConfig.user_id` at create time; persist owner column.
3. Scope every list/get/update/delete/Twilio action by `user_id`; admin role
   path requires explicit role check + audit log entry.
4. Migrate ownerless rows through a reviewed mapping; ambiguous rows are
   denied rather than assigned arbitrarily.
5. Disable public registration OR require email verification + role assignment.

## Sub-agents

- `sec-002a-tenant-ownership-migration` — schema + data migration.
- `sec-002b-admin-role-policy` — define role table + admin check helper.
- `sec-002c-registration-gate` — gate /auth/register behind verification.

## Dependencies

- Existing `require_auth` dependency; new `require_admin` helper.

## Verification

```python
# tests/test_tenant_authorization.py
def test_user_cannot_list_other_tenants():
    user_a = make_user("a@test")
    user_b = make_user("b@test")
    t = create_tenant(user_a)
    client_b = login(user_b)
    r = client_b.get("/tenants")
    assert t.id not in [x["id"] for x in r.json()]

def test_user_cannot_update_other_tenant():
    ...
def test_user_cannot_invoke_validation_other_tenant():
    ...
def test_admin_can_list_all():
    ...
```

## Acceptance

- Two-user negative tests pass for every tenant endpoint.
- Ownerless rows are migrated or rejected, never randomly claimed.
- Admin role is a separate, auditable capability.
- Public registration either disabled or behind verification.
