# sec-001-bootstrap-credential

**Severity:** CRITICAL — Secret exposure / Auth gap.

## Scope (files)

- `Agent_IA_Server/start.sh` — remove plaintext password, drop insert block.
- `Agent_IA_Server/db/migrations/002_seed_admin.sql` — drop migration from
  default path; require an explicit `seed.sql` provided by operator.
- `Agent_IA_Server/data/users.json` — never commit real users; move to gitignore
  + runtime mount or generate on first boot.
- `Agent_IA_Server/STT_server/routes/auth.py` — remove or strictly gate the
  public diagnostic that hashes the bootstrap password.

## Approach (SAFE_CHANGE)

1. Replace the literal password in `start.sh` with a read from
   `ADMIN_BOOTSTRAP_PASSWORD` env var; abort startup if unset in non-dev.
2. Delete `002_seed_admin.sql` from the default migration chain and add an
   operator-supplied `seed_admin.sql.template` instead.
3. Delete the public diagnostic that reveals the hash of the known password
   (and the endpoint itself if not otherwise used).
4. Move `data/users.json` to runtime path; add a `.gitignore` entry.

## Sub-agents

- (none)

## Dependencies

- A documented runbook for rotating the deployed admin credential and revoking
  existing sessions.

## Verification

```bash
# 1. no literal password in source
! grep -E 'admin.*pass|Admin123|administrator' start.sh STT_server/routes/auth.py

# 2. seed migration removed from default chain
! test -f Agent_IA_Server/db/migrations/002_seed_admin.sql

# 3. diagnostic gone or guarded
! grep -E 'admin_hash|sha256.*admin' STT_server/routes/auth.py

# 4. fresh start without ADMIN_BOOTSTRAP_PASSWORD in prod exits non-zero
ENVIRONMENT=production ADMIN_BOOTSTRAP_PASSWORD= ./start.sh ; test $? -ne 0
```

## Acceptance

- Repository contains zero references to the known password literal.
- Diagnostic endpoint is removed or returns 404 unless caller is admin.
- Default migration does not seed a known account.
- New `pytest tests/test_auth_admin.py::test_no_default_password` passes.
