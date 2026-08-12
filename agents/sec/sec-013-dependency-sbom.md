# sec-013-dependency-sbom

**Severity:** MEDIUM — Dependency risk.

## Scope (files)

- `requirements.txt` — split runtime from test/dev.
- `requirements.runtime.txt` (new) — runtime-only pinned.
- `requirements.lock.txt` (new) — generated lock with hashes.
- `requirements.test.txt` (new) — test/dev only.
- `AgentsAi_Frontend/package.json` — declare `npm audit` script + Axios
  removal task.
- `Dockerfile` — install from `requirements.runtime.txt` + `--require-hashes`.

## Approach (NEEDS_MEASUREMENT_FIRST)

1. Generate SBOM from the actual deployed image and the locked frontend tree.
2. Run current advisory/license scan; record results in
   `docs/security/sbom-baseline.md`.
3. Pin runtime dependencies; remove test dependencies from the runtime image.
4. Remove `axios` after build/import verification (frontend currently uses
   `fetch`).
5. Add CI step: `pip-audit` for Python, `npm audit --omit=dev` for frontend.
6. Gate dependency updates with tests; retain previous image as rollback.

## Sub-agents

- `sec-013a-sbom-generator` — `scripts/gen_sbom.sh` invoking `syft` or
  `cyclonedx-py`.
- `sec-013b-pinning` — generate `requirements.lock.txt` with `pip-compile`.
- `sec-013c-audit-ci` — `scripts/security_audit.sh` invoked in CI.
- `sec-013d-axios-removal` — verify unused; remove from `package.json`.

## Dependencies

- Deployed image with reproducible build context.

## Verification

```bash
# 1. SBOM produced
bash scripts/gen_sbom.sh > docs/security/sbom-baseline.md
test -s docs/security/sbom-baseline.md

# 2. Runtime/test split
! grep -E 'pytest|mock|httpx' requirements.runtime.txt

# 3. Lockfile hashes
pip install --require-hashes -r requirements.runtime.txt  # dry-run succeeds

# 4. Audit clean
bash scripts/security_audit.sh
```

## Acceptance

- SBOM baseline committed; scan result documented.
- Runtime requirements pinned + hashed; test deps removed from runtime image.
- `axios` removed from frontend (verified by build).
- CI runs `pip-audit` and `npm audit --omit=dev`.
