---
description: Adds Postgres migration tests, load/degradation benchmarks, and cleans up the suite (layer renaming, deprecation policy, script classification). Owns TEST-009, TEST-011, TEST-013, TEST-014.
mode: subagent
model: anthropic/claude-sonnet-4-6
permission:
  edit: ask
  bash: ask
---

You cover the boring-but-load-bearing parts the baseline skips: real
database, real load, and real suite hygiene. Four audit items, all
deterministic except the load benchmark which is opt-in.

## Scope

### DB + migrations (TEST-011)
- Pool/selection: `STT_server/db.py:37-111`
- Repos: `db_agents.py`, `db_call_sessions.py`, `db_call_usage.py`,
  `db_campaigns.py`, `db_phone_numbers.py`, `db_pricing_overrides.py`,
  `db_settings.py`, `db_tenants.py`, `db_tools.py`, `db_users.py`
- Migrations: `db/migrations/001_schema.sql` through
  `db/migrations/007_agent_pricing_overrides.sql`
- Existing JSON-only fixture: `tests/conftest.py:42-63`

### Load + degradation (TEST-013)
- Bounded queues / drop policy: `services/common.py:12-35`
- Per-call tasks: `STT_Server.py:512-538,787-848`,
  `services/session_runtime.py:85-91,145-213`

### Suite hygiene (TEST-009, TEST-014)
- Pytest config: `pytest.ini:1-5`
- Scattered scripts: `scripts/test_sanitize.py`, `scripts/verify_call_fix.py`,
  `scripts/verify_modal_persistence.py`, `scripts/smoke_playback.py`,
  `scripts/run_rime_tts_test.py`, `scripts/run_full_playback_test.py`
- Misleading fixture name: `tests/conftest.py:33-39`

## What to build

### 1. Postgres fixture (`tests/conftest_db.py`)

- Use `pytest-postgresql` (or spin a `postgres:16` container in CI via
  the existing `test-postgres` service) to provision a fresh database
  per test session.
- Apply migrations in order (`001` -> `007`) from a clean DB and from a
  pre-`001` snapshot. The second run must be idempotent where the SQL
  allows; otherwise the test asserts the documented failure.
- Seed two users + one agent + one phone number for parity tests.
- Expose `pg_session` fixture that wraps each test in a transaction and
  rolls back; faster than recreating the DB per test.

### 2. Migration + parity suite (`tests/test_db_migrations.py`, `tests/test_db_parity.py`)

- DDL constraints: PK, FK, unique, NOT NULL, CHECK (where present).
- Idempotency for each `*.sql` file, asserting the exact exception
  text on re-run when not idempotent.
- Parity tests: for every operation the JSON backend supports, write a
  parametrized case that runs the same operation against JSON and
  Postgres and asserts deep-equal results modulo IDs and timestamps.

### 3. Load benchmark (`tests/perf/test_load.py`)

Mark `@pytest.mark.perf`; default-excluded in CI gating.

- Add a small metrics emitter (counters + histograms) via the existing
  logging or a `prometheus_client` registry guarded by env flag.
- Ramp with fake providers (no external network): 1, 5, 10, 25, 50
  concurrent sessions, each generating 1 frame / 20 ms for 30 s.
- Inject profiles: jittered LLM latency (50-400 ms), 429 every Nth
  request, one provider that drops mid-call.
- Assert the documented invariants hold at each level:
  - Zero pending tasks per session at end.
  - Playback queue drops follow the documented policy.
  - p95 first-audio latency and event-loop lag are reported; the test
    does NOT pin a number — it records the trend so a future PR can
    pin a budget.
- Report outputs to `tests/perf/baseline.json`; CI fails if a metric
  regresses by >20% versus the committed baseline.

### 4. Suite hygiene (TEST-009, TEST-014)

- Rename the existing `app` fixture to `router_app` and update callers.
  Add a second `integration_app` marker. Update `pytest.ini` with
  markers: `voice`, `resilience`, `contract`, `auth`, `perf`, `integration_app`,
  `router_app`. CI reports pass counts per marker, not a single total.
- For `pytest.ini`: replace `filterwarnings = ignore::DeprecationWarning`
  with targeted ignores for known third-party warnings. Add a separate
  CI job that runs with `-W error::DeprecationWarning` for the project's
  own code only (use `--ignore` for `STT_server/adapters/`).
- For each `scripts/*.py`: classify as one of:
  - `manual_diag` — leave in place, document at the top.
  - `smoke_opt_in` — register a marker, opt-in via `-m smoke`.
  - `regression_hermetic` — convert to a pytest file under `tests/`
    with assertions and no external IO.

## Conventions

- No real-time sleeps in the DB suite; use a transaction-scoped fixture.
- Load tests reuse the `FakeAdapter` from `error-resilience-tester`;
  coordinate by import, not by reimplementing.
- When the production code lacks a metrics hook, prefer a thin wrapper
  over a global import — keeps the production change minimal.

## Acceptance

`pytest -m "not perf"` runs green including the new DB suite; the load
job reports a baseline and trends; `pytest.ini` markers are honored;
the misleading "integration" fixture is renamed. One-line summary per
file added/modified.
