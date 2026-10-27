"""The test suite must never write to the operator's real data dir.

ponytail: this exists because it already happened. The suite seeded
integrations straight into STT_server/data/integrations.json and
replaced the operator's live store with test rows (ids like
"int-salesforce-test"). The store was lost.

Two layers protect it now:

  1. tests/conftest.py redirects EVERY JSON store to a per-test tmp dir,
     autouse, so a test cannot opt out by not requesting `data_dir`.
  2. the session-scoped `_guard_real_data_dir` fixture hashes the real
     dir before and after the run and fails if anything moved.

These tests cover layer 1 explicitly and prove layer 2 is armed.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

REAL_DATA_DIR = Path(__file__).resolve().parent.parent / "STT_server" / "data"

PROBE_NAME = "isolation-probe-do-not-ship"


def _real_file_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    if not REAL_DATA_DIR.exists():
        return out
    for path in sorted(REAL_DATA_DIR.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(REAL_DATA_DIR))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return out


def test_every_json_store_is_redirected(data_dir: Path, json_store_paths):
    """Each store's path constant must point INSIDE the tmp dir."""
    for module_name, attr, filename in json_store_paths:
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), (
            f"{module_name}.{attr} does not exist — the store table is "
            f"stale, so {filename} may no longer be redirected"
        )
        value = getattr(module, attr)
        assert Path(value).parent == data_dir, (
            f"{module_name}.{attr} still points at {value} — a test writing "
            f"through it would overwrite the operator's real {filename}"
        )


def test_the_routes_layer_agrees_with_the_db_layer(data_dir: Path):
    """api.py keeps its own copy of the paths. If the two layers are
    redirected inconsistently, a route writes to one file while the db
    layer reads another — which surfaces as a bogus "Agent not found"
    rather than as the obvious "a path was not patched"."""
    from STT_server import db_agents, db_phone_numbers, db_users
    from STT_server.routes import api as api_mod

    assert Path(api_mod.AGENTS_FILE) == Path(db_agents.AGENTS_FILE)
    assert Path(api_mod.USERS_FILE) == Path(db_users.USERS_FILE)
    assert Path(api_mod.SESSIONS_FILE) == Path(db_users.SESSIONS_FILE)
    assert Path(api_mod.NUMBERS_FILE) == Path(db_phone_numbers.NUMBERS_FILE)


def test_integration_store_is_redirected(data_dir: Path):
    """The store that was actually destroyed, pinned on its own so a
    future edit to the store table cannot silently drop it."""
    from STT_server import db_integrations

    assert Path(db_integrations._INTEGRATIONS_FILE).parent == data_dir
    assert db_integrations._INTEGRATIONS_FILE != REAL_DATA_DIR / "integrations.json"


def test_writing_an_integration_does_not_touch_the_real_store(data_dir: Path):
    """Seed a row the way the API does and prove the real file is
    untouched."""
    from STT_server.db_integrations import create_integration, list_integrations

    before = _real_file_hashes()

    create_integration(
        "user-test-001",
        {
            "provider": "zendesk",
            "name": PROBE_NAME,
            "agent_id": "__shared__",
            "configuration": {},
            "connection_status": "connected",
        },
    )
    rows = list_integrations("user-test-001")
    assert any(r["name"] == PROBE_NAME for r in rows), (
        "the probe row was not written at all — the tmp store is not wired up"
    )

    after = _real_file_hashes()
    changed = sorted(
        name for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    )
    assert not changed, (
        "a test write reached the operator's real data dir: "
        + ", ".join(changed)
    )

    real_file = REAL_DATA_DIR / "integrations.json"
    if real_file.exists():
        real_rows = json.loads(real_file.read_text(encoding="utf-8"))
        assert not any(r.get("name") == PROBE_NAME for r in real_rows), (
            "a test row reached the operator's real integrations.json"
        )


def test_the_real_store_is_never_a_write_target():
    """The guard in conftest is armed for the whole session: it must be
    possible to snapshot the real dir at all, and the snapshot must be
    stable, or the tripwire would report false positives and get
    ignored."""
    first = _real_file_hashes()
    second = _real_file_hashes()
    assert first == second


def test_the_encryption_key_is_pinned_for_the_session():
    """The suite must not depend on a CREDENTIAL_ENCRYPTION_KEY exported
    by the developer's shell.

    It used to: conftest used setdefault, so a shell variable silently
    won, and other test files overwrote the variable mid-session. Every
    mismatch surfaced as "Could not decrypt with any configured key",
    which reads like a data problem but is really a non-reproducible
    environment.
    """
    import os

    from STT_server.security import credentials as credentials_mod

    key = os.environ["CREDENTIAL_ENCRYPTION_KEY"]
    assert credentials_mod.primary_key_fingerprint() == (
        credentials_mod.key_fingerprint(key)
    ), (
        "the credentials module is using a different key than the one the "
        "session pinned — a test changed it without rebuilding the cache"
    )
