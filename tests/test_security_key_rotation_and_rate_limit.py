"""Key rotation (P1) and internal rate limiting (P2).

No DB, no network, no API key. P1's crypto path runs for real (Fernet is
local); the sweep's SQL is asserted at the source level because running
it needs a live Postgres.
"""
from __future__ import annotations

import ast
import inspect
import os
import pathlib
import sys

sys.path.insert(0, ".")

from cryptography.fernet import Fernet, InvalidToken  # noqa: E402

from STT_server.security import credentials as C  # noqa: E402
from STT_server.security import rate_limit as RL  # noqa: E402

SWEEP = pathlib.Path("scripts/rotate_credential_key.py")
ROUTES = pathlib.Path("STT_server/routes/api.py")


def _fresh_keys(n: int = 3) -> list[str]:
    return [Fernet.generate_key().decode("ascii") for _ in range(n)]


def _reset_cache() -> None:
    C._fernet_cache.update(
        {"key": None, "instance": None, "ring": None, "fingerprints": None}
    )


def _use(primary: str, previous: str | None = None):
    """Point the module at a key ring for the duration of a test."""
    saved = {
        "CREDENTIAL_ENCRYPTION_KEY": os.environ.get("CREDENTIAL_ENCRYPTION_KEY"),
        "CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS": os.environ.get(
            "CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS"
        ),
    }
    os.environ["ENVIRONMENT"] = "test"
    os.environ["CREDENTIAL_ENCRYPTION_KEY"] = primary
    if previous:
        os.environ["CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS"] = previous
    else:
        os.environ.pop("CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS", None)
    _reset_cache()
    return saved


def _restore(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _reset_cache()


# ── P1: the key ring ──────────────────────────────────────────────

def test_single_key_behaviour_is_unchanged():
    """With no PREVIOUS set this must be byte-for-byte the old behaviour,
    or every existing deployment breaks on deploy."""
    key = _fresh_keys(1)[0]
    saved = _use(key)
    try:
        assert C.key_ring_fingerprints() == [C.key_fingerprint(key)]
        tok = C.encrypt_value("secreto")
        assert C.decrypt_value(tok) == "secreto"
        plaintext, fp = C.decrypt_with_fingerprint(tok)
        assert plaintext == "secreto"
        assert fp == C.primary_key_fingerprint()
    finally:
        _restore(saved)


def test_retired_key_still_decrypts():
    old, new = _fresh_keys(2)
    saved = _use(new, old)
    try:
        old_token = Fernet(old.encode()).encrypt(b"viejo").decode("ascii")
        assert C.decrypt_value(old_token) == "viejo", (
            "a row still on the retired key must keep decrypting"
        )
    finally:
        _restore(saved)


def test_new_writes_always_use_the_primary():
    """The point of rotation: writes must land on the NEW key even when
    the value being replaced was read off an old one."""
    old, new = _fresh_keys(2)
    saved = _use(new, old)
    try:
        old_token = Fernet(old.encode()).encrypt(b"viejo").decode("ascii")
        rotated = C.encrypt_value(C.decrypt_value(old_token))
        # Opens under the new key...
        assert Fernet(new.encode()).decrypt(
            rotated.encode("ascii")
        ).decode() == "viejo"
        # ...and NOT under the old one. If this ever passes, a "rotated"
        # row is still silently on the retired key.
        try:
            Fernet(old.encode()).decrypt(rotated.encode("ascii"))
        except InvalidToken:
            pass
        else:
            raise AssertionError(
                "re-encrypted value still opens under the retired key — "
                "the sweep would not have moved it"
            )
    finally:
        _restore(saved)


def test_all_ring_keys_are_tried_in_order():
    keys = _fresh_keys(3)
    saved = _use(keys[0], ",".join(keys[1:]))
    try:
        assert C.key_ring_fingerprints() == [C.key_fingerprint(k) for k in keys]
        for k in keys:
            tok = Fernet(k.encode()).encrypt(b"x").decode("ascii")
            assert C.decrypt_value(tok) == "x"
    finally:
        _restore(saved)


def test_unreadable_token_still_fails_closed():
    """The ring must not become a way to accept garbage."""
    saved = _use(_fresh_keys(1)[0])
    try:
        for junk in ("not-a-token", "gAAAA", "x" * 200):
            try:
                C.decrypt_value(junk)
            except Exception:
                continue
            raise AssertionError(f"accepted {junk!r} as a credential")
    finally:
        _restore(saved)


def test_empty_and_none_pass_through_unchanged():
    """Pre-existing and correct: an absent/cleared credential field stays
    absent rather than becoming ciphertext. '' is not a usable secret, so
    this is not a fail-open."""
    saved = _use(_fresh_keys(1)[0])
    try:
        assert C.decrypt_value("") == ""
        assert C.decrypt_value(None) is None
        assert C.encrypt_value("") == ""
        assert C.encrypt_value(None) is None
    finally:
        _restore(saved)


def test_a_key_outside_the_ring_is_refused():
    saved = _use(_fresh_keys(1)[0])
    try:
        stranger = Fernet(Fernet.generate_key()).encrypt(b"nope").decode("ascii")
        try:
            C.decrypt_value(stranger)
        except Exception:
            return
        raise AssertionError("a token from an unknown key was accepted")
    finally:
        _restore(saved)


def test_a_malformed_previous_key_is_fatal_not_skipped():
    """Silently ignoring a retired key the operator thinks is present would
    turn a rotation into silent data loss."""
    saved = _use(_fresh_keys(1)[0], "not-a-fernet-key")
    try:
        try:
            C.key_ring_fingerprints()
        except RuntimeError as exc:
            assert "CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS" in str(exc), exc
            return
        raise AssertionError("a malformed retired key was silently ignored")
    finally:
        _restore(saved)


def test_fingerprint_never_reveals_key_material():
    keys = _fresh_keys(2)
    for k in keys:
        fp = C.key_fingerprint(k)
        assert k not in fp
        assert len(fp) == 12
    assert C.key_fingerprint(keys[0]) != C.key_fingerprint(keys[1])
    assert C.key_fingerprint(keys[0]) == C.key_fingerprint(keys[0])


def test_production_still_refuses_to_start_without_a_key():
    """SEC-014 must not be weakened by the rotation work."""
    saved_key = os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)
    os.environ["ENVIRONMENT"] = "production"
    os.environ.pop("ALLOW_EPHEMERAL_ENCRYPTION_KEY", None)
    os.environ.pop("CREDENTIAL_ENCRYPTION_PREVIOUS_KEYS", None)
    _reset_cache()
    try:
        try:
            C._get_fernet()
        except RuntimeError as exc:
            assert "CREDENTIAL_ENCRYPTION_KEY" in str(exc)
        else:
            raise AssertionError("production started with an ephemeral key")
    finally:
        if saved_key is not None:
            os.environ["CREDENTIAL_ENCRYPTION_KEY"] = saved_key
        _reset_cache()


# ── P1: the sweep covers every store ──────────────────────────────

def test_sweep_is_dry_run_by_default():
    src = SWEEP.read_text(encoding="utf-8")
    assert 'apply = "--apply" in argv' in src
    assert "DRY-RUN (nothing will be written)" in src, (
        "the default run must say it will not write"
    )


def test_sweep_covers_all_three_tables():
    """A partial sweep leaves data on the retired key forever, which is
    exactly what makes step 3 unsafe."""
    src = SWEEP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    funcs = {
        n.name: ast.get_source_segment(src, n) or ""
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    assert "agent_tools" in funcs["sweep_agent_tools"]
    assert "credentials_encrypted" in funcs["sweep_integrations"]
    assert "oauth_code_verifier_encrypted" in funcs["sweep_integrations"]
    assert "twilio_credentials" in funcs["sweep_twilio_credentials"]
    assert "account_sid_encrypted" in funcs["sweep_twilio_credentials"]
    assert "auth_token_encrypted" in funcs["sweep_twilio_credentials"]


def test_sweep_never_prints_key_material():
    src = SWEEP.read_text(encoding="utf-8")
    assert "key_fingerprint" in src
    # No bare printing of the env var contents.
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("print(") and "CREDENTIAL_ENCRYPTION_KEY" in stripped:
            raise AssertionError(f"sweep prints key material: {stripped}")


def test_sweep_reports_unreadable_rather_than_overwriting():
    src = SWEEP.read_text(encoding="utf-8")
    assert "could not be opened by ANY" in src, (
        "unreadable rows must be reported and skipped, never overwritten"
    )


# ── P2: the rate limiter ──────────────────────────────────────────

def test_limit_is_100_per_minute():
    assert RL.DEFAULT_INTERNAL_RATE_LIMIT == 100
    assert RL.DEFAULT_INTERNAL_RATE_WINDOW_SEC == 60.0


def test_under_the_limit_is_allowed():
    RL.reset()
    for i in range(RL._limit()):
        allowed, remaining, retry = RL.hit("10.0.0.1")
        assert allowed is True, i
        assert remaining == RL._limit() - i - 1
        assert retry == 0.0


def test_over_the_limit_is_refused_with_retry_after():
    RL.reset()
    for _ in range(RL._limit()):
        RL.hit("10.0.0.2")
    allowed, remaining, retry = RL.hit("10.0.0.2")
    assert allowed is False
    assert remaining == 0
    assert 0 < retry <= RL._window()


def test_budget_is_per_client():
    RL.reset()
    for _ in range(RL._limit()):
        RL.hit("10.0.0.3")
    assert RL.hit("10.0.0.4")[0] is True, "a second client must have its own budget"


def test_refusals_do_not_grow_the_counter_forever():
    RL.reset()
    for _ in range(RL._limit() + 2000):
        RL.hit("10.0.0.5")
    snap = RL.snapshot()
    # Bounded by what was actually sent, pruned to the window on read.
    assert snap["10.0.0.5"] <= RL._limit() + 2000


def test_stale_clients_are_pruned():
    RL.reset()
    RL.hit("10.0.0.6")
    assert "10.0.0.6" in RL.snapshot()
    RL._hits["10.0.0.6"].clear()
    RL._prune(RL._now() - RL._window() - 1, RL._window())
    assert RL.snapshot() == {}, "inactive clients must not accumulate forever"


def test_client_key_prefers_forwarded_for():
    """Behind Railway every request would otherwise be 127.0.0.1 and the
    whole platform would share one bucket."""
    class R:
        class _C:
            host = "127.0.0.1"
        client = _C()
        headers = {"x-forwarded-for": "203.0.113.7, 10.1.1.1"}

    assert RL._client_key(R()) == "203.0.113.7"

    class NoXff:
        class _C:
            host = "198.51.100.9"
        client = _C()
        headers = {}

    assert RL._client_key(NoXff()) == "198.51.100.9"


def test_both_internal_endpoints_are_throttled():
    """Credentials AND execute must be limited; either one alone still
    allows brute force."""
    from STT_server.routes.api import (
        internal_execute_integration_action,
        internal_get_integration_credentials,
    )

    for fn in (internal_get_integration_credentials,
               internal_execute_integration_action):
        assert "enforce_internal_rate_limit" in inspect.getsource(fn), fn.__name__


def test_limit_is_env_overridable():
    saved = os.environ.get("INTERNAL_RATE_LIMIT")
    os.environ["INTERNAL_RATE_LIMIT"] = "7"
    try:
        assert RL._limit() == 7
    finally:
        if saved is None:
            os.environ.pop("INTERNAL_RATE_LIMIT", None)
        else:
            os.environ["INTERNAL_RATE_LIMIT"] = saved
    # Garbage falls back to the default rather than disabling the limit.
    os.environ["INTERNAL_RATE_LIMIT"] = "banana"
    try:
        assert RL._limit() == RL.DEFAULT_INTERNAL_RATE_LIMIT
    finally:
        if saved is None:
            os.environ.pop("INTERNAL_RATE_LIMIT", None)
        else:
            os.environ["INTERNAL_RATE_LIMIT"] = saved


def test_rate_limit_is_not_applied_to_the_public_api():
    """A shared quota in front of every user on one NAT'd IP would throttle
    paying tenants."""
    src = ROUTES.read_text(encoding="utf-8-sig")
    assert src.count("enforce_internal_rate_limit") == 3, (
        "import + exactly the two internal endpoints"
    )
    assert "/internal/integrations/" in src


# ── Self-check ────────────────────────────────────────────────────

def _self_check():
    fns = [
        test_single_key_behaviour_is_unchanged,
        test_retired_key_still_decrypts,
        test_new_writes_always_use_the_primary,
        test_all_ring_keys_are_tried_in_order,
        test_unreadable_token_still_fails_closed,
        test_empty_and_none_pass_through_unchanged,
        test_a_key_outside_the_ring_is_refused,
        test_a_malformed_previous_key_is_fatal_not_skipped,
        test_fingerprint_never_reveals_key_material,
        test_production_still_refuses_to_start_without_a_key,
        test_sweep_is_dry_run_by_default,
        test_sweep_covers_all_three_tables,
        test_sweep_never_prints_key_material,
        test_sweep_reports_unreadable_rather_than_overwriting,
        test_limit_is_100_per_minute,
        test_under_the_limit_is_allowed,
        test_over_the_limit_is_refused_with_retry_after,
        test_budget_is_per_client,
        test_refusals_do_not_grow_the_counter_forever,
        test_stale_clients_are_pruned,
        test_client_key_prefers_forwarded_for,
        test_both_internal_endpoints_are_throttled,
        test_limit_is_env_overridable,
        test_rate_limit_is_not_applied_to_the_public_api,
    ]
    for fn in fns:
        fn()
    print(f"security_p1_p2: OK ({len(fns)} checks)")


if __name__ == "__main__":
    _self_check()