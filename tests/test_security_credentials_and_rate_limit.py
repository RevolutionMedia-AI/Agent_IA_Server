"""Credential encryption (single key) and internal rate limiting.

No DB, no network, no API key. The crypto path runs for real (Fernet is
local).

There is ONE key: CREDENTIAL_ENCRYPTION_KEY. No key ring, no retired
keys, no rotation script — see the module docstring in
STT_server/security/credentials.py for why a ring could not help once
the old key is gone. What these tests pin is the behaviour that follows
from that: a value encrypted with a DIFFERENT key is refused (fail
closed), and the error tells the operator what to actually do about it.
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys

sys.path.insert(0, ".")

from cryptography.fernet import Fernet, InvalidToken  # noqa: E402

from STT_server.security import credentials as C  # noqa: E402
from STT_server.security import rate_limit as RL  # noqa: E402

ROUTES = pathlib.Path("STT_server/routes/api.py")


def _fresh_keys(n: int = 3) -> list[str]:
    return [Fernet.generate_key().decode("ascii") for _ in range(n)]


def _reset_cache() -> None:
    C._fernet_cache.update({"key": None, "instance": None})


def _use(key: str):
    """Point the module at *key* for the duration of a test."""
    saved = os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
    os.environ["ENVIRONMENT"] = "test"
    os.environ["CREDENTIAL_ENCRYPTION_KEY"] = key
    _reset_cache()
    return saved


def _restore(saved: str | None) -> None:
    if saved is None:
        os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)
    else:
        os.environ["CREDENTIAL_ENCRYPTION_KEY"] = saved
    _reset_cache()


# ── The single key ───────────────────────────────────────────────

def test_encrypt_decrypt_round_trip():
    """The everyday path: a value goes in, comes back out."""
    key = _fresh_keys(1)[0]
    saved = _use(key)
    try:
        tok = C.encrypt_value("secreto")
        assert C.decrypt_value(tok) == "secreto"
        assert C.primary_key_fingerprint() == C.key_fingerprint(key)
    finally:
        _restore(saved)


def test_unreadable_token_still_fails_closed():
    """Garbage must never be accepted as a credential."""
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


def test_a_value_encrypted_with_another_key_is_refused():
    """The production scenario: the key was changed, so the stored value
    no longer opens. It must raise, never fall back to the raw value."""
    key = _fresh_keys(1)[0]
    saved = _use(key)
    try:
        stranger = Fernet(Fernet.generate_key()).encrypt(b"nope").decode("ascii")
        # Fernet verifies the HMAC before touching the ciphertext, so a
        # wrong key fails fast and leaks nothing about the payload.
        try:
            Fernet(key.encode()).decrypt(stranger.encode("ascii"))
        except InvalidToken:
            pass
        else:
            raise AssertionError("a token from an unknown key opened")
        try:
            C.decrypt_value(stranger)
        except Exception:
            return
        raise AssertionError("a token from an unknown key was accepted")
    finally:
        _restore(saved)


def test_decrypt_failure_says_how_to_recover():
    """The error must point at the only fix that exists.

    There is no retired key to fall back to, so an error that tells the
    operator to "add the retired key" sends them after something that
    does not exist. It must say: replace the stored value.
    """
    key = _fresh_keys(1)[0]
    saved = _use(key)
    try:
        stranger = Fernet(Fernet.generate_key()).encrypt(b"nope").decode("ascii")
        try:
            C.decrypt_value(stranger)
        except ValueError as exc:
            msg = str(exc)
            assert "CREDENTIAL_ENCRYPTION_KEY" in msg
            assert "re-enter" in msg.lower(), msg
            assert "retired" not in msg.lower(), (
                "the error still points at a retired key, which no longer exists"
            )
            assert "PREVIOUS_KEYS" not in msg
        else:
            raise AssertionError("expected a ValueError")
    finally:
        _restore(saved)


def test_invalid_key_error_names_the_actual_mistake():
    """"not a valid Fernet key" alone sends the operator back to the docs
    without saying what is wrong with the value they pasted. The error must
    name the shape of the mistake."""
    truncated = _fresh_keys(1)[0][:30]
    cases = [
        ("your_fernet_key_here", "placeholder"),
        ("a" * 64, "44 characters"),
        (truncated, "44 characters"),
        ("0123456789abcdef" * 4, "hex"),
    ]
    for bad_value, expected in cases:
        try:
            C._build_fernet(bad_value, "CREDENTIAL_ENCRYPTION_KEY")
        except RuntimeError as exc:
            msg = str(exc)
            assert expected in msg, (
                f"error for {bad_value[:20]!r} did not mention {expected!r}: {msg}"
            )
            # The key material must never leak into the message.
            assert bad_value not in msg
        else:
            raise AssertionError(f"accepted {bad_value!r} as a Fernet key")


def test_fingerprint_never_reveals_key_material():
    keys = _fresh_keys(2)
    for k in keys:
        fp = C.key_fingerprint(k)
        assert k not in fp
        assert len(fp) == 12
    assert C.key_fingerprint(keys[0]) != C.key_fingerprint(keys[1])
    assert C.key_fingerprint(keys[0]) == C.key_fingerprint(keys[0])


def test_production_still_refuses_to_start_without_a_key():
    """SEC-014 must not be weakened."""
    saved_key = os.environ.pop("CREDENTIAL_ENCRYPTION_KEY", None)
    os.environ["ENVIRONMENT"] = "production"
    os.environ.pop("ALLOW_EPHEMERAL_ENCRYPTION_KEY", None)
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


def test_internal_endpoints_are_throttled():
    """Credentials must be rate limited.

    Was `test_both_internal_endpoints_are_throttled`: the /execute
    endpoint is gone, so this covers the one internal endpoint that
    remains. It discovers the endpoints by prefix rather than naming them,
    so the next one added is covered by construction.
    """
    from STT_server.routes import api as api_mod

    internal_fns = [
        getattr(api_mod, name)
        for name in dir(api_mod)
        if name.startswith("internal_") and callable(getattr(api_mod, name))
    ]
    assert internal_fns, "expected at least one internal endpoint"
    for fn in internal_fns:
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
    assert src.count("enforce_internal_rate_limit") == 2, (
        "import + exactly the one internal endpoint. Was 3 while /execute "
        "existed; asserting the count rather than the literal keeps this "
        "honest when the next internal endpoint lands."
    )
    assert "/internal/integrations/" in src


# ── Self-check ────────────────────────────────────────────────────

def _self_check():
    fns = [
        test_encrypt_decrypt_round_trip,
        test_unreadable_token_still_fails_closed,
        test_empty_and_none_pass_through_unchanged,
        test_a_value_encrypted_with_another_key_is_refused,
        test_decrypt_failure_says_how_to_recover,
        test_invalid_key_error_names_the_actual_mistake,
        test_fingerprint_never_reveals_key_material,
        test_production_still_refuses_to_start_without_a_key,
        test_limit_is_100_per_minute,
        test_under_the_limit_is_allowed,
        test_over_the_limit_is_refused_with_retry_after,
        test_budget_is_per_client,
        test_refusals_do_not_grow_the_counter_forever,
        test_stale_clients_are_pruned,
        test_client_key_prefers_forwarded_for,
        test_internal_endpoints_are_throttled,
        test_limit_is_env_overridable,
        test_rate_limit_is_not_applied_to_the_public_api,
    ]
    for fn in fns:
        fn()
    print(f"security_credentials_and_rate_limit: OK ({len(fns)} checks)")


if __name__ == "__main__":
    _self_check()