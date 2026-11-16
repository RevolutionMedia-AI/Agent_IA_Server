"""Unit tests for the two pieces that took the DB off the voice path.

  - STT_server.services.db_async.db_call: runs a sync DB function on a
    dedicated thread pool so it never blocks the event loop that serves
    live calls.
  - STT_server.services.credentials_resolver: the per-call credential
    snapshot (PERF-002) — one read per call, dict hits afterwards.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from STT_server.domain.session import CallSession
from STT_server.security.credentials import encrypt_credentials
from STT_server.services import credentials_resolver
from STT_server.services.db_async import db_call


# ── db_call ──────────────────────────────────────────────────────────────


async def test_db_call_returns_the_value() -> None:
    assert await db_call(lambda a, b: a + b, 2, 3) == 5


async def test_db_call_passes_kwargs() -> None:
    assert await db_call(lambda a, b=0: a + b, 2, b=3) == 5


async def test_db_call_propagates_exceptions() -> None:
    def boom() -> None:
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        await db_call(boom)


async def test_db_call_runs_off_the_event_loop_thread() -> None:
    """The whole point: the sync work must NOT run on the loop thread.

    If it did, the event loop that receives every call's audio frames
    would stall for the duration of the query.
    """
    loop_thread = threading.get_ident()
    seen: dict[str, Any] = {}

    def capture() -> None:
        seen["thread"] = threading.get_ident()

    await db_call(capture)
    assert seen["thread"] != loop_thread, "DB work ran on the event loop thread"


async def test_db_call_does_not_block_the_event_loop() -> None:
    """A slow DB call must let other coroutines keep running."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    task = asyncio.create_task(ticker())
    try:
        await db_call(time.sleep, 0.2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert ticks > 3, f"event loop was blocked during the DB call (ticks={ticks})"


async def test_db_call_actually_awaits_the_result() -> None:
    """The value must be resolved, not a coroutine/future handed back."""
    result = await db_call(str.upper, "abc")
    assert result == "ABC"
    assert not asyncio.iscoroutine(result)


# ── credential snapshot ──────────────────────────────────────────────────


def _encrypted(creds: dict) -> dict:
    return encrypt_credentials(creds)


@pytest.fixture
def counted_list_tools(monkeypatch: pytest.MonkeyPatch) -> list:
    """Patch db_tools.list_tools and count the calls."""
    calls: list = []

    def fake_list_tools(user_id: str, agent_id: str | None = None) -> list[dict]:
        calls.append((user_id, agent_id))
        return [
            {
                "id": "openai",
                "agent_id": "__shared__",
                "user_id": user_id,
                "credentials": _encrypted({"api_key": "sk-user-openai"}),
            },
            {
                "id": "elevenlabs",
                "agent_id": "__shared__",
                "user_id": user_id,
                "credentials": _encrypted({"api_key": "sk-user-eleven", "voice_id": "v1"}),
            },
        ]

    monkeypatch.setattr("STT_server.db_tools.list_tools", fake_list_tools)
    return calls


def test_load_credential_snapshot_reads_the_db_once(counted_list_tools: list) -> None:
    snap = credentials_resolver.load_credential_snapshot("alice")
    assert len(counted_list_tools) == 1
    assert snap.get("openai") == {"api_key": "sk-user-openai"}
    assert snap.get("elevenlabs") == {"api_key": "sk-user-eleven", "voice_id": "v1"}
    assert snap.get("nope") == {}


def test_load_credential_snapshot_without_user_reads_nothing(
    counted_list_tools: list,
) -> None:
    snap = credentials_resolver.load_credential_snapshot(None)
    assert counted_list_tools == []
    assert snap.get("openai") == {}


def test_resolve_for_session_uses_the_snapshot_and_skips_the_db(
    counted_list_tools: list,
) -> None:
    """The whole point of the snapshot: no DB read per lookup.

    Before the snapshot, every TTS segment / LLM turn / STT session
    re-read the user's integrations and re-ran the Fernet decrypt.
    """
    session = CallSession(session_key="s1", user_id="alice")
    session.tts_use_own_key = True
    session.cred_snapshot = credentials_resolver.load_credential_snapshot("alice")
    assert len(counted_list_tools) == 1

    for _ in range(10):
        creds = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")
        assert creds["api_key"] == "sk-user-eleven"
    assert len(counted_list_tools) == 1, "snapshot lookups must not hit the DB"


def test_resolve_for_session_without_snapshot_falls_back_to_the_db(
    counted_list_tools: list,
) -> None:
    """Sessions without a snapshot (admin paths, legacy) still work."""
    session = CallSession(session_key="s2", user_id="alice")
    session.tts_use_own_key = True
    creds = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")
    assert creds["api_key"] == "sk-user-eleven"
    assert len(counted_list_tools) == 1


def test_snapshot_resolution_matches_the_db_resolution(
    counted_list_tools: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Snapshot and DB paths must return the same thing.

    Same merge rules, same "first row wins" tie-break, same
    use_own_key handling — otherwise the snapshot would silently
    resolve a different key than the code it replaced.
    """
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-platform-eleven")

    resolved = {}
    for use_own in (True, False):
        session = CallSession(session_key="s3", user_id="alice")
        session.tts_use_own_key = use_own

        from_db = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")

        session.cred_snapshot = credentials_resolver.load_credential_snapshot("alice")
        from_snapshot = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")

        assert from_snapshot == from_db, f"diverged with use_own_key={use_own}"
        resolved[use_own] = from_snapshot
    # The user's key wins over the platform one, with the toggle on...
    assert resolved[True]["api_key"] == "sk-user-eleven"
    # ...and with the toggle OFF too: the toggle means "the platform key
    # is the fallback when I have no key of my own", not "ignore my key".
    assert resolved[False]["api_key"] == "sk-user-eleven"


def test_snapshot_respects_the_use_own_key_toggle(
    counted_list_tools: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The toggle must mean the same thing on both resolution paths.

    use_own_key=False means "the platform key is the fallback when I
    have no key of my own" -- the user's stored credential still wins.
    use_own_key=True means "only my own key".
    """
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-platform-eleven")
    session = CallSession(session_key="s4", user_id="alice")
    session.tts_use_own_key = False
    session.cred_snapshot = credentials_resolver.load_credential_snapshot("alice")

    creds = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")
    assert creds["api_key"] == "sk-user-eleven"

    session.tts_use_own_key = True
    creds = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")
    assert creds["api_key"] == "sk-user-eleven"
    # The toggle on means ONLY the user's own credential.
    assert "voice_id" not in creds or creds.get("voice_id") == "v1"


def test_snapshot_is_ignored_without_a_user_id(
    counted_list_tools: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """user_id=None means "platform keys only" — the snapshot must not leak.

    The greeting pre-generation builds a session with user_id=None on
    purpose. A snapshot present on such a session must not hand it the
    user's key.
    """
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-platform-eleven")
    session = CallSession(session_key="warmup-es")
    session.user_id = None
    session.cred_snapshot = credentials_resolver.load_credential_snapshot("alice")

    creds = credentials_resolver.resolve_for_session(session, "tts", "elevenlabs")
    assert creds["api_key"] == "sk-platform-eleven"


def test_find_first_configured_provider_with_snapshot_does_not_read_the_db(
    counted_list_tools: list,
) -> None:
    """Auto-detect used to run three full credential reads at call start."""
    snap = credentials_resolver.load_credential_snapshot("alice")
    assert len(counted_list_tools) == 1

    for category in ("stt", "tts", "llm"):
        provider = credentials_resolver.find_first_configured_provider(
            "alice", category, snapshot=snap
        )
        assert provider in ("openai", "elevenlabs"), f"unexpected provider for {category}"
    assert len(counted_list_tools) == 1, "auto-detect must not read the DB"


def test_find_first_configured_provider_matches_the_db_path(
    counted_list_tools: list,
) -> None:
    snap = credentials_resolver.load_credential_snapshot("alice")
    for category in ("stt", "tts", "llm"):
        with_db = credentials_resolver.find_first_configured_provider("alice", category)
        with_snapshot = credentials_resolver.find_first_configured_provider(
            "alice", category, snapshot=snap
        )
        assert with_snapshot == with_db


def test_snapshot_survives_a_failing_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DB error must degrade to platform env vars, never raise."""

    def boom(user_id: str, agent_id: str | None = None) -> list:
        raise RuntimeError("db is down")

    monkeypatch.setattr("STT_server.db_tools.list_tools", boom)
    snap = credentials_resolver.load_credential_snapshot("alice")
    assert snap.get("openai") == {}


def test_snapshot_keeps_the_newest_row_per_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider saved per-agent AND as a shared row resolves to the newest.

    db_tools.list_tools orders by created_at DESC and the old per-provider
    read took the first match. The snapshot must keep that rule or a call
    would authenticate with a stale key.
    """

    def fake_list_tools(user_id: str, agent_id: str | None = None) -> list[dict]:
        return [
            {
                "id": "openai",
                "agent_id": "agent-a",
                "user_id": user_id,
                "credentials": _encrypted({"api_key": "sk-newest"}),
            },
            {
                "id": "openai",
                "agent_id": "__shared__",
                "user_id": user_id,
                "credentials": _encrypted({"api_key": "sk-older"}),
            },
        ]

    monkeypatch.setattr("STT_server.db_tools.list_tools", fake_list_tools)
    snap = credentials_resolver.load_credential_snapshot("alice")
    assert snap.get("openai") == {"api_key": "sk-newest"}


def test_snapshot_keeps_an_undecryptable_provider_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale ciphertext must resolve to {} — not fall through to an older row.

    Matches the old behaviour: the newest row wins, and if it cannot be
    decrypted the provider simply has no per-user credential.
    """

    def fake_list_tools(user_id: str, agent_id: str | None = None) -> list[dict]:
        return [
            {
                "id": "openai",
                "agent_id": "agent-a",
                "user_id": user_id,
                "credentials": {"api_key": "not-a-fernet-token"},
            },
            {
                "id": "openai",
                "agent_id": "__shared__",
                "user_id": user_id,
                "credentials": _encrypted({"api_key": "sk-older"}),
            },
        ]

    monkeypatch.setattr("STT_server.db_tools.list_tools", fake_list_tools)
    snap = credentials_resolver.load_credential_snapshot("alice")
    assert snap.get("openai") == {}
