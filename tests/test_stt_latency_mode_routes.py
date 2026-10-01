"""Route-level validation for stt_latency_mode (B8 / B9 / B10 end to end).

The unit tests cover openai_stt_models.validate; these go through the
actual HTTP handlers, because the thing that must never happen is an
invalid pair reaching the agent row and then failing OpenAI's
session.update at call time.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from STT_server.routes import api


def _create(**over):
    base = {
        "name": "Latency Dial Agent",
        "stt_provider": "openai",
        "stt_model": "gpt-live-transcribe",
        "stt_latency_mode": "low",
    }
    base.update(over)
    return api.AgentCreate(**base)


AUTH = {"user_id": "user-admin-001"}


def test_valid_pair_persists(monkeypatch):
    """The happy path: the normalized value reaches the payload."""
    seen = {}

    def fake_create(user_id, payload):
        seen.update(payload)
        return {"id": "agent-x", **payload}

    monkeypatch.setattr(api, "db_create_agent", fake_create)
    out = api.create_agent(_create(stt_latency_mode="xhigh"), auth=AUTH)
    assert out["stt_latency_mode"] == "xhigh"


def test_absent_mode_persists_null(monkeypatch):
    """B10 — an agent saved without the field stores NULL, and the adapter
    substitutes the platform default at call time. Nothing is invented at
    write time, so a legacy row and a deliberately-default row are the
    same thing on disk."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    api.create_agent(
        api.AgentCreate(name="A", stt_provider="openai",
                        stt_model="gpt-live-transcribe", stt_latency_mode=None),
        auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_invalid_mode_is_400(monkeypatch):
    """B8 — a bogus level is rejected with a message naming the legal ones."""
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda *a, **k: pytest.fail("must not reach the store"),
    )
    with pytest.raises(HTTPException) as exc:
        api.create_agent(_create(stt_latency_mode="turbo"), auth=AUTH)
    assert exc.value.status_code == 400
    assert "turbo" in str(exc.value.detail)
    assert "minimal" in str(exc.value.detail)


def test_mode_illegal_for_that_specific_model_is_400(monkeypatch):
    """The interesting case: a level that is legal for the OTHER model but
    not this one. A global enum check would let this through."""
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda *a, **k: pytest.fail("must not reach the store"),
    )
    # gpt-live-transcribe accepts all five; a future model with fewer
    # levels must still 400. Assert the mechanism with the real models:
    # gpt-transcribe accepts none at all.
    with pytest.raises(HTTPException) as exc:
        api.create_agent(
            _create(stt_model="gpt-transcribe", stt_latency_mode="low"),
            auth=AUTH,
        )
    assert exc.value.status_code == 400
    assert "no latency" in str(exc.value.detail)


def test_committed_turn_model_rejects_an_explicit_mode(monkeypatch):
    """gpt-transcribe stores NULL, and an explicitly-sent mode is a 400
    rather than a silent drop — otherwise a client that always posts the
    field looks like it saved when it did not."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    with pytest.raises(HTTPException) as exc:
        api.create_agent(
            _create(stt_model="gpt-transcribe", stt_latency_mode="high"),
            auth=AUTH,
        )
    assert exc.value.status_code == 400
    assert "committed-turn" in str(exc.value.detail)
    assert "stt_latency_mode" not in seen

    # ...and omitting the field is a clean save with NULL.
    api.create_agent(
        api.AgentCreate(name="A", stt_provider="openai",
                        stt_model="gpt-transcribe", stt_latency_mode=None),
        auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_non_openai_provider_never_stores_a_mode(monkeypatch):
    """The dial is an OpenAI concept. Deepgram and Inworld rows must not
    carry one, whatever the client sends."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    api.create_agent(
        api.AgentCreate(name="A", stt_provider="deepgram", stt_model="nova-3",
                        stt_latency_mode="low"),
        auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_unknown_stt_model_stores_null_rather_than_400(monkeypatch):
    """An stt_model outside the OpenAI lineup (a legacy row, or a model
    from another provider) must not be blocked on the dial."""
    seen = {}
    monkeypatch.setattr(
        api, "db_create_agent",
        lambda u, p: (seen.update(p), {"id": "a", **p})[1],
    )
    api.create_agent(
        _create(stt_model="whisper-1", stt_latency_mode="low"), auth=AUTH,
    )
    assert seen["stt_latency_mode"] is None


def test_update_validates_against_the_stored_model(monkeypatch):
    """A partial update that only carries the mode is checked against the
    model the agent actually has, not against a missing field."""
    monkeypatch.setattr(api, "_agent_stt_provider", lambda a, u: "openai")
    monkeypatch.setattr(api, "_agent_stt_model", lambda a, u: "gpt-live-transcribe")
    monkeypatch.setattr(
        api, "db_update_agent",
        lambda *a, **k: pytest.fail("must not reach the store"),
    )
    with pytest.raises(HTTPException) as exc:
        api.update_agent(
            "agent-1",
            api.AgentUpdate(stt_latency_mode="nope"),
            auth=AUTH,
        )
    assert exc.value.status_code == 400
    assert "nope" in str(exc.value.detail)
