"""Regression tests for the per-agent runtime knobs (llm_temperature,
llm_max_tokens, tts_speed).

The operator reported: "I set LLM Temperature to 0.2, LLM Max Tokens
to 200, TTS Speed to 1.1 — every time I save the agent the values
go back to empty." Root cause: the `AgentCreate` and `AgentUpdate`
Pydantic schemas didn't declare these fields, so Pydantic
silently dropped them from the request body before the BE
persisted anything. The DB column existed (006_agent_runtime_
params.sql), the FE was sending the values, the DB update query
included the columns — only the Pydantic layer was eating the
fields between FE and BE.

These tests pin the contract at every layer: schema accepts the
field, the round-trip persists it, and the missing field on the
schema is detected by a regression guard so it can't come back.
"""
from __future__ import annotations

import pytest


# ── Regression guard: the field must be in the schema ──────────────
# If a future refactor drops these fields from AgentCreate /
# AgentUpdate the test fails loud BEFORE the FE ships a save that
# silently loses the value. The operator symptom (200 → 0.2 →
# save → reload → 0.2) is exactly the silent-drop class we want to
# prevent.


def test_agent_create_schema_accepts_runtime_knobs():
    """Pydantic must declare the runtime knobs. Otherwise the FE
    payload reaches the route handler, gets dropped on `.dict()`,
    and the DB row never sees the value."""
    from STT_server.routes.api import AgentCreate

    fields = set(AgentCreate.model_fields.keys())
    for knob in ("llm_temperature", "llm_max_tokens", "tts_speed"):
        assert knob in fields, (
            f"AgentCreate is missing `{knob}`. The 2026-08-26 "
            f"regression: this omission made Pydantic silently drop "
            f"the field on save. Add it as `Optional[float]` (or "
            f"`Optional[int]` for llm_max_tokens) and re-run this "
            f"test."
        )


def test_agent_update_schema_accepts_runtime_knobs():
    """Same regression guard for the update path. ModalAgents
    (Edit) calls PUT /agents/{id}; the response of that call has
    to include the runtime knob field for the operator to see
    their value persisted on reload."""
    from STT_server.routes.api import AgentUpdate

    fields = set(AgentUpdate.model_fields.keys())
    for knob in ("llm_temperature", "llm_max_tokens", "tts_speed"):
        assert knob in fields, (
            f"AgentUpdate is missing `{knob}`. The 2026-08-26 "
            f"regression: this omission made PUT silently drop the "
            f"field, so opening the edit modal after save always "
            f"showed the runtime knobs as blank."
        )


def test_agent_update_schema_runtime_knobs_round_trip():
    """End-to-end schema check. The route receives
    ``{"name": "X", "llm_temperature": 0.2, ...}``; Pydantic
    must round-trip the runtime knobs onto the resulting
    ``data.dict()`` so the DB update query sees them. The
    regression was that the field never made it past Pydantic.
    """
    from STT_server.routes.api import AgentUpdate

    payload = {
        "name": "Test",
        "llm_temperature": 0.2,
        "llm_max_tokens": 200,
        "tts_speed": 1.1,
    }
    parsed = AgentUpdate(**payload)
    dumped = parsed.model_dump(exclude_none=True)
    for knob, expected in (
        ("llm_temperature", 0.2),
        ("llm_max_tokens", 200),
        ("tts_speed", 1.1),
    ):
        assert dumped.get(knob) == expected, (
            f"AgentUpdate round-trip lost `{knob}`: "
            f"input={payload.get(knob)!r} output={dumped.get(knob)!r}"
        )


def test_agent_create_schema_runtime_knobs_round_trip():
    """Same round-trip for the create path. ModalNewAgent (Create)
    posts the runtime knobs in step 4; the BE must persist them
    on insert."""
    from STT_server.routes.api import AgentCreate

    payload = {
        "name": "Test",
        "llm_temperature": 0.3,
        "llm_max_tokens": 250,
        "tts_speed": 0.95,
    }
    parsed = AgentCreate(**payload)
    dumped = parsed.model_dump(exclude_none=True)
    for knob, expected in (
        ("llm_temperature", 0.3),
        ("llm_max_tokens", 250),
        ("tts_speed", 0.95),
    ):
        assert dumped.get(knob) == expected, (
            f"AgentCreate round-trip lost `{knob}`: "
            f"input={payload.get(knob)!r} output={dumped.get(knob)!r}"
        )


def test_runtime_knobs_default_to_none_so_platform_default_applies():
    """When the operator doesn't override a knob, the BE must see
    None for that field. db_update_agent's exclude_none logic
    means a None column is left alone at the DB level. The
    downstream adapter (openai_llm / inworld_tts / elevenlabs_tts)
    reads `getattr(session, knob, None)` and falls back to the
    platform default when the value is None. Pin the contract so a
    future refactor can't accidentally send 0.0 instead of None.
    """
    from STT_server.routes.api import AgentUpdate

    parsed = AgentUpdate(name="Test")
    dumped = parsed.model_dump(exclude_none=True)
    assert "llm_temperature" not in dumped, (
        "Default AgentUpdate must omit runtime knobs so the BE leaves "
        "the DB column untouched (inherit platform default)."
    )
    assert "llm_max_tokens" not in dumped
    assert "tts_speed" not in dumped


# ── End-to-end: PUT /agents/{id} persists the runtime knobs ──────
# These tests hit the full route → Pydantic → db layer to make sure
# the operator's symptom ("I set 0.2 but it resets every save")
# can't come back via a regression in any of those layers.


async def test_put_agent_persists_runtime_knobs(client, data_dir, auth_token):
    """Full round-trip: PUT /agents/{id} with runtime knobs →
    reload the agent → knobs are still there. Regression for the
    2026-08-26 "I set 0.2 but it resets every save" bug.

    The Pydantic schema gap was the actual cause: AgentCreate /
    AgentUpdate didn't declare the fields, so Pydantic dropped them
    on `.dict()` before the route handler saw them. Now that the
    fields are declared, the round-trip works end-to-end.
    """
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Create the agent first (the create flow also exercises the
    # schema; we use the round-trip on update because that's the
    # operator's actual path — Edit modal Save).
    r = await client.post(
        "/agents",
        json={"name": "Runtime Knob Test"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    agent_id = r.json()["id"]

    r = await client.put(
        f"/agents/{agent_id}",
        json={
            "llm_temperature": 0.2,
            "llm_max_tokens": 200,
            "tts_speed": 1.1,
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text

    # Reload via the GET endpoint and confirm the knobs round-tripped.
    r = await client.get("/agents", headers=headers)
    assert r.status_code == 200
    agents = r.json()
    saved = next(a for a in agents if a["id"] == agent_id)
    assert saved.get("llm_temperature") == 0.2, (
        f"llm_temperature not persisted: {saved.get('llm_temperature')!r}"
    )
    assert saved.get("llm_max_tokens") == 200, (
        f"llm_max_tokens not persisted: {saved.get('llm_max_tokens')!r}"
    )
    assert saved.get("tts_speed") == 1.1, (
        f"tts_speed not persisted: {saved.get('tts_speed')!r}"
    )


async def test_post_agent_persists_runtime_knobs(client, data_dir, auth_token):
    """Same round-trip via the create path. The wizard (ModalNewAgent)
    posts the knobs in step 4 — they must land in the DB row."""
    headers = {"Authorization": f"Bearer {auth_token}"}

    r = await client.post(
        "/agents",
        json={
            "name": "Create Test",
            "llm_temperature": 0.3,
            "llm_max_tokens": 250,
            "tts_speed": 0.95,
        },
        headers=headers,
    )
    assert r.status_code == 200, r.text
    created = r.json()
    assert created.get("llm_temperature") == 0.3
    assert created.get("llm_max_tokens") == 250
    assert created.get("tts_speed") == 0.95
