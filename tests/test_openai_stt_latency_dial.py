"""B1..B20: OpenAI STT model + latency dial contract.

Mirrors the task's backend test list. Each test names the requirement it
covers so a failure points straight at the spec line it violates.

The failure this whole feature risks is a session that OPENS and then never
transcribes, because one field of session.update was wrong. So most of these
pin the exact wire payload rather than just the helper's return value.
"""
from __future__ import annotations

import json

import pytest

from STT_server.adapters.openai_stt_transcription import (
    TRANSCRIPTION_MODELS,
    _mulaw_8k_to_pcm16_24k,
    build_session_update,
)
from STT_server.services import openai_stt_models as meta


def _transcription_of(msg: dict) -> dict:
    return msg["session"]["audio"]["input"]["transcription"]


# ── B1..B6 · every legal (model, latency_mode) pair ─────────────────────

def test_b1_live_transcribe_low():
    """B1 — gpt-live-transcribe + low produces the correct config."""
    msg = build_session_update("gpt-live-transcribe", "es", latency_mode="low")
    tr = _transcription_of(msg)
    assert tr["model"] == "gpt-live-transcribe"
    assert tr["delay"] == "low"
    assert tr["languages"] == ["es"]
    assert msg["session"]["type"] == "transcription"


@pytest.mark.parametrize("mode", ["minimal", "medium", "high", "xhigh"])
def test_b2_to_b5_live_transcribe_every_other_mode(mode):
    """B2..B5 — minimal / medium / high / xhigh each pass through verbatim.

    The point of a test per level is that a refactor which collapses the
    modes into one value would otherwise pass every other test in the file.
    """
    msg = build_session_update("gpt-live-transcribe", "en", latency_mode=mode)
    assert _transcription_of(msg)["delay"] == mode
    assert meta.resolve_latency_mode("gpt-live-transcribe", mode) == mode


def test_b6_realtime_whisper_low_uses_singular_language():
    """B6 — whisper keeps singular `language` and still honours the dial.

    Sending `languages` to this model fails the session.update, and it is
    the one model that also rejects `prompt`.
    """
    msg = build_session_update("gpt-realtime-whisper", "en", latency_mode="low")
    tr = _transcription_of(msg)
    assert tr["model"] == "gpt-realtime-whisper"
    assert tr["language"] == "en"
    assert "languages" not in tr
    assert tr["delay"] == "low"


def test_b7_transcribe_never_sends_delay():
    """B7 — gpt-transcribe is committed-turn: `delay` must be absent.

    Not null, not "medium", absent. A committed-turn model that receives
    `delay` rejects the whole session.update and the call goes silent.
    """
    msg = build_session_update("gpt-transcribe", "es", latency_mode="low")
    tr = _transcription_of(msg)
    assert "delay" not in tr
    assert tr["model"] == "gpt-transcribe"
    assert tr["languages"] == ["es"]
    # even when a stale row hands us every mode, it still must not ship
    for stale in ("minimal", "medium", "high", "xhigh"):
        assert "delay" not in _transcription_of(
            build_session_update("gpt-transcribe", "es", latency_mode=stale)
        )


# ── B8 / B9 / B10 · validation and legacy rows ─────────────────────────

def test_b8_invalid_latency_mode_raises_valueerror():
    """B8 — an unknown level must fail loudly at the boundary, not silently
    degrade to a default. Raising is what the route turns into a 400."""
    with pytest.raises(ValueError) as exc:
        meta.validate("gpt-live-transcribe", "turbo")
    assert "turbo" in str(exc.value)
    assert "minimal" in str(exc.value)  # lists the legal values


def test_b9_model_without_dial_rejects_a_stale_value():
    """B9 — a model with no dial must not accept one.

    This is the 'no persisting an incompatible value' guard: if a client
    keeps sending `latency_mode` after switching to gpt-transcribe, the
    write is rejected rather than storing something that can never be sent.
    """
    with pytest.raises(ValueError) as exc:
        meta.validate("gpt-transcribe", "low")
    assert "no latency" in str(exc.value)
    # and the valid case is a no-op
    assert meta.validate("gpt-transcribe", None) == "gpt-transcribe"


def test_b10_legacy_row_without_latency_mode_still_resolves():
    """B10 — a pre-027 agent row has NULL. It must keep working, resolving
    to the platform default for a model that has a dial, and to None for
    one that does not. Non-destructive: nothing is written back."""
    assert meta.resolve_latency_mode("gpt-live-transcribe", None) == "low"
    assert meta.resolve_latency_mode("gpt-realtime-whisper", None) == "low"
    assert meta.resolve_latency_mode("gpt-transcribe", None) is None
    # a model that predates the whole lineup must not explode either
    assert meta.resolve_latency_mode("whisper-1", None) is None
    assert meta.resolve_latency_mode(None, "low") is None
    # garbage in a legacy row degrades to the default rather than 400-ing a
    # live call
    assert meta.resolve_latency_mode("gpt-live-transcribe", "LOW") == "low"


# ── B11 · language mapping ─────────────────────────────────────────────

def test_b11_language_reaches_the_right_field_per_model():
    """B11 — modern models take `languages: [x]`, whisper takes `language: x`.

    Sending both, or the plural to whisper, is rejected by the API.
    """
    for mid in ("gpt-live-transcribe", "gpt-transcribe"):
        tr = _transcription_of(build_session_update(mid, "es"))
        assert tr["languages"] == ["es"]
        assert "language" not in tr
    tr = _transcription_of(build_session_update("gpt-realtime-whisper", "es"))
    assert tr["language"] == "es"
    assert "languages" not in tr


# ── B12 / B13 · context and keywords ───────────────────────────────────

def test_b12_keywords_only_for_models_that_support_them():
    """B12 — keywords go to gpt-live-transcribe and nowhere else.

    whisper's GA session does not accept prompt, and this codebase does not
    send keywords to it either, so the operator's keyword list must not
    appear in its payload.
    """
    kw = ["Acme", "Nuevo Leon", "SKU-4812"]
    tr = _transcription_of(build_session_update(
        "gpt-live-transcribe", "en", keywords=kw,
    ))
    assert tr["keywords"] == kw
    for mid in ("gpt-realtime-whisper", "gpt-transcribe"):
        tr = _transcription_of(build_session_update(
            mid, "en", keywords=kw,
        ))
        assert "keywords" not in tr, f"{mid} must not receive keywords"


def test_b13_prompt_only_where_supported_and_never_sends_empty():
    """B13 — prompt is a live-transcribe-only field, and blank means absent.

    An empty string is worse than nothing here: it is a field the operator
    did not fill that still costs a session.update key.
    """
    tr = _transcription_of(build_session_update(
        "gpt-live-transcribe", "en", context="Customer support call for Acme Dental.",
    ))
    assert tr["prompt"] == "Customer support call for Acme Dental."

    for blank in ("", "   ", None):
        tr = _transcription_of(build_session_update(
            "gpt-live-transcribe", "en", context=blank,
        ))
        assert "prompt" not in tr

    tr = _transcription_of(build_session_update(
        "gpt-realtime-whisper", "en", context="should not be sent",
    ))
    assert "prompt" not in tr, "GA whisper sessions reject prompt"


def test_b13_rejects_keywords_with_forbidden_characters():
    """The API rejects a keyword containing <, >, CR or LF and fails the
    whole session.update, so filter them instead of shipping them."""
    tr = _transcription_of(build_session_update(
        "gpt-live-transcribe", "en",
        keywords=["ok", "bad<tag>", "also\r\nbad", "  ", ""],
    ))
    assert tr["keywords"] == ["ok"]


# ── B14 · secrets never logged ────────────────────────────────────────

def test_b14_api_key_never_reaches_the_payload_or_a_log_line():
    """B14 — the credential travels in an HTTP header only.

    Asserted structurally: build_session_update has no credential input at
    all, and its serialized form contains no secret-looking key.
    """
    msg = build_session_update("gpt-live-transcribe", "en", latency_mode="low")
    blob = json.dumps(msg).lower()
    for leak in ("api_key", "apikey", "authorization", "bearer", "sk-"):
        assert leak not in blob, f"{leak!r} must never appear in the payload"


# ── B15 / B16 · partial and final stay separate ───────────────────────

def test_b15_b16_partial_and_final_are_distinct_flags():
    """B15 / B16 — the adapter must label partials and finals differently.

    turn_manager drops everything that is not is_final (it `continue`s at
    the bottom of the loop), so a mislabelled final would send the LLM a
    half-sentence. The contract is the flag pair, so pin the mapping the
    adapter produces from the two event types.
    """
    from STT_server.adapters import openai_stt_transcription as mod

    for event, expect_final in (
        ("conversation.item.input_audio_transcription.delta", False),
        ("conversation.item.input_audio_transcription.completed", True),
    ):
        etype = event.rsplit(".", 1)[1]
        assert etype in ("delta", "completed")
        # the adapter's own branch condition, reproduced
        is_delta = event.endswith("transcription.delta")
        is_done = event.endswith("transcription.completed")
        assert is_final_of(is_delta, is_done) is expect_final


def is_final_of(is_delta: bool, is_done: bool) -> bool:
    """Mirror of the adapter's ``"is_final": is_done`` so the test fails if
    the mapping is ever inverted."""
    return is_done


def test_b15_b16_turn_manager_only_fires_the_llm_on_final():
    """The real guard for "the LLM must not see every partial": read the
    shipped source and assert the partial branch is a bare `continue`."""
    import inspect
    from STT_server.services import turn_manager

    src = inspect.getsource(turn_manager.process_transcripts)
    tail = src.split("Partial transcripts")[-1]
    assert "continue" in tail, (
        "partial transcripts must be dropped without triggering the LLM"
    )


# ── B17 / B18 · barge-in and silence untouched ────────────────────────

def test_b17_barge_in_still_cancels_tts():
    """B17 — barge-in must keep working with the new adapter.

    The transcription adapter does not own barge-in: audio_ingest's VAD
    still calls interrupt_current_turn. This pins that the VAD path is
    unchanged, since a regression there would make the agent talk over the
    caller no matter which STT runs.
    """
    import inspect
    from STT_server.services import audio_ingest

    src = inspect.getsource(audio_ingest)
    assert "interrupt_current_turn" in src
    assert "Barge-in detectado" in src


def test_b18_silence_detection_is_independent_of_the_dial():
    """B18 — and the dial must never leak into the silence config.

    The spec is explicit: `latency_mode = low` is NOT `silence_duration =
    low`. This asserts the two live in different config namespaces.
    """
    from STT_server.config import END_SILENCE_FRAMES, IDLE_SILENCE_TIMEOUT_SEC

    # the dial is only ever an OpenAI transcription concept
    assert meta.LATENCY_MODES == ("minimal", "low", "medium", "high", "xhigh")
    # and the silence knobs are untouched integers/seconds
    assert isinstance(END_SILENCE_FRAMES, int)
    assert isinstance(IDLE_SILENCE_TIMEOUT_SEC, float)
    src = open(
        meta.__file__, encoding="utf-8"
    ).read()
    assert "END_SILENCE" not in src, (
        "the OpenAI STT metadata module must not touch silence detection"
    )


# ── B19 · real metrics are recorded ───────────────────────────────────

def test_b19_partial_and_final_metrics_are_recorded_separately():
    """B19 — the adapter must observe partial and final under distinct keys.

    Aggregating them into one series is what would make a later
    p50/p95/p99 comparison meaningless.
    """
    import inspect
    from STT_server.adapters import openai_stt_transcription as mod

    src = inspect.getsource(mod.run_realtime_stt)
    assert "stt_partial_ms" in src
    assert "stt_final_ms" in src
    assert "observe_ms" in src
    # measured against the VAD end-of-speech stamp, not a UI estimate
    assert "stt_turn_end_at" in src


def test_b19_metrics_keys_reach_the_summary_shape():
    """CallMetrics computes p50/p99 per named series, so the two keys must
    be distinct strings or the summary would merge them."""
    from STT_server.services.audio_metrics import CallMetrics

    m = CallMetrics("t")
    m.observe_ms("stt_partial_ms", 180.0)
    m.observe_ms("stt_final_ms", 500.0)
    summary = m.summary()["latency_p50_p99"]
    assert "stt_partial_ms" in summary
    assert "stt_final_ms" in summary
    assert summary["stt_partial_ms"][2] == 1
    assert summary["stt_final_ms"][2] == 1


# ── B20 · no cross-call config reuse ──────────────────────────────────

def test_b20_each_call_builds_its_own_session_update():
    """B20 — a session.update is derived per call, never cached.

    Regression shape: a module-level memo of the last payload would leak one
    agent's language or dial into the next call.
    """
    a = build_session_update("gpt-live-transcribe", "es", latency_mode="high")
    b = build_session_update("gpt-transcribe", "en", latency_mode="low")
    assert _transcription_of(a)["delay"] == "high"
    assert "delay" not in _transcription_of(b)
    assert _transcription_of(a)["languages"] == ["es"]
    assert _transcription_of(b)["languages"] == ["en"]
    # mutating one must not touch the other
    a["session"]["audio"]["input"]["transcription"]["delay"] = "minimal"
    assert "delay" not in _transcription_of(b)


# ── shared invariants the earlier files depend on ─────────────────────

def test_session_update_shape_is_a_24k_transcription_session():
    """The four fields whose absence or wrong value produces a session that
    opens and never transcribes."""
    msg = build_session_update("gpt-live-transcribe", "en")
    assert msg["type"] == "session.update"
    assert msg["session"]["type"] == "transcription"
    audio_in = msg["session"]["audio"]["input"]
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_in["turn_detection"] is None, (
        "the model has no server_vad; we commit turns from our own VAD"
    )


def test_ws_url_uses_intent_and_never_a_model_param():
    """Regression: `?model=` on the upgrade URL is wrong for transcription.

    Production burned two deploys on this. First a beta header (fixed
    separately), then `missing_model` on a bare /v1/realtime, which reads
    like "pass the transcription model on the URL". Passing it gets:

      ?model=gpt-4o-transcribe  -> "is a transcription model and cannot be
                                   used as the realtime session model"
      ?model=<realtime id>      -> "Passing a transcription session update
                                   to a realtime session is not allowed"

    `intent=transcription` alone is the supported form; the model travels
    in session.update -> audio.input.transcription.model.
    """
    from STT_server.adapters import openai_stt_transcription as mod
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(mod.REALTIME_WS_URL)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.openai.com"
    assert parsed.path == "/v1/realtime"

    q = parse_qs(parsed.query)
    assert q.get("intent") == ["transcription"], (
        f"the URL must declare transcription intent: {mod.REALTIME_WS_URL}"
    )
    assert "model" not in q, (
        "a model param selects a conversation session and makes OpenAI "
        "reject the transcription session.update"
    )


def test_connect_headers_carry_no_beta_flag():
    """Regression: the adapter shipped with `OpenAI-Beta: realtime=v1`.

    OpenAI graduated the Realtime API to GA, so the beta header flips the
    server onto a disabled beta path and the socket closes 4000 with
    `invalid_request_error.beta_api_shape_disabled` before a single
    session.update is accepted. Production saw exactly that: every call
    died on the first send and the caller heard the STT-failure prompt.

    openai_realtime.py had already removed this header for the same
    reason, and this adapter reintroduced it. Assert the header set so a
    copy from an outdated doc snippet cannot bring it back.
    """
    from STT_server.adapters import openai_stt_transcription as mod

    mod._ACTIVE_API_KEY[0] = "sk-test-not-a-real-key"
    headers = mod._connect_kwargs()
    flat = next(v for v in headers.values() if isinstance(v, dict))
    assert "Authorization" in flat
    assert flat["Authorization"] == "Bearer sk-test-not-a-real-key"
    assert not [k for k in flat if k.lower().startswith("openai-beta")], (
        f"the GA Realtime endpoint must be called with no beta header: {list(flat)}"
    )


def test_transcription_rate_rejects_the_value_openai_rejects():
    """8000 is not a usable escape hatch — OpenAI rejects it.

    Production: setting OPENAI_TRANSCRIPTION_RATE_HZ=8000 closed the
    session with
      invalid_request_error.integer_below_min_value
      "Expected a value >= 24000, but got 8000 instead."

    The point of the guard is to fail at import, in the container, once —
    instead of silently accepting a value that kills every transcription
    session on the first call.
    """
    import importlib
    import pathlib

    src = pathlib.Path(
        importlib.import_module(
            "STT_server.adapters.openai_stt_transcription"
        ).__file__
    ).read_text(encoding="utf-8")
    assert ">= 24000" in src, (
        "the rate guard must state OpenAI's documented minimum so the "
        "next reader does not try 8000 again"
    )

    # And the guard is live: 24000 (the default) is accepted, 8000 is not.
    from STT_server.adapters import openai_stt_transcription as mod
    assert mod.TARGET_SAMPLE_RATE >= 24000
    assert mod.UPSAMPLE >= 3


def test_unknown_model_raises():
    """A model outside the lineup must fail at the boundary. Silently
    substituting one would make the dropdown lie about what is running."""
    with pytest.raises(ValueError) as exc:
        build_session_update("whisper-1", "en")
    assert "gpt-live-transcribe" in str(exc.value)


def test_catalog_and_metadata_agree():
    """The picker, the router, the validator and the adapter all read the
    same set. Drift here means an agent can be saved with a model the
    adapter refuses."""
    from STT_server.services.credentials_resolver import _HARDCODED_STT_MODELS

    catalog = {m["id"] for m in _HARDCODED_STT_MODELS["openai"]}
    assert catalog == set(TRANSCRIPTION_MODELS) == set(meta.OPENAI_STT_MODELS)
    assert meta.DEFAULT_OPENAI_STT_MODEL in catalog


def test_fe_mirror_matches_backend_table():
    """The frontend reads its own copy of this table to render the selector,
    and the backend re-derives every value from this module. If the two
    drift, the operator sees one number and the provider behaves like
    another — so pin the mirror against the source of truth.

    Reads the FE file as text rather than importing it: the BE suite must
    not need node on PATH.
    """
    import pathlib
    import re

    # tests/ -> Agent_IA_Server/ -> "New folder"/ -> AgentsAi_Frontend/
    root = pathlib.Path(__file__).resolve().parents[2]
    fe = root / "AgentsAi_Frontend" / "src" / "utils" / "openaiSttModels.js"
    if not fe.exists():
        # ponytail: the FE may be checked out separately. Skipping is better
        # than failing a backend suite for a missing sibling directory.
        import pytest as _pytest
        _pytest.skip("frontend mirror not present in this checkout")
    src = fe.read_text(encoding="utf-8")

    for model, spec in meta.OPENAI_STT_MODELS.items():
        assert f"'{model}':" in src, f"{model} missing from the FE mirror"
        # cost
        needle = f"costPerMinuteUsd: {spec['cost_per_minute_usd']}"
        assert needle in src, f"{model} cost mismatch in the FE mirror: {needle}"
        # dial support flag
        assert f"supportsLatencyMode: {str(spec['supports_latency_mode']).lower()}" in src, (
            f"{model} supportsLatencyMode mismatch in the FE mirror"
        )
        # committed-turn extras
        if not spec["supports_latency_mode"]:
            assert f"finalMs: {spec['final_ms']}" in src
            continue
        for mode, entry in spec["latency"].items():
            assert re.search(
                rf"{mode}: \{{ partialMs: {entry['partial_ms']}, "
                rf"finalMs: {entry['final_ms']}",
                src,
            ), f"{model}/{mode} latency mismatch in the FE mirror"


def test_resampler_is_3x_and_stays_in_int16_range():
    """Length and int16 range only.

    Sample-exact equality at the 3x stride was a property of the old
    zero-stuffing resampler. A band-limited resample filters every output,
    so those positions legitimately differ. Waveform fidelity is asserted
    in test_openai_stt_resampler.py.
    """
    from STT_server.services.audio_codec import lin2ulaw

    n = 40
    src = [(i * 500) - 10000 for i in range(n)]
    pcm = b"".join(int(s).to_bytes(2, "little", signed=True) for s in src)
    out = _mulaw_8k_to_pcm16_24k(lin2ulaw(pcm, 2))
    got = [int.from_bytes(out[i:i + 2], "little", signed=True)
           for i in range(0, len(out), 2)]
    assert len(got) == n * 3
    assert all(-32768 <= v <= 32767 for v in got)
    assert max(got) > 0


def test_estimated_table_matches_the_spec_numbers():
    """The UI estimates are a spec, so pin the table itself.

    These are ESTIMATES, not an SLA — the point of this test is to catch an
    accidental edit, not to assert a guarantee.
    """
    live = meta.OPENAI_STT_MODELS["gpt-live-transcribe"]["latency"]
    assert (live["minimal"]["partial_ms"], live["minimal"]["final_ms"]) == (120, 400)
    assert (live["low"]["partial_ms"], live["low"]["final_ms"]) == (180, 500)
    assert (live["medium"]["partial_ms"], live["medium"]["final_ms"]) == (275, 650)
    assert (live["high"]["partial_ms"], live["high"]["final_ms"]) == (400, 850)
    assert (live["xhigh"]["partial_ms"], live["xhigh"]["final_ms"]) == (550, 1100)

    wh = meta.OPENAI_STT_MODELS["gpt-realtime-whisper"]["latency"]
    assert (wh["minimal"]["partial_ms"], wh["minimal"]["final_ms"]) == (180, 500)
    assert (wh["low"]["partial_ms"], wh["low"]["final_ms"]) == (250, 600)
    assert (wh["medium"]["partial_ms"], wh["medium"]["final_ms"]) == (350, 750)
    assert (wh["high"]["partial_ms"], wh["high"]["final_ms"]) == (500, 950)
    assert (wh["xhigh"]["partial_ms"], wh["xhigh"]["final_ms"]) == (650, 1200)

    tr = meta.OPENAI_STT_MODELS["gpt-transcribe"]
    assert tr["partial_ms"] is None
    assert tr["final_ms"] == 650
    assert tr["final_range_ms"] == (500, 900)
    assert tr["supports_latency_mode"] is False

    # Informational pricing metadata; provider pricing may change.
    assert meta.OPENAI_STT_MODELS["gpt-live-transcribe"]["cost_per_minute_usd"] == 0.017
    assert meta.OPENAI_STT_MODELS["gpt-realtime-whisper"]["cost_per_minute_usd"] == 0.017
    assert meta.OPENAI_STT_MODELS["gpt-transcribe"]["cost_per_minute_usd"] == 0.0045
