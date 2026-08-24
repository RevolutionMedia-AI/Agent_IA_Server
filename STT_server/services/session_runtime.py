import asyncio
import contextlib
import json
import logging
import os
import time

from fastapi import WebSocket

from STT_server.config import IDLE_SILENCE_TIMEOUT_SEC, MAX_CALL_DURATION_SEC
from STT_server.db_tools import list_tools as db_list_tools
from STT_server.domain.session import CallSession
from STT_server.services.common import enqueue_with_drop
from STT_server.services.usage_store import has_user_stored_key, record_call


log = logging.getLogger("stt_server")

sessions: dict[str, CallSession] = {}

# ponytail: idle/duration monitor polling cadence. Landed here (instead of
# config.py) so the constant is colocated with the only two functions that
# read it. Kept tiny — the goal is "responsive shutdown on tear-down",
# not a config knob for operators.
IDLE_MONITOR_POLL_SEC = 5.0
MAX_DURATION_MONITOR_POLL_SEC = 10.0

# ponytail: agents.json lookup for the per-call usage record. We need
# the agent's stt_provider/llm_provider (the session only carries
# tts_provider); loading the whole file here is fine — it's small and
# the lookup happens once per call end.
_AGENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "agents.json",
)


def _load_agent_providers(agent_id: str | None) -> tuple[str | None, str | None]:
    if not agent_id:
        return (None, None)
    try:
        with open(_AGENTS_FILE, "r", encoding="utf-8") as f:
            agents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return (None, None)
    if not isinstance(agents, list):
        return (None, None)
    for a in agents:
        if isinstance(a, dict) and a.get("id") == agent_id:
            return (a.get("stt_provider"), a.get("llm_provider"))
    return (None, None)


def _load_agent_tools(agent_id: str | None, user_id: str | None = None) -> list[dict]:
    """Load the tools available to one agent at call start.

    ponytail: 010_agent_tools.sql moved the storage layer to Postgres
    via STT_server.db_tools.list_tools(agent_id=...). The JSONB `?|`
    operator inside list_tools handles the "shared tool whose
    assignments array contains this agent_id" branch in a single
    indexed query — no more client-side post-processing of the full
    tool list, and crucially no more on-the-fly backfill writes to
    a file the operator might restart away.

    Two scopes (unchanged from the legacy contract):
      1. Per-agent tools — rows with agent_id == agent_id. Always
         included; the agent that owns the row gets it implicitly.
      2. Shared tools — rows with agent_id == "__shared__" and a
         matching user_id. ONLY included when the agent id is in the
         tool's `assignments` list. Operators pick exactly which
         agents can invoke each shared tool via the Assign / Unassign
         buttons in the edit modal.

    Legacy "auto-include on first read" behaviour is gone — the
    one-time backfill in db_tools.backfill_from_json() handles
    pre-migration rows before this function ever runs.
    """
    if not agent_id or not user_id:
        return []
    return db_list_tools(user_id, agent_id=agent_id)


def track_task(session: CallSession, task: asyncio.Task) -> asyncio.Task:
    session.tasks.add(task)
    task.add_done_callback(session.tasks.discard)
    return task


async def register_session(session: CallSession) -> None:
    sessions[session.session_key] = session
    # ponytail: lazy-init the per-call StageTimer here, exactly once
    # per session. STT_Server.py's start-event handler is the only
    # caller, so this runs before any adapter can emit its first
    # stage. wait_signals.mark_stage() also lazy-inits as a safety
    # net for code paths that bypass register_session (debug
    # endpoints, tests).
    if session.stage_timer is None:
        from STT_server.services._instrumentation import StageTimer
        session.stage_timer = StageTimer(
            call_id=session.session_key,
            turn_id=0,
            generation=session.active_generation,
        )
        # ponytail: alias the underscore-prefixed name so audio_ingest.py
        # (which already attaches to session._stage_timer) reuses the
        # same timer instead of constructing a parallel one. Same
        # instance, same stage timestamps — the dashboard sees one
        # timeline per call.
        session._stage_timer = session.stage_timer  # type: ignore[attr-defined]
    # ponytail: P2 round-3 — attach the per-call metrics container so
    # all adapters can `session.metrics.incr(...)`/`.observe_ms(...)`
    # without the attach_metrics idempotency dance. Idempotent.
    from STT_server.services.audio_metrics import attach_metrics
    attach_metrics(session, call_id=session.session_key)
    # ponytail: H2 from the call-flow audit. Mirror the in-memory
    # registration to Postgres so a server crash doesn't orphan
    # the CallSession forever (memory leak across restarts). Best
    # effort: the in-memory registration already happened and the
    # call works without the DB write. A subsequent redeploy will
    # see the row via list_open_sessions() and can mark it closed.
    #
    # db_call_sessions.register_session is a sync function (psycopg2
    # is sync) returning a dict on success / None on skip. We
    # previously `await`-ed it, which raised
    # `object dict can't be used in 'await' expression` and broke
    # every call. Just call it.
    try:
        from STT_server import db_call_sessions
        db_call_sessions.register_session(
            session.session_key,
            tenant_id=session.tenant_id,
            call_sid=session.call_sid,
            preferred_language=session.preferred_language,
            tts_provider=session.tts_provider,
            custom_prompt=session.custom_prompt,
            started_at=session.started_at,
        )
    except Exception as exc:  # noqa: BLE001 — DB write must never block a call
        log.warning("[runtime] DB register_session failed for %s: %s",
                    session.session_key, exc)


async def cleanup_session(session: CallSession, ws: WebSocket) -> None:
    if session.closed:
        return

    session.closed = True

    # ponytail: P2 round-3 — emit the consolidated end-of-call summary
    # BEFORE the usage record and queue teardown so all adapter-recorded
    # counters are still attached to the session. Idempotent + best-effort:
    # nothing here can crash cleanup.
    try:
        from STT_server.services.call_summary import emit_call_summary
        await emit_call_summary(log, session)
    except Exception:
        log.exception("emit_call_summary failed")

    # ponytail: write the per-call usage record BEFORE we lose the
    # session data. The aggregation in /api/usage reads from this
    # ledger, so dropping the write means the call is invisible to
    # billing.
    try:
        _record_usage_for(session)
    except Exception as exc:  # noqa: BLE001 — never crash cleanup on billing
        log.warning("[usage] record failed for %s: %s", session.session_key, exc)

    with contextlib.suppress(Exception):
        await enqueue_with_drop(session.stt_audio_queue, None, "stt_audio_queue")

    with contextlib.suppress(Exception):
        await enqueue_with_drop(session.realtime_audio_queue, None, "realtime_audio_queue")

    session.generation_changed.set()

    # AUDIO-005: free every per-call audio buffer so the dataclass holds
    # zero audio bytes after cleanup. speech_frames is already a bounded
    # deque (maxlen=SPEECH_FRAMES_MAX) but we still .clear() to release
    # the bytes eagerly. vad_buffer is an unbounded bytearray in the happy
    # path it drains to ~zero, but a misbehaving caller could leave a
    # few KB behind; clear it too. pre_speech_frames is bounded by
    # PRE_SPEECH_FRAMES but the slot-holding bytes can be large.
    session.speech_frames.clear()
    session.vad_buffer.clear()
    session.pre_speech_frames.clear()
    session.stt_mute_buffer.clear()
    # ponytail: AUDIO-005 — reset the once-per-session cap warning
    # flag. cleanup_session may run after a subsequent call reuses
    # the same session_key (rare but possible in tests); reset
    # defensively.
    session._speech_frames_cap_warned = False

    for task in list(session.tasks):
        task.cancel()

    sessions.pop(session.session_key, None)

    try:
        await asyncio.gather(*session.tasks, return_exceptions=True)
    except Exception:
        pass

    # ponytail: H2 from the call-flow audit. Mark the call closed
    # in Postgres so list_open_sessions() at startup can recover
    # any sessions that didn't close cleanly (server crash, deploy).
    # Best effort — if the DB write fails, the in-memory pop already
    # ran and the call is fully torn down. The row will be recovered
    # by a future startup sweep.
    #
    # db_call_sessions.close_session is sync (psycopg2 is sync);
    # don't await it.
    try:
        from STT_server import db_call_sessions
        db_call_sessions.close_session(session.session_key)
    except Exception as exc:  # noqa: BLE001 — DB write must never block cleanup
        log.warning("[runtime] DB close_session failed for %s: %s",
                    session.session_key, exc)

    try:
        await ws.close()
    except Exception:
        pass

    # ponytail: 2026-08-14 audio review — flush + close the per-call
    # capture files (A_inworld_<callSid>.mulaw + B_twilio_<callSid>.mulaw)
    # so the B file sees the LAST frame and the operator can diff
    # A vs B vs the AMR recording without waiting for the
    # process to exit. No-op when TTS_AUDIO_CAPTURE_DIR is unset.
    try:
        from STT_server.services.audio_capture import close_all
        close_all()
    except Exception:
        pass


def _record_usage_for(session: CallSession) -> None:
    """Best-effort: write one usage row per call. Skipped silently
    when there's no user_id (anonymous test calls) or no start ts.
    """
    user_id = session.user_id
    if not user_id:
        return
    if session.started_at is None:
        return
    stt_provider, llm_provider = _load_agent_providers(session.agent_id)
    providers = {
        "stt": stt_provider,
        "llm": llm_provider,
        "tts": session.tts_provider,
    }
    used_platform = any(
        p and not has_user_stored_key(user_id, p)
        for p in providers.values()
    )
    record_call(
        user_id=user_id,
        agent_id=session.agent_id,
        tenant_id=session.tenant_id,
        call_sid=session.call_sid,
        started_at=session.started_at,
        ended_at=time.time(),
        providers=providers,
        used_platform_keys=used_platform,
    )


async def monitor_idle_silence(session: CallSession, ws: WebSocket) -> None:
    """Detect prolonged silence and (optionally) prompt the caller, then hang up.

    Two modes:
      * Legacy / global (idle_enabled is None or False, or any required
        field missing): the same single-timeout-then-close behaviour the
        platform had before 008_agent_idle_settings.sql. Configured by
        IDLE_SILENCE_TIMEOUT_SEC. Existing agents are unaffected.
      * Per-agent (idle_enabled=True): a small state machine that
        increments ``attempts_played`` each time the configured silence
        interval elapses, speaks the configured message via TTS, and
        closes the websocket after ``idle_max_attempts`` prompts plus
        ``idle_disconnect_timeout_sec`` of continued silence.

    State transitions (per-agent mode):
      t=0                          attempts=0, deadline=first_timeout
      deadline expires             attempts+=1; speak first_message;
                                   deadline=subsequent_timeout
      deadline expires             attempts+=1; speak final_message;
                                   deadline=subsequent_timeout
      …repeat until attempts==max_attempts…
      deadline expires once more   wait disconnect_timeout; close WS

    Any user or assistant activity that bumps ``last_activity_at`` resets
    attempts to 0 and the deadline to ``idle_first_timeout_sec`` — the
    caller came back, start the silence clock from scratch.
    """
    # ponytail: per-agent flow only runs when the agent row opted in AND
    # every required field is positive. Negative / None values mean the
    # operator either left the feature off or set a knob to a bad value;
    # the safest fallback is the global single-timeout behaviour the
    # platform had before this feature shipped.
    per_agent = bool(
        session.idle_enabled
        and (session.idle_first_timeout_sec or 0) > 0
        and (session.idle_subsequent_timeout_sec or 0) > 0
        and (session.idle_max_attempts or 0) > 0
    )

    if not per_agent:
        if IDLE_SILENCE_TIMEOUT_SEC <= 0:
            return
        try:
            while not session.closed:
                if session.assistant_speaking:
                    await asyncio.sleep(IDLE_MONITOR_POLL_SEC)
                    continue
                remaining = IDLE_SILENCE_TIMEOUT_SEC - (time.monotonic() - session.last_activity_at)
                if remaining <= 0:
                    log.info(
                        "Idle silence timeout (%.0fs) en %s, cerrando llamada",
                        IDLE_SILENCE_TIMEOUT_SEC,
                        session.session_key,
                    )
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    break
                await asyncio.sleep(min(remaining, IDLE_MONITOR_POLL_SEC))
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("Error en monitor_idle_silence (global)")
        return

    # ── per-agent state machine ──────────────────────────────────────
    # ponytail: attempts_played and last_seen_activity live in local
    # closure state, NOT on the session. They are diagnostic / private
    # to this monitor and don't need cross-task visibility — the session
    # already exposes last_activity_at for the activity check. Putting
    # them on the session would have polluted the dataclass with
    # monitor-only state and made the schema harder to reason about.
    first_timeout = session.idle_first_timeout_sec or 0
    sub_timeout = session.idle_subsequent_timeout_sec or 0
    disc_timeout = session.idle_disconnect_timeout_sec or 0
    max_attempts = session.idle_max_attempts or 1
    first_message = (session.idle_first_message or "").strip() or "Are you still there?"
    final_message = (session.idle_final_message or "").strip() or "I'm not hearing a response, so I'm going to disconnect the call."

    attempts_played = 0
    deadline = first_timeout
    last_seen_activity = session.last_activity_at

    # ponytail: lazily imported here so the module doesn't take the
    # circular-import cost when this monitor is never reached (e.g. the
    # global-mode early-return above on a legacy agent).
    from STT_server.services.turn_manager import run_tts_with_retries

    try:
        while not session.closed:
            # 1) Wait while the assistant is mid-utterance (VAD also
            #    blocks user input here, so we don't fire an extra
            #    prompt on top of the TTS we just queued).
            if session.assistant_speaking:
                await asyncio.sleep(IDLE_MONITOR_POLL_SEC)
                continue

            # 2) Did the user just speak (or the assistant did outside
            #    of our prompt)? Reset the silence clock.
            if session.last_activity_at != last_seen_activity:
                last_seen_activity = session.last_activity_at
                attempts_played = 0
                deadline = first_timeout

            # 3) Has the current deadline elapsed?
            remaining = deadline - (time.monotonic() - last_seen_activity)
            if remaining > 0:
                await asyncio.sleep(min(remaining, IDLE_MONITOR_POLL_SEC))
                continue

            # 4) Deadline elapsed. Speak or disconnect.
            if attempts_played >= max_attempts:
                log.info(
                    "Idle silence: %d attempts played + %.0fs disconnect timeout in %s, cerrando llamada",
                    attempts_played, disc_timeout, session.session_key,
                )
                # The spec says "I'm going to disconnect the call" — the
                # disconnect-timeout window gives the caller a chance to
                # hear that warning. We don't speak again here, we just
                # wait then close.
                if disc_timeout > 0:
                    await asyncio.sleep(disc_timeout)
                    if session.closed:
                        break
                    # user came back during the disconnect window
                    if session.last_activity_at != last_seen_activity:
                        last_seen_activity = session.last_activity_at
                        attempts_played = 0
                        deadline = first_timeout
                        continue
                try:
                    await ws.close()
                except Exception:
                    pass
                break

            attempts_played += 1
            # First attempt uses the first_message; every subsequent
            # attempt uses the final_message (the operator can still
            # keep them identical if they want a single repeated prompt).
            text = first_message if attempts_played == 1 else final_message
            log.info(
                "Idle silence: speaking prompt %d/%d in %s (len=%d)",
                attempts_played, max_attempts, session.session_key, len(text),
            )
            # ponytail: mark the assistant as speaking BEFORE the TTS
            # provider responds so the VAD ignores the user's voice
            # during the TTFB window. Same pattern as play_initial_greeting.
            session.assistant_speaking = True
            session.assistant_started_at = time.perf_counter()
            session.last_activity_at = time.monotonic()
            last_seen_activity = session.last_activity_at
            try:
                await run_tts_with_retries(
                    session, text, session.active_generation,
                )
            except Exception:
                log.exception(
                    "Idle prompt TTS failed in %s — falling back to close",
                    session.session_key,
                )
            finally:
                session.assistant_speaking = False
                session.assistant_started_at = None

            # After TTS, reset the silence clock so the user has the
            # full next interval to respond. If they spoke while we were
            # playing (impossible — VAD blocks them — but defensive),
            # last_activity_at would have moved and the next tick resets.
            last_seen_activity = session.last_activity_at
            deadline = sub_timeout
            await asyncio.sleep(IDLE_MONITOR_POLL_SEC)
    except asyncio.CancelledError:
        return
    except Exception:
        log.exception("Error en monitor_idle_silence (per-agent)")


async def monitor_max_call_duration(session: CallSession, ws: WebSocket) -> None:
    """Hard-timeout: close the call if it exceeds MAX_CALL_DURATION_SEC.

    This prevents phantom calls that consume infinite STT/LLM units when
    the idle-silence monitor cannot fire (e.g. one party is always speaking,
    or the audio stream stays active without meaningful conversation).
    """
    if MAX_CALL_DURATION_SEC <= 0:
        return
    try:
        while not session.closed:
            await asyncio.sleep(MAX_DURATION_MONITOR_POLL_SEC)
            if session.closed:
                return
            if session.started_at is None:
                continue
            elapsed = time.time() - session.started_at
            if elapsed >= MAX_CALL_DURATION_SEC:
                log.warning(
                    "Max call duration (%.0fs) exceeded in %s, force-closing. "
                    "started_at=%.1f elapsed=%.1f",
                    MAX_CALL_DURATION_SEC,
                    session.session_key,
                    session.started_at,
                    elapsed,
                )
                try:
                    await ws.close()
                except Exception:
                    pass
                break
    except asyncio.CancelledError:
        return
    except Exception:
        log.exception("Error en monitor_max_call_duration")


# ── Self-test ──────────────────────────────────────────────────────────
# ponytail: keep one runnable check behind a real audio+TTS dependency is
# impractical. This drives the state machine with a fake session + ws and
# uses an injected speak hook so we can verify the attempts / messages /
# disconnect ordering without spinning up a TTS provider. If the state
# machine regresses (e.g. wrong message on attempt 2, never disconnects,
# activity-reset stops working) the assert fails immediately.
async def _demo() -> None:
    import dataclasses

    @dataclasses.dataclass
    class _FakeWS:
        closed: bool = False
        close_calls: int = 0
        async def close(self):
            self.closed = True
            self.close_calls += 1

    # fake session: only the fields the monitor touches need real values.
    sess = CallSession(session_key="demo", active_generation=1)
    sess.idle_enabled = True
    sess.idle_first_timeout_sec = 1
    sess.idle_first_message = "are you still there?"
    sess.idle_subsequent_timeout_sec = 1
    sess.idle_final_message = "hanging up now."
    sess.idle_disconnect_timeout_sec = 1
    sess.idle_max_attempts = 2
    ws = _FakeWS()

    # The monitor's poll cadence is 5s. For the demo we don't want to
    # wait 5s between every state-machine tick — shrink it to 50ms so
    # the whole flow (3 timeouts + 1 disconnect window) finishes in
    # ~3-4s instead of ~20s.
    import STT_server.services.session_runtime as _self
    orig_poll = _self.IDLE_MONITOR_POLL_SEC
    _self.IDLE_MONITOR_POLL_SEC = 0.05

    # replace run_tts_with_retries for the duration of the test. The
    # monitor imports it lazily from STT_server.services.turn_manager
    # inside its body, so we patch that namespace directly (NOT the
    # session_runtime module — the symbol doesn't live there).
    # turn_manager has transitive deps (openai_llm → openai SDK) that may
    # not be installed in a thin dev env; pre-seed sys.modules with a
    # stub so the import doesn't blow up the demo. Real deployments
    # already have the SDK installed and the stub is a no-op there.
    import sys as _sys
    import types as _types
    for _name in ("openai", "STT_server.adapters.openai_llm"):
        if _name not in _sys.modules:
            _sys.modules[_name] = _types.ModuleType(_name)
    _fake_openai = _sys.modules["openai"]
    _fake_openai.OpenAI = lambda *a, **kw: None  # not invoked in demo
    _sys.modules["STT_server.adapters.openai_llm"].build_messages = lambda *a, **kw: []
    _sys.modules["STT_server.adapters.openai_llm"].call_llm = lambda *a, **kw: ""
    _sys.modules["STT_server.adapters.openai_llm"].stream_llm_reply_sync = lambda *a, **kw: iter(())

    spoken: list[str] = []
    async def fake_tts(_sess, text, _gen):
        spoken.append(text)
        await asyncio.sleep(0.001)  # simulate a fast TTS round-trip
    import STT_server.services.turn_manager as _tm
    _orig_tts = _tm.run_tts_with_retries
    _tm.run_tts_with_retries = fake_tts
    try:
        await _self.monitor_idle_silence(sess, ws)
    finally:
        _tm.run_tts_with_retries = _orig_tts
        _self.IDLE_MONITOR_POLL_SEC = orig_poll

    assert ws.close_calls == 1, f"expected 1 close, got {ws.close_calls}"
    # max_attempts=2 → first prompt, then final prompt, then disconnect.
    assert spoken == ["are you still there?", "hanging up now."], (
        f"unexpected prompt order: {spoken}"
    )
    log.info(
        "[demo] monitor_idle_silence OK — closed=%d prompts=%s",
        ws.close_calls, spoken,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_demo())