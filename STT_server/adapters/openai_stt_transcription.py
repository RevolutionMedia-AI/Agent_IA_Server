"""OpenAI STT adapter: Realtime TRANSCRIPTION session (cascade, no assistant audio).

Twilio hands us mu-law 8 kHz on ``session.stt_audio_queue``. We open the
OpenAI Realtime WebSocket in ``type: "transcription"`` mode — text out, no
spoken reply — and stream the caller's audio in. The LLM turn, tool calls
and TTS stay in ``turn_manager`` exactly as they do for Deepgram, Inworld
and AssemblyAI; this adapter only turns audio into text.

This is the cascade counterpart to ``openai_realtime.py``, which is a
speech-to-speech session (one model does STT + LLM + TTS). Do not confuse
the two: this one must NOT be handed tools or an output modality.

Contract notes that are easy to get wrong, all from
https://developers.openai.com/api/docs/guides/realtime-transcription:

- ``audio.input.format`` is ``{"type": "audio/pcm", "rate": 24000}``.
  24 kHz, not the 8 kHz Twilio gives us, so every chunk is resampled.
- ``turn_detection`` must be null/omitted. The model does not support
  ``server_vad`` or ``semantic_vad``; we commit the turn ourselves via
  ``input_audio_buffer.commit`` when our own VAD says the caller stopped
  talking. The trigger is ``session.stt_turn_end_seq``, which
  ``audio_ingest`` bumps at end-of-speech — same VAD, same adaptive noise
  floor, no second detector to drift out of sync.
- ``gpt-live-transcribe`` / ``gpt-transcribe`` take PLURAL ``languages``.
  Sending ``language`` alongside it is rejected.
- ``gpt-realtime-whisper`` is the legacy model: it keeps singular
  ``language`` and does NOT accept ``prompt`` on GA Realtime sessions.
  Getting that wrong fails the whole ``session.update``.
- ``delay`` (the latency/accuracy dial) applies to the two streaming models
  only. ``gpt-transcribe`` is committed-turn and must never receive it.

Model metadata, the latency dial, cost and the UI estimates all live in
``services/openai_stt_models.py``. Nothing about a specific model is
duplicated in this file.
"""
import asyncio
import base64
import json
import logging
import struct
import time

import websockets
from websockets.exceptions import ConnectionClosed

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a hard dep in practice
    HAVE_NUMPY = False

from STT_server.config import (
    DEFAULT_CALL_LANGUAGE,
    STT_RECONNECT_BASE_DELAY_MS,
    STT_RECONNECT_MAX_ATTEMPTS,
    STT_RECONNECT_MAX_DELAY_MS,
)
from STT_server.domain.language import normalize_supported_language
from STT_server.domain.session import CallSession
from STT_server.services import openai_stt_models as meta
from STT_server.services.audio_codec import ulaw2lin
from STT_server.services.credentials_resolver import resolve_for_session

log = logging.getLogger("stt_server")

REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"
TARGET_SAMPLE_RATE = 24000
SOURCE_SAMPLE_RATE = 8000
UPSAMPLE = TARGET_SAMPLE_RATE // SOURCE_SAMPLE_RATE  # 3

DEFAULT_MODEL_ID = meta.DEFAULT_OPENAI_STT_MODEL

# Numeric encoding of the dial so it can ride along as a gauge. -1 = the
# model has no dial (committed-turn), so the series reads "not applicable"
# rather than colliding with a real level.
LATENCY_MODE_CODES = {
    "minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4,
}

# Transcription sessions, not voice agents. Re-exported because the routing
# in STT_Server and the catalog tests import these names from here.
TRANSCRIPTION_MODELS = tuple(meta.OPENAI_STT_MODELS)

# Same ceiling as inworld_stt_realtime: close the WS if the connection
# opens but never yields a transcript, so the reconnect loop and
# announce_stt_failure_once TTS fallback both get a chance to run.
STT_INACTIVITY_TIMEOUT_S = 60


def _mulaw_8k_to_pcm16_24k(mulaw: bytes) -> bytes:
    """mu-law 8 kHz mono -> LINEAR16 PCM 24 kHz mono (int16 little-endian).

    Same shape as the Inworld adapter's 8k->16k helper: original samples
    land on every Nth output slot and the slots between are filled by
    linear interpolation, so the rate is right without inventing
    high-frequency content that was never in the source.
    """
    pcm_8k = ulaw2lin(mulaw, 2)
    n_8k = len(pcm_8k) // 2
    if n_8k == 0:
        return b""

    if HAVE_NUMPY:
        src = np.frombuffer(pcm_8k, dtype="<i2")
        n_out = UPSAMPLE * n_8k
        out = np.zeros(n_out, dtype="<i2")
        out[0::UPSAMPLE] = src
        if n_8k > 1:
            pair = src.astype(np.int32)
            a = pair[:-1]
            b = pair[1:]
            for k in range(1, UPSAMPLE):
                # linear ramp from a to b across the intermediate slots
                out[k::UPSAMPLE][: n_8k - 1] = (
                    a + ((b - a) * k) // UPSAMPLE
                ).astype("<i2")
        return out.tobytes()

    src = struct.unpack(f"<{n_8k}h", pcm_8k)
    out = [0] * (UPSAMPLE * n_8k)
    for i, s in enumerate(src):
        nxt = src[i + 1] if i + 1 < n_8k else s
        for k in range(UPSAMPLE):
            out[UPSAMPLE * i + k] = s + ((nxt - s) * k) // UPSAMPLE
    return struct.pack(f"<{len(out)}h", *out)


def _resolve_api_key(session: CallSession) -> str:
    creds = resolve_for_session(session, "stt", "openai")
    return (creds.get("api_key") or "").strip()


def _resolve_model(session: CallSession) -> str:
    model = (getattr(session, "stt_model", None) or "").strip()
    return model or DEFAULT_MODEL_ID


def _resolve_latency_mode(session: CallSession, model_id: str) -> str | None:
    """The delay value to actually send, or None to omit the field.

    Reads the agent's stored dial and normalizes it. A committed-turn model
    always yields None, which is how we guarantee `delay` is never sent to
    gpt-transcribe even if a stale row carries a value.
    """
    stored = getattr(session, "stt_latency_mode", None)
    return meta.resolve_latency_mode(model_id, stored)


def build_session_update(
    model_id: str,
    language: str,
    latency_mode: str | None = None,
    context: str | None = None,
    keywords: list[str] | None = None,
) -> dict:
    """The one session.update that opens a transcription session.

    Split out so tests can pin the exact wire shape — the failure mode this
    whole file exists to avoid is a session that opens and then never
    transcribes because one field was wrong.

    Field-inclusion rules, driven by services/openai_stt_models.py:
      - ``delay``      only for models with a dial (never gpt-transcribe)
      - ``language``   singular for gpt-realtime-whisper, plural otherwise
      - ``prompt``     only for models that support context
      - ``keywords``   only for models that support keywords
    """
    spec = meta.get(model_id)
    if spec is None:
        raise ValueError(
            f"Unsupported OpenAI STT model {model_id!r}; expected one of "
            + ", ".join(sorted(meta.OPENAI_STT_MODELS))
        )

    transcription: dict = {"model": model_id}

    if model_id in meta.SINGULAR_LANGUAGE_MODELS:
        transcription["language"] = language
    else:
        transcription["languages"] = [language]

    # Committed-turn models have no dial; resolve_latency_mode returns None
    # for them so the key is omitted entirely rather than sent as null.
    delay = meta.resolve_latency_mode(model_id, latency_mode)
    if delay is not None:
        transcription["delay"] = delay

    if spec.get("supports_context"):
        text = (context or "").strip()
        if text:
            transcription["prompt"] = text
    if spec.get("supports_keywords"):
        # The API rejects a keyword containing <, >, CR or LF, so filter
        # rather than let one bad row kill the whole session.update.
        clean = [
            k.strip() for k in (keywords or [])
            if k and k.strip() and not any(
                bad in k for bad in ("<", ">", "\r", "\n")
            )
        ]
        if clean:
            transcription["keywords"] = clean[:50]

    return {
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": TARGET_SAMPLE_RATE,
                    },
                    "transcription": transcription,
                    # gpt-live-transcribe has no server_vad / semantic_vad.
                    # We commit turns from our own VAD instead.
                    "turn_detection": None,
                }
            },
        },
    }


async def _audio_sender(ws, session: CallSession) -> None:
    """Pump mu-law into the transcription session, committing turns.

    Commit is driven by ``session.stt_turn_end_seq`` rather than a local
    energy detector: audio_ingest already decides end-of-speech against
    the per-call adaptive noise floor, and a second detector here would
    disagree with it on exactly the noisy lines that matter.
    """
    last_committed_seq = session.stt_turn_end_seq
    queue = session.stt_audio_queue

    while not session.closed:
        # ponytail: the seq is checked on a short poll, not only when
        # audio arrives. audio_ingest enqueues the media payload BEFORE it
        # runs the VAD that bumps stt_turn_end_seq, so a sender that only
        # checked after each append could read the seq before the bump and
        # then block on an empty queue forever — the turn would never be
        # committed, no .completed event would arrive, and the agent would
        # go silent mid-conversation.
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            chunk = None
        else:
            if chunk is None:
                # Cleanup sentinel from the session teardown path.
                return
            pcm = _mulaw_8k_to_pcm16_24k(chunk)
            if pcm:
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }))
                sent_this_turn = True

        seq = session.stt_turn_end_seq
        if seq != last_committed_seq:
            last_committed_seq = seq
            # Commit even with no audio of our own this turn: the append
            # for the turn may already be on the wire and OpenAI needs the
            # commit to emit the final transcript.
            try:
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.commit",
                }))
            except ConnectionClosed:
                return


def _connect_kwargs() -> dict:
    """Header kwarg name differs across websockets versions (<10 used
    ``extra_headers``). Probe once instead of carrying two copies of the
    receive loop.
    """
    import inspect

    try:
        params = inspect.signature(websockets.connect).parameters
        name = (
            "additional_headers"
            if "additional_headers" in params
            else "extra_headers"
        )
    except (TypeError, ValueError):  # pragma: no cover
        name = "additional_headers"
    return {
        name: {
            "Authorization": "Bearer " + _ACTIVE_API_KEY[0],
            "OpenAI-Beta": "realtime=v1",
        },
        "ping_interval": 20,
        "ping_timeout": 20,
    }


# ponytail: single-slot holder so _connect_kwargs can read the key without
# threading it through every call site. Set once per connect.
_ACTIVE_API_KEY = [""]


async def run_realtime_stt(
    session: CallSession,
    on_transcript,
    on_failure,
) -> None:
    """Same signature as the other cascade STT adapters so the dispatcher
    in STT_Server can route this one in unchanged.
    """
    api_key = _resolve_api_key(session)
    if not api_key:
        # Never return silently: process_transcripts would block on an
        # empty queue and the caller would hear the greeting then nothing.
        log.error(
            "[OPENAI_STT] session=%s: no OpenAI api_key resolved "
            "(user_id=%s agent_id=%s). Add the key in Settings -> API or "
            "change stt_provider on the agent.",
            getattr(session, "session_key", "?"),
            getattr(session, "user_id", None),
            getattr(session, "agent_id", None),
        )
        await on_failure(session)
        return

    _ACTIVE_API_KEY[0] = api_key

    model_id = _resolve_model(session)
    if model_id not in TRANSCRIPTION_MODELS:
        log.warning(
            "[OPENAI_STT] session=%s stt_model=%r is not a supported "
            "transcription id; using the default %s. The agent row needs "
            "saving from the modal.",
            session.session_key, model_id, DEFAULT_MODEL_ID,
        )
        model_id = DEFAULT_MODEL_ID

    latency_mode = _resolve_latency_mode(session, model_id)
    language = normalize_supported_language(
        session.preferred_language or DEFAULT_CALL_LANGUAGE or "en"
    )
    session.stt_model = model_id
    session.stt_latency_mode = latency_mode

    attempt = 0
    while not session.closed:
        sender_task: asyncio.Task | None = None
        received_any = False
        watchdog_task: asyncio.Task | None = None
        try:
            async with websockets.connect(
                REALTIME_WS_URL, **_connect_kwargs()
            ) as ws:
                await ws.send(json.dumps(build_session_update(
                    model_id, language, latency_mode=latency_mode,
                )))
                sender_task = asyncio.create_task(_audio_sender(ws, session))
                log.info(
                    "[OPENAI_STT] transcription session open for %s "
                    "(model=%s latency_mode=%s lang=%s rate=%d)",
                    session.session_key, model_id,
                    latency_mode or "n/a", language, TARGET_SAMPLE_RATE,
                )
                _metrics = getattr(session, "metrics", None)
                if _metrics is not None:
                    try:
                        _metrics.gauge("stt_rate_hz", float(TARGET_SAMPLE_RATE))
                    except Exception:
                        pass

                async def _inactivity_watchdog() -> None:
                    try:
                        await asyncio.sleep(STT_INACTIVITY_TIMEOUT_S)
                        if not received_any and not session.closed:
                            log.warning(
                                "[OPENAI_STT] no transcripts in %ss for "
                                "%s (model=%s) — closing WS to trigger "
                                "reconnect / announce_stt_failure_once",
                                STT_INACTIVITY_TIMEOUT_S,
                                session.session_key, model_id,
                            )
                            await ws.close(
                                code=1000,
                                reason="inactivity-no-transcripts",
                            )
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        log.exception(
                            "[OPENAI_STT] watchdog error in %s",
                            session.session_key,
                        )

                watchdog_task = asyncio.create_task(_inactivity_watchdog())

                while not session.closed:
                    try:
                        raw = await ws.recv()
                    except ConnectionClosed:
                        if not received_any:
                            log.warning(
                                "[OPENAI_STT] WS closed without any "
                                "transcript for %s",
                                session.session_key,
                            )
                        break

                    if isinstance(raw, bytes):
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(msg, dict):
                        continue

                    etype = msg.get("type") or ""

                    if etype == "error" or isinstance(msg.get("error"), dict):
                        log.error(
                            "[OPENAI_STT] session error in %s: %s",
                            session.session_key,
                            msg.get("error") or msg,
                        )
                        break

                    is_delta = etype.endswith(
                        "transcription.delta"
                    )
                    is_done = etype.endswith(
                        "transcription.completed"
                    )
                    if not (is_delta or is_done):
                        # session.created/updated and keepalives carry
                        # nothing we act on. Log the rest so a contract
                        # change is visible instead of silent.
                        if etype not in (
                            "session.created",
                            "session.updated",
                            "transcription_session.created",
                            "transcription_session.updated",
                            "ping",
                            "pong",
                        ):
                            log.debug(
                                "[OPENAI_STT] unhandled event %s in %s",
                                etype, session.session_key,
                            )
                        continue

                    text = (msg.get("delta") if is_delta
                            else msg.get("transcript")) or ""
                    text = text.strip()
                    if not text:
                        continue

                    received_any = True
                    if watchdog_task is not None and not watchdog_task.done():
                        watchdog_task.cancel()
                    attempt = 0

                    # 2026-10-01 — REAL measured STT latency, keyed so the
                    # numbers can be aggregated per model and per dial.
                    # t0 is the VAD end-of-speech bump in audio_ingest, so
                    # these are true end-of-speech -> transcript delays,
                    # not the UI's static estimates.
                    _now = time.monotonic()
                    if _metrics is None:
                        _metrics = getattr(session, "metrics", None)
                    if _metrics is not None:
                        _since_vad = (
                            (_now - session.stt_turn_end_at) * 1000.0
                            if getattr(session, "stt_turn_end_at", None)
                            else None
                        )
                        if _since_vad is not None and _since_vad >= 0:
                            # Distinct series per kind: merging them is what
                            # would make a later p50/p95/p99 useless.
                            _metrics.observe_ms(
                                "stt_final_ms" if is_done else "stt_partial_ms",
                                _since_vad,
                            )
                            if not is_done:
                                # record the dial once per turn, on the
                                # first partial
                                _metrics.gauge(
                                    f"stt_latency_mode:{model_id}",
                                    float(
                                        LATENCY_MODE_CODES.get(latency_mode, -1)
                                    ),
                                )

                    await on_transcript({
                        "text": text,
                        "language": language,
                        "is_final": is_done,
                        "speech_final": is_done,
                        "source": "openai_transcription",
                    })

                if watchdog_task is not None and not watchdog_task.done():
                    watchdog_task.cancel()

            if session.closed:
                return

            if not received_any:
                await on_failure(session)
                return

        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "[OPENAI_STT] error in %s (model=%s)",
                session.session_key, model_id,
            )
        finally:
            for task in (sender_task, watchdog_task):
                if task is not None:
                    task.cancel()
            for task in (sender_task, watchdog_task):
                if task is not None:
                    try:
                        await task
                    except BaseException:
                        pass

        if session.closed:
            return

        attempt += 1
        if attempt > STT_RECONNECT_MAX_ATTEMPTS:
            await on_failure(session)
            return
        delay_ms = min(
            STT_RECONNECT_MAX_DELAY_MS,
            STT_RECONNECT_BASE_DELAY_MS * (2 ** (attempt - 1)),
        )
        log.warning(
            "[OPENAI_STT] reconnecting in %sms (attempt %s/%s) for %s",
            delay_ms, attempt, STT_RECONNECT_MAX_ATTEMPTS,
            session.session_key,
        )
        await asyncio.sleep(delay_ms / 1000.0)
