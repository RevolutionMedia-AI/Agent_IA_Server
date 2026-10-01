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
import os
import struct
import time

import websockets
from websockets.exceptions import ConnectionClosed

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a hard dep in practice
    HAVE_NUMPY = False

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    _HAVE_SCIPY = False

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

# ponytail: 2026-10-01 — `intent=transcription`, NOT `?model=`.
# The GA Realtime endpoint answers a bare /v1/realtime with
# `invalid_request_error.missing_model`, which reads like "pass the
# transcription model on the URL". It does not:
#   ?model=gpt-4o-transcribe -> "is a transcription model and cannot be
#                               used as the realtime session model"
#   ?intent=transcription&model=... -> also rejected
#   ?model=<realtime model> + a `type: transcription` session.update ->
#       "Passing a transcription session update to a realtime session is
#        not allowed"
# `intent=transcription` alone is the supported form: OpenAI picks the
# transcription model from session.update -> audio.input.transcription.model,
# which is exactly where build_session_update() puts it.
REALTIME_WS_URL = (
    "wss://api.openai.com/v1/realtime?intent=transcription"
)

# ponytail: 2026-10-01 — 24000 is a HARD MINIMUM, not a documentation
# example. OpenAI rejected 8000 outright:
#   invalid_request_error.integer_below_min_value
#   "Invalid 'session.audio.input.format.rate': integer below minimum
#    value. Expected a value >= 24000, but got 8000 instead."
# So there is no "skip the resample" escape hatch: the 8 kHz mu-law Twilio
# hands us must always be converted. I previously suggested 8000 here as a
# first thing to try; that was wrong and it cost a deploy.
#
# The knob is kept only because the API accepts anything >= 24000 and
# 48000 is a legitimate higher-fidelity choice, which costs 2x the CPU.
TARGET_SAMPLE_RATE = int(
    os.getenv("OPENAI_TRANSCRIPTION_RATE_HZ", "24000")
)
SOURCE_SAMPLE_RATE = 8000
if TARGET_SAMPLE_RATE % SOURCE_SAMPLE_RATE or TARGET_SAMPLE_RATE < 24000:
    # Fail at import, not on the first call: a bad value would otherwise
    # cost every call in the container a dead transcription session.
    raise ValueError(
        f"OPENAI_TRANSCRIPTION_RATE_HZ={TARGET_SAMPLE_RATE} is invalid. "
        f"OpenAI's transcription session requires a rate >= 24000 that is a "
        f"multiple of {SOURCE_SAMPLE_RATE} (Twilio's native rate); "
        f"8000 is rejected by the API."
    )
UPSAMPLE = TARGET_SAMPLE_RATE // SOURCE_SAMPLE_RATE

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


class _StreamingResampler:
    """mu-law 8k -> PCM16 24k across chunk boundaries, without seams.

    resample_poly is a finite-impulse-response filter, so the first
    `half_len` output samples of any call depend on samples that were
    filtered away in the previous call. Resampling 20 ms chunks in
    isolation therefore puts an audible step at every 20 ms seam.

    Keeping the input history and dropping the output samples that
    correspond to it makes the stream equivalent to one resample_poly over
    the whole signal — which is what the test asserts.

    ponytail: half_len mirrors resample_poly's default filter length
    (10 * max(up, down) + 1). If that default ever changes, the seam
    assertion in the test is what catches it.
    """

    def __init__(self) -> None:
        # resample_poly's default filter is 10 * max(up, down) + 1 taps,
        # so it reaches HALF that many taps either side of a boundary.
        # 30 is a multiple of UPSAMPLE, which the phase math needs.
        self._half = 10 * UPSAMPLE
        self._hist = b""      # last _half input samples (left filter context)
        self._carry = b""     # 0..UPSAMPLE-1 samples held for a whole group
        self._in_seen = 0     # input samples consumed, including _hist
        self._emitted = 0     # output samples yielded so far

    def feed(self, mulaw: bytes) -> bytes:
        if not mulaw:
            return b""
        if UPSAMPLE == 1 or not _HAVE_SCIPY:
            return _mulaw_8k_to_pcm16_24k(mulaw)

        pcm_8k = ulaw2lin(mulaw, 2)
        if not pcm_8k:
            return b""

        # ponytail: 2026-10-01 — the phase correction. resample_poly puts
        # block-output k at block-input k/up, so its phase grid depends on
        # the block length. _half (30) + a 20 ms chunk (160) = 190, which
        # is NOT a multiple of 3, so the grid slid a third of a sample
        # every chunk. Over a second of speech that is a full cycle of
        # drift and the waveform stops matching its input. Consuming only
        # whole groups of UPSAMPLE samples keeps every block length a
        # multiple of 3 and the grid fixed.
        buf = self._carry + pcm_8k
        n_avail = len(buf) // 2
        use = (n_avail // UPSAMPLE) * UPSAMPLE
        if use == 0:
            self._carry = buf
            return b""
        self._carry = buf[use * 2:]
        self._in_seen += use

        block = self._hist + buf[:use * 2]
        n_block = len(block) // 2

        out = resample_poly(
            np.frombuffer(block, dtype="<i2"), up=UPSAMPLE, down=1
        )
        out = np.clip(out, -32768, 32767).astype("<i2")

        # Block output k corresponds to GLOBAL output S*UPSAMPLE + k, so
        # tracking _emitted in global terms is what makes "which of these
        # are new" answerable. Dropping `len(_hist)//2 * UPSAMPLE` from
        # the head every call instead removes the samples the previous
        # call HELD, and the stream runs at 2.43x instead of 3x.
        S = self._in_seen - n_block
        start_k = self._emitted - S * UPSAMPLE
        if start_k < 0:
            start_k = 0
        # Emit only outputs whose filter support ends at or before the last
        # input sample we hold; the rest are recomputed next call.
        limit_k = (n_block - 1 - self._half) * UPSAMPLE
        if limit_k < start_k:
            return b""

        self._emitted = S * UPSAMPLE + limit_k + 1
        self._hist = block[-self._half * 2:]
        return out[start_k:limit_k + 1].tobytes()


def _mulaw_8k_to_pcm16_24k(mulaw: bytes) -> bytes:
    """mu-law 8 kHz mono -> LINEAR16 PCM at TARGET_SAMPLE_RATE.

    ponytail: 2026-10-01 — this used to zero-stuff and linearly
    interpolate. That is NOT band-limited resampling: it leaves spectral
    images between 8 kHz and 24 kHz, and production heard the result as
    garbage transcripts ("One of the" from a Spanish caller, then a
    hallucination-rejected "No no no no"). scipy is already a dependency
    (requirements.txt), so use resample_poly's proper polyphase FIR with
    an anti-aliasing filter.

    At UPSAMPLE == 1 there is nothing to convert and the bytes pass
    through untouched, which is the point of the env-var escape hatch.

    Falls back to the naive path only if scipy is somehow missing, so a
    broken deploy degrades instead of dropping every call.
    """
    if UPSAMPLE == 1:
        return ulaw2lin(mulaw, 2)
    pcm_8k = ulaw2lin(mulaw, 2)  # PCM16 @ 8 kHz mono
    n_8k = len(pcm_8k) // 2
    if n_8k == 0:
        return b""
    if _HAVE_SCIPY:
        out = resample_poly(
            np.frombuffer(pcm_8k, dtype="<i2"), up=UPSAMPLE, down=1
        )
        return np.clip(out, -32768, 32767).astype("<i2").tobytes()

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
    # ponytail: 2026-10-01 — the resampler is stateful. resample_poly's
    # anti-aliasing FIR reaches 10*max(up,down) taps either side, so
    # resampling each 20 ms chunk in isolation left an unfiltered edge at
    # every boundary: audible clicks, and the transcribe model sees
    # discontinuities. Carry the filter history across chunks so the
    # concatenation is identical to resampling the whole stream at once.
    resampler = _StreamingResampler()

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
            pcm = resampler.feed(chunk)
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
    # ponytail: 2026-10-01 — DO NOT add `OpenAI-Beta: realtime=v1` here.
    # It was removed from openai_realtime.py because OpenAI graduated the
    # Realtime API to GA; the beta header now flips the server onto a
    # disabled beta path and the socket closes 4000 with
    # `invalid_request_error.beta_api_shape_disabled`. This adapter shipped
    # with the header (copied from an outdated docs snippet) and every call
    # failed on the first session.update. The GA endpoint accepts the exact
    # same session.update payload with no beta header at all.
    return {
        name: {
            "Authorization": "Bearer " + _ACTIVE_API_KEY[0],
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
                # ponytail: 2026-10-01 — this says "sent", not "open".
                # The previous wording logged the session as open
                # immediately after ws.send(), before the server had
                # accepted anything, so a rejected session.update still
                # printed "transcription session open". Reading that line
                # in production is how a hard rejection got mistaken for a
                # success twice. `session.updated` from the server is the
                # real confirmation and is logged when it arrives.
                log.info(
                    "[OPENAI_STT] session.update sent for %s "
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
                        # session.updated is the server confirming it
                        # accepted the config — the first hard evidence the
                        # session is actually transcribing.
                        if etype in (
                            "session.updated", "transcription_session.updated",
                        ):
                            log.info(
                                "[OPENAI_STT] session confirmed for %s "
                                "(model=%s rate=%d)",
                                session.session_key, model_id,
                                TARGET_SAMPLE_RATE,
                            )
                        # session.created/updated and keepalives carry
                        # nothing we act on. Log the rest so a contract
                        # change is visible instead of silent.
                        elif etype not in (
                            "session.created",
                            "transcription_session.created",
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
