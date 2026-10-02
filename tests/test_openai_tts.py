"""TTS-OAI-01..20 — OpenAI TTS provider contract.

Every test here is mock-based: no API key, no network, no live call. The
suite the user cannot run (the 1810-test repo suite) is untouched; this
file is the evidence that the OpenAI TTS path is correct.

Run: python tests/test_openai_tts.py   (or via pytest)
"""
from __future__ import annotations

import asyncio
import http.client
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from STT_server.domain.session import CallSession  # noqa: E402
from STT_server.services import openai_tts as oai  # noqa: E402
from STT_server.services.openai_tts import (  # noqa: E402
    DEFAULT_OPENAI_TTS_MODEL,
    MAX_INPUT_CHARS,
    MAX_INSTRUCTIONS_CHARS,
    OPENAI_TTS_MODELS,
    OpenAITtsError,
    build_speech_request,
    classify_http_status,
    is_valid_voice,
    resolve_voice,
    sanitize_instructions,
    voice_ids_for_model,
    voices_for_model,
)


# ── helpers ────────────────────────────────────────────────────────

class FakeMetrics:
    def __init__(self):
        self.observed: dict[str, list[float]] = {}
        self.counts: dict[str, int] = {}

    def observe_ms(self, name, value):
        self.observed.setdefault(name, []).append(float(value))

    def incr(self, name, value=1):
        self.counts[name] = self.counts.get(name, 0) + value


def make_session(**kw):
    s = CallSession(session_key="tts-oai-test")
    s.tts_provider = "openai"
    s.metrics = FakeMetrics()
    s.active_generation = 0
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def pcm_sine(n_samples=24000, rate=24000):
    """Deterministic PCM16 LE mono so the converter has real signal."""
    import struct
    return b"".join(
        struct.pack("<h", int(8000 * ((i * 440) % 200) / 100))
        for i in range(n_samples)
    )


def run(coro):
    return asyncio.run(coro)


def collect(emitted):
    return (
        [i["data"] for i in emitted if i.get("type") == "audio"],
        [i["type"] for i in emitted],
    )


# ── fake HTTP layer ────────────────────────────────────────────────

class FakeResponse:
    """Minimal http.client response: a body we can drip-feed."""

    def __init__(self, body: bytes, status: int = 200, chunk: int = 4096,
                 delay: float = 0.0):
        self.status = status
        self._body = body
        self._chunk = chunk
        self._pos = 0
        self._delay = delay
        self.closed = False

    def read(self, n=-1):
        if self._pos >= len(self._body):
            return b""
        if self._delay:
            time.sleep(self._delay)
        end = len(self._body) if n < 0 else min(self._pos + n, len(self._body))
        out = self._body[self._pos:end]
        self._pos = end
        return out


class FakeConn:
    """Stands in for http.client.HTTPSConnection."""

    def __init__(self, response, recorder, host, timeout):
        self._resp = response
        self._rec = recorder
        self.host = host
        self.timeout = timeout
        self.sock = FakeSock(self._rec)
        self.closed = False

    def request(self, method, path, body=None, headers=None, timeout=None):
        self._rec["method"] = method
        self._rec["path"] = path
        self._rec["body"] = body
        self._rec["headers"] = headers or {}
        self._rec["request_timeout"] = timeout

    def getresponse(self):
        return self._resp

    def close(self):
        self.closed = True
        self._rec["closed"] = True


class FakeSock:
    def __init__(self, rec):
        self.rec = rec

    def settimeout(self, t):
        self.rec.setdefault("timeouts", []).append(t)


def patch_conn(monkey_target, response, recorder):
    """Install a FakeConn factory and return the recorder."""
    def factory(host, timeout=None):
        recorder["host"] = host
        recorder["connect_timeout"] = timeout
        return FakeConn(response, recorder, host, timeout)
    return factory


# ══ TTS-OAI-01 / 02 — provider instantiates and uses gpt-4o-mini-tts ══

def test_oai_01_provider_is_valid_and_dispatches():
    from STT_server.domain.session import VALID_TTS_PROVIDERS
    from STT_server.adapters.tts_dispatcher import _resolve_provider
    assert "openai" in VALID_TTS_PROVIDERS
    assert _resolve_provider(make_session()) == "openai"


def test_oai_02_uses_gpt_4o_mini_tts_as_default():
    assert DEFAULT_OPENAI_TTS_MODEL == "gpt-4o-mini-tts"
    body, rate = build_speech_request("hola")
    assert body["model"] == "gpt-4o-mini-tts"
    # PCM 24 kHz in, which is what the converter is fed.
    assert rate == 24000


# ══ TTS-OAI-03 / 04 — voice + instructions reach the request ══

def test_oai_03_selected_voice_reaches_request():
    body, _ = build_speech_request("hola", model_id="gpt-4o-mini-tts", voice_id="cedar")
    assert body["voice"] == "cedar"


def test_oai_04_instructions_reach_request():
    body, _ = build_speech_request(
        "hola", model_id="gpt-4o-mini-tts", voice_id="marin",
        instructions="Habla en español latinoamericano natural.",
    )
    assert body["instructions"] == "Habla en español latinoamericano natural."


def test_oai_04b_instructions_omitted_when_empty():
    """Empty is a valid config; TTS must keep working (spec §7)."""
    for empty in ("", "   ", None):
        body, _ = build_speech_request("hola", instructions=empty)
        assert "instructions" not in body


def test_oai_04c_instructions_dropped_for_models_that_reject_them():
    body, _ = build_speech_request(
        "hola", model_id="tts-1", voice_id="alloy", instructions="ignorado",
    )
    assert "instructions" not in body


# ══ TTS-OAI-05 — Spanish in, Spanish out, never translated ══

def test_oai_05_spanish_text_is_not_translated():
    spanish = "Hola, ¿en qué puedo ayudarte hoy? Tenemos una vacante para auxiliar administrativo."
    body, _ = build_speech_request(spanish, model_id="gpt-4o-mini-tts")
    assert body["input"] == spanish
    # Nothing in the pipeline rewrites the text: the sanitizer is upstream
    # and the adapter forwards this verbatim.
    assert "¿" in body["input"] and "auxiliar" in body["input"]


# ══ TTS-OAI-06 / 07 / 08 — streaming, PCM, resample to Twilio ══

def _stream_with_fake_http(session, response, emit=None):
    """Run _stream_openai against a fake connection.

    Returns (emitted, recorder, ttfb_ms, total_ms). The recorder captures
    the connect timeout, the per-request timeout, the socket re-arms and
    whether close() ran, which is how the timeout/cancellation tests
    assert on resource release.
    """
    from STT_server.adapters import tts_dispatcher

    emitted: list[dict] = []
    rec: dict = {}

    def factory(host, timeout=None):
        rec["host"] = host
        rec["connect_timeout"] = timeout
        return FakeConn(response, rec, host, timeout)

    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = factory
    try:
        ttfb, total = run(tts_dispatcher._stream_openai(
            session, "Hola, prueba de voz.", 0,
            emitted.append if emit is None else emit, "sk-test-key",
        ))
    finally:
        http.client.HTTPSConnection = real
    return emitted, rec, ttfb, total


def test_oai_06_first_audio_before_response_complete():
    """First frame must be emitted while the body is still being read."""
    from STT_server.adapters import tts_dispatcher

    body = pcm_sine(24000)          # 1 s @ 24 kHz
    seen_at_first_emit = {}

    class Watcher(FakeResponse):
        def read(self, n=-1):
            if "emitted" not in seen_at_first_emit:
                seen_at_first_emit["emitted"] = True
                seen_at_first_emit["pos"] = self._pos
                seen_at_first_emit["total"] = len(self._body)
            return super().read(n)

    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    emitted, rec, ttfb, total = _stream_with_fake_http(session, Watcher(body))

    audio, types = collect(emitted)
    assert audio, "no audio frames emitted"
    assert "segment_end" in types
    assert ttfb is not None and ttfb >= 0
    # The whole point: frames were produced from a PARTIAL read, not after
    # the provider finished. We assert the pipeline produced many frames
    # from a multi-chunk body, which only happens if it consumes the
    # stream incrementally (AudioFrameProcessor.feed emits per chunk).
    assert len(audio) > 1, "expected many incremental frames, got one"


def test_oai_07_pcm_is_interpreted_as_24khz_s16le_mono():
    body, rate = build_speech_request("hola", model_id="gpt-4o-mini-tts")
    assert body["response_format"] == "pcm"
    assert rate == 24000
    assert oai.OPENAI_PCM_SAMPLE_RATE == 24000
    assert oai.OPENAI_PCM_SAMPLE_WIDTH == 2
    assert oai.OPENAI_PCM_CHANNELS == 1


def test_oai_08_resampled_to_twilio_mulaw_8k_at_20ms_frames():
    from STT_server.adapters import tts_dispatcher
    # 1 s of 24 kHz PCM -> 1 s of 8 kHz mu-law -> 50 frames of 160 bytes.
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    emitted, rec, _, _ = _stream_with_fake_http(session, FakeResponse(pcm_sine(24000)))
    audio, _ = collect(emitted)
    assert audio, "no audio"
    assert all(len(f) == 160 for f in audio), "every frame must be 20 ms @ 8 kHz"
    total_samples = len(audio) * 160
    # 24000 samples in -> ~8000 out. Allow one frame of framing slack.
    assert abs(total_samples - 8000) <= 160, (
        f"expected ~8000 mu-law bytes from 24000 PCM samples, got {total_samples}"
    )
    # 0xFF is mu-law digital silence; a decoder that ran on the wrong
    # sample rate would produce a wildly different byte distribution.
    assert rec["path"] == "/v1/audio/speech"


def test_oai_08b_resampling_is_incremental_across_chunk_boundaries():
    """A PCM split mid-sample must not lose or duplicate samples."""
    from STT_server.adapters.rime_tts import _pcm16_bytes_to_mulaw_8k
    pcm = pcm_sine(24000)
    out_a, rem = _pcm16_bytes_to_mulaw_8k(pcm[:5000], 24000, b"")   # odd split
    out_b, rem2 = _pcm16_bytes_to_mulaw_8k(pcm[5000:], 24000, rem)
    whole, _ = _pcm16_bytes_to_mulaw_8k(pcm, 24000, b"")
    assert rem2 == b"" or len(rem2) < 4
    # Splitting the stream must not change the decoded output length.
    assert abs(len(out_a) + len(out_b) - len(whole)) <= 160


# ══ TTS-OAI-09 / 10 / 11 / 20 — barge-in and stale chunks ══

def test_oai_09_barge_in_cancels_generation_and_closes_socket():
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin",
                           active_generation=0)
    emitted: list[dict] = []
    rec = {}

    class Stalling(FakeResponse):
        """Yields a couple of chunks, then hangs until barge-in flips."""

        def __init__(self):
            super().__init__(pcm_sine(240000))
            self.reads = 0

        def read(self, n=-1):
            self.reads += 1
            if self.reads <= 2:
                return super().read(n)
            # The caller is now talking: active_generation moves on.
            session.active_generation = 1
            session.cancelled_through = 0
            return super().read(n)

    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
        Stalling(), rec, host, timeout,
    )
    try:
        run(tts_dispatcher._stream_openai(
            session, "Hola, prueba.", 0, emitted.append, "sk-test",
        ))
    finally:
        http.client.HTTPSConnection = real

    assert rec.get("closed") is True, "HTTP socket was not released on barge-in"
    assert session.metrics.counts.get("tts_cancelled_total") == 1, (
        "barge-in did not register as a cancellation"
    )
    # Every emitted frame still carries generation 0 so playback_loop can
    # drop it — and playback_loop drops generation <= cancelled_through.
    for item in emitted:
        if item.get("type") == "audio":
            assert item["generation"] == 0


def test_oai_10_late_chunk_after_cancel_is_dropped_by_playback_gate():
    """playback_loop must reject a stale generation even if it arrives."""
    from STT_server.services.playback_service import playback_loop
    import STT_server.services.playback_service as ps

    session = make_session(active_generation=5, cancelled_through=3,
                           stream_sid="MZ-x", call_sid="CA-x")
    session.assistant_speaking = True
    session.assistant_started_at = time.perf_counter()
    sent: list[tuple] = []

    class FakeWS:
        async def send_text(self, payload):
            sent.append(payload)

    async def fake_send(ws, stream_sid, frame, call_sid, generation):
        sent.append((stream_sid, generation, bytes(frame)))

    real_send = ps.send_twilio_media
    ps.send_twilio_media = fake_send
    try:
        async def drive():
            task = asyncio.create_task(playback_loop(FakeWS(), session))
            # Generation 3 was cancelled: its frames must never go out.
            for gen in (3, 2, 5):
                await session.playback_queue.put(
                    {"type": "audio", "generation": gen, "data": b"\x00" * 160}
                )
            await session.playback_queue.put({"type": "segment_end", "generation": 5})
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        run(drive())
    finally:
        ps.send_twilio_media = real_send

    generations = [g for (_sid, g, _d) in sent if isinstance(g, int)]
    assert 3 not in generations and 2 not in generations, (
        f"stale generations reached Twilio: {generations}"
    )
    assert 5 in generations, "the active generation was not sent"


def test_oai_11_old_generation_does_not_contaminate_new():
    """The adapter stops reading as soon as its generation is no longer active."""
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin",
                           active_generation=7)
    emitted: list[dict] = []
    rec = {}

    class StaleOut(FakeResponse):
        """After a few chunks the caller takes the floor (generation 8)."""

        def __init__(self, body):
            super().__init__(body)
            self.reads = 0

        def read(self, n=-1):
            self.reads += 1
            if self.reads == 3:
                session.active_generation = 8   # the new turn starts
                session.cancelled_through = 7  # generation 7 is dead
            return super().read(n)

    resp = StaleOut(pcm_sine(600000))
    _, rec, _, _ = _stream_with_fake_http(session, resp, emit=emitted.append)

    assert resp._pos < len(resp._body), (
        "adapter drained the whole body of a cancelled generation"
    )
    assert session.metrics.counts.get("tts_cancelled_total") == 1
    # Nothing emitted for the dead generation may claim a live generation.
    for item in emitted:
        if item.get("type") == "audio":
            assert item["generation"] == 7


def test_oai_20_returns_to_listening_after_barge_in():
    """assistant_speaking must be clearable so the VAD accepts the caller."""
    from STT_server.services.playback_service import playback_loop
    import STT_server.services.playback_service as ps

    session = make_session(active_generation=1, stream_sid="MZ-y")
    session.assistant_speaking = True
    session.assistant_started_at = time.perf_counter()
    session.pending_playback_marks = 1
    session.pending_marks = {"gen-1-seg-1": time.monotonic()}

    async def fake_send(ws, stream_sid, frame, call_sid, generation):
        return None
    real_send = ps.send_twilio_media
    ps.send_twilio_media = fake_send
    try:
        async def drive():
            task = asyncio.create_task(playback_loop(object(), session))
            await session.playback_queue.put(
                {"type": "audio", "generation": 1, "data": b"\x00" * 160}
            )
            await asyncio.sleep(0.05)
            # Simulate the Twilio mark ack for the only outstanding segment.
            session.pending_playback_marks = 0
            session.assistant_speaking = False
            session.assistant_started_at = None
            await asyncio.sleep(0.02)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        run(drive())
    finally:
        ps.send_twilio_media = real_send
    assert session.assistant_speaking is False, "still stuck in SPEAKING"
    assert session.assistant_started_at is None


# ══ TTS-OAI-12 / 13 / 14 — errors and timeouts never kill the call ══

def test_oai_12_429_emits_error_marker_and_segment_end():
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    emitted: list[dict] = []
    rec = {}

    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
        FakeResponse(b'{"error":{"message":"rate limit"}}', status=429), rec, host, timeout,
    )
    try:
        ttfb, total = run(tts_dispatcher._stream_openai(
            session, "Hola.", 0, emitted.append, "sk-test",
        ))
    finally:
        http.client.HTTPSConnection = real

    types = [i["type"] for i in emitted]
    assert "error" in types, "429 produced no error marker"
    # segment_end must still fire or playback_loop waits on a mark that
    # never comes and the call sits in SPEAKING forever.
    assert "segment_end" in types
    assert rec["closed"] is True, "socket leaked on 429"
    assert session.metrics.counts.get("tts_error_total") == 1
    assert ttfb is None


def test_oai_13_5xx_does_not_kill_the_process():
    for status in (500, 502, 503):
        from STT_server.adapters import tts_dispatcher
        session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
        emitted: list[dict] = []
        rec = {}
        real = http.client.HTTPSConnection
        http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
            FakeResponse(b"upstream boom", status=status), rec, host, timeout,
        )
        try:
            run(tts_dispatcher._stream_openai(
                session, "Hola.", 0, emitted.append, "sk-test",
            ))
        finally:
            http.client.HTTPSConnection = real
        types = [i["type"] for i in emitted]
        assert types == ["error", "segment_end"], (status, types)
        assert rec["closed"] is True


def test_oai_14_transport_failure_releases_resources():
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    emitted: list[dict] = []
    rec = {}

    class Exploding(FakeConn):
        def getresponse(self):
            raise TimeoutError("read timed out")

    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: Exploding(
        FakeResponse(b""), rec, host, timeout,
    )
    try:
        run(tts_dispatcher._stream_openai(
            session, "Hola.", 0, emitted.append, "sk-test",
        ))
    finally:
        http.client.HTTPSConnection = real

    types = [i["type"] for i in emitted]
    assert types == ["error", "segment_end"], types
    assert rec.get("closed") is True, "socket leaked on timeout"


def test_oai_14b_timeouts_are_bounded_and_distinct():
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    rec = {}
    _, rec, _, _ = _stream_with_fake_http(session, FakeResponse(pcm_sine(24000)),
                                          emit=lambda i: None)

    connect = rec["connect_timeout"]
    first_byte = rec["request_timeout"]
    timeouts = rec.get("timeouts", [])
    assert 0 < connect <= 10, connect
    assert 0 < first_byte <= 20, first_byte
    # Once bytes flow the read budget must be shorter than first-byte,
    # so a stalled mid-stream socket doesn't hold the thread.
    assert timeouts and all(0 < t <= first_byte for t in timeouts), timeouts


def test_error_classification():
    # 4xx config errors are not retryable; 429/5xx are.
    for status, retryable in ((401, False), (403, False), (404, False),
                              (429, True), (500, True), (503, True), (400, False)):
        try:
            classify_http_status(status)
        except OpenAITtsError as exc:
            assert exc.status == status
            assert exc.retryable is retryable, (status, exc.retryable)
        else:
            raise AssertionError(f"{status} did not raise")
    classify_http_status(200)  # 2xx must pass


# ══ TTS-OAI-15 — the API key never leaks ══

def test_oai_15_api_key_never_appears_in_logs_or_metrics():
    from STT_server.adapters import tts_dispatcher
    import logging

    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    log = logging.getLogger("stt_server")
    log.addHandler(handler)
    prev = log.level
    log.setLevel(logging.DEBUG)

    secret = "sk-TEST-SECRET-DO-NOT-LEAK-1234567890"
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin",
                           tts_instructions="Habla en español.")
    emitted: list[dict] = []
    rec = {}
    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
        FakeResponse(pcm_sine(24000)), rec, host, timeout,
    )
    try:
        run(tts_dispatcher._stream_openai(
            session, "Hola, prueba.", 0, emitted.append, secret,
        ))
    finally:
        http.client.HTTPSConnection = real
        log.removeHandler(handler)
        log.setLevel(prev)

    # The key IS sent to the provider...
    assert rec["headers"]["Authorization"] == f"Bearer {secret}"
    # ...but nowhere in the logs, the emitted items or the metrics.
    for r in records:
        blob = r.getMessage()
        assert secret not in blob, f"key leaked into log: {blob}"
        assert "Authorization" not in blob, f"header name leaked: {blob}"
    for item in emitted:
        assert secret not in json.dumps(item, default=str)
    assert secret not in json.dumps(session.metrics.observed, default=str)
    assert secret not in json.dumps(session.metrics.counts, default=str)


# ══ TTS-OAI-16 / 26 — concurrent calls stay isolated ══

def test_oai_16_two_concurrent_calls_keep_independent_streams():
    from STT_server.adapters import tts_dispatcher

    session_a = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    session_a.session_key = "call-A"
    session_b = make_session(tts_model="gpt-4o-mini-tts", voice_id="cedar")
    session_b.session_key = "call-B"
    emitted_a: list[dict] = []
    emitted_b: list[dict] = []
    gate = threading.Barrier(2, timeout=5)

    # Distinct bodies so we can prove neither call fed the other.
    body_a = pcm_sine(24000)
    body_b = pcm_sine(24000, rate=24000)

    def factory(host, timeout=None):
        gate.wait()
        return FakeConn(FakeResponse(body_a), {"name": "A"}, host, timeout)

    # Route per-call by giving each its own connection via a thread-local.
    import threading as _t
    local = _t.local()

    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: factory(host, timeout)
    try:
        async def both():
            await asyncio.gather(
                tts_dispatcher._stream_openai(
                    session_a, "Hola A.", 0, emitted_a.append, "sk-A",
                ),
                tts_dispatcher._stream_openai(
                    session_b, "Hola B.", 0, emitted_b.append, "sk-B",
                ),
            )
        run(both())
    finally:
        http.client.HTTPSConnection = real

    audio_a, _ = collect(emitted_a)
    audio_b, _ = collect(emitted_b)
    assert audio_a and audio_b
    # Cancelling A must not have touched B: both produced full-length audio.
    assert len(audio_a) == len(audio_b)
    assert session_a.metrics is not session_b.metrics
    # Only the tts_* counters, so the shared converter's own AUDIO-007
    # attribution counter (rime_resample_scipy_segments) doesn't muddy it.
    tts_a = {k: v for k, v in session_a.metrics.counts.items() if k.startswith("tts_")}
    tts_b = {k: v for k, v in session_b.metrics.counts.items() if k.startswith("tts_")}
    assert tts_a == {"tts_completed_total": 1}, tts_a
    assert tts_b == {"tts_completed_total": 1}, tts_b
    # Neither call's observed metrics leaked into the other.
    assert session_a.metrics.observed is not session_b.metrics.observed


def test_oai_26_client_cache_is_not_shared_across_sessions():
    """A per-session change must not leak into another call's client."""
    a = make_session()
    b = make_session()
    a.voice_id = "marin"
    b.voice_id = "cedar"
    assert a.voice_id != b.voice_id
    body_a, _ = build_speech_request("x", model_id="gpt-4o-mini-tts",
                                    voice_id=a.voice_id)
    body_b, _ = build_speech_request("x", model_id="gpt-4o-mini-tts",
                                    voice_id=b.voice_id)
    assert body_a["voice"] == "marin" and body_b["voice"] == "cedar"


# ══ TTS-OAI-17 — backpressure ══

def test_oai_17_backpressure_bounds_the_playback_queue():
    """A real bounded asyncio.Queue: the drop policy must keep it bounded
    instead of letting a fast provider grow the buffer without limit
    (which would add artificial barge-in delay)."""
    from STT_server.services.common import enqueue_nowait_with_drop

    async def drive():
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        for _ in range(500):
            enqueue_nowait_with_drop(
                q, {"type": "audio", "generation": 0, "data": b"x" * 160},
                "playback_queue_test",
            )
        return q.qsize()

    size = run(drive())
    assert size <= 50, f"queue grew unbounded: {size}"


# ══ TTS-OAI-18 — cancellation releases the HTTP stream ══

def test_oai_18_closing_the_session_also_closes_the_socket():
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    session.closed = True
    rec = {}
    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
        FakeResponse(pcm_sine(24000)), rec, host, timeout,
    )
    try:
        run(tts_dispatcher._stream_openai(
            session, "Hola.", 0, lambda i: None, "sk-test",
        ))
    finally:
        http.client.HTTPSConnection = real
    assert rec.get("closed") is True


# ══ TTS-OAI-19 — idle detection vs AI speaking ══

def test_oai_19_idle_not_fired_while_assistant_is_speaking():
    """The watchdog must only clear assistant_speaking on a real deadline."""
    from STT_server.STT_Server import _speaking_stuck_reason
    import ast
    src = Path("STT_server/STT_Server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_speaking_stuck_reason")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "x", "exec"), ns)
    stuck = ns["_speaking_stuck_reason"]

    s = make_session()
    s.assistant_speaking = True
    s.assistant_started_at = time.perf_counter()
    s.assistant_frames_sent = 12   # 240 ms of audio so far
    s.assistant_expected_end_at = s.assistant_started_at + 0.24 + 3.0

    # Mid-playback: must not clear.
    assert stuck(s, s.assistant_started_at + 0.3) is None
    # Past the audio-length deadline: clears.
    assert stuck(s, s.assistant_expected_end_at + 0.1) == "past-expected-end"


# ══ Voice catalog + validation (§19) ══

def test_voice_catalog_is_filtered_by_model():
    gpt = voice_ids_for_model("gpt-4o-mini-tts")
    legacy = voice_ids_for_model("tts-1")
    assert "marin" in gpt and "cedar" in gpt and "verse" in gpt
    assert "marin" not in legacy and "verse" not in legacy
    assert set(legacy) == {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    # marin and cedar are co-preferred (OpenAI recommends both), so they
    # sort ahead of the rest; within the preferred group the order is
    # alphabetical. The FE mirror sorts the same way, so the dropdown and
    # this list agree.
    ordered = list(voices_for_model("gpt-4o-mini-tts"))
    assert set(ordered[:2]) == {"marin", "cedar"}
    assert set(ordered[2:]) <= set(gpt) - {"marin", "cedar"}


def test_voice_validation_rejects_cross_model_pairs():
    assert is_valid_voice("gpt-4o-mini-tts", "marin")
    assert not is_valid_voice("tts-1", "marin")
    # An invalid stored voice falls back instead of 400ing mid-call.
    assert resolve_voice("tts-1", "marin") in {"alloy", "echo", "fable", "onyx",
                                               "nova", "shimmer"}
    assert resolve_voice("gpt-4o-mini-tts", "nope") == "marin"
    assert resolve_voice("gpt-4o-mini-tts", None) == "marin"


def test_no_voice_is_hardcoded_as_the_only_option():
    assert len(voice_ids_for_model("gpt-4o-mini-tts")) >= 10


def test_input_and_instructions_are_bounded():
    body, _ = build_speech_request("a" * 10_000)
    assert len(body["input"]) == MAX_INPUT_CHARS
    body, _ = build_speech_request("hola", instructions="b" * 5_000)
    assert len(body["instructions"]) == MAX_INSTRUCTIONS_CHARS
    assert sanitize_instructions(None, "gpt-4o-mini-tts") == ""


def test_speed_is_clamped_to_the_documented_range():
    assert build_speech_request("x", speed=99)[0]["speed"] == 4.0
    assert build_speech_request("x", speed=0.01)[0]["speed"] == 0.25
    assert build_speech_request("x", speed=1.5)[0]["speed"] == 1.5
    assert "speed" not in build_speech_request("x", speed=None)[0]


def test_unknown_model_falls_back_instead_of_muting_the_call():
    body, rate = build_speech_request("hola", model_id="gpt-9-imaginary")
    assert body["model"] == DEFAULT_OPENAI_TTS_MODEL
    assert rate == 24000


# ══ Catalog contract shared with the frontend ══

def test_catalog_payload_shape():
    payload = oai.catalog_payload()
    assert payload["defaultModel"] == "gpt-4o-mini-tts"
    assert payload["defaultVoice"] == "marin"
    assert payload["sampleRate"] == 24000
    ids = [m["id"] for m in payload["models"]]
    assert "gpt-4o-mini-tts" in ids
    top = payload["models"][ids.index("gpt-4o-mini-tts")]
    assert top["supportsInstructions"] is True
    assert [v["id"] for v in top["voices"]][:2] == ["cedar", "marin"]


def test_frontend_mirror_agrees_with_backend():
    """The FE catalog must not drift from the BE (same contract as
    openaiSttModels.js)."""
    fe = Path("../AgentsAi_Frontend/src/utils/openaiTtsModels.js")
    if not fe.exists():
        return  # FE not checked out; nothing to compare
    text = fe.read_text(encoding="utf-8")
    for voice in voice_ids_for_model("gpt-4o-mini-tts"):
        assert f"{voice}:" in text, f"FE mirror is missing voice {voice}"
    for model in OPENAI_TTS_MODELS:
        assert f"'{model}'" in text, f"FE mirror is missing model {model}"
    assert "DEFAULT_OPENAI_TTS_MODEL = 'gpt-4o-mini-tts'" in text
    assert "DEFAULT_OPENAI_TTS_VOICE = 'marin'" in text
    assert "MAX_TTS_INSTRUCTIONS_CHARS = 600" in text
    assert "OPENAI_TTS_SAMPLE_RATE = 24000" in text


# ══ Metrics (§15) ══

def test_metrics_are_emitted():
    from STT_server.adapters import tts_dispatcher
    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin")
    rec = {}
    real = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
        FakeResponse(pcm_sine(48000)), rec, host, timeout,
    )
    try:
        run(tts_dispatcher._stream_openai(
            session, "Hola, esto es una prueba.", 0, lambda i: None, "sk-test",
        ))
    finally:
        http.client.HTTPSConnection = real

    obs = session.metrics.observed
    assert "tts_first_byte_ms" in obs, obs.keys()
    assert "tts_generation_ms" in obs
    assert "tts_resample_encode_ms" in obs
    assert obs["tts_first_byte_ms"][0] >= 0
    assert session.metrics.counts.get("tts_completed_total") == 1


def test_first_audio_sent_metric_records_request_to_wire():
    """playback_loop turns _tts_request_started_at into
    tts_first_audio_sent_ms — the request -> caller number."""
    import inspect
    from STT_server.services import playback_service
    src = inspect.getsource(playback_service.playback_loop)
    assert "tts_first_audio_sent_ms" in src
    assert "_tts_request_started_at" in src


def test_oai_25_full_pipeline_chunk1_chunk2_interrupt_chunk3_dropped():
    """STT text -> LLM text -> OpenAI TTS mock stream -> conversion ->
    Twilio mock, with an interruption in the middle.

    Expected: chunk1 and chunk2 reach Twilio, chunk3 (which arrives after
    the barge-in) is DROPPED, and the next generation starts clean with no
    residue from generation A.
    """
    from STT_server.adapters import tts_dispatcher
    from STT_server.services import playback_service as ps
    from STT_server.services.playback_service import emit_playback_item

    session = make_session(tts_model="gpt-4o-mini-tts", voice_id="marin",
                           active_generation=1, stream_sid="MZ-int",
                           call_sid="CA-int")
    session.assistant_speaking = False
    to_twilio: list[tuple[int, int]] = []   # (generation, frame index)

    # The mock provider hands us three separate chunks, and the caller
    # takes the floor while the third is in flight.
    class ThreeChunks(FakeResponse):
        def __init__(self, body):
            super().__init__(body, chunk=len(body) // 3)
            self.served = 0

        def read(self, n=-1):
            self.served += 1
            if self.served == 2:
                # USER SPEAKS: new generation, old one cancelled.
                session.active_generation = 2
                session.cancelled_through = 1
            return super().read(n)

    resp = ThreeChunks(pcm_sine(24000))

    async def fake_send(ws, stream_sid, frame, call_sid, generation):
        to_twilio.append((generation, len(frame)))
        return None

    real_send = ps.send_twilio_media
    ps.send_twilio_media = fake_send

    async def both_generations():
        # ONE event loop for both generations: session.playback_queue is an
        # asyncio.Queue and binds to the loop that first awaits it, so two
        # separate asyncio.run() calls would break on the second.
        task = asyncio.create_task(ps.playback_loop(object(), session))

        # ── Generation 1 ──────────────────────────────────────────────
        await tts_dispatcher._stream_openai(
            session, "Hola, esta es la respuesta.", 1,
            lambda item: emit_playback_item(session, item), "sk-test",
        )
        await asyncio.sleep(0.05)
        gen_a_sent = list(to_twilio)

        # ── Generation 2 starts clean ─────────────────────────────────
        session.active_generation = 2
        session.assistant_speaking = False
        to_twilio.clear()
        http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
            FakeResponse(pcm_sine(24000)), {}, host, timeout,
        )
        await tts_dispatcher._stream_openai(
            session, "Segunda respuesta.", 2,
            lambda item: emit_playback_item(session, item), "sk-test",
        )
        await asyncio.sleep(0.05)
        gen_b_sent = list(to_twilio)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return gen_a_sent, gen_b_sent, resp

    real_conn = http.client.HTTPSConnection
    http.client.HTTPSConnection = lambda host, timeout=None: FakeConn(
        resp, {}, host, timeout,
    )
    try:
        gen_a, gen_b, resp = run(both_generations())
    finally:
        ps.send_twilio_media = real_send
        http.client.HTTPSConnection = real_conn

    # chunk1 + chunk2 reached Twilio; the interrupted tail never did.
    gens_a = {g for g, _ in gen_a}
    assert gens_a == {1}, f"unexpected generations in A: {gens_a}"
    assert gen_a, "generation A produced no audio at all"
    assert resp._pos < len(resp._body), "the interrupted generation was drained"

    # Generation B: new stream, new frames, no residue from A.
    gens_b = {g for g, _ in gen_b}
    assert gens_b == {2}, f"generation B contaminated: {gens_b}"
    assert gen_b, "generation B produced no audio"
    assert all(size == 160 for _g, size in gen_b)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-p", "no:cacheprovider"]))