import json
import logging
import time
import urllib.error
import urllib.request

from openai import OpenAI

from STT_server.config import MAX_HISTORY_MESSAGES, MAX_RESPONSE_TOKENS
from STT_server.domain.language import detect_language, get_language_instruction,                  pop_streaming_segments
from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider, resolve_for_session
from STT_server.services.thread_pool import to_thread as _to_thread
from STT_server.utils.safe_http import UnsafeURLError, validate_public_url


log = logging.getLogger("stt_server")


# SSRF guard. Every per-user base_url (anthropic / gemini / minimax
# custom-tenant endpoints) is funneled through _safe_base() so a
# malicious user can't point the SDK at loopback / cloud metadata /
# internal VPC IPs. Raises UnsafeURLError on rejection.
def _safe_base(creds: dict | None, default: str) -> str:
    raw = (creds or {}).get("base_url")
    candidate = (raw.strip().rstrip("/") if raw else default.rstrip("/"))
    validate_public_url(candidate)
    return candidate


# ponytail: per-user client cache. Keyed by (provider, key, base_url)
# so two users on the same deploy each get their own client, and the
# MiniMax client (which carries a base_url) doesn't collide with the
# default OpenAI one (which doesn't). Busting happens implicitly when
# the entry's credentials change — the cache key includes them.
_client_cache: dict[str, object] = {}

_DEFAULT_BASE_URLS = {
    "openai":    "",                            # SDK default
    "minimax":   "https://api.minimax.io/v1",   # OpenAI-compat endpoint
    "anthropic": "https://api.anthropic.com",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta",
}
# ponytail: _DEFAULT_MODELS is the absolute last-resort fallback for
# sessions where the agent row has llm_model=NULL (which the backfill
# in migration 005 should never produce again). Kept here so legacy
# sessions don't crash; new sessions must have llm_model from the
# agent row.
_DEFAULT_MODELS = {
    "minimax":   "MiniMax-M3",
    "anthropic": "claude-sonnet-4-5",
    "gemini":    "gemini-2-5-flash",
}


def _session_provider(session: CallSession) -> str:
    """Read llm_provider off the session, defaulting to 'openai' for
    any session that pre-dates the field (tenants, test calls, etc).
    Unknown values fall through to OpenAI — better than blowing up
    a live call because someone fat-fingered a provider id.
    """
    raw = getattr(session, "llm_provider", None)
    if not raw:
        return "openai"
    p = str(raw).strip().lower()
    return p if p in _DEFAULT_BASE_URLS else "openai"


def _resolve_model(session: CallSession, provider: str) -> str:
    """Pick the model id to send to the provider API.

    ponytail: per-agent only. The agent row's llm_model is the single
    source of truth. The per-provider defaults in _DEFAULT_MODELS are
    only consulted if llm_model is empty/null, which the backfill
    migration guarantees won't happen for any new agent. If both are
    missing, raise — fail loud, never silently fall back to a
    system-wide default that would mask user config errors.
    """
    explicit = getattr(session, "llm_model", None)
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    fallback = _DEFAULT_MODELS.get(provider)
    if fallback:
        return fallback
    raise RuntimeError(
        f"LLM provider '{provider}' has no model configured for this "
        f"session. Agent row must set llm_model."
    )


# ─── OpenAI ────────────────────────────────────────────────────────

def _openai_client(session: CallSession) -> OpenAI:
    creds = resolve_for_session(session, "llm", "openai")
    key = creds.get("api_key")
    # ponytail: per-user key only. No more `_default_openai_client`
    # that silently fell back to a system env var — if the user didn't
    # upload their key, the caller has to handle None and bail loudly.
    if not key:
        raise RuntimeError(
            "OpenAI API key not configured for this user. Upload your "
            "OpenAI key via Settings → API or the inline field in ModalAgents."
        )
    cached = _client_cache.get(("openai", key, ""))
    if cached is not None:
        return cached
    client = OpenAI(api_key=key)
    _client_cache[("openai", key, "")] = client
    return client


# ─── MiniMax (OpenAI-compatible, custom base_url) ────────────────

def _minimax_client(session: CallSession) -> OpenAI:
    creds = resolve_for_session(session, "llm", "minimax")
    key = creds.get("api_key")
    if not key:
            raise RuntimeError(
                "MiniMax not configured. Upload your key via Settings → API or the ModalAgents inline field."
            )
    try:
        base_url = _safe_base(creds, _DEFAULT_BASE_URLS["minimax"])
    except UnsafeURLError as exc:
        raise RuntimeError(
            f"MiniMax base_url rejected by SSRF guard: {exc}"
        )
    cache_key = ("minimax", key, base_url)
    cached = _client_cache.get(cache_key)
    if cached is not None:
        return cached
    # ponytail: the same openai SDK works against MiniMax because they
    # expose an OpenAI-compatible surface. The base_url is what tells
    # the SDK to route there instead of api.openai.com.
    client = OpenAI(api_key=key, base_url=base_url)
    _client_cache[cache_key] = client
    return client


# ─── Anthropic (REST, no extra SDK dep) ───────────────────────────

def _anthropic_config(session: CallSession) -> tuple[str, str, str]:
    creds = resolve_for_session(session, "llm", "anthropic")
    key = creds.get("api_key")
    if not key:
        raise RuntimeError(
            "Anthropic not configured. Upload your key via Settings → API or the ModalAgents inline field."
        )
    try:
        base = _safe_base(creds, _DEFAULT_BASE_URLS["anthropic"])
    except UnsafeURLError as exc:
        raise RuntimeError(f"Anthropic base_url rejected by SSRF guard: {exc}")
    return key, base, _resolve_model(session, "anthropic")


def _anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split Anthropic's required structure: top-level `system` string +
    `messages` list of {role, content}. Returns (system, msgs).
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            out.append({"role": role, "content": content})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, out


def _anthropic_call_sync(messages: list[dict], config: tuple[str, str, str],
                         session: CallSession | None = None) -> str:
    """One-shot Anthropic POST to /v1/messages. Returns the assistant
    text or a friendly fallback on any error.
    """
    key, base, model = config
    system_text, msgs = _anthropic_messages(messages)
    # ponytail: per-agent knobs override the platform defaults.
    # None (legacy agents, no override) → keep the 0.2 / MAX_RESPONSE_TOKENS
    # we've been shipping for months. The exact numbers are in the
    # 006_agent_runtime_params.sql migration + AGENTS / Settings UI.
    body = {
        "model": model,
        "max_tokens": getattr(session, "llm_max_tokens", None) or MAX_RESPONSE_TOKENS,
        "temperature": getattr(session, "llm_temperature", None)
            if getattr(session, "llm_temperature", None) is not None else 0.2,
        "messages": msgs,
    }
    if system_text:
        body["system"] = system_text
    try:
        req = urllib.request.Request(
            f"{base}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.warning("[LLM] anthropic HTTP %s: %s", exc.code, _safe_body(exc)[:200])
        return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
    except Exception:
        log.exception("[LLM] anthropic error")
        return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
    content = data.get("content") or []
    chunks = [c.get("text", "") for c in content if c.get("type") == "text"]
    return " ".join(chunks).strip() or "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"


def _anthropic_stream_sync(
    messages: list[dict],
    config: tuple[str, str, str],
    should_stop,
    on_first_segment,
    emit_segment,
    emit_done,
    session: CallSession | None = None,
):
    """Anthropic streams via SSE on /v1/messages with stream=true. Each
    event line starts with `event: <type>\ndata: <json>\n\n`. We only
    care about content_block_delta (text deltas) and message_stop.
    """
    key, base, model = config
    system_text, msgs = _anthropic_messages(messages)
    body = {
        "model": model,
        "max_tokens": getattr(session, "llm_max_tokens", None) or MAX_RESPONSE_TOKENS,
        "temperature": getattr(session, "llm_temperature", None)
            if getattr(session, "llm_temperature", None) is not None else 0.2,
        "messages": msgs,
        "stream": True,
    }
    if system_text:
        body["system"] = system_text
    full_reply = ""
    pending = ""
    try:
        req = urllib.request.Request(
            f"{base}/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            current_event = None
            for raw_line in resp:
                if should_stop():
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    current_event = None
                    continue
                if line.startswith("event: "):
                    current_event = line[7:].strip()
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                etype = evt.get("type") or current_event
                if etype == "content_block_delta":
                    delta = evt.get("delta") or {}
                    text = delta.get("text") or ""
                    if not text:
                        continue
                    full_reply += text
                    pending += text
                    ready, pending = pop_streaming_segments(pending)
                    for seg in ready:
                        on_first_segment()
                        emit_segment(seg)
                elif etype == "message_stop":
                    break
        if not should_stop():
            ready, _ = pop_streaming_segments(pending, force=True)
            for seg in ready:
                on_first_segment()
                emit_segment(seg)
        emit_done(full_reply.strip())
        # ponytail: log the LLM's full reply so the operator can
        # confirm the steering tags are actually being emitted.
        # If you see "(sighs)" / "(pause)" / etc. in this log,
        # the LLM is generating tags and the TTS should render
        # them. If you see plain prose, the prepend didn't work
        # or the model isn't following the hint.
        log.info(
            "[LLM] anthropic reply session=%s len=%d preview=%r",
            session.session_key,
            len(full_reply),
            full_reply[:300],
        )
        return full_reply.strip(), None
    except Exception as exc:
        log.exception("[LLM] anthropic stream error")
        return full_reply.strip(), str(exc)


# ─── Gemini (REST, no extra SDK dep) ──────────────────────────────

def _gemini_config(session: CallSession) -> tuple[str, str, str]:
    creds = resolve_for_session(session, "llm", "gemini")
    key = creds.get("api_key")
    if not key:
        raise RuntimeError(
            "Gemini not configured. Upload your key via Settings → API or the ModalAgents inline field."
        )
    try:
        base = _safe_base(creds, _DEFAULT_BASE_URLS["gemini"])
    except UnsafeURLError as exc:
        raise RuntimeError(f"Gemini base_url rejected by SSRF guard: {exc}")
    return key, base, _resolve_model(session, "gemini")


def _gemini_contents(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Gemini wants `contents` with role=user|model and a separate
    `systemInstruction` for the system prompt.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return contents, system_text


def _gemini_call_sync(messages: list[dict], config: tuple[str, str, str],
                      session: CallSession | None = None) -> str:
    key, base, model = config
    contents, system_text = _gemini_contents(messages)
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": getattr(session, "llm_temperature", None)
                if getattr(session, "llm_temperature", None) is not None else 0.2,
            "maxOutputTokens": getattr(session, "llm_max_tokens", None) or MAX_RESPONSE_TOKENS,
        },
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    url = f"{base}/models/{model}:generateContent?key={key}"
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.warning("[LLM] gemini HTTP %s: %s", exc.code, _safe_body(exc)[:200])
        return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
    except Exception:
        log.exception("[LLM] gemini error")
        return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
    candidates = data.get("candidates") or []
    if not candidates:
        return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    chunks = [p.get("text", "") for p in parts if p.get("text")]
    return " ".join(chunks).strip() or "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"


def _gemini_stream_sync(
    messages: list[dict],
    config: tuple[str, str, str],
    should_stop,
    on_first_segment,
    emit_segment,
    emit_done,
    session: CallSession | None = None,
):
    """Gemini streams via streamGenerateContent?alt=sse. Same SSE shape
    as Anthropic but the JSON envelope differs (candidates[0].content.parts).
    """
    key, base, model = config
    contents, system_text = _gemini_contents(messages)
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": getattr(session, "llm_temperature", None)
                if getattr(session, "llm_temperature", None) is not None else 0.2,
            "maxOutputTokens": getattr(session, "llm_max_tokens", None) or MAX_RESPONSE_TOKENS,
        },
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    url = f"{base}/models/{model}:streamGenerateContent?alt=sse&key={key}"
    full_reply = ""
    pending = ""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw_line in resp:
                if should_stop():
                    break
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    evt = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                candidates = evt.get("candidates") or []
                if not candidates:
                    continue
                parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
                for p in parts:
                    text = p.get("text")
                    if not text:
                        continue
                    full_reply += text
                    pending += text
                    ready, pending = pop_streaming_segments(pending)
                    for seg in ready:
                        on_first_segment()
                        emit_segment(seg)
        if not should_stop():
            ready, _ = pop_streaming_segments(pending, force=True)
            for seg in ready:
                on_first_segment()
                emit_segment(seg)
        emit_done(full_reply.strip())
        return full_reply.strip(), None
    except Exception as exc:
        log.exception("[LLM] gemini stream error")
        return full_reply.strip(), str(exc)


def _safe_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "(no body)"


# ─── Message building (provider-agnostic) ──────────────────────────

def build_messages(session: CallSession, user_text: str) -> list[dict]:
    # ponytail: the previous version fell back to a hardcoded Tigo
    # Panama / Camila system prompt when session.custom_prompt was
    # empty. The user explicitly asked for "ningún prompt genérico de
    # prueba" — drop the fallback. If the agent row has no prompt,
    # raise so the call fails loud instead of speaking as Camila.
    custom_prompt = getattr(session, 'custom_prompt', None)
    if not custom_prompt or not custom_prompt.strip():
        raise RuntimeError(
            f"Agent {getattr(session, 'agent_id', '<none>')} has no system_prompt "
            f"configured. Set one in the FE (Agents → Edit → System prompt) "
            f"and redeploy."
        )
    log.info("[LLM] Using custom_prompt for session=%s (len=%d)", session.session_key, len(custom_prompt))
    # ponytail: detect whether the TTS steering hint is at the top of
    # the system message. If you see "False" in this log, the prepend
    # in STT_Server.py didn't run and the LLM will produce flat
    # prose without steering tags.
    _hint_marker = "[TTS Steering"
    log.info(
        "[LLM] custom_prompt HEAD session=%s tts_hint_present=%s preview=%r",
        session.session_key,
        _hint_marker in custom_prompt,
        custom_prompt[:160],
    )
    messages = [
        {"role": "system", "content": custom_prompt.strip()},
    ]

    if session.collected_data:
        collected_items = ", ".join(f"{k}: {v}" for k, v in session.collected_data.items())
        messages.append(
            {
                "role": "system",
                "content": (
                    "User state already collected in this session: "
                    + collected_items
                    + ". Do not ask for these details again."
                ),
            }
        )

    # Count how many times the assistant has already asked for the order
    # number so the LLM can decide to escalate rather than loop.
    _ORDER_PHRASES = ("order number", "order #", "número de orden", "numero de pedido")
    ask_count = 0
    for entry in session.history:
        if entry["role"] == "assistant":
            lowered = entry["content"].lower()
            if any(phrase in lowered for phrase in _ORDER_PHRASES):
                ask_count += 1
    if ask_count >= 2:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"WARNING: You have already asked for the order number {ask_count} times in this call. "
                    "The speech recognition system is having difficulty capturing the digits. "
                    "Do NOT ask again. Transfer the caller to a live agent immediately using TRANSFER_AGENT."
                ),
            }
        )

    # ponytail: anti-loop genérico. gpt-4o-mini a temperature baja
    # tiende a imitar literalmente los diálogos de ejemplo que el
    # custom_prompt del agente pueda contener. Cuando el caller
    # responde "sí, adelante" después de un intro, el modelo produce
    # el mismo intro en vez de avanzar la conversación. Este system
    # message cita el último reply del asistente y le prohíbe
    # repetirlo — el LLM sigue el system prompt (prepended) más
    # que el ejemplo embebido (background).
    n_assistant = sum(1 for e in session.history if e["role"] == "assistant")
    if n_assistant >= 1:
        last_assistant = next(
            (e["content"] for e in reversed(session.history) if e["role"] == "assistant"),
            "",
        )
        messages.append({
            "role": "system",
            "content": (
                "CRITICAL ANTI-LOOP: You have already spoken in this call. "
                "Read the conversation history before responding. "
                "NEVER repeat the same opening greeting, self-introduction, "
                "or permission request if the user has already responded. "
                "If the user just said 'yes' / 'go ahead' / 'adelante' / 'si' / "
                "'ok', MOVE DIRECTLY to the next question or action — do NOT "
                "re-introduce yourself, re-disclose call recording, or "
                "re-ask permission to proceed. Your previous reply was:\n\n"
                f"\"{last_assistant[:400]}\""
            ),
        })

    messages.extend(session.history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_text})
    return messages


# ─── Entry points (dispatch by session.llm_provider) ──────────────

async def call_llm(session: CallSession, user_text: str) -> str:
    messages = build_messages(session, user_text)
    provider = _session_provider(session)
    if provider == "anthropic":
        config = _anthropic_config(session)
        return await _to_thread(_anthropic_call_sync, messages, config, session)
    if provider == "gemini":
        config = _gemini_config(session)
        return await _to_thread(_gemini_call_sync, messages, config, session)
    # OpenAI-compatible: openai or MiniMax.
    try:
        client = _openai_client(session) if provider == "openai" else _minimax_client(session)
    except Exception:
        log.exception("[LLM] %s client init failed", provider)
        return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"
    if client is None:
        raise RuntimeError(
            "No LLM provider configured. Define the right env var or upload your key in Settings → API."
        )
    model = _resolve_model(session, provider)

    def sync_call() -> str:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=getattr(session, "llm_temperature", None)
                    if getattr(session, "llm_temperature", None) is not None else 0.2,
                max_tokens=getattr(session, "llm_max_tokens", None) or MAX_RESPONSE_TOKENS,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception:
            log.exception("LLM ERROR")
            return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"

    return await _to_thread(sync_call)


def stream_llm_reply_sync(
    messages: list[dict],
    should_stop,
    emit_segment,
    emit_done,
    on_first_segment,
    client,
    *,
    provider: str = "openai",
    tools: list[dict] | None = None,
    execute_tool_callback=None,
    session: CallSession | None = None,
) -> tuple[str, str | None, list[dict] | None]:
    """Streaming LLM call.

    `client` is passed in by the wrapper (turn_manager.stream_llm_reply_with_tts)
    so the right per-user OpenAI client is used. The previous version tried
    to read the session from a module-level single-element list, which is a
    latent bug — only one call at a time could be in flight.

    `provider` lets the caller (which knows the session.llm_provider) tell
    us whether to route through Anthropic/Gemini SSE instead of the
    OpenAI streaming endpoint.

    `tools` is a list of OpenAI-style function definitions that the LLM
    can call during the conversation.

    `execute_tool_callback` is an async function that executes a tool and
    returns the result. If provided and the LLM calls a tool, the third
    return value will be the list of tool calls instead of None.

    `session` is optional but, when passed, lets the per-agent runtime
    knobs (llm_temperature, llm_max_tokens) reach the outbound request.
    When None, we use the platform defaults (0.2 / MAX_RESPONSE_TOKENS).
    """
    if provider == "anthropic":
        config = _anthropic_config_from_client(client)
        result = _anthropic_stream_sync(messages, config, should_stop, on_first_segment, emit_segment, emit_done, session)
        return result[0], result[1], None
    if provider == "gemini":
        config = _gemini_config_from_client(client)
        result = _gemini_stream_sync(messages, config, should_stop, on_first_segment, emit_segment, emit_done, session)
        return result[0], result[1], None
    if provider == "minimax":
        # The client passed in might be an OpenAI-compat one — fall through
        # to the standard chat.completions.create path.
        pass

    if client is None:
        return "", "No LLM provider configured. Define the right env var or upload your key in Settings → API."

    full_reply = ""
    pending = ""
    tool_calls = []

    try:
        # ponytail: model MUST come from the session (per-agent
        # llm_model column is the source of truth). _model_from_client
        # only worked for the Anthropic/Gemini tuple branches and
        # returned "" for the OpenAI SDK client — which is exactly
        # what OpenAI complained about with "you must provide a
        # model parameter". Use _resolve_model() which honors
        # session.llm_model and raises loudly if missing.
        model_id = _resolve_model(session, provider) if session is not None else _model_from_client(client)
        kwargs = {
            "model": model_id,
            "messages": messages,
            "temperature": getattr(session, "llm_temperature", None)
                if getattr(session, "llm_temperature", None) is not None else 0.2,
            "max_tokens": getattr(session, "llm_max_tokens", None) or MAX_RESPONSE_TOKENS,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        stream = client.chat.completions.create(**kwargs)

        for chunk in stream:
            if should_stop():
                break
            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            # Handle content
            content = delta.content or ""
            if content:
                full_reply += content
                pending += content
                ready_segments, pending = pop_streaming_segments(pending)
                for segment in ready_segments:
                    on_first_segment()
                    emit_segment(segment)

            # Handle tool calls (OpenAI / MiniMax)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.index >= len(tool_calls):
                        tool_calls.append({"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        tool_calls[tc.index]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[tc.index]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[tc.index]["arguments"] += tc.function.arguments

        if not should_stop():
            final_segments, _ = pop_streaming_segments(pending, force=True)
            for segment in final_segments:
                on_first_segment()
                emit_segment(segment)

        # If we have tool calls, return them for processing
        if tool_calls and execute_tool_callback:
            parsed_calls = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                parsed_calls.append({"id": tc["id"], "name": tc["name"], "arguments": args})
            # ponytail: P3 — emit the None sentinel so play_tts_from_text_queue's
            # "wait for segment OR None" loop wakes up and dispatches the tool
            # calls via execute_tool_callback. Without this the consumer is
            # blocked forever after a tool-call turn (the "hang after 3 turns"
            # symptom from the Phase 1 audit).
            emit_done(None)
            return full_reply.strip(), None, parsed_calls

        # ponytail: log the LLM's full reply so the operator can
        # confirm the steering tags are actually being emitted.
        # Same as the anthropic path: "[sighs]" / "[pause]" etc.
        # in this log means the LLM is generating tags and the
        # TTS should render them. Plain prose means the prepend
        # didn't work or the model isn't following the hint.
        log.info(
            "[LLM] openai reply session=%s len=%d preview=%r",
            session.session_key,
            len(full_reply),
            full_reply[:300],
        )
        # ponytail: emit_done(None) is the sentinel that ends the
        # segmenting loop in play_tts_from_text_queue. The Anthropic
        # path emits it (line 298); the Gemini path emits it (line 458);
        # the openai/minimax path WAS NOT, which left the consumer
        # blocked forever after the last segment — so reply_task
        # never completed and every subsequent launch_reply_pipeline
        # silent-returned at line 784 of turn_manager.py. The hang
        # after 3 turns was this exact bug.
        emit_done(full_reply.strip())
        return full_reply.strip(), None, None
    except Exception as exc:
        log.exception("LLM STREAM ERROR")
        # ponytail: same bug as above on the error path. If the
        # LLM crashes mid-stream the consumer still needs the None
        # sentinel to wake up and play the failure TTS, otherwise
        # the call hangs on silence until Twilio cuts the stream.
        emit_done(full_reply.strip())
        return full_reply.strip(), str(exc), None


# ponytail: stream_llm_reply_sync was originally called with the OpenAI
# client only. To keep turn_manager.py unchanged when the provider is
# OpenAI, we use the client object as a sentinel when the provider is
# not anthropic/gemini/minimax. For the dispatching providers we stash
# the config tuple on the client via these helpers below. They're a
# little hacky — turn_manager doesn't have the session at this layer
# — so the cleanest fix is to thread session through. Until then, the
# helpers below read creds off the cached client when possible.
def _anthropic_config_from_client(client) -> tuple[str, str, str]:
    # The "client" here is actually a stashed (key, base, model) tuple
    # we wrote in the cache via _stashed_anthropic / _stashed_gemini.
    if isinstance(client, tuple) and len(client) == 3:
        return client
    return _anthropic_config_from_user_id(None)


def _gemini_config_from_client(client) -> tuple[str, str, str]:
    if isinstance(client, tuple) and len(client) == 3:
        return client
    return _gemini_config_from_user_id(None)


def _anthropic_config_from_user_id(user_id):
    creds = resolve_provider(user_id, "anthropic")
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("Anthropic not configured.")
    try:
        base = _safe_base(creds, _DEFAULT_BASE_URLS["anthropic"])
    except UnsafeURLError as exc:
        raise RuntimeError(f"Anthropic base_url rejected by SSRF guard: {exc}")
    return key, base, _DEFAULT_MODELS["anthropic"]


def _gemini_config_from_user_id(user_id):
    creds = resolve_provider(user_id, "gemini")
    key = creds.get("api_key")
    if not key:
        raise RuntimeError("Gemini not configured.")
    try:
        base = _safe_base(creds, _DEFAULT_BASE_URLS["gemini"])
    except UnsafeURLError as exc:
        raise RuntimeError(f"Gemini base_url rejected by SSRF guard: {exc}")
    return key, base, _DEFAULT_MODELS["gemini"]


def _model_from_client(client) -> str:
    # ponytail: was OPENAI_MODEL env fallback. Caller passes the right
    # session-derived model via _resolve_model(); this helper is only
    # called when the openai_compat branch needs a string from a
    # client that pre-stashed a config tuple. In practice the openai /
    # MiniMax branches get the model from _resolve_model directly.
    # Returning empty here is fine — those branches ignore it.
    return ""


# ─── Backwards-compat shims used by turn_manager.prefetch_agent_reply
# and turn_manager.stream_llm_reply_with_tts. These build the right
# "client" handle for the streaming entry point above. Keep them tiny
# so the dispatch logic stays in one place.

def client_for_session(session: CallSession):
    """Returns a handle the streaming wrapper can use. For OpenAI /
    MiniMax this is the OpenAI SDK client. For Anthropic / Gemini
    this is a (key, base, model) tuple we recognize in the helpers
    above. None means no provider is configured.
    """
    provider = _session_provider(session)
    try:
        if provider == "anthropic":
            return _anthropic_config(session)
        if provider == "gemini":
            return _gemini_config(session)
        if provider == "minimax":
            return _minimax_client(session)
        return _openai_client(session)
    except Exception:
        log.exception("[LLM] client_for_session failed for provider=%s", provider)
        return None


# ponytail: /list-models is a debug endpoint (only behind
# ENABLE_DEBUG_ENDPOINTS). It requires the caller to pass the OpenAI
# key as a query param — the env fallback that used to feed this is
# gone, so we don't have any other source for the key.
async def list_models(api_key: str | None = None) -> dict:
    if not api_key:
        return {"error": "api_key query param is required (env fallback removed)"}

    def sync_list() -> dict:
        try:
            client = OpenAI(api_key=api_key)
            models_page = client.models.list()
            if hasattr(models_page, "data"):
                models = [model.id for model in models_page.data]
            else:
                models = [model.id for model in models_page]
            return {"models": models}
        except Exception as exc:
            return {"error": str(exc)}

    return await _to_thread(sync_list)