import asyncio
import json
import logging
import time
import urllib.error
import urllib.request

from openai import OpenAI

from STT_server.config import MAX_HISTORY_MESSAGES, MAX_RESPONSE_TOKENS, OPENAI_MODEL
from STT_server.domain.language import detect_language, get_language_instruction, get_system_prompt, pop_streaming_segments
from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider


log = logging.getLogger("stt_server")


# ponytail: per-user client cache. Keyed by (provider, key, base_url)
# so two users on the same deploy each get their own client, and the
# MiniMax client (which carries a base_url) doesn't collide with the
# default OpenAI one (which doesn't). Busting happens implicitly when
# the entry's credentials change — the cache key includes them.
_client_cache: dict[str, object] = {}

_DEFAULT_BASE_URLS = {
    "openai":    "",                            # SDK default
    "minimax":   "https://api.MiniMax.com/v1",  # OpenAI-compat endpoint
    "anthropic": "https://api.anthropic.com",
    "gemini":    "https://generativelanguage.googleapis.com/v1beta",
}
_DEFAULT_MODELS = {
    "openai":    None,                          # fall back to OPENAI_MODEL env
    "minimax":   "minimax",
    "anthropic": "claude-3-5-sonnet-20241022",
    "gemini":    "gemini-1.5-pro",
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

    Priority: session.llm_model (set by media_stream from agent config)
    > provider default > OPENAI_MODEL env (only relevant for openai).
    """
    explicit = getattr(session, "llm_model", None)
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    fallback = _DEFAULT_MODELS.get(provider)
    if fallback:
        return fallback
    return OPENAI_MODEL


# ─── OpenAI ────────────────────────────────────────────────────────

def _openai_client(session: CallSession) -> OpenAI:
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "openai")
    key = creds.get("api_key")
    if not key:
        return _default_openai_client()
    cached = _client_cache.get(("openai", key, ""))
    if cached is not None:
        return cached
    client = OpenAI(api_key=key)
    _client_cache[("openai", key, "")] = client
    return client


def _default_openai_client() -> OpenAI:
    key = resolve_provider(None, "openai").get("api_key")
    return OpenAI(api_key=key) if key else None


# ─── MiniMax (OpenAI-compatible, custom base_url) ────────────────

def _minimax_client(session: CallSession) -> OpenAI:
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "minimax")
    key = creds.get("api_key")
    if not key:
        raise RuntimeError(
            "MiniMax not configured. Define MINIMAX_API_KEY or upload your key in Settings → API."
        )
    base_url = (creds.get("base_url") or _DEFAULT_BASE_URLS["minimax"]).rstrip("/")
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
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "anthropic")
    key = creds.get("api_key")
    base = (creds.get("base_url") or _DEFAULT_BASE_URLS["anthropic"]).rstrip("/")
    if not key:
        raise RuntimeError(
            "Anthropic not configured. Define ANTHROPIC_API_KEY or upload your key in Settings → API."
        )
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


def _anthropic_call_sync(messages: list[dict], config: tuple[str, str, str]) -> str:
    """One-shot Anthropic POST to /v1/messages. Returns the assistant
    text or a friendly fallback on any error.
    """
    key, base, model = config
    system_text, msgs = _anthropic_messages(messages)
    body = {
        "model": model,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": 0.2,
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
):
    """Anthropic streams via SSE on /v1/messages with stream=true. Each
    event line starts with `event: <type>\\ndata: <json>\\n\\n`. We only
    care about content_block_delta (text deltas) and message_stop.
    """
    key, base, model = config
    system_text, msgs = _anthropic_messages(messages)
    body = {
        "model": model,
        "max_tokens": MAX_RESPONSE_TOKENS,
        "temperature": 0.2,
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
        return full_reply.strip(), None
    except Exception as exc:
        log.exception("[LLM] anthropic stream error")
        return full_reply.strip(), str(exc)


# ─── Gemini (REST, no extra SDK dep) ──────────────────────────────

def _gemini_config(session: CallSession) -> tuple[str, str, str]:
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "gemini")
    key = creds.get("api_key")
    base = (creds.get("base_url") or _DEFAULT_BASE_URLS["gemini"]).rstrip("/")
    if not key:
        raise RuntimeError(
            "Gemini not configured. Define GEMINI_API_KEY or upload your key in Settings → API."
        )
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


def _gemini_call_sync(messages: list[dict], config: tuple[str, str, str]) -> str:
    key, base, model = config
    contents, system_text = _gemini_contents(messages)
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": MAX_RESPONSE_TOKENS,
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
):
    """Gemini streams via streamGenerateContent?alt=sse. Same SSE shape
    as Anthropic but the JSON envelope differs (candidates[0].content.parts).
    """
    key, base, model = config
    contents, system_text = _gemini_contents(messages)
    body: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": MAX_RESPONSE_TOKENS,
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
    # Include structured user state, not as a memory, but as a guide for LLM
    lang = session.preferred_language or detect_language(user_text)
    custom_prompt = getattr(session, 'custom_prompt', None)

    if custom_prompt and custom_prompt.strip():
        # Custom prompt replaces EVERYTHING — user has full control over
        # the agent's behavior, rules, and language. No default prompt
        # or language instruction is appended.
        log.info("[LLM] Using custom_prompt for session=%s (len=%d)", session.session_key, len(custom_prompt))
        messages = [
            {"role": "system", "content": custom_prompt.strip()},
        ]
    else:
        system_prompt = get_system_prompt(lang)
        log.info("[LLM] Using default system prompt for session=%s lang=%s prompt_len=%d", session.session_key, lang, len(system_prompt))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": get_language_instruction(lang)},
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

    messages.extend(session.history[-MAX_HISTORY_MESSAGES:])
    messages.append({"role": "user", "content": user_text})
    return messages


# ─── Entry points (dispatch by session.llm_provider) ──────────────

async def call_llm(session: CallSession, user_text: str) -> str:
    messages = build_messages(session, user_text)
    provider = _session_provider(session)
    if provider == "anthropic":
        config = _anthropic_config(session)
        return await asyncio.to_thread(_anthropic_call_sync, messages, config)
    if provider == "gemini":
        config = _gemini_config(session)
        return await asyncio.to_thread(_gemini_call_sync, messages, config)
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
                temperature=0.2,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception:
            log.exception("LLM ERROR")
            return "Lo siento, tuve un problema momentaneo. Puedes repetirlo?"

    return await asyncio.to_thread(sync_call)


def stream_llm_reply_sync(
    messages: list[dict],
    should_stop,
    emit_segment,
    emit_done,
    on_first_segment,
    client,
    *,
    provider: str = "openai",
) -> tuple[str, str | None]:
    """Streaming LLM call.

    `client` is passed in by the wrapper (turn_manager.stream_llm_reply_with_tts)
    so the right per-user OpenAI client is used. The previous version tried
    to read the session from a module-level single-element list, which is a
    latent bug — only one call at a time could be in flight.

    `provider` lets the caller (which knows the session.llm_provider) tell
    us whether to route through Anthropic/Gemini SSE instead of the
    OpenAI streaming endpoint.
    """
    if provider == "anthropic":
        config = _anthropic_config_from_client(client)
        return _anthropic_stream_sync(messages, config, should_stop, on_first_segment, emit_segment, emit_done)
    if provider == "gemini":
        config = _gemini_config_from_client(client)
        return _gemini_stream_sync(messages, config, should_stop, on_first_segment, emit_segment, emit_done)
    if provider == "minimax":
        # The client passed in might be an OpenAI-compat one — fall through
        # to the standard chat.completions.create path.
        pass

    if client is None:
        return "", "No LLM provider configured. Define the right env var or upload your key in Settings → API."

    full_reply = ""
    pending = ""

    try:
        stream = client.chat.completions.create(
            model=_model_from_client(client),
            messages=messages,
            temperature=0.2,
            max_tokens=MAX_RESPONSE_TOKENS,
            stream=True,
        )

        for chunk in stream:
            if should_stop():
                break
            if not getattr(chunk, "choices", None):
                continue

            delta = chunk.choices[0].delta.content or ""
            if not delta:
                continue

            full_reply += delta
            pending += delta
            ready_segments, pending = pop_streaming_segments(pending)
            for segment in ready_segments:
                on_first_segment()
                emit_segment(segment)

        if not should_stop():
            final_segments, _ = pop_streaming_segments(pending, force=True)
            for segment in final_segments:
                on_first_segment()
                emit_segment(segment)

        return full_reply.strip(), None
    except Exception as exc:
        log.exception("LLM STREAM ERROR")
        return full_reply.strip(), str(exc)


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
    base = (creds.get("base_url") or _DEFAULT_BASE_URLS["anthropic"]).rstrip("/")
    if not key:
        raise RuntimeError("Anthropic not configured.")
    return key, base, _DEFAULT_MODELS["anthropic"]


def _gemini_config_from_user_id(user_id):
    creds = resolve_provider(user_id, "gemini")
    key = creds.get("api_key")
    base = (creds.get("base_url") or _DEFAULT_BASE_URLS["gemini"]).rstrip("/")
    if not key:
        raise RuntimeError("Gemini not configured.")
    return key, base, _DEFAULT_MODELS["gemini"]


def _model_from_client(client) -> str:
    # Best-effort: if the client is a stashed config tuple (Anthropic /
    # Gemini) we still want the openai-compat branches to work for the
    # openai and MiniMax providers. Caller passes the right client for
    # their provider.
    return OPENAI_MODEL


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