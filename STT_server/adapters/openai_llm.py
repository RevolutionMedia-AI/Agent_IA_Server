import asyncio
import logging

from openai import OpenAI

from STT_server.config import MAX_HISTORY_MESSAGES, MAX_RESPONSE_TOKENS, OPENAI_MODEL
from STT_server.domain.language import detect_language, get_language_instruction, get_system_prompt, pop_streaming_segments
from STT_server.domain.session import CallSession
from STT_server.services.credentials_resolver import resolve_provider


log = logging.getLogger("stt_server")


# ponytail: per-user OpenAI client cache. Keyed by (api_key,) so two users
# in the same deploy each get their own client. We bust the entry when the
# key changes — cheap, and avoids leaking a previous user's key if they
# disconnect (the new entry will have the env-var key and a fresh client).
_client_cache: dict[str, OpenAI] = {}


def _client_for_session(session: CallSession):
    """Returns an OpenAI client for the session's user (per-user key) or the
    system default (OPENAI_API_KEY env var) if no per-user key is configured.

    A single shared default client backs all sessions without a user_id —
    `OPENAI_API_KEY` is set once on the host.
    """
    user_id = getattr(session, "user_id", None)
    creds = resolve_provider(user_id, "openai")
    key = creds.get("api_key")
    if not key:
        return _default_client()
    cached = _client_cache.get(key)
    if cached is not None:
        return cached
    client = OpenAI(api_key=key)
    _client_cache[key] = client
    return client


def _default_client():
    """System-default OpenAI client. Built lazily so the module imports
    even when no key is configured.
    """
    key = resolve_provider(None, "openai").get("api_key")
    return OpenAI(api_key=key) if key else None


def build_messages(session: CallSession, user_text: str) -> list[dict[str, str]]:
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

async def call_llm(session: CallSession, user_text: str) -> str:
    client = _client_for_session(session)
    if client is None:
        raise RuntimeError("OpenAI no configurada. Define OPENAI_API_KEY o sube tu key en Settings → API.")

    messages = build_messages(session, user_text)

    def sync_call() -> str:
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
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
    messages: list[dict[str, str]],
    should_stop,
    emit_segment,
    emit_done,
    on_first_segment,
    client,
) -> tuple[str, str | None]:
    """Streaming LLM call.

    `client` is passed in by the wrapper (turn_manager.stream_llm_reply_with_tts)
    so the right per-user OpenAI client is used. The previous version tried
    to read the session from a module-level single-element list, which is a
    latent bug — only one call at a time could be in flight.
    """
    if client is None:
        return "", "OpenAI no configurada. Define OPENAI_API_KEY o sube tu key en Settings → API."

    full_reply = ""
    pending = ""

    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
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
    finally:
        emit_done()


async def list_models() -> dict:
    client = _default_client()
    if client is None:
        return {"error": "OpenAI no configurada"}

    def sync_list() -> dict:
        try:
            models_page = client.models.list()
            if hasattr(models_page, "data"):
                models = [model.id for model in models_page.data]
            else:
                models = [model.id for model in models_page]
            return {"models": models}
        except Exception as exc:
            return {"error": str(exc)}

    return await asyncio.to_thread(sync_list)
