import asyncio
import logging

from openai import OpenAI

from STT_server.config import MAX_HISTORY_MESSAGES, MAX_RESPONSE_TOKENS, OPENAI_API_KEY, OPENAI_MODEL
from STT_server.domain.language import SYSTEM_PROMPT, detect_language, get_language_instruction, pop_streaming_segments
from STT_server.domain.session import CallSession


log = logging.getLogger("stt_server")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def build_messages(session: CallSession, user_text: str) -> list[dict[str, str]]:
    # Include structured user state, not as a memory, but as a guide for LLM
    lang = session.preferred_language or detect_language(user_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
    if openai_client is None:
        raise RuntimeError("OpenAI no configurada. Define OPENAI_API_KEY.")

    messages = build_messages(session, user_text)

    def sync_call() -> str:
        try:
            response = openai_client.chat.completions.create(
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
) -> tuple[str, str | None]:
    if openai_client is None:
        return "", "OpenAI no configurada. Define OPENAI_API_KEY."

    full_reply = ""
    pending = ""

    try:
        stream = openai_client.chat.completions.create(
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
    if openai_client is None:
        return {"error": "OpenAI no configurada"}

    def sync_list() -> dict:
        try:
            models_page = openai_client.models.list()
            if hasattr(models_page, "data"):
                models = [model.id for model in models_page.data]
            else:
                models = [model.id for model in models_page]
            return {"models": models}
        except Exception as exc:
            return {"error": str(exc)}

    return await asyncio.to_thread(sync_list)