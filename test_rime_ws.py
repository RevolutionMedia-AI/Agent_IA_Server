"""
Quick local test: Rime WebSocket TTS protocol debugging.
Tries multiple message formats to find what actually returns audio.
"""
import asyncio
import base64
import json
import sys
from urllib.parse import urlencode

import websockets

API_KEY = "FsYIZi_FflfbSaPHp9NaVDSOuMvlyGelnY7UkAWY8J8"
SPEAKER = "lyra"
MODEL_ID = "arcana"
TEXT = "Hello, this is a test of the Rime text to speech system."

ENDPOINTS = [
    "wss://users-ws.rime.ai/ws3",
    "wss://users-ws.rime.ai/ws",
]

# Different message formats to try
MESSAGE_VARIANTS = {
    "full_json": {
        "text": TEXT,
        "speaker": SPEAKER,
        "modelId": MODEL_ID,
        "lang": "eng",
        "audioFormat": "pcm",
        "samplingRate": 8000,
    },
    "text_only": {
        "text": TEXT,
    },
    "raw_text": TEXT,  # maybe just send the text directly
    "with_speed": {
        "text": TEXT,
        "speaker": SPEAKER,
        "modelId": MODEL_ID,
        "lang": "eng",
        "audioFormat": "pcm",
        "samplingRate": 8000,
        "speedAlpha": 1.0,
    },
    "wav_format": {
        "text": TEXT,
        "speaker": SPEAKER,
        "modelId": MODEL_ID,
        "lang": "eng",
        "audioFormat": "wav",
        "samplingRate": 8000,
    },
    "mp3_format": {
        "text": TEXT,
        "speaker": SPEAKER,
        "modelId": MODEL_ID,
        "lang": "eng",
        "audioFormat": "mp3",
        "samplingRate": 22050,
    },
    "all_qs_text_msg": "qs_only",  # all in query params, text in message
}

QS_VARIANTS = {
    "speaker_only": {"speaker": SPEAKER},
    "full_qs": {
        "speaker": SPEAKER,
        "modelId": MODEL_ID,
        "lang": "eng",
        "audioFormat": "pcm",
        "samplingRate": "8000",
    },
}


async def try_ws(endpoint: str, qs_params: dict, message, label: str):
    qs = urlencode(qs_params) if qs_params else ""
    url = f"{endpoint}?{qs}" if qs else endpoint
    headers = {"Authorization": f"Bearer {API_KEY}"}

    msg_str = message if isinstance(message, str) else json.dumps(message)
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  URL: {url}")
    print(f"  Headers: Authorization: Bearer <key>")
    print(f"  Message: {msg_str[:100]}...")

    try:
        async with websockets.connect(
            url,
            additional_headers=headers,
            close_timeout=3,
            open_timeout=5,
        ) as ws:
            print(f"  ✓ Handshake OK")
            await ws.send(msg_str)
            print(f"  → Message sent, waiting for response (3s timeout)...")

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                if isinstance(raw, bytes):
                    print(f"  ✓ Got BINARY frame: {len(raw)} bytes")
                    print(f"    First 20 bytes: {raw[:20].hex()}")
                    return True
                else:
                    print(f"  ✓ Got TEXT frame: {raw[:200]}")
                    try:
                        obj = json.loads(raw)
                        if "audio" in obj:
                            print(f"    Audio field present ({len(obj['audio'])} chars)")
                            return True
                        if "error" in obj:
                            print(f"    ERROR: {obj['error']}")
                    except json.JSONDecodeError:
                        pass
                    # Try to get a second message
                    try:
                        raw2 = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        if isinstance(raw2, bytes):
                            print(f"  ✓ 2nd frame BINARY: {len(raw2)} bytes")
                            return True
                        else:
                            print(f"  ✓ 2nd frame TEXT: {raw2[:200]}")
                    except asyncio.TimeoutError:
                        print(f"  ✗ No 2nd frame (timeout)")
            except asyncio.TimeoutError:
                print(f"  ✗ No response within 3s (timeout)")
                return False

    except websockets.exceptions.InvalidStatus as exc:
        body = ""
        if hasattr(exc, "response") and exc.response:
            try:
                body = exc.response.body.decode("utf-8", errors="replace") if exc.response.body else ""
            except Exception:
                pass
        status = exc.response.status_code if hasattr(exc, "response") and exc.response else "?"
        print(f"  ✗ Handshake rejected: HTTP {status} — {body[:200]}")
        return False
    except Exception as exc:
        print(f"  ✗ Exception: {type(exc).__name__}: {exc}")
        return False
    return False


async def drain_all(label, endpoint, qs_params, message):
    """Connect and collect ALL frames to understand full protocol."""
    qs = urlencode(qs_params) if qs_params else ""
    url = f"{endpoint}?{qs}" if qs else endpoint
    headers = {"Authorization": f"Bearer {API_KEY}"}
    msg_str = message if isinstance(message, str) else json.dumps(message)

    print(f"\n{'='*60}")
    print(f"DRAIN: {label}")
    print(f"  URL: {url}")
    print(f"  Message: {msg_str[:80]}")

    try:
        async with websockets.connect(
            url, additional_headers=headers, close_timeout=5, open_timeout=5,
        ) as ws:
            print("  ✓ Handshake OK")
            await ws.send(msg_str)
            print("  → Sent, draining all frames (5s total timeout)...")

            frame_num = 0
            total_audio_bytes = 0
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    frame_num += 1
                    if isinstance(raw, bytes):
                        total_audio_bytes += len(raw)
                        print(f"    frame {frame_num}: BINARY {len(raw)} bytes (total audio: {total_audio_bytes})")
                        if frame_num <= 2:
                            print(f"      First 20 bytes hex: {raw[:20].hex()}")
                    else:
                        obj = json.loads(raw)
                        ftype = obj.get("type", "?")
                        if ftype == "chunk":
                            audio_b64 = obj.get("data", "")
                            decoded = base64.b64decode(audio_b64)
                            total_audio_bytes += len(decoded)
                            print(f"    frame {frame_num}: chunk — b64 len={len(audio_b64)} → decoded={len(decoded)} PCM bytes (total: {total_audio_bytes})")
                            if frame_num <= 2:
                                print(f"      First 20 bytes hex: {decoded[:20].hex()}")
                        elif ftype == "timestamps":
                            print(f"    frame {frame_num}: timestamps — words: {obj.get('word_timestamps',{}).get('words',[])}") 
                        else:
                            print(f"    frame {frame_num}: TEXT type={ftype} keys={list(obj.keys())}")
            except asyncio.TimeoutError:
                print(f"  ✗ Timeout after {frame_num} frames, {total_audio_bytes} total audio bytes")
            except websockets.exceptions.ConnectionClosed as exc:
                print(f"  Connection closed after {frame_num} frames: {exc}")
            duration_sec = total_audio_bytes / 2 / 8000 if total_audio_bytes > 0 else 0
            print(f"  Summary: {frame_num} frames, {total_audio_bytes} audio bytes = {duration_sec:.2f}s @ 8kHz PCM16")
    except Exception as exc:
        print(f"  ✗ {type(exc).__name__}: {exc}")


async def main():
    # Full drain of /ws3 with full QS + text-only message
    await drain_all(
        "ws3 FULL",
        "wss://users-ws.rime.ai/ws3",
        {"speaker": SPEAKER, "modelId": MODEL_ID, "lang": "eng",
         "audioFormat": "pcm", "samplingRate": "8000"},
        {"text": TEXT},
    )

    # Full drain of /ws with full QS + text-only message
    await drain_all(
        "ws FULL",
        "wss://users-ws.rime.ai/ws",
        {"speaker": SPEAKER, "modelId": MODEL_ID, "lang": "eng",
         "audioFormat": "pcm", "samplingRate": "8000"},
        {"text": TEXT},
    )

    print("\n" + "="*60)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
