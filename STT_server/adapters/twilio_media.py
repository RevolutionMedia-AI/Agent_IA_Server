import base64
from fastapi import WebSocket
from fastapi.responses import Response

async def send_twilio_media(ws: WebSocket, stream_sid: str, mulaw_audio: bytes) -> None:
    payload = base64.b64encode(mulaw_audio).decode("ascii")
    await ws.send_json(
        {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload},
        }
    )


async def send_twilio_mark(ws: WebSocket, stream_sid: str, mark_name: str) -> None:
    await ws.send_json(
        {
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {"name": mark_name},
        }
    )


async def send_twilio_clear(ws: WebSocket, stream_sid: str) -> None:
    await ws.send_json({"event": "clear", "streamSid": stream_sid})
