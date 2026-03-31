#!/usr/bin/env python3
import sys, os
# Ensure repo root is on sys.path so `STT_server` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import asyncio
import audioop
import wave
import sys
try:
    import winsound
except Exception:
    winsound = None

from STT_server.domain.session import CallSession
from STT_server.adapters.rime_tts import stream_tts_segment


async def main():
    session = CallSession(session_key="localtest")
    text = "Hola, esto es una prueba de síntesis de voz. Esta frase sirve para verificar si hay ruido o estática."

    def emit_item(item):
        t = item.get("type")
        if t == "audio":
            print(f"EMIT audio gen={item.get('generation')} bytes={len(item.get('data', b''))}")
        else:
            print("EMIT", t, item)

    try:
        ttfb, total = await stream_tts_segment(session, text, 1, emit_item)
        print("TTFB:", ttfb, "total_ms:", total)
    except Exception as e:
        print("Error running stream_tts_segment:", repr(e))
        return 1

    fname = f"rime_tts_{session.session_key}_1.mulaw"
    if not os.path.exists(fname):
        print("No .mulaw file written:", fname)
        return 2

    wavname = fname + ".wav"
    with open(fname, "rb") as f:
        mulaw = f.read()
    try:
        pcm = audioop.ulaw2lin(mulaw, 2)
    except Exception as e:
        print("Error converting ulaw->pcm:", e)
        return 3
    with wave.open(wavname, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(pcm)
    print("WAV written:", wavname)

    if winsound:
        print("Playing via winsound:", wavname)
        winsound.PlaySound(wavname, winsound.SND_FILENAME)
    else:
        print("winsound not available; please play the WAV file manually:", wavname)

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
