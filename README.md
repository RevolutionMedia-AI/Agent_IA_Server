# STT Server — Agente de Voz (Cialix)

Este repositorio contiene el servidor STT/TTS en tiempo real usado por el agente de voz de Cialix.

Resumen rápido
- FastAPI + Uvicorn
- Deepgram (STT), Rime (TTS), OpenAI (LLM), Twilio (voz/stream)
- Soporta barge-in (interrupción del usuario), reproducción en mu-law 8k a Twilio, y warm-up TTS.

Requisitos
- Python 3.11+ (probado con 3.13)
- `requirements.txt` contiene las dependencias (instalar con pip)

Instalación (local)
Windows (PowerShell):
```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn STT_server.STT_Server:app --host 0.0.0.0 --port 8080
```
Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn STT_server.STT_Server:app --host 0.0.0.0 --port 8080
```

Variables de entorno
- Crea `STT_server/entornoLocal.env` (no subas claves a git).
- Variables principales:
```
PUBLIC_URL=https://mi-server-publico.example
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=dg-...
RIME_API_KEY=rime-...
# Rime TTS (valores recomendados)
RIME_TTS_MODEL_ID=arcana
RIME_TTS_SPEAKER_EN=lyra
RIME_TTS_SPEAKER_ES=celestino
RIME_TTS_SAMPLE_RATE=8000
# Timeouts / comportamiento
TTS_TTFB_TIMEOUT_SEC=15.0
INITIAL_GREETING_ENABLED=true
INITIAL_GREETING_TEXT="Thank you for calling the Cialix Support Line"
DEFAULT_CALL_LANGUAGE=en
```
No es necesario definir todas las variables; el servidor tiene valores por defecto, pero las claves (`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `RIME_API_KEY`) son necesarias para activar los servicios.

Cómo funciona el saludo inicial
- En `STT_server.STT_Server:media_stream` se dispara `play_initial_greeting(session)` cuando Twilio envía el evento `start`.
- Para evitar cold-starts de Rime, el servidor ejecuta un *warm-up* TTS en startup y guarda archivos `rime_tts_warmup-en_0.mulaw`/`rime_tts_warmup-es_0.mulaw`.
- `play_initial_greeting` ahora intenta reproducir esos archivos pre-generados inmediatamente (si existen), y si no, genera TTS en vivo.
- Archivos TTS generados se guardan como `rime_tts_<session>_<gen>.mulaw` para debugging.

Endpoints de prueba
- `/test-llm-tts?q=texto` — genera el reply y muestra segmentos TTS.
- `/test-stt` — endpoint debug para STT.
- `/` — estado básico.

Integración con Twilio
- Configura el webhook de Twilio para llamar al endpoint `/voice` del servidor (`PUBLIC_URL/voice`). Twilio abrirá un stream hacia `/media-stream`.
- El servidor envía audio a Twilio en paquetes mu-law 8k codificados en base64.

Logs y resolución de problemas (rápido)
- Buscar en logs: `[WARMUP] Ejecutando warm-up TTS` — confirma warm-up al inicio.
- `[PLAYBACK] Found warm-up file ...` — indica que el archivo warm-up se encoló y se intentó reproducir inmediatamente.
- `Playback error ...: Rime WS timeout while waiting for audio` — timeout en la generación de TTS (aumentar `TTS_TTFB_TIMEOUT_SEC` o revisar la conectividad con Rime).
- Si no se oye el saludo:
  - Verifica `PUBLIC_URL` y la configuración del webhook en Twilio.
  - Revisa que Twilio reciba los eventos `media`/`mark` y que `streamSid` esté presente en el session.
  - Revisa si aparecen logs de `No stream_sid for audio item, skipping`.

Archivos clave
- Configuración: [STT_server/config.py](STT_server/config.py)
- Servidor / rutas: [STT_server/STT_Server.py](STT_server/STT_Server.py)
- Adaptador TTS Rime: [STT_server/adapters/rime_tts.py](STT_server/adapters/rime_tts.py)
- Servicio de reproducción: [STT_server/services/playback_service.py](STT_server/services/playback_service.py)
- Envío a Twilio: [STT_server/adapters/twilio_media.py](STT_server/adapters/twilio_media.py)

Notas sobre cambios recientes
- Se aumentó `TTS_TTFB_TIMEOUT_SEC` a `15.0` por defecto.
- Se añadió warm-up TTS en `startup` y reproducción de warm-up si existe.
- Se eliminó el intento de enviar archivos TTS por correo; los archivos siguen guardándose para debugging.

Contribuir
- Abre un issue o PR con cambios. Mantén las claves fuera del repo.

¿Necesitas que suba `README.md` al repositorio remoto (`git add/commit/push`) o que lo adapte con más detalles (diagramas, ejemplo de TwiML, etc.)?
