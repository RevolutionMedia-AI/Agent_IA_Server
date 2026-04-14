# 1. Resumen del proyecto

## Descripcion general

Este proyecto implementa un servidor de voz conversacional en tiempo real orientado a telefonia, especializado como agente de atencion al cliente para **Cialix Customer Support**. La aplicacion expone endpoints HTTP y WebSocket mediante FastAPI para integrarse con Twilio Media Streams. El sistema soporta dos modos de operacion mutuamente excluyentes:

- **Modo OpenAI Realtime** (`USE_OPENAI_REALTIME=true`, por defecto): el audio se envia directamente a la API Realtime de OpenAI via WebSocket, que realiza STT + deteccion de turnos + generacion LLM en un solo pipeline. Las respuestas textuales se sintetizan con **Rime TTS**.
- **Modo Deepgram STT** (`USE_OPENAI_REALTIME=false`): el audio se transcribe con Deepgram STT en tiempo real via WebSocket, se genera la respuesta con OpenAI Chat Completions, y se sintetiza con Rime TTS (o Deepgram TTS como alternativa disponible).

El sistema esta disenado para operar como backend de un agente telefonico.El estado de cada llamada se mantiene en memoria mientras la sesion permanece activa.

En modo Deepgram, el backend emplea una arquitectura de STT con transcripciones parciales para prefetch de respuestas y transcripciones finales para procesamiento definitivo. Adicionalmente implementa un mecanismo de ventana de gracia para diferir transcripciones finales cortas o incompletas, evitando interrupciones prematuras durante el habla del usuario. En modo OpenAI Realtime, la deteccion de turnos la realiza el servidor VAD de OpenAI.

## Proposito del sistema

El objetivo principal es habilitar una experiencia de atencion conversacional por voz sobre telefonia IP para el servicio al cliente de Cialix, con las siguientes capacidades:

- recibir audio de una llamada telefonica mediante Twilio;
- detectar turnos de habla del usuario (VAD local o VAD del servidor OpenAI);
- transcribir el audio a texto en tiempo real;
- generar respuestas conversacionales contextuales con un modelo LLM;
- sintetizar audio de salida con Rime TTS y reenviarlo a la misma llamada;
- gestionar flujos de atencion al cliente: consultas de pedidos, devoluciones, informacion de producto, precios y transferencias a agentes humanos.

## Problema que resuelve

El proyecto resuelve la orquestacion de un flujo de voz telefonico que normalmente involucra multiples sistemas desacoplados:

- ingesta de audio en tiempo real desde una plataforma de telefonia;
- deteccion de actividad de voz y cierre de utterances;
- conversion de voz a texto;
- generacion de respuesta con un modelo conversacional;
- conversion de texto a voz en un formato compatible con Twilio (mu-law 8 kHz);
- coordinacion de reproduccion, interrupcion y continuidad de la llamada;
- recopilacion estructurada de datos del cliente (numero de orden, nombre, email, telefono).

## Alcance funcional

El alcance actual incluye:

- endpoint HTTP para devolver TwiML a Twilio con soporte opcional de saludo pregrabado via `<Play>`;
- endpoint WebSocket bidireccional para intercambio de audio y eventos;
- manejo de sesiones activas por llamada en memoria;
- VAD local con `webrtcvad` (agresividad configurable 0-3) para detectar inicio y fin de turno;
- modo dual de operacion: OpenAI Realtime (STT+LLM integrado) o Deepgram STT + OpenAI Chat;
- transcripcion STT realtime via WebSocket de Deepgram con sistema de candidatos y fallback;
- prefetch silencioso de respuestas LLM a partir de transcripciones parciales;
- ventana de gracia configurable para diferir finals cortos o linguisticamente incompletos;
- ventana de gracia extendida para dictado de digitos (numeros de orden);
- deteccion de utterances incompletas mediante marcadores linguisticos;
- normalizacion de digitos dictados oralmente (`"four five one"` → `"451"`);
- generacion de respuesta mediante OpenAI con streaming y segmentacion progresiva;
- sintesis TTS principal mediante **Rime** via WebSocket con conversion PCM→mu-law integrada;
- sintesis TTS alternativa mediante **Deepgram** via HTTP;
- filler de espera configurable por idioma;
- modo ingles completo por defecto (deteccion de espanol deshabilitada, reactivable);
- barge-in configurable con umbral RMS promedio y proteccion contra eco;
- filtrado de utterances no accionables (~60 frases: saludos, fillers, confirmaciones, pronombres);
- deteccion y descarte de alucinaciones STT y ecos del TTS del asistente;
- memoria estructurada en `session.collected_data` para evitar repetir solicitudes de datos ya capturados;
- extraccion automatica de datos estructurados (numero de orden, email, telefono, nombre);
- deteccion de preguntas repetidas del asistente con escalacion automatica;
- monitoreo de inactividad con cierre automatico de llamada;
- reconexion automatica de STT realtime con backoff exponencial;
- muteo de STT durante reproduccion del asistente con buffer de replay configurable;
- proteccion de endpoints de diagnostico mediante flag configurable;
- servicio de archivos estaticos (`/static`) para saludo pregrabado WAV;
- endpoints de prueba y diagnostico;
- soporte Docker con Dockerfile y `start.sh`;
- despliegue en Railway y Azure App Service.

Quedan fuera del alcance actual:

- frontend o panel administrativo;
- base de datos o persistencia historica;
- autenticacion de usuarios finales;
- control de acceso por roles;
- pruebas automatizadas formales;
- observabilidad centralizada y metricas persistentes;
- integracion real con herramientas/tools de backend (ship_information, order_information, cialix_rma, cialix_transfer_call_tool estan descritos en el prompt del sistema pero no implementados como herramientas funcionales).

# 2. Tecnologias utilizadas

## Frontend

No existe frontend en el estado actual del repositorio.

## Backend

| Tecnologia | Uso principal |
| --- | --- |
| Python 3.11+ | Lenguaje principal del proyecto (Dockerfile usa 3.11-slim) |
| FastAPI | Exposicion de endpoints HTTP y WebSocket, servicio de archivos estaticos |
| Uvicorn | Servidor ASGI para ejecucion local y despliegue |
| asyncio | Coordinacion asincrona de tareas, colas y eventos |
| python-dotenv | Carga de configuracion local desde archivo `.env` |
| webrtcvad | Deteccion de actividad de voz con agresividad configurable (modo 0-3) |
| audioop | Conversion mu-law↔PCM16 y calculo de RMS para deteccion de energia |
| OpenAI SDK | Cliente para Chat Completions (streaming) y API Realtime (WebSocket) |
| websockets | Conexion WebSocket con Deepgram STT realtime, Rime TTS y OpenAI Realtime |
| struct | Empaquetado/desempaquetado de muestras PCM16-LE para conversion de audio |

## Base de datos

No se utiliza base de datos. Toda la informacion operacional se conserva temporalmente en memoria del proceso mediante la clase `CallSession`.

## Servicios externos

| Servicio | Proposito | Protocolo |
| --- | --- | --- |
| Twilio | Transporte de llamadas y Media Streams bidireccionales | WebSocket + HTTP (TwiML) |
| OpenAI Realtime API | STT + deteccion de turnos + LLM integrado (modo principal) | WebSocket (`wss://api.openai.com/v1/realtime`) |
| OpenAI Chat Completions | Generacion de respuesta del asistente (modo Deepgram) | HTTP con streaming |
| Deepgram STT Realtime | Transcripcion de audio en tiempo real (modo alternativo) | WebSocket (`wss://api.deepgram.com/v1/listen`) |
| Deepgram STT Batch | Transcripcion autoritativa via HTTP con deteccion de idioma | HTTP POST |
| Deepgram TTS | Sintesis de texto a voz alternativa (formato mu-law directo) | HTTP POST |
| Rime TTS | Sintesis de texto a voz principal via WebSocket streaming | WebSocket (`wss://users-ws.rime.ai/ws3`) |
| Railway | Plataforma de despliegue (configuracion activa) | Git + panel web |
| Azure App Service | Plataforma de despliegue principal mediante `git push` | Git |

## Herramientas de desarrollo

| Herramienta | Uso |
| --- | --- |
| Git | Control de versiones |
| venv | Entorno virtual Python local |
| Docker | Contenerizacion del servicio (Dockerfile incluido) |
| Railway CLI o panel web | Despliegue y configuracion del servicio |
| Azure App Service + Git | Despliegue principal mediante `git push Azure main` |

## Dependencias relevantes

| Dependencia | Rol en el proyecto |
| --- | --- |
| fastapi | Framework principal del backend |
| uvicorn[standard] | Ejecucion del servidor ASGI |
| python-dotenv | Carga de variables locales |
| webrtcvad | Deteccion de voz por frames PCM |
| websockets | Conexion WebSocket con Deepgram STT, Rime TTS y OpenAI Realtime |
| openai | Cliente del modelo conversacional (Chat Completions y Realtime) |
| twilio | Dependencia declarada para integracion con el ecosistema Twilio |
| scipy>=1.17.1 | Resampleo de audio de alta calidad (`resample_poly`); opcional con fallback a interpolacion lineal |
| numpy | Soporte numerico para resampleo de audio (usado por Rime TTS adapter) |
| setuptools<81 | Restriccion de compatibilidad declarada |

Dependencias declaradas en el repositorio como `faster-whisper`, `librosa` y `soundfile` aparecen asociadas a scripts auxiliares o a configuracion heredada y no forman parte del flujo principal actual del servidor telefonico. `websockets` se instala como dependencia transitiva de `uvicorn[standard]` y tambien se requiere directamente por los adaptadores de Deepgram, Rime y OpenAI Realtime.

**Dependencias Docker adicionales** (`requirements.docker.txt`): `scipy>=1.17.1` para resampleo de audio en el contenedor.

# 3. Arquitectura general

## Enfoque arquitectonico

La aplicacion sigue un enfoque de backend orientado a eventos y estado en memoria por sesion. Cada llamada activa se representa mediante una estructura `CallSession` que encapsula buffers, colas, historial, datos recopilados y metadatos de reproduccion. El procesamiento del audio se divide en etapas desacopladas coordinadas con `asyncio`. El sistema soporta dos modos de operacion principales controlados por `USE_OPENAI_REALTIME`.

## Modos de operacion

### Modo OpenAI Realtime (por defecto)

```
Twilio → WebSocket → Audio mu-law → OpenAI Realtime API (STT+VAD+LLM) → Texto → Rime TTS → mu-law → Twilio
```

- El audio crudo se envia directamente al WebSocket de OpenAI Realtime.
- OpenAI maneja STT, deteccion de turnos (server_vad) y generacion de respuesta.
- Las respuestas textuales se segmentan y sintetizan con Rime TTS.
- El VAD local con `webrtcvad` sigue activo para detectar barge-in.

### Modo Deepgram STT

```
Twilio → WebSocket → Audio mu-law → Deepgram STT Realtime (parciales) → Prefetch LLM
                                   → VAD local → Transcripcion final → LLM Streaming → Rime TTS → mu-law → Twilio
```

- El audio crudo se envia continuamente a Deepgram STT Realtime via WebSocket.
- El VAD local con `webrtcvad` detecta inicio/fin de habla en paralelo.
- Las transcripciones parciales disparan prefetch silencioso de respuesta LLM.
- Las transcripciones finales se procesan con la logica de turnos (ventana de gracia, defer, merge).

## Relacion entre componentes

El sistema se compone de los siguientes bloques funcionales:

| Componente | Responsabilidad |
| --- | --- |
| Endpoint `/voice` | Entregar TwiML para iniciar el Media Stream, opcionalmente con `<Play>` de saludo pregrabado |
| Endpoint `/media-stream` | Gestionar eventos WebSocket de Twilio y orquestar el ciclo de vida de la sesion |
| VAD local | Detectar inicio y fin de una utterance mediante `webrtcvad` y calcular RMS para barge-in |
| STT realtime (Deepgram) | Transcripcion en tiempo real via WebSocket con candidatos fallback (modo Deepgram) |
| OpenAI Realtime | STT + deteccion de turnos + generacion LLM unificada (modo Realtime) |
| Turn manager | Coordinar transcripciones, prefetch, ventana de gracia, filtrado de ecos/alucinaciones y pipeline de respuesta |
| LLM (Chat) | Generar respuesta con streaming y construccion de contexto enriquecido |
| Rime TTS | Sintetizar audio via WebSocket streaming con conversion PCM→mu-law a 8 kHz |
| Deepgram TTS | Sintetizar audio via HTTP con mu-law nativo (alternativa) |
| Playback | Fragmentar audio en frames de 160 bytes, enviar con pacing a Twilio, gestionar marcas |
| Gestion de sesion | Mantener estado por llamada, registro/limpieza y monitoreo de inactividad |
| Extraccion de datos | Reconocer y almacenar datos estructurados (order_number, email, phone, name) |
| Filtrado inteligente | Descartar utterances no accionables, detectar ecos/alucinaciones STT, evitar duplicados |
| Muteo STT | Silenciar el flujo STT durante reproduccion del asistente con buffer de replay |

## Estructura de comunicacion

### Flujo general (Modo OpenAI Realtime)

1. Twilio invoca `POST /voice`.
2. El backend responde con TwiML que opcionalmente reproduce un saludo pregrabado y abre un stream WebSocket hacia `/media-stream`.
3. Twilio abre el WebSocket bidireccional y comienza a enviar eventos `connected`, `start` y `media`.
4. El servidor abre una conexion WebSocket paralela hacia la API Realtime de OpenAI.
5. La tarea `_audio_sender` lee chunks mu-law de `realtime_audio_queue` y los envia a OpenAI.
6. La tarea `_barge_in_watcher` escucha el evento `generation_changed` del VAD local para cancelar respuestas en curso.
7. La tarea `_event_receiver` procesa eventos de OpenAI: transcripciones de entrada, deltas de texto de respuesta.
8. Los deltas de texto se acumulan, se segmentan con `pop_streaming_segments` y se envian a una cola de texto.
9. La tarea `play_tts_from_text_queue` lee segmentos de texto, los sintetiza con Rime TTS y emite audio a la cola de playback.
10. La tarea `playback_loop` lee items de la cola de playback y los envia a Twilio como frames mu-law de 160 bytes con pacing.
11. El VAD local sigue procesando frames para detectar barge-in basado en RMS promedio.
12. Los datos estructurados se extraen de las transcripciones del usuario y se almacenan en `collected_data`.

### Flujo general (Modo Deepgram STT)

1. Twilio invoca `POST /voice`.
2. El backend responde con TwiML.
3. Twilio abre el WebSocket bidireccional y envia eventos.
4. El servidor acumula audio en frames PCM lineales y aplica VAD local con `webrtcvad`.
5. En paralelo, el audio crudo se envia al canal STT realtime de Deepgram via WebSocket.
6. Las transcripciones parciales del realtime disparan un prefetch silencioso de respuesta LLM.
7. Cuando el VAD local detecta fin de turno (silencio >= `END_SILENCE_FRAMES`), se marca fin de utterance.
8. La transcripcion final del realtime se procesa por el turn manager.
9. El turn manager evalua si el transcript final debe diferirse (ventana de gracia), filtrarse (no accionable, eco, alucinacion) o procesarse inmediatamente.
10. Si existe un prefetch de respuesta LLM compatible (delta <= `PARTIAL_PREFETCH_MAX_DELTA_CHARS`), se reutiliza directamente; de lo contrario se genera una nueva respuesta con streaming.
11. La respuesta textual se divide en segmentos aptos para TTS via `pop_streaming_segments`.
12. Cada segmento se sintetiza con Rime TTS via WebSocket, se convierte de PCM a mu-law a 8 kHz.
13. El backend envia el audio a Twilio mediante eventos `media` con pacing proporcional y sincroniza con eventos `mark`.
14. Si el usuario interrumpe y el barge-in esta habilitado, el turno actual se invalida y se limpia.

### Gestion del muteo STT

Durante la reproduccion del asistente (`assistant_speaking=True`), los chunks de audio del usuario NO se envian directamente a la cola STT. En su lugar, se almacenan en un buffer circular (`stt_mute_buffer`, por defecto 25 chunks = 500 ms). Cuando el asistente deja de hablar, el buffer se vacia hacia la cola STT para que el reconocedor procese los ultimos 500 ms de audio del usuario, evitando perdida de habla en la transicion.

## Capas o modulos principales

| Capa | Implementacion principal | Descripcion |
| --- | --- | --- |
| Entrada HTTP | `STT_Server.voice()` | Inicializa la llamada de voz con TwiML |
| Archivos estaticos | `StaticFiles(/static)` | Sirve el saludo pregrabado `greeting.wav` |
| Entrada WebSocket | `STT_Server.media_stream()` | Orquesta el ciclo de vida del stream |
| Audio/VAD | `audio_ingest.handle_incoming_media()` | Convierte mu-law→PCM, segmenta, detecta voz y gestiona muteo STT |
| OpenAI Realtime | `openai_realtime.run_realtime_session()` | STT+LLM via audio-in/text-out WebSocket |
| STT Realtime | `deepgram_stt_realtime.run_realtime_stt()` | Transcripcion en tiempo real via WebSocket con candidatos |
| Turn Manager | `turn_manager.process_transcripts()` | Coordina prefetch, gracia, filtrado, ecos y pipeline de respuesta |
| LLM | `openai_llm.call_llm()` / `stream_llm_reply_sync()` | Genera respuesta con streaming y contexto enriquecido |
| Rime TTS | `rime_tts.stream_tts_segment()` | Sintesis via WebSocket con conversion PCM→mu-law |
| Deepgram TTS | `deepgram_tts.stream_tts_segment()` | Sintesis via HTTP con mu-law nativo |
| Playback | `playback_service.playback_loop()` | Emite audio fragmentado con pacing y marcas a Twilio |
| Sesion | `session.CallSession` | Aisla el estado de una llamada |
| Idioma | `language.*` | Deteccion de idioma, analisis linguistico, segmentacion TTS, extraccion de datos |
| Runtime de sesion | `session_runtime.*` | Registro, limpieza, tracking de tareas y monitoreo de inactividad |
| Utilidades | `common.*` | Proteccion de endpoints, operaciones de cola con drop policy |
| Email | `utils/send_audio_email.py` | Envio de audio por email via SMTP (utilidad auxiliar) |

## Relacion con frontend y base de datos

No existe frontend ni base de datos. Toda la comunicacion del sistema ocurre entre servicios externos y el backend:

- Twilio consume los endpoints de voz y WebSocket;
- OpenAI Realtime o Deepgram procesan STT;
- OpenAI Chat Completions genera respuestas conversacionales;
- Rime o Deepgram procesan TTS;
- Railway o Azure App Service hospeda la aplicacion.

# 4. Estructura del proyecto

## Organizacion de carpetas

```text
.
├── main.py
├── Dockerfile
├── start.sh
├── railway.toml
├── requirements.txt
├── requirements.docker.txt
├── Documentacion-Server.md
├── README.md
├── ConvertLocalSTT/
│   ├── ConvertText.py
│   ├── ConvertText.md
│   ├── RealTimeTranscription.py
│   └── RealTimeTranscription.md
├── scripts/
│   ├── analyze_mulaw.py
│   ├── analyze_wav.py
│   ├── check_problematic_llm.py
│   ├── compare_wavs.py
│   ├── convert_to_8k.py
│   ├── debug_sanitize.py
│   ├── inspect_wav.py
│   ├── mulaw_to_wav.py
│   ├── parse_twilio_timings.py
│   ├── plot_audio_compare.py
│   ├── run_full_playback_test.py
│   ├── run_rime_tts_test.py
│   ├── smoke_playback.py
│   └── test_sanitize.py
└── STT_server/
    ├── __init__.py
    ├── config.py
    ├── STT_Server.py
    ├── entornoLocal.env
    ├── adapters/
    │   ├── deepgram_stt_batch.py
    │   ├── deepgram_stt_realtime.py
    │   ├── deepgram_tts.py
    │   ├── openai_llm.py
    │   ├── openai_realtime.py
    │   ├── rime_tts.py
    │   └── twilio_media.py
    ├── domain/
    │   ├── language.py
    │   └── session.py
    ├── services/
    │   ├── audio_ingest.py
    │   ├── common.py
    │   ├── playback_service.py
    │   ├── session_runtime.py
    │   └── turn_manager.py
    ├── static/
    │   └── greeting.wav
    └── utils/
        └── send_audio_email.py
```

## Responsabilidad de cada carpeta o modulo

| Ruta | Responsabilidad |
| --- | --- |
| `main.py` | Punto de entrada ASGI para Uvicorn — importa y expone `app` |
| `Dockerfile` | Imagen Docker basada en Python 3.11-slim con dependencias de build |
| `start.sh` | Script de arranque para Docker y Railway |
| `railway.toml` | Configuracion de build (RAILPACK) y arranque en Railway |
| `requirements.txt` | Dependencias principales del proyecto |
| `requirements.docker.txt` | Dependencias adicionales para Docker (`scipy`) |
| `STT_server/config.py` | Todas las constantes y parametros configurables (>100 variables) |
| `STT_server/STT_Server.py` | Punto de entrada FastAPI con endpoints HTTP y WebSocket |
| `STT_server/entornoLocal.env` | Configuracion local de desarrollo (**no versionar con credenciales reales**) |
| `STT_server/adapters/` | Integraciones con servicios externos (Deepgram, OpenAI, Rime, Twilio) |
| `STT_server/adapters/openai_realtime.py` | Adaptador OpenAI Realtime API — STT+LLM via audio-in/text-out |
| `STT_server/adapters/openai_llm.py` | Chat Completions con streaming y construccion de contexto |
| `STT_server/adapters/deepgram_stt_realtime.py` | STT realtime con sistema de candidatos y reconexion |
| `STT_server/adapters/deepgram_stt_batch.py` | STT batch via HTTP con reintentos |
| `STT_server/adapters/rime_tts.py` | TTS via WebSocket con conversion PCM→mu-law y resampleo |
| `STT_server/adapters/deepgram_tts.py` | TTS via HTTP con mu-law nativo |
| `STT_server/adapters/twilio_media.py` | Helpers para enviar media, marks y clear a Twilio |
| `STT_server/domain/` | Modelo de datos (`CallSession`) y logica de dominio |
| `STT_server/domain/language.py` | System prompt, deteccion de idioma, segmentacion TTS, extraccion de datos |
| `STT_server/domain/session.py` | Dataclass `CallSession` con todo el estado por llamada |
| `STT_server/services/` | Orquestacion de audio, reproduccion, turnos y sesiones |
| `STT_server/static/` | Archivos estaticos servidos via `/static` (saludo pregrabado) |
| `STT_server/utils/` | Utilidades auxiliares (envio de email) |
| `ConvertLocalSTT/` | Scripts auxiliares de transcripcion local (heredado) |
| `scripts/` | Scripts de diagnostico, analisis de audio, testing y depuracion |

## Archivos principales

| Archivo | Descripcion |
| --- | --- |
| `main.py` | Importa y expone la instancia `app` de FastAPI (`from STT_server.STT_Server import app`) |
| `STT_server/config.py` | Centraliza >100 constantes y parametros configurables con valores por defecto |
| `STT_server/STT_Server.py` | Define los endpoints HTTP y WebSocket, orquesta tareas por sesion |
| `STT_server/domain/session.py` | Define la clase `CallSession` con ~40 campos de estado por llamada |
| `STT_server/domain/language.py` | System prompt de Cialix (~3KB), logica linguistica, extraccion de datos, segmentacion TTS |
| `STT_server/services/turn_manager.py` | Modulo mas complejo (~800 lineas): toda la logica de turnos conversacionales |
| `STT_server/adapters/rime_tts.py` | Adaptador TTS principal con conversion de audio integrada (mu-law, resampleo) |
| `STT_server/adapters/openai_realtime.py` | Adaptador de la API Realtime de OpenAI para STT+LLM unificado |
| `Dockerfile` | Imagen Docker con Python 3.11-slim |
| `start.sh` | Script de arranque con port por defecto 8080 |
| `railway.toml` | RAILPACK builder y comando de arranque |
| `requirements.txt` | Dependencias Python principales |
| `Documentacion-Server.md` | Documentacion tecnica del repositorio |

## Convenciones de nombres

- Las constantes globales se expresan en `UPPER_SNAKE_CASE` y se definen en `config.py`.
- Las funciones internas usan prefijo `_` (ej: `_has_excessive_repetition`, `_echoes_agent_speech`).
- Las rutas HTTP y WebSocket usan nombres breves y semanticos (`/voice`, `/media-stream`).
- La sesion por llamada se modela con la clase `CallSession` en `domain/session.py`.
- Los adaptadores de servicios externos se nombran por proveedor (ej: `rime_tts.py`, `deepgram_stt_batch.py`).
- La logica de negocio/orquestacion se agrupa en `services/`.
- El dominio (modelo de datos y reglas linguisticas) se agrupa en `domain/`.
- Los logs usan el logger `stt_server` con nivel `WARNING` por defecto.

# 5. Modelo de datos

## Enfoque general

El sistema no dispone de un modelo relacional ni de persistencia. El unico modelo estructurado relevante es el estado en memoria por llamada, modelado como un `dataclass` de Python.

## Entidad principal: `CallSession`

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| `session_key` | `str` | Identificador interno de la sesion (se reemplaza por `call_sid` cuando disponible) |
| `call_sid` | `str \| None` | Identificador de llamada de Twilio |
| `stream_sid` | `str \| None` | Identificador del stream activo |
| `preferred_language` | `str \| None` | Idioma preferente de la conversacion |
| `vad_buffer` | `bytearray` | Buffer incremental de audio PCM para VAD |
| `pre_speech_frames` | `deque[bytes]` | Frames previos a la deteccion de voz (configurable via `PRE_SPEECH_FRAMES`) |
| `speech_frames` | `list[bytes]` | Frames de la utterance activa |
| `speech_frame_count` | `int` | Conteo de frames validos de voz |
| `voice_streak` | `int` | Racha de frames consecutivos con voz |
| `silence_frames` | `int` | Conteo de frames en silencio |
| `active_generation` | `int` | Version logica del turno actual (se incrementa en cada interrupcion o nuevo turno) |
| `response_active` | `bool` | Indica si hay una respuesta Realtime en curso |
| `history` | `list[dict[str, str]]` | Historial reciente de la conversacion (acotado a `MAX_HISTORY_MESSAGES`) |
| `utterance_queue` | `asyncio.Queue` | Cola de utterances completas para STT batch |
| `playback_queue` | `asyncio.Queue` | Cola de audio/eventos de salida (maxsize configurable) |
| `stt_audio_queue` | `asyncio.Queue` | Cola de chunks de audio crudo para STT Deepgram |
| `stt_mute_buffer` | `deque[bytes]` | Buffer circular para almacenar audio durante muteo STT (configurable via `STT_MUTE_BUFFER_CHUNKS`) |
| `transcript_queue` | `asyncio.Queue` | Cola de eventos de transcripcion |
| `realtime_audio_queue` | `asyncio.Queue` | Cola de chunks mu-law para OpenAI Realtime |
| `realtime_text_queue` | `asyncio.Queue \| None` | Cola de segmentos de texto desde OpenAI Realtime hacia TTS |
| `generation_changed` | `asyncio.Event` | Evento para notificar barge-in al adaptador Realtime |
| `tasks` | `set[asyncio.Task]` | Tareas asociadas a la sesion con auto-limpieza via callback |
| `pending_marks` | `set[str]` | Marcas pendientes de confirmacion por Twilio |
| `mark_counter` | `int` | Contador incremental de marcas |
| `assistant_speaking` | `bool` | Indica si el asistente esta reproduciendo audio |
| `assistant_started_at` | `float \| None` | Timestamp (`perf_counter`) del inicio de la reproduccion |
| `current_transcript` | `str` | Transcripcion parcial o final en curso |
| `reply_source_text` | `str` | Texto fuente del pipeline de respuesta activo |
| `reply_task` | `asyncio.Task \| None` | Tarea del pipeline de respuesta activo |
| `partial_reply_task` | `asyncio.Task \| None` | Tarea de debounce para prefetch desde parciales |
| `prefetched_reply_source_text` | `str` | Texto fuente del prefetch LLM |
| `prefetched_reply_text` | `str` | Respuesta prefetched lista para reutilizar |
| `prefetched_reply_task` | `asyncio.Task \| None` | Tarea del prefetch LLM en curso |
| `deferred_final_text` | `str` | Texto final diferido en ventana de gracia |
| `deferred_final_language` | `str \| None` | Idioma del final diferido |
| `deferred_final_flush_task` | `asyncio.Task \| None` | Timer de la ventana de gracia |
| `collected_data` | `dict[str, str]` | Datos estructurados extraidos del usuario (order_number, email, phone, name) |
| `last_processed_user_text` | `str` | Ultimo texto de usuario procesado (para deteccion de duplicados) |
| `stt_failure_announced` | `bool` | Indica si ya se anuncio fallo de STT al usuario |
| `closed` | `bool` | Indica si la sesion fue cerrada |
| `last_activity_at` | `float` | Timestamp monotonic de la ultima actividad; usado para monitoreo de inactividad |

## Relaciones

- una llamada activa corresponde a una instancia de `CallSession`;
- una sesion puede contener multiples turnos de usuario y asistente;
- un turno puede producir varios segmentos TTS;
- cada sesion mantiene su propio historial, colas, datos recopilados y estado de reproduccion;
- las sesiones se registran en un diccionario global `sessions` en `session_runtime.py`.

## Datos estructurados extraidos automaticamente

El modulo `language.extract_structured_data()` reconoce los siguientes patrones en las transcripciones:

| Dato | Patron de reconocimiento | Ejemplo |
| --- | --- | --- |
| `order_number` | 5-6 digitos contiguos (con normalizacion de digitos dictados) | `"45108"`, `"four five one oh eight"` |
| `email` | Direccion de correo electronico valida | `"user@example.com"` |
| `phone` | 7-15 digitos contiguos (sin separadores) | `"+18882425491"` |
| `name` | Frase "my name is" / "mi nombre es" seguida de 1-3 palabras | `"my name is John Smith"` |

Los datos extraidos se almacenan en `session.collected_data` y se inyectan en el contexto del LLM para evitar preguntas repetidas.

## Reglas de negocio

| Regla | Descripcion |
| --- | --- |
| Aislamiento por llamada | Cada llamada mantiene estado propio en memoria |
| Historial acotado | Solo se conservan los ultimos `MAX_HISTORY_MESSAGES` (12 por defecto) |
| Idioma operativo | Modo ingles completo por defecto; espanol deshabilitado pero reactivable |
| Respuesta contextual | El prompt del sistema incluye reglas extensas de atencion al cliente Cialix |
| Turno invalidable | Un turno previo puede cancelarse mediante `active_generation` |
| Escalacion automatica | Tras 2 solicitudes fallidas del numero de orden, se transfiere a agente humano |
| Filtrado de utterances | Se descartan saludos, fillers, confirmaciones y ecos/alucinaciones |
| Deduplicacion | Si los datos extraidos ya existen en `collected_data`, no se repite la pregunta |

## Restricciones y validaciones

| Restriccion | Descripcion |
| --- | --- |
| Umbral minimo de utterance | El audio debe superar `MIN_UTTERANCE_MS` (180 ms) y `MIN_SPEECH_FRAMES` (5) |
| Formato de salida | El TTS se convierte a mu-law a 8 kHz para compatibilidad con Twilio |
| Idiomas normalizados | Solo se aceptan `en` y `es` como idiomas operativos |
| Estado efimero | Al reiniciar el proceso se pierde todo el estado de llamadas |
| Cola con drop | Las colas usan politica de drop (descarte del elemento mas antiguo) cuando estan llenas |

## Enumeraciones o catalogos

| Elemento | Valores |
| --- | --- |
| `SUPPORTED_LANGUAGES` | `en`, `es` |
| Eventos de Twilio procesados | `connected`, `start`, `media`, `mark`, `dtmf`, `stop` |
| Eventos internos de playback | `audio`, `mark`, `clear`, `segment_end`, `error` |
| `NON_ACTIONABLE_PHRASES` | Conjunto de ~60 frases (saludos, fillers, confirmaciones, pronombres) que se filtran |
| `INCOMPLETE_TRAILING_MARKERS` | ~35 palabras/conectores en ingles y espanol que indican utterance incompleta |
| `INCOMPLETE_TRAILING_PHRASES` | ~15 frases de 2 palabras que indican utterance incompleta |
| `WORD_TO_DIGIT` | Mapeo de palabras inglesas a digitos (`"zero"`→`"0"`, `"nine"`→`"9"`, etc.) |

# 6. Funcionalidades del sistema

## Modulos principales

### 6.1 Recepcion de llamadas

El endpoint `/voice` responde con TwiML para que Twilio conecte un Media Stream hacia el servidor. Si existe un archivo `static/greeting.wav` o `TWIML_INITIAL_GREETING_ENABLED=true`, se incluye un `<Play>` previo al `<Connect>` para reproducir un saludo pregrabado antes de abrir el stream WebSocket.

### 6.2 Procesamiento de audio entrante

El endpoint `/media-stream` recibe audio en base64 (formato mu-law de Twilio), lo convierte a PCM16 con `audioop.ulaw2lin` y lo analiza frame a frame (20 ms, 160 muestras a 8 kHz) para identificar voz y silencio. El audio crudo se enruta a la cola STT apropiada segun el modo de operacion (`realtime_audio_queue` para OpenAI Realtime, `stt_audio_queue` para Deepgram). Durante la reproduccion del asistente, los chunks se almacenan en `stt_mute_buffer` en vez de enviarse directamente.

### 6.3 Segmentacion de turnos

El sistema usa `webrtcvad` (agresividad configurable via `WEBRTC_VAD_MODE`) y umbrales configurables para detectar:

- inicio de habla (racha de `SPEECH_START_FRAMES` frames con voz y RMS >= `MIN_VOICE_RMS`);
- continuidad de voz;
- fin de utterance por silencio (`END_SILENCE_FRAMES` frames consecutivos de silencio);
- barge-in durante reproduccion del asistente: requiere `MIN_BARGE_IN_FRAMES` frames de voz con RMS promedio >= `BARGE_IN_MIN_RMS`, y al menos 600 ms de reproduccion del asistente para evitar self-barge-in;
- proteccion contra eco: `ASSISTANT_ECHO_IGNORE_MS` define una ventana al inicio de la reproduccion donde se ignora la deteccion de voz para evitar que el eco del TTS active barge-in.

### 6.4 Transcripcion STT

El sistema soporta dos backends de STT:

#### Modo OpenAI Realtime
- El audio mu-law se envia directamente via WebSocket a la API Realtime de OpenAI.
- OpenAI realiza STT internamente usando Whisper-1 y devuelve transcripciones.
- La deteccion de turnos la maneja el server_vad de OpenAI (threshold 0.5, silence 500 ms).
- Las respuestas LLM se generan automaticamente al fin de turno.

#### Modo Deepgram STT Realtime
- El audio crudo se envia continuamente a Deepgram via WebSocket.
- Se obtienen transcripciones parciales e intermedias para anticipar la respuesta.
- La conexion utiliza un sistema de candidatos con fallback: se prueban multiples combinaciones de modelo (`nova-3`, `nova-2`, `phonecall`), idioma (especifico, multi, sin idioma) y parametros (con/sin vad_events).
- KeepAlive messages se envian cuando no hay audio por 5 segundos para mantener la conexion.
- Keywords boosting configurable via `DEEPGRAM_STT_KEYWORDS` para mejorar reconocimiento de digitos.
- Numeral formatting habilitado para convertir numeros hablados a digitos.

#### Reconexion automatica
- El adaptador de STT realtime implementa reconexion exponencial con hasta `STT_RECONNECT_MAX_ATTEMPTS` intentos.
- En cada reconexion se drenan los chunks de audio stale del buffer para evitar procesar audio antiguo.
- Si la reconexion falla definitivamente, se anuncia al usuario un mensaje de fallo de STT en su idioma.

### 6.5 Gestion de turnos y ventana de gracia

El turn manager (`turn_manager.py`) implementa los siguientes mecanismos:

- **Prefetch silencioso**: las transcripciones parciales con al menos `PARTIAL_TRANSCRIPT_START_CHARS` caracteres disparan una consulta LLM anticipada tras un debounce de `PARTIAL_TRANSCRIPT_DEBOUNCE_MS`. Si la transcripcion final coincide (delta <= `PARTIAL_PREFETCH_MAX_DELTA_CHARS`), se reutiliza la respuesta sin latencia adicional.

- **Ventana de gracia**: cuando una transcripcion final cumple alguna de estas condiciones, se difiere su procesamiento durante `FINAL_TRANSCRIPT_GRACE_MS`:
  - es de una sola palabra sin puntuacion terminal;
  - parece dictado de digitos con menos de 5 digitos acumulados (usa `DIGIT_DICTATION_GRACE_MS` mayor);
  - termina en un marcador linguistico incompleto (ej: "and", "because", "para");
  - el asistente sigue hablando.
  
  Si llega un nuevo final en ese periodo, ambos se combinan. Si no, se procesa el texto acumulado. Antes de procesar, se aplica una extension de medio grace period si el texto aun parece incompleto.

- **Deteccion linguistica de incompletitud**: marcadores de trailing en ingles y espanol (preposiciones, conjunciones, pronombres) y frases incompletas de 2 palabras permiten identificar utterances que probablemente no han terminado.

- **Normalizacion de digitos dictados**: `normalize_digits_in_text()` convierte secuencias como `"four five one zero eight six"` a `"451086"` y colapsa digitos separados por espacios.

- **Filtrado de utterances no accionables**: un conjunto de ~60 frases (`NON_ACTIONABLE_PHRASES`) permite identificar y diferir saludos, fillers, confirmaciones y pronombres aislados. Estos se acumulan con el siguiente texto del usuario.

- **Deteccion de alucinaciones/ecos**: `is_echo_hallucination()` detecta:
  - Repeticion excesiva: cualquier palabra repetida 3+ veces consecutivas.
  - Eco del asistente: >60% de las palabras coinciden con la ultima respuesta del asistente.

- **Deduplicacion**: se descartan transcripciones que duplican exactamente el ultimo texto procesado o la ultima entrada de usuario en el historial.

### 6.6 Generacion conversacional

El texto transcrito se combina con un contexto enriquecido que incluye:

- el **system prompt de Cialix** (~3KB), que define personalidad (Tessa), reglas de atencion, precios, politicas de devolucion, informacion de producto, instrucciones de transferencia y restricciones de comportamiento;
- una instruccion explicita de idioma (actualmente solo ingles);
- los datos ya recopilados en `collected_data` como contexto de sesion para que el LLM no repita preguntas;
- un contador de veces que el asistente ha preguntado por el numero de orden — si >= 2, se inyecta un WARNING para forzar transferencia a agente humano;
- el historial reciente de la sesion (ultimos `MAX_HISTORY_MESSAGES` mensajes).

La generacion se realiza mediante:
- **Modo streaming** (`stream_llm_reply_sync`): genera respuesta por chunks, segmenta progresivamente con `pop_streaming_segments` y alimenta la cola TTS en paralelo.
- **Modo sincrono** (`call_llm`): genera respuesta completa de una vez (usado para prefetch).

Ambos modos usan `temperature=0.2` y `max_tokens=MAX_RESPONSE_TOKENS`.

### 6.7 Sintesis TTS y playback

#### Rime TTS (principal)
La respuesta textual se segmenta y cada segmento se sintetiza con Rime TTS via WebSocket streaming. El adaptador:
- Selecciona el speaker segun el idioma: `RIME_TTS_SPEAKER_EN` (Astra) para ingles, `RIME_TTS_SPEAKER_ES` (celestino) para espanol.
- Solicita audio PCM a `RIME_TTS_SAMPLE_RATE` Hz (8000 por defecto).
- Convierte PCM→mu-law usando una lookup table integrada (sin dependencia de `audioop`).
- Si el sample rate de respuesta difiere de 8 kHz, resamplea con `scipy.signal.resample_poly` (si disponible) o interpolacion lineal como fallback.
- Emite frames de 160 bytes (20 ms @ 8 kHz) alineados.
- Soporta timeouts granulares: `TTS_TTFB_TIMEOUT_SEC` para primer chunk y `TTS_IDLE_TIMEOUT_SEC` entre chunks.

#### Deepgram TTS (alternativa)
Disponible como adaptador alternativo. Genera mu-law a 8 kHz directamente via HTTP POST sin necesidad de conversion.

#### Playback
El playback loop lee items de la cola `playback_queue` y:
- Fragmenta audio en chunks de `TWILIO_OUTBOUND_CHUNK_BYTES` (160 bytes).
- Envia cada frame a Twilio con pacing proporcional (`TWILIO_OUTBOUND_PACING_MS`).
- Emite marcas `mark` al final de cada segmento para sincronizar el estado.
- Procesa eventos `clear` para detener la reproduccion ante barge-in.
- Soporta reintentos TTS con `TTS_MAX_RETRIES` y backoff de `TTS_RETRY_BACKOFF_MS`.
- Opcionalmente guarda frames a disco (`SAVE_TWILIO_FRAMES`) y registra timings diagnosticos.

#### Filler de espera
Antes de la primera respuesta real, el sistema puede emitir un filler de espera configurable por idioma (`FILLER_TEXT_EN` / `FILLER_TEXT_ES`) si la generacion LLM tarda mas de `FILLER_DELAY_MS` milisegundos. El filler se cancela automaticamente si la primera respuesta LLM llega antes.

### 6.8 Monitoreo de inactividad

El modulo `session_runtime` implementa un monitor de silencio por sesion (`monitor_idle_silence`). Si no se detecta actividad durante `IDLE_SILENCE_TIMEOUT_SEC` segundos (por defecto 45), la llamada se cierra automaticamente. El monitor no cierra la llamada mientras el asistente esta hablando.

### 6.9 Proteccion de endpoints de diagnostico

Los endpoints `/test-llm-tts`, `/test-stt` y `/list-models` estan protegidos por la variable `ENABLE_DEBUG_ENDPOINTS`. Si esta en `false` (valor por defecto), devuelven HTTP 404. Esto evita la exposicion de funcionalidad de diagnostico en produccion.

### 6.10 Reconexion automatica de STT realtime

El adaptador de STT realtime implementa un mecanismo de reconexion exponencial con hasta `STT_RECONNECT_MAX_ATTEMPTS` intentos. Antes de cada reconexion se drenan los chunks de audio stale. Si la reconexion falla definitivamente, el sistema anuncia al usuario un mensaje de fallo de STT configurable por idioma e interrumpe el turno actual.

### 6.11 Extraccion y gestion de datos estructurados

Durante cada transcripcion final, el sistema:
1. Extrae datos estructurados (numero de orden, email, telefono, nombre) mediante regex.
2. Normaliza digitos dictados oralmente antes de la extraccion.
3. Almacena los datos en `session.collected_data`.
4. Inyecta los datos como contexto del LLM para evitar preguntas redundantes.
5. Detecta y descarta duplicados puros; los duplicados con contexto conversacional se procesan normalmente.

### 6.12 Barge-in (interrupcion del asistente)

Cuando se detecta barge-in:
1. Se incrementa `active_generation` para invalidar el turno en curso.
2. Se cancelan las tareas de respuesta y prefetch.
3. Se limpian las marcas pendientes y se marcan `assistant_speaking = False`.
4. Se drena la cola de playback y se envia un evento `clear` a Twilio.
5. Se senala `generation_changed` para que el adaptador OpenAI Realtime cancele la respuesta en curso.

## Flujo de uso por modulo

| Modulo | Flujo principal |
| --- | --- |
| Entrada de voz | Twilio llama a `/voice` y recibe TwiML (opcionalmente con `<Play>`) |
| WebSocket | Twilio abre `/media-stream` y envia eventos |
| Routing audio | El audio se enruta a OpenAI Realtime o Deepgram STT segun `USE_OPENAI_REALTIME` |
| VAD | El backend detecta voz, silencio y barge-in frame a frame |
| STT | Se obtienen transcripciones parciales y finales |
| Turn Manager | Se evalua diferir, filtrar, combinar o procesar la transcripcion |
| LLM | Se construye contexto enriquecido y se genera respuesta con streaming |
| TTS | Se sintetiza audio con Rime (o Deepgram) y se convierte a mu-law |
| Playback | Se fragmenta y envia audio a Twilio con pacing |

## Comportamiento esperado

- las respuestas deben ser naturales, conversacionales y contextuales para atencion al cliente de Cialix;
- el sistema opera en modo ingles completo por defecto;
- el sistema debe mantener continuidad de contexto en una misma llamada;
- la reproduccion del asistente debe poder detenerse si el turno deja de ser valido;
- los datos del usuario se extraen y almacenan automaticamente para evitar preguntas repetidas;
- el asistente escala a agente humano tras 2 solicitudes fallidas del numero de orden.

## Validaciones importantes

- se ignoran utterances demasiado cortas (< `MIN_UTTERANCE_MS`);
- se filtran utterances no accionables (saludos, fillers, pronombres aislados);
- se detectan y descartan alucinaciones STT y ecos del TTS;
- se descartan duplicados exactos de transcripciones previas;
- se descartan resultados de turnos cuya `generation` ya no sea la activa;
- el servidor no arranca sin `PUBLIC_URL`;
- si faltan credenciales externas, el sistema registra advertencias y ciertas funciones dejan de estar disponibles;
- las colas usan drop policy para evitar bloqueos.

## Casos de uso relevantes

1. Atencion automatizada de llamadas entrantes para Cialix Customer Support.
2. Consulta de estado de pedidos mediante numero de orden dictado por voz.
3. Procesamiento de solicitudes de devolucion con generacion de RMA.
4. Informacion de producto, ingredientes y precios.
5. Transferencia a agente humano cuando el sistema no puede resolver la consulta.
6. Pruebas de pipeline STT, LLM y TTS mediante endpoints diagnosticos.

# 7. API o comunicacion entre modulos

## Endpoints HTTP y WebSocket

| Metodo | Ruta | Proposito |
| --- | --- | --- |
| `POST` | `/voice` | Devuelve TwiML para abrir el stream de voz (opcionalmente con `<Play>`) |
| `WS` | `/media-stream` | Gestiona audio y eventos de Twilio |
| `GET` | `/test-llm-tts` | Prueba la generacion textual, sanitizacion y segmentacion TTS (requiere `ENABLE_DEBUG_ENDPOINTS=true`) |
| `POST` | `/test-stt` | Prueba la transcripcion STT (requiere `ENABLE_DEBUG_ENDPOINTS=true`) |
| `GET` | `/list-models` | Lista modelos accesibles desde OpenAI (requiere `ENABLE_DEBUG_ENDPOINTS=true`) |
| `GET` | `/` | Health check basico |
| `GET` | `/static/{file}` | Sirve archivos estaticos (saludo pregrabado) |

## Detalle de endpoints

### `POST /voice`

- Entrada: no requiere body especifico.
- Logica: construye la URL WebSocket a partir de `PUBLIC_URL`. Si existe `static/greeting.wav` o `TWIML_INITIAL_GREETING_ENABLED=true`, incluye `<Play>` antes de `<Connect>`.
- Salida: XML TwiML con `<Connect><Stream>` hacia el WebSocket del servicio.
- Respuesta esperada: `200 OK` con `application/xml`.

### `WS /media-stream`

- Entrada: eventos de Twilio en formato JSON.
- Salida: eventos JSON hacia Twilio como `media`, `mark` y `clear`.
- Responsabilidad: coordinar toda la sesion de audio de una llamada.
- Tareas que se crean por sesion:
  - `playback_loop` — siempre activa
  - `process_transcripts` — siempre activa (segundo listener en modo Deepgram)
  - `run_realtime_session` o `run_realtime_stt` — segun modo de operacion
  - `monitor_idle_silence` — monitoreo de inactividad

Ejemplo simplificado de evento `media` entrante:

```json
{
  "event": "media",
  "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXX",
  "media": {
    "payload": "<audio-base64-mulaw>"
  }
}
```

### `GET /test-llm-tts`

- Query param: `q` obligatorio.
- Proposito: validar respuesta textual del LLM, sanitizacion y cantidad de segmentos TTS.
- Respuesta: JSON con `input`, `reply`, `sanitized_reply`, `tts_segments` y `tts_ready`.

### `POST /test-stt`

- Entrada: usa audio dummy interno (1 segundo de silencio PCM).
- Proposito: validar disponibilidad del flujo STT.
- Respuesta: JSON con `text`, `segments`, `language`, `stt_ready` y `model`.

### `GET /list-models`

- Entrada: no requiere parametros.
- Proposito: listar modelos visibles mediante la API de OpenAI.
- Respuesta: JSON con `models` o `error`.

### `GET /`

- Entrada: no requiere parametros.
- Proposito: verificar disponibilidad del servicio.
- Respuesta: `{"status": "ok", "message": "STT server running"}`.

## Comunicacion interna entre modulos

### Colas asyncio

El sistema usa multiples colas `asyncio.Queue` con tamanos maximos configurables para desacoplar los componentes:

| Cola | Flujo | Tamano por defecto |
| --- | --- | --- |
| `stt_audio_queue` | Audio → Deepgram STT | 300 |
| `realtime_audio_queue` | Audio → OpenAI Realtime | 300 |
| `transcript_queue` | STT → Turn Manager | 32 |
| `playback_queue` | TTS/Control → Playback | 1024 |

Todas las colas usan una politica de **drop del elemento mas antiguo** cuando estan llenas (implementada en `enqueue_nowait_with_drop`).

### Eventos y senales

| Mecanismo | Proposito |
| --- | --- |
| `generation_changed` (Event) | Notifica barge-in al adaptador Realtime |
| `active_generation` (int) | Invalida turnos previos en todo el pipeline |
| `pending_marks` (set) | Sincroniza fin de reproduccion con confirmacion de Twilio |

## Formato estandar de respuestas

No existe un contrato de respuesta unificado. El sistema usa:

- XML para `/voice`;
- JSON para endpoints de prueba y diagnostico;
- mensajes JSON de eventos para el WebSocket de Twilio.

## Manejo de errores

| Area | Estrategia |
| --- | --- |
| STT Realtime | Reconexion exponencial con candidatos fallback; anuncio de fallo si agotados |
| STT Batch | Retry con backoff (`_BATCH_MAX_RETRIES=2`); captura de `HTTPError`, `URLError` y timeouts |
| LLM | Captura generica con mensaje de fallback ("Lo siento, tuve un problema momentaneo...") |
| LLM Timeout | Si `LLM_TIMEOUT_SEC > 0`, la tarea se cancela y se emite fallback |
| TTS | Retry con `TTS_MAX_RETRIES` y backoff; timeouts granulares (TTFB, idle, total) |
| WebSocket Twilio | Captura de `WebSocketDisconnect` y `RuntimeError` con limpieza de sesion |
| WebSocket OpenAI | Captura global con log de excepcion |
| Colas llenas | Drop del elemento mas antiguo (no bloqueo) |

## Codigos de estado o respuestas relevantes

| Contexto | Respuesta esperada |
| --- | --- |
| `/` | `200 OK` |
| `/voice` | `200 OK` |
| `/test-llm-tts` | `200 OK` o error controlado si falta OpenAI |
| `/test-stt` | `200 OK` o fallo si falta Deepgram |
| `/list-models` | `200 OK` con `error` en payload si falla la consulta |
| Endpoints debug deshabilitados | `404 Not Found` |

# 8. Autenticacion y seguridad

## Metodo de autenticacion

No existe autenticacion de usuarios en el backend actual.

## Autorizacion por roles o permisos

No existe un esquema de roles o permisos.

## Proteccion de rutas

Las rutas expuestas no tienen proteccion aplicativa. La seguridad depende principalmente del control de acceso a las credenciales y de la exposicion publica del servicio. Los endpoints de diagnostico se protegen con un flag booleano (`ENABLE_DEBUG_ENDPOINTS`).

## Manejo de sesiones o tokens

- Twilio mantiene la sesion de llamada y stream.
- OpenAI se autentica mediante API key en variables de entorno (header `Authorization: Bearer`).
- Deepgram se autentica mediante API key en variables de entorno (header `Authorization: Token`).
- Rime se autentica mediante API key en variables de entorno.
- El backend no genera tokens propios ni maneja sesiones autenticadas de usuario.

## Buenas practicas implementadas

- uso de variables de entorno para credenciales;
- aislamiento de estado por llamada;
- cierre y cancelacion de tareas al finalizar una sesion;
- restriccion del formato TTS para compatibilidad con Twilio;
- proteccion de endpoints de diagnostico via flag;
- no se imprimen credenciales en logs.

## Riesgos o pendientes de seguridad

| Riesgo | Estado |
| --- | --- |
| Credenciales en `entornoLocal.env` | **CRITICO**: el archivo contiene API keys reales que deben rotarse y nunca versionarse |
| Validacion de origen de Twilio | No implementada; no se verifica firma de Twilio |
| Rate limiting | No implementado |
| Persistencia segura de logs | No especificada |
| Sanitizacion de entrada del usuario | Las transcripciones se procesan tal cual — no hay sanitizacion XSS/injection (irrelevante para canal de voz pero relevante si se agrega persistencia) |

# 9. Variables de entorno y configuracion

## Variables requeridas

| Variable | Obligatoria | Proposito | Ejemplo seguro |
| --- | --- | --- | --- |
| `PUBLIC_URL` | Si | URL publica del servicio para construir el WebSocket Twilio | `https://mi-servicio.up.railway.app` |
| `OPENAI_API_KEY` | Si | Credencial para OpenAI (Chat y Realtime) | `sk-xxxxxxxxxxxxxxxx` |
| `DEEPGRAM_API_KEY` | Condicional | Credencial para STT y TTS en Deepgram (requerida si `USE_OPENAI_REALTIME=false`) | `dg_XXXXXXXXXXXXXXXX` |
| `RIME_API_KEY` | Si | Credencial para Rime TTS (proveedor principal de TTS) | `xxxxxxxxxxxx` |

## Variables de modo de operacion

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `USE_OPENAI_REALTIME` | `true` | Habilita modo OpenAI Realtime (STT+LLM integrado); si `false`, usa Deepgram STT + OpenAI Chat |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo conversacional para Chat Completions |
| `OPENAI_REALTIME_MODEL` | `gpt-4o-mini-realtime-preview` | Modelo para OpenAI Realtime API |
| `OPENAI_REALTIME_TEMPERATURE` | `0.7` | Temperatura del modelo Realtime (minimo API: 0.6) |

## Variables de Deepgram STT

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `DEEPGRAM_STT_MODEL` | `nova-3` | Modelo STT |
| `DEEPGRAM_STT_PUNCTUATE` | `true` | Inserta puntuacion en transcript |
| `DEEPGRAM_STT_SMART_FORMAT` | `true` | Mejora el formato del transcript |
| `DEEPGRAM_STT_DETECT_LANGUAGE` | `false` | Habilita deteccion de idioma (deshabilitada — modo ingles completo) |
| `DEEPGRAM_STT_LANGUAGE_HINT` | `en` | Fuerza idioma de entrada |
| `DEEPGRAM_STT_ENDPOINTING_MS` | `500` | Endpointing de Deepgram realtime |
| `DEEPGRAM_UTTERANCE_END_MS` | `1000` | Utterance end timeout de Deepgram |
| `DEEPGRAM_STT_NUMERALS` | `true` | Convierte numeros hablados a digitos |
| `DEEPGRAM_STT_KEYWORDS` | `zero:2,one:2,...,order number:2` | Keywords boosting para mejorar reconocimiento de digitos y frases clave |

## Variables de Rime TTS

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `RIME_TTS_MODEL_ID` | `arcana` | Modelo TTS de Rime |
| `RIME_TTS_SPEAKER_EN` | `Astra` | Voz para ingles |
| `RIME_TTS_SPEAKER_ES` | `celestino` | Voz para espanol |
| `RIME_TTS_SAMPLE_RATE` | `8000` | Sample rate solicitado a Rime |

## Variables de Deepgram TTS (alternativa)

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `DEEPGRAM_TTS_ENCODING` | `mulaw` | Encoding del audio saliente |
| `DEEPGRAM_TTS_SAMPLE_RATE` | `8000` | Sample rate del TTS |

## Variables de idioma y saludo

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `DEFAULT_CALL_LANGUAGE` | `en` | Idioma fallback de la sesion |
| `INITIAL_GREETING_ENABLED` | `true` | Activa el saludo inicial (legacy, funcion deshabilitada) |
| `INITIAL_GREETING_TEXT` | `Thank you for calling Cialix Customer Support` | Texto del saludo inicial en ingles |
| `INITIAL_GREETING_TEXT_ES` | `Gracias por llamar a la linea de atencion al cliente de Cialix` | Texto del saludo en espanol |
| `TWIML_INITIAL_GREETING_ENABLED` | `false` | Incluye `<Play>` del saludo pregrabado en TwiML |
| `TWIML_INITIAL_GREETING_LANG` | `en` | Idioma del saludo pregrabado |

## Variables operativas de audio y VAD

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `WEBRTC_VAD_MODE` | `1` | Agresividad del VAD (0=menos agresivo, 3=mas agresivo) |
| `MIN_UTTERANCE_MS` | `180` | Duracion minima aceptada para una utterance |
| `MIN_SPEECH_FRAMES` | `5` | Frames minimos de voz |
| `END_SILENCE_FRAMES` | `14` | Frames de silencio para cierre de turno (280 ms) |
| `SPEECH_START_FRAMES` | `1` | Frames requeridos para detectar inicio |
| `MIN_BARGE_IN_FRAMES` | `12` | Frames minimos para interrupcion |
| `PRE_SPEECH_FRAMES` | `5` | Frames previos retenidos |
| `TRIM_TRAILING_SILENCE_FRAMES` | `6` | Recorte final de silencio |
| `MIN_VOICE_RMS` | `260` | Umbral RMS minimo de voz |
| `BARGE_IN_MIN_RMS` | `900` | Umbral RMS minimo promedio para barge-in |
| `ENABLE_BARGE_IN` | `true` | Activa interrupcion del asistente |
| `ASSISTANT_ECHO_IGNORE_MS` | `2000` | Ventana para ignorar eco al inicio de la reproduccion |

## Variables de timeouts y limites

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `STT_TIMEOUT_SEC` | `0` | Timeout de STT batch; `0` implica timeout del cliente |
| `LLM_TIMEOUT_SEC` | `5.0` | Timeout maximo de la llamada al LLM |
| `TTS_TTFB_TIMEOUT_SEC` | `15.0` | Timeout para primer chunk TTS |
| `TTS_IDLE_TIMEOUT_SEC` | `45.0` | Timeout maximo entre chunks TTS |
| `TTS_TIMEOUT_SEC` | `45.0` | Timeout total maximo por segmento TTS |
| `TTS_MAX_RETRIES` | `1` | Reintentos de TTS (total intentos = retries + 1) |
| `TTS_RETRY_BACKOFF_MS` | `250` | Backoff entre reintentos TTS |
| `MAX_HISTORY_MESSAGES` | `12` | Longitud maxima del historial |
| `MAX_RESPONSE_TOKENS` | `150` | Limite de tokens de respuesta LLM |
| `IDLE_SILENCE_TIMEOUT_SEC` | `45` | Segundos de inactividad antes de cerrar la llamada |

## Variables de streaming y turnos

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `REALTIME_TTS_STREAMING` | `false` | Si true, envia segmentos TTS progresivamente; si false, buferea la respuesta completa |
| `STREAMING_SEGMENT_MAX_CHARS` | `200` | Caracteres maximos por segmento de streaming TTS |
| `STREAMING_FIRST_SEGMENT_CHARS` | `200` | Caracteres maximos del primer segmento (para TTFT rapido) |
| `FILLER_TTS_ENABLED` | `true` | Habilita la emision de filler de espera TTS |
| `FILLER_TEXT_EN` | `""` | Texto del filler en ingles (vacio = sin filler) |
| `FILLER_TEXT_ES` | `""` | Texto del filler en espanol |
| `FILLER_DELAY_MS` | `1200` | Milisegundos antes de emitir el filler de espera |
| `FINAL_TRANSCRIPT_GRACE_MS` | `800` | Ventana de gracia para diferir finals cortos/incompletos |
| `DIGIT_DICTATION_GRACE_MS` | `2000` | Ventana de gracia extendida para dictado de digitos |
| `SHORT_FINAL_MAX_WORDS` | `3` | Umbral de palabras para considerar un final como corto |
| `PARTIAL_TRANSCRIPT_START_CHARS` | `20` | Caracteres minimos para iniciar prefetch desde parciales |
| `PARTIAL_TRANSCRIPT_DEBOUNCE_MS` | `200` | Debounce antes de lanzar prefetch |
| `FINAL_RESTART_DELTA_CHARS` | `12` | Delta de caracteres para reiniciar pipeline desde un final nuevo |
| `PARTIAL_PREFETCH_MAX_DELTA_CHARS` | `40` | Delta maximo de caracteres para reutilizar prefetch parcial |

## Variables de colas y buffers

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `STT_AUDIO_QUEUE_MAXSIZE` | `300` | Tamano maximo de la cola de audio para STT Deepgram |
| `REALTIME_AUDIO_QUEUE_MAXSIZE` | `300` | Tamano maximo de la cola de audio para OpenAI Realtime |
| `STT_MUTE_BUFFER_CHUNKS` | `25` | Chunks de audio retenidos durante muteo (25 × 20 ms = 500 ms) |
| `TRANSCRIPT_QUEUE_MAXSIZE` | `32` | Tamano maximo de la cola de transcripciones |
| `PLAYBACK_QUEUE_MAXSIZE` | `1024` | Tamano maximo de la cola de playback |
| `TEXT_SEGMENT_QUEUE_MAXSIZE` | `16` | Tamano maximo de la cola de segmentos de texto |

## Variables de reconexion STT

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `STT_RECONNECT_MAX_ATTEMPTS` | `3` | Intentos maximos de reconexion STT realtime |
| `STT_RECONNECT_BASE_DELAY_MS` | `250` | Retardo base para reconexion exponencial |
| `STT_RECONNECT_MAX_DELAY_MS` | `2000` | Retardo maximo para reconexion exponencial |
| `STT_FAILURE_PROMPT_EN` | `"I'm having trouble hearing you right now."` | Mensaje al usuario cuando falla STT (ingles) |
| `STT_FAILURE_PROMPT_ES` | `"Estoy teniendo problemas para escucharte en este momento."` | Mensaje cuando falla STT (espanol) |

## Variables de diagnostico

| Variable | Default actual | Proposito |
| --- | --- | --- |
| `ENABLE_DEBUG_ENDPOINTS` | `false` | Habilita los endpoints de diagnostico |
| `LOG_TWILIO_PLAYBACK` | `false` | Activa logs detallados de playback |
| `SAVE_TWILIO_FRAMES` | `false` | Guarda frames de audio saliente a disco para diagnostico |
| `TWILIO_OUTBOUND_PACING_MS` | `20` | Pausa entre frames salientes |
| `TWILIO_OUTBOUND_CHUNK_BYTES` | `160` | Tamano de frame saliente (20 ms @ 8 kHz mu-law) |

## Configuracion por ambiente

### Desarrollo local

- uso de `STT_server/entornoLocal.env`;
- ejecucion local con Uvicorn;
- configuracion manual de llaves y URL publica de pruebas.

### Produccion en Railway

- configuracion mediante variables del servicio en Railway;
- `PORT` es provisto por Railway y no debe fijarse manualmente;
- `PUBLIC_URL` debe apuntar al dominio publico real del servicio;
- builder RAILPACK, arranque via `start.sh`.

### Produccion en Docker

- uso de `Dockerfile` basado en Python 3.11-slim;
- dependencias adicionales via `requirements.docker.txt` (scipy);
- arranque via `start.sh` con `PORT` configurable (default 8080).

## Ejemplo seguro de configuracion

```dotenv
PUBLIC_URL=https://mi-servicio.up.railway.app
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o-mini
USE_OPENAI_REALTIME=true
OPENAI_REALTIME_MODEL=gpt-4o-mini-realtime-preview
DEEPGRAM_API_KEY=dg_xxxxxxxxxxxxxxxx
DEEPGRAM_STT_MODEL=nova-3
DEEPGRAM_STT_LANGUAGE_HINT=en
RIME_API_KEY=xxxxxxxxxxxxxxxxxxxx
RIME_TTS_MODEL_ID=arcana
RIME_TTS_SPEAKER_EN=Astra
DEFAULT_CALL_LANGUAGE=en
LLM_TIMEOUT_SEC=5.0
TTS_TTFB_TIMEOUT_SEC=15.0
TTS_IDLE_TIMEOUT_SEC=45.0
WEB_CONCURRENCY=1
```

# 10. Despliegue y ejecucion

## Requisitos para correr el proyecto

- Python 3.11 o superior (Dockerfile usa 3.11-slim);
- acceso a credenciales validas de OpenAI y Rime TTS;
- opcionalmente, credenciales de Deepgram para modo STT alternativo;
- una cuenta Twilio con capacidad de Media Streams;
- dominio publico HTTPS accesible desde Twilio;
- entorno virtual Python para desarrollo local.

## Instalacion local

1. Crear y activar un entorno virtual.
2. Instalar dependencias desde `requirements.txt`.
3. Configurar variables de entorno en `STT_server/entornoLocal.env`.
4. Verificar que `PUBLIC_URL` apunte a una URL publica valida si se va a probar con Twilio.

Ejemplo:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Ejecucion en desarrollo

La aplicacion se inicia mediante:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

O via el script de arranque:

```bash
sh start.sh
```

## Despliegue con Docker

```bash
docker build -t agent-ai .
docker run -p 8080:8080 --env-file .env agent-ai
```

El Dockerfile:
- Usa Python 3.11-slim como base.
- Instala dependencias de build (autoconf, automake, etc.) para compilaciones nativas.
- Copia el proyecto completo e instala dependencias de `requirements.txt` y `requirements.docker.txt`.
- Ejecuta `start.sh` como CMD.

## Despliegue en Railway

El archivo `railway.toml` define el arranque del servicio con:

```toml
[build]
builder = "RAILPACK"

[deploy]
startCommand = "sh -lc '/app/start.sh'"
```

Pasos recomendados:

1. Conectar el repositorio a Railway.
2. Configurar variables de entorno obligatorias y recomendadas.
3. Habilitar dominio publico del servicio.
4. Establecer `PUBLIC_URL` con el dominio real asignado por Railway.
5. Desplegar y validar el endpoint `/`.
6. Configurar en Twilio el webhook de voz hacia `POST /voice`.

## Despliegue en Azure (App Service con Git)

1. Crear una Azure Web App con runtime Python 3.11+.
2. Configurar el deployment center con Git local.
3. Agregar el remote de Azure: `git remote add Azure <url>`.
4. Configurar variables de entorno en la Web App.
5. Desplegar con `git push Azure main`.
6. Verificar el endpoint de health check `/`.

## Consideraciones para produccion

- usar un solo worker para priorizar estabilidad de sesiones WebSocket;
- **nunca versionar** archivos con credenciales reales (`entornoLocal.env`);
- monitorear tiempos de respuesta de STT, LLM y TTS;
- revisar limites de costo y cuota de OpenAI, Deepgram, Rime y Twilio;
- considerar limpieza de dependencias heredadas (`faster-whisper`, `librosa`, `soundfile`);
- `ENABLE_DEBUG_ENDPOINTS` debe estar en `false` en produccion.

# 11. Pruebas y mantenimiento

## Pruebas existentes o recomendadas

No se identifican pruebas automatizadas versionadas en el repositorio principal. El directorio `scripts/` contiene scripts de diagnostico y testing manual:

| Script | Proposito |
| --- | --- |
| `run_rime_tts_test.py` | Prueba de TTS via Rime |
| `run_full_playback_test.py` | Prueba completa del pipeline de playback |
| `smoke_playback.py` | Smoke test de playback |
| `test_sanitize.py` | Prueba de sanitizacion de texto TTS |
| `analyze_mulaw.py` / `analyze_wav.py` | Analisis de archivos de audio |
| `compare_wavs.py` / `compare_mulaw_pair.py` | Comparacion de archivos de audio |
| `inspect_wav.py` | Inspeccion de formato WAV |
| `mulaw_to_wav.py` | Conversion de mu-law a WAV |
| `parse_twilio_timings.py` | Analisis de timings de playback |

Se recomiendan al menos las siguientes validaciones:

- prueba de health check en `/`;
- prueba de conectividad con OpenAI mediante `/list-models`;
- prueba STT mediante `/test-stt`;
- prueba de respuesta textual con `/test-llm-tts?q=...`;
- prueba end-to-end con una llamada real de Twilio.

## Estrategia de validacion

| Nivel | Validacion sugerida |
| --- | --- |
| Unitario | Normalizacion de idioma, segmentacion de texto, extraccion de datos, conversion mu-law, filtrado de utterances |
| Integracion | Deepgram STT, Rime TTS, OpenAI Realtime, OpenAI Chat y WebSocket de Twilio |
| End-to-end | Flujo completo de llamada telefonica con diferentes escenarios de atencion al cliente |

## Buenas practicas de mantenimiento

- mantener separadas credenciales de configuracion local y de produccion;
- revisar cambios de API en OpenAI, Deepgram y Rime;
- documentar cualquier cambio de modelo, voces o parametros de audio;
- controlar el crecimiento del modulo `turn_manager.py` (~800 lineas);
- eliminar dependencias heredadas cuando dejen de ser necesarias;
- rotar credenciales periodicamente y nunca exponerlas en el repositorio.

## Puntos criticos a revisar

- estabilidad del WebSocket con Twilio y OpenAI Realtime;
- cancelacion correcta de tareas al cerrar una llamada;
- consistencia del idioma detectado y del idioma de respuesta;
- latencia total por turno (STT + LLM + TTS + playback);
- manejo de errores de red con servicios externos;
- comportamiento del buffer de muteo STT en transiciones rapidas;
- consumo de memoria con multiples llamadas simultaneas.

# 12. Estado actual y pendientes

## Funcionalidades implementadas

- recepcion de llamadas con Twilio Media Streams;
- VAD local con webrtcvad (agresividad configurable) y segmentacion por turnos;
- modo dual de operacion: OpenAI Realtime (STT+LLM integrado) y Deepgram STT + OpenAI Chat;
- transcripcion STT realtime via WebSocket de Deepgram con sistema de candidatos y fallback;
- prefetch silencioso de respuesta LLM desde transcripciones parciales;
- ventana de gracia para diferir finals cortos, incompletos o durante dictado de digitos;
- deteccion linguistica de utterances incompletas (marcadores en ingles y espanol);
- normalizacion de digitos dictados oralmente;
- generacion LLM con OpenAI GPT-4o-mini con streaming y segmentacion progresiva;
- TTS principal con Rime via WebSocket con conversion PCM→mu-law integrada;
- TTS alternativo con Deepgram via HTTP;
- filler de espera configurable por idioma;
- barge-in con umbral RMS promedio y proteccion contra eco;
- historial corto por sesion (hasta 12 mensajes);
- saludo pregrabado via TwiML `<Play>`;
- endpoints de diagnostico y health check;
- proteccion de endpoints de diagnostico via flag;
- filtrado de utterances no accionables (~60 frases);
- deteccion y descarte de alucinaciones STT y ecos del TTS;
- extraccion automatica de datos estructurados (order_number, email, phone, name);
- deteccion de preguntas repetidas con escalacion a agente humano;
- monitoreo de inactividad con cierre automatico;
- reconexion automatica STT con backoff exponencial y drenado de audio stale;
- muteo de STT durante reproduccion del asistente con buffer de replay;
- anuncio de fallo de STT al usuario en su idioma;
- soporte Docker con Dockerfile y start.sh;
- servicio de archivos estaticos;
- utilidad de envio de audio por email;
- scripts de diagnostico y analisis de audio;
- arquitectura modular: adapters, domain, services, utils.

## Funcionalidades faltantes o no especificadas

- autenticacion y proteccion de endpoints (validacion de firma Twilio);
- persistencia de conversaciones en base de datos;
- implementacion real de herramientas/tools del LLM (ship_information, order_information, cialix_rma, cialix_transfer_call_tool);
- trazabilidad centralizada de llamadas;
- pruebas automatizadas formales;
- panel de monitoreo u operacion;
- rate limiting;
- soporte multilingue completo (espanol deshabilitado actualmente).

## Mejoras futuras sugeridas

1. implementar pruebas unitarias e integracion;
2. agregar validacion de origen o firma para Twilio;
3. implementar las herramientas/tools referenciadas en el system prompt como funciones reales;
4. incorporar observabilidad estructurada (metricas, traces);
5. persistir metricas y eventos de llamada;
6. agregar persistencia de conversaciones en base de datos;
7. reactivar soporte bilingue (ingles/espanol) cuando sea necesario;
8. limpiar scripts y dependencias heredadas;
9. rotar las credenciales expuestas en `entornoLocal.env` y agregar el archivo a `.gitignore`.

## Limitaciones conocidas

- el estado se pierde ante reinicios del proceso;
- el sistema depende de varios servicios externos para operar (OpenAI, Rime, Twilio);
- las herramientas del LLM (ship_information, order_information, cialix_rma, cialix_transfer_call_tool) no estan implementadas como funciones reales — el LLM las simula;
- el modo espanol esta deshabilitado (hardcoded a ingles);
- el archivo `entornoLocal.env` contiene credenciales reales que deben rotarse;
- el repositorio contiene artefactos y dependencias heredadas que no forman parte del flujo principal;
- la funcion `sanitize_tts_text()` esta deshabilitada (no-op) para evitar problemas de corrupcion de audio.

# 13. Glosario

| Termino | Definicion |
| --- | --- |
| ASGI | Interfaz estandar para aplicaciones Python asincronas |
| Barge-in | Interrupcion de la respuesta del asistente cuando el usuario comienza a hablar |
| Deepgram STT | Servicio de reconocimiento de voz a texto (Speech-to-Text) |
| Deepgram TTS | Servicio de sintesis de texto a voz (Text-to-Speech) |
| Frame | Unidad corta de audio (20 ms, 160 muestras a 8 kHz) procesada por el VAD |
| Generation | Version logica del turno actual; se incrementa en cada interrupcion para invalidar turnos previos |
| Grace period | Ventana de tiempo configurable para esperar mas input antes de procesar una transcripcion final |
| LLM | Modelo de lenguaje usado para generar respuestas (Large Language Model) |
| Media Stream | Canal bidireccional de audio y eventos de Twilio sobre WebSocket |
| Mu-law | Codificacion logaritmica de audio usada por Twilio para telefonia (G.711) |
| PCM | Representacion lineal sin compresion del audio (Pulse Code Modulation) |
| Prefetch | Consulta anticipada al LLM a partir de transcripciones parciales para reducir latencia |
| Rime TTS | Servicio de sintesis de texto a voz via WebSocket streaming |
| RMA | Return Merchandise Authorization; codigo de autorizacion para devolucion de producto |
| RMS | Root Mean Square; medida de la energia/volumen del audio |
| STT | Conversion de voz a texto (Speech-to-Text) |
| TTS | Conversion de texto a voz (Text-to-Speech) |
| TwiML | XML usado por Twilio para describir el comportamiento de una llamada |
| Utterance | Segmento de audio que representa un turno de habla del usuario |
| VAD | Voice Activity Detection, mecanismo para detectar voz y silencio |
| WebSocket | Protocolo de comunicacion bidireccional en tiempo real sobre TCP |
