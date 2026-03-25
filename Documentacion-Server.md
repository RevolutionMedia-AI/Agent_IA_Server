# 1. Resumen del proyecto

  

## Descripcion general

  

Este proyecto implementa un servidor de voz conversacional en tiempo real orientado a telefonia. La aplicacion expone endpoints HTTP y WebSocket mediante FastAPI para integrarse con Twilio Media Streams, transcribe el audio entrante con Deepgram, genera una respuesta breve mediante OpenAI y sintetiza la salida de voz nuevamente con Deepgram para devolverla a la llamada.

  

El sistema esta disenado para operar como backend de un agente telefonico sin interfaz grafica ni persistencia en base de datos. El estado de cada llamada se mantiene en memoria mientras la sesion permanece activa.

El backend emplea una arquitectura de STT dual: un canal en tiempo real mediante WebSocket de Deepgram para obtener transcripciones parciales y prefetch de respuestas, y un canal batch mediante HTTP para obtener transcripciones finales autoritativas con deteccion de idioma. Adicionalmente implementa un mecanismo de ventana de gracia para diferir transcripciones finales cortas o incompletas, evitando interrupciones prematuras durante el habla del usuario.

  

## Proposito del sistema

  

El objetivo principal es habilitar una experiencia de atencion conversacional por voz sobre telefonia IP con las siguientes capacidades:

  

- recibir audio de una llamada telefonica mediante Twilio;

- detectar turnos de habla del usuario;

- transcribir el audio a texto en tiempo casi real;

- generar respuestas breves y compatibles con un canal de voz;

- sintetizar audio de salida y reenviarlo a la misma llamada.

  

## Problema que resuelve

  

El proyecto resuelve la orquestacion de un flujo de voz telefonico que normalmente involucra multiples sistemas desacoplados:

  

- ingesta de audio en tiempo real desde una plataforma de telefonia;

- deteccion de actividad de voz y cierre de utterances;

- conversion de voz a texto;

- generacion de respuesta con un modelo conversacional;

- conversion de texto a voz en un formato compatible con Twilio;

- coordinacion de reproduccion, interrupcion y continuidad de la llamada.

  

## Alcance funcional

  

El alcance actual incluye:

  

- endpoint HTTP para devolver TwiML a Twilio;

- endpoint WebSocket bidireccional para intercambio de audio y eventos;

- manejo de sesiones activas por llamada en memoria;

- VAD local con `webrtcvad` para detectar inicio y fin de turno;

- transcripcion STT dual: realtime por WebSocket y batch por HTTP mediante Deepgram;

- sistema de fallback con candidatos multiples para la conexion STT realtime;

- prefetch silencioso de respuestas LLM a partir de transcripciones parciales;

- ventana de gracia configurable para diferir finals cortos o linguisticamente incompletos;

- deteccion de utterances incompletas mediante marcadores linguisticos en ingles y espanol;

- generacion de respuesta mediante OpenAI con streaming;

- sintesis TTS mediante Deepgram con voces diferenciadas por idioma;

- filler de espera configurable por idioma;

- soporte bilingue en ingles y espanol con persistencia de idioma por sesion;

- barge-in configurable para permitir interrupcion del asistente (cancelacion de turnos, tareas y cola de reproducción);

- filtrado de utterances no accionables (saludos, fillers, confirmaciones);

- memoria estructurada en `session.collected_data` para evitar repetir solicitudes de datos ya capturados;

- monitoreo de inactividad con cierre automatico de llamada;

- reconexion automatica de STT realtime con backoff exponencial;

- proteccion de endpoints de diagnostico mediante flag configurable;

- endpoints de prueba y diagnostico.

  

Quedan fuera del alcance actual:

  

- frontend o panel administrativo;

- base de datos o persistencia historica;

- autenticacion de usuarios finales;

- control de acceso por roles;

- pruebas automatizadas formales;

- observabilidad centralizada y metricas persistentes.

  

# 2. Tecnologias utilizadas

  

## Frontend

  

No existe frontend en el estado actual del repositorio.

  

## Backend

  

| Tecnologia | Uso principal |

| --- | --- |

| Python | Lenguaje principal del proyecto |

| FastAPI | Exposicion de endpoints HTTP y WebSocket |

| Uvicorn | Servidor ASGI para ejecucion local y despliegue |

| asyncio | Coordinacion asincrona de tareas y colas |

| python-dotenv | Carga de configuracion local desde archivo `.env` |

| webrtcvad | Deteccion de actividad de voz |

| OpenAI SDK | Cliente para generacion de respuestas conversacionales |

  

## Base de datos

  

No se utiliza base de datos. Toda la informacion operacional se conserva temporalmente en memoria del proceso.

  

## Servicios externos

  

| Servicio | Proposito |

| --- | --- |

| Twilio | Transporte de llamadas y Media Streams bidireccionales |

| Deepgram STT | Transcripcion de audio a texto |

| Deepgram TTS | Sintesis de texto a voz en formato compatible con Twilio |

| OpenAI | Generacion de respuesta del asistente |

| Railway | Plataforma de despliegue (configuracion heredada) |

| Azure App Service | Plataforma de despliegue principal mediante `git push` |

  

## Herramientas de desarrollo

  

| Herramienta | Uso |

| --- | --- |

| Git | Control de versiones |

| venv | Entorno virtual Python local |

| Railway CLI o panel web | Despliegue y configuracion del servicio (heredado) |

| Azure App Service + Git | Despliegue principal mediante `git push Azure main` |

  

## Dependencias relevantes

  

| Dependencia | Rol en el proyecto |

| --- | --- |

| fastapi | Framework principal del backend |

| uvicorn[standard] | Ejecucion del servidor ASGI |

| python-dotenv | Carga de variables locales |

| webrtcvad | Deteccion de voz por frames |

| websockets | Conexion WebSocket con Deepgram STT realtime |

| httpx | Cliente HTTP asincrono (dependencia indirecta via OpenAI SDK) |

| openai | Cliente del modelo conversacional |

| twilio | Dependencia declarada para integracion con el ecosistema Twilio |

| setuptools<81 | Restriccion de compatibilidad declarada |

  

Dependencias declaradas en el repositorio como `faster-whisper`, `numpy`, `librosa` y `soundfile` aparecen asociadas a scripts auxiliares o a configuracion heredada y no forman parte del flujo principal actual del servidor telefonico. Nota: `websockets` no esta declarado en `requirements.txt` pero es requerido por `deepgram_stt_realtime.py`; se instala como dependencia transitiva de `uvicorn[standard]`.

  

# 3. Arquitectura general

  

## Enfoque arquitectonico

  

La aplicacion sigue un enfoque de backend orientado a eventos y estado en memoria por sesion. Cada llamada activa se representa mediante una estructura `CallSession` que encapsula buffers, colas, historial y metadatos de reproduccion. El procesamiento del audio se divide en etapas desacopladas coordinadas con `asyncio`.

  

## Relacion entre componentes

  

El sistema se compone de los siguientes bloques funcionales:

  

| Componente | Responsabilidad |

| --- | --- |

| Endpoint `/voice` | Entregar TwiML para iniciar el Media Stream |

| Endpoint `/media-stream` | Gestionar eventos WebSocket de Twilio |

| VAD local | Detectar inicio y fin de una utterance mediante `webrtcvad` |

| STT realtime | Transcribir audio en tiempo real via WebSocket de Deepgram |

| STT batch | Transcribir utterances completas via HTTP de Deepgram con deteccion de idioma |

| Turn manager | Coordinar transcripciones, prefetch, ventana de gracia y pipeline de respuesta |

| LLM | Generar respuesta textual breve con streaming |

| TTS | Sintetizar audio de salida con Deepgram |

| Playback | Reenviar audio, marcas y comandos de limpieza a Twilio |

| Gestion de sesion | Mantener estado por llamada |

| Monitoreo de inactividad | Cerrar llamadas sin actividad tras timeout configurable |

| Filtrado de utterances | Descartar utterances no accionables (saludos, fillers) |

  

## Estructura de comunicacion

  

### Flujo general

  

1. Twilio invoca `POST /voice`.

2. El backend responde con TwiML que indica abrir un stream WebSocket hacia `/media-stream`.

3. Twilio abre el WebSocket bidireccional y comienza a enviar eventos `connected`, `start` y `media`.

4. El servidor acumula audio en frames PCM lineales y aplica VAD local con `webrtcvad`.

5. En paralelo, el audio crudo se envia al canal STT realtime de Deepgram via WebSocket.

6. Las transcripciones parciales del realtime disparan un prefetch silencioso de respuesta LLM.

7. Cuando el VAD local detecta fin de turno (silencio >= `END_SILENCE_FRAMES`), agrupa la utterance y la envia al STT batch de Deepgram via HTTP.

8. El STT batch devuelve el transcript final autoritativo con deteccion de idioma.

9. El turn manager evalua si el transcript final debe diferirse (ventana de gracia) o procesarse inmediatamente.

10. Si existe un prefetch de respuesta LLM compatible, se reutiliza directamente; de lo contrario se genera una nueva respuesta con streaming.

11. La respuesta textual se divide en segmentos aptos para TTS.

12. Cada segmento se sintetiza con Deepgram TTS en `mulaw` a `8000 Hz`, seleccionando la voz segun el idioma detectado.

13. El backend envia el audio a Twilio mediante eventos `media` y sincroniza el fin de segmento con eventos `mark`.

14. Si el usuario interrumpe y el barge-in esta habilitado, el turno actual puede invalidarse y limpiarse.

  

## Capas o modulos principales

  

| Capa | Implementacion principal | Descripcion |

| --- | --- | --- |

| Entrada HTTP | `STT_Server.voice()` | Inicializa la llamada de voz |

| Entrada WebSocket | `STT_Server.media_stream()` | Orquesta el ciclo de vida del stream |

| Audio/VAD | `audio_ingest.handle_incoming_media()` | Convierte, segmenta y detecta utterances |

| STT Realtime | `deepgram_stt_realtime.run_realtime_stt()` | Transcripcion en tiempo real via WebSocket |

| STT Batch | `deepgram_stt_batch.transcribe_block()` | Transcripcion autoritativa via HTTP |

| Turn Manager | `turn_manager.process_transcripts()` | Coordina prefetch, gracia y pipeline de respuesta |

| Batch Worker | `turn_manager.process_local_utterances()` | Procesa utterances del VAD local con STT batch |

| LLM | `openai_llm.call_llm()` / `stream_llm_reply_sync()` | Genera respuesta con streaming |

| TTS | `deepgram_tts.stream_tts_segment()` | Solicita sintesis a Deepgram |

| Playback | `playback_service.playback_loop()` | Emite audio y marcas a Twilio |

| Sesion | `session.CallSession` | Aisla el estado de una llamada |

| Idioma | `language.detect_language()` / `looks_like_incomplete_utterance()` / `is_non_actionable_utterance()` | Deteccion de idioma, analisis linguistico y filtrado de utterances |

| Runtime de sesion | `session_runtime.register_session()` / `cleanup_session()` | Registro, limpieza y monitoreo de inactividad |

| Utilidades | `common.require_debug_endpoints()` | Proteccion de endpoints y operaciones de cola |

  

## Relacion con frontend y base de datos

  

No existe frontend ni base de datos. Toda la comunicacion del sistema ocurre entre servicios externos y el backend:

  

- Twilio consume los endpoints de voz y WebSocket;

- Deepgram procesa STT y TTS;

- OpenAI responde al contexto conversacional;

- Railway o Azure App Service hospeda la aplicacion.

  

# 4. Estructura del proyecto

  

## Organizacion de carpetas

  

```text

.

├── main.py

├── railway.toml

├── requirements.txt

├── CAMBIOS_2026-03-23.md

├── Documentacion-Server.md

├── ConvertLocalSTT/

│   ├── ConvertText.py

│   ├── ConvertText.md

│   ├── RealTimeTranscription.py

│   └── RealTimeTranscription.md

└── STT_server/

    ├── __init__.py

    ├── config.py

    ├── STT_Server.py

    ├── entornoLocal.env

    ├── adapters/

    │   ├── deepgram_stt_batch.py

    │   ├── deepgram_stt_realtime.py

    │   ├── deepgram_tts.py

    │   ├── openai_llm.py

    │   └── twilio_media.py

    ├── domain/

    │   ├── language.py

    │   └── session.py

    └── services/

        ├── audio_ingest.py

        ├── common.py

        ├── playback_service.py

        ├── session_runtime.py

        └── turn_manager.py

```

  

## Responsabilidad de cada carpeta o modulo

  

| Ruta | Responsabilidad |

| --- | --- |

| `main.py` | Punto de entrada ASGI para Uvicorn y Railway/Azure |

| `railway.toml` | Configuracion de build y arranque en Railway |

| `requirements.txt` | Dependencias del proyecto |

| `STT_server/config.py` | Constantes y parametros configurables del sistema |

| `STT_server/STT_Server.py` | Punto de entrada FastAPI con definicion de endpoints |

| `STT_server/entornoLocal.env` | Configuracion local de desarrollo |

| `STT_server/adapters/` | Integraciones con servicios externos (Deepgram, OpenAI, Twilio) |

| `STT_server/domain/` | Modelo de datos (`CallSession`) y logica de idioma |

| `STT_server/services/` | Orquestacion de audio, reproduccion, turnos y sesiones |

| `ConvertLocalSTT/` | Scripts auxiliares o heredados de transcripcion local |

  

## Archivos principales

  

| Archivo | Descripcion |

| --- | --- |

| `main.py` | Importa y expone la instancia `app` de FastAPI |

| `STT_server/config.py` | Centraliza todas las constantes y parametros configurables |

| `STT_server/STT_Server.py` | Define los endpoints HTTP y WebSocket del servidor |

| `STT_server/domain/session.py` | Define la clase `CallSession` con todo el estado por llamada |

| `STT_server/domain/language.py` | Deteccion de idioma, marcadores linguisticos y segmentacion TTS |

| `STT_server/services/turn_manager.py` | Orquesta transcripciones, prefetch, ventana de gracia y pipeline de respuesta |

| `STT_server/services/session_runtime.py` | Registro de sesiones, limpieza y monitoreo de inactividad |

| `railway.toml` | Define el comando de arranque para despliegue |

| `requirements.txt` | Lista dependencias Python |

| `Documentacion-Server.md` | Documentacion tecnica del repositorio |

| `CAMBIOS_2026-03-23.md` | Registro de cambios del 23 de marzo de 2026 |

  

## Convenciones de nombres

  

- Las constantes globales se expresan en mayusculas y se definen en `config.py`.

- Las rutas HTTP y WebSocket usan nombres breves y semanticos.

- La sesion por llamada se modela con la clase `CallSession` en `domain/session.py`.

- Los adaptadores de servicios externos se agrupan en `adapters/`.

- La logica de negocio se agrupa en `services/`.

- El dominio (modelo de datos y reglas linguisticas) se agrupa en `domain/`.

  

# 5. Modelo de datos

  

## Enfoque general

  

El sistema no dispone de un modelo relacional ni de persistencia. El unico modelo estructurado relevante es el estado en memoria por llamada.

  

## Entidad principal: `CallSession`

  

| Campo | Tipo | Descripcion |

| --- | --- | --- |

| `session_key` | `str` | Identificador interno de la sesion |

| `call_sid` | `str | None` | Identificador de llamada de Twilio |

| `stream_sid` | `str | None` | Identificador del stream activo |

| `preferred_language` | `str | None` | Idioma preferente de la conversacion |

| `vad_buffer` | `bytearray` | Buffer incremental de audio PCM |

| `pre_speech_frames` | `deque[bytes]` | Frames previos a la deteccion de voz |

| `speech_frames` | `list[bytes]` | Frames de la utterance activa |

| `speech_frame_count` | `int` | Conteo de frames validos de voz |

| `voice_streak` | `int` | Racha de frames con voz |

| `silence_frames` | `int` | Conteo de frames en silencio |

| `active_generation` | `int` | Version logica del turno actual |

| `history` | `list[dict[str, str]]` | Historial reciente de la conversacion |

| `utterance_queue` | `asyncio.Queue` | Cola de utterances completas para STT batch |

| `playback_queue` | `asyncio.Queue` | Cola de audio/eventos de salida |

| `stt_audio_queue` | `asyncio.Queue` | Cola de chunks de audio crudo para STT realtime |

| `transcript_queue` | `asyncio.Queue` | Cola de eventos de transcripcion |

| `tasks` | `set[asyncio.Task]` | Tareas asociadas a la sesion |

| `pending_marks` | `set[str]` | Marcas pendientes de confirmacion |

| `mark_counter` | `int` | Contador incremental de marcas |

| `assistant_speaking` | `bool` | Indica si el asistente esta reproduciendo audio |

| `assistant_started_at` | `float | None` | Timestamp del inicio de la reproduccion |

| `current_transcript` | `str` | Transcripcion parcial o final en curso |

| `reply_source_text` | `str` | Texto fuente del pipeline de respuesta activo |

| `reply_task` | `asyncio.Task | None` | Tarea del pipeline de respuesta activo |

| `partial_reply_task` | `asyncio.Task | None` | Tarea de debounce para prefetch desde parciales |

| `prefetched_reply_source_text` | `str` | Texto fuente del prefetch LLM |

| `prefetched_reply_text` | `str` | Respuesta prefetched lista para reutilizar |

| `prefetched_reply_task` | `asyncio.Task | None` | Tarea del prefetch LLM en curso |

| `awaiting_local_final` | `bool` | Indica que el VAD local detecto fin de utterance y espera resultado batch |

| `pending_realtime_final` | `dict | None` | Final realtime almacenado como fallback mientras se espera batch |

| `deferred_final_text` | `str` | Texto final diferido en ventana de gracia |

| `deferred_final_language` | `str | None` | Idioma del final diferido |

| `deferred_final_flush_task` | `asyncio.Task | None` | Timer de la ventana de gracia |

| `stt_failure_announced` | `bool` | Indica si ya se anuncio fallo de STT al usuario |

| `closed` | `bool` | Indica si la sesion fue cerrada |

| `last_activity_at` | `float` | Timestamp monotonic de la ultima actividad; usado para monitoreo de inactividad |

  

## Relaciones

  

- una llamada activa corresponde a una instancia de `CallSession`;

- una sesion puede contener multiples turnos de usuario y asistente;

- un turno puede producir varios segmentos TTS;

- cada sesion mantiene su propio historial, colas y estado de reproduccion.

  

## Reglas de negocio

  

| Regla | Descripcion |

| --- | --- |

| Aislamiento por llamada | Cada llamada mantiene estado propio en memoria |

| Historial acotado | Solo se conserva una ventana limitada de mensajes |

| Idioma soportado | El sistema opera en ingles y espanol |

| Respuesta breve | El prompt impone respuestas cortas y aptas para telefonia |

| Turno invalidable | Un turno previo puede cancelarse mediante `active_generation` |

  

## Restricciones y validaciones

  

| Restriccion | Descripcion |

| --- | --- |

| Umbral minimo de utterance | El audio debe superar duracion y numero minimo de frames |

| Formato de salida | El TTS se fuerza a `mulaw` a `8000 Hz` |

| Idiomas normalizados | Solo se aceptan `en` y `es` como idiomas operativos |

| Estado efimero | Al reiniciar el proceso se pierde todo el estado de llamadas |

  

## Enumeraciones o catalogos

  

| Elemento | Valores |

| --- | --- |

| `SUPPORTED_LANGUAGES` | `en`, `es` |

| Eventos de Twilio procesados | `connected`, `start`, `media`, `mark`, `dtmf`, `stop` |

| Eventos internos de playback | `audio`, `mark`, `clear`, `segment_end`, `error` |

| `NON_ACTIONABLE_PHRASES` | Conjunto de ~60 frases (saludos, fillers, confirmaciones) que se filtran silenciosamente |

  

# 6. Funcionalidades del sistema

  

## Modulos principales

  

### 6.1 Recepcion de llamadas

  

El endpoint `/voice` responde con TwiML para que Twilio conecte un Media Stream hacia el servidor.

  

### 6.2 Procesamiento de audio entrante

  

El endpoint `/media-stream` recibe audio en base64, lo convierte a PCM lineal y lo analiza frame a frame para identificar voz y silencio.

  

### 6.3 Segmentacion de turnos

  

El sistema usa `webrtcvad` y umbrales configurables para detectar:

  

- inicio de habla;

- continuidad de voz;

- fin de utterance por silencio;

- opcion de barge-in durante reproduccion del asistente.

  

### 6.4 Transcripcion STT

  

El sistema emplea una arquitectura de STT dual:

  

- **STT Realtime**: el audio crudo se envia continuamente a Deepgram via WebSocket. Se obtienen transcripciones parciales e intermedias que permiten al sistema anticipar la respuesta del asistente. La conexion utiliza un sistema de candidatos con fallback automatico ante rechazos HTTP 400.

- **STT Batch**: cuando el VAD local detecta fin de utterance, el bloque de audio completo se envia a Deepgram via HTTP. Esta transcripcion es autoritativa y soporta deteccion de idioma, lo cual es esencial para el soporte bilingue.

  

El turn manager coordina ambas fuentes: si el batch STT tarda, el final del realtime se almacena como fallback.

  

### 6.5 Gestion de turnos y ventana de gracia

  

El turn manager implementa los siguientes mecanismos:

  

- **Prefetch silencioso**: las transcripciones parciales disparan una consulta LLM anticipada. Si la transcripcion final coincide con el parcial, se reutiliza la respuesta sin espera adicional.

- **Ventana de gracia**: cuando una transcripcion final es corta (hasta `SHORT_FINAL_MAX_WORDS` palabras) o termina en un marcador linguistico incompleto (como "and", "because", "para"), el sistema difiere su procesamiento durante `FINAL_TRANSCRIPT_GRACE_MS` milisegundos. Si llega un nuevo final en ese periodo, ambos se combinan. Si no, se procesa el texto acumulado.

- **Deteccion linguistica de incompletitud**: un conjunto de marcadores en ingles y espanol permite identificar utterances que probablemente no han terminado.

- **Filtrado de utterances no accionables**: un conjunto de mas de 60 frases (`NON_ACTIONABLE_PHRASES`) permite identificar y descartar saludos, fillers, confirmaciones y pronombres aislados que no requieren respuesta del asistente.

  

### 6.6 Generacion conversacional

  

El texto transcrito se combina con:

  

- un prompt del sistema;

- una instruccion explicita de idioma;

- el historial reciente de la sesion.

  

Con ello se genera una respuesta breve mediante OpenAI.

  

### 6.7 Sintesis TTS y playback

  

La respuesta textual se divide en segmentos y cada uno se sintetiza con Deepgram TTS. El sistema selecciona automaticamente la voz segun el idioma detectado: `aura-2-thalia-en` para ingles y `aura-2-estrella-es` para espanol. El audio resultante se fragmenta y se envia a Twilio con eventos `media`. Al final de cada segmento se emite un `mark` para sincronizar el estado del playback.

  

Antes de la primera respuesta real, el sistema puede emitir un filler de espera configurable por idioma ("Okay, one moment." / "Claro, un momento.") si la generacion LLM tarda mas de `FILLER_DELAY_MS` milisegundos.

### 6.8 Monitoreo de inactividad

El modulo `session_runtime` implementa un monitor de silencio por sesion (`monitor_idle_silence`). Si no se detecta actividad durante `IDLE_SILENCE_TIMEOUT_SEC` segundos (por defecto 45), la llamada se cierra automaticamente para liberar recursos.

### 6.9 Proteccion de endpoints de diagnostico

Los endpoints `/test-llm-tts`, `/test-stt` y `/list-models` estan protegidos por la variable `ENABLE_DEBUG_ENDPOINTS`. Si esta en `false` (valor por defecto), devuelven HTTP 404. Esto evita la exposicion de funcionalidad de diagnostico en produccion.

### 6.10 Reconexion automatica de STT realtime

El adaptador de STT realtime implementa un mecanismo de reconexion exponencial con hasta `STT_RECONNECT_MAX_ATTEMPTS` intentos. Si la reconexion falla definitivamente, el sistema anuncia al usuario un mensaje de fallo de STT configurable por idioma.

  

## Flujo de uso por modulo

  

| Modulo | Flujo principal |

| --- | --- |

| Entrada de voz | Twilio llama a `/voice` y recibe TwiML |

| WebSocket | Twilio abre `/media-stream` y envia eventos |

| VAD | El backend detecta una utterance completa |

| STT | Se solicita transcripcion a Deepgram |

| LLM | Se construye contexto y se genera respuesta |

| TTS | Se sintetiza audio y se reenvia a Twilio |

  

## Comportamiento esperado

  

- las respuestas deben ser breves y naturales;

- el idioma de respuesta debe coincidir con el idioma del usuario;

- el sistema debe mantener continuidad de contexto en una misma llamada;

- la reproduccion del asistente debe poder detenerse si el turno deja de ser valido.

  

## Validaciones importantes

  

- se ignoran utterances demasiado cortas;

- se filtran utterances no accionables (saludos, fillers, pronombres aislados);

- se descartan resultados de turnos cuya `generation` ya no sea la activa;

- el servidor no arranca sin `PUBLIC_URL`;

- si faltan credenciales externas, el sistema registra advertencias y ciertas funciones dejan de estar disponibles.

  

## Casos de uso relevantes

  

1. Atencion automatizada de llamadas entrantes.

2. Recepcion de consultas breves en ingles o espanol.

3. Pruebas de pipeline STT, LLM y TTS mediante endpoints diagnosticos.

  

# 7. API o comunicacion entre modulos

  

## Endpoints HTTP y WebSocket

  

| Metodo | Ruta | Proposito |

| --- | --- | --- |

| `POST` | `/voice` | Devuelve TwiML para abrir el stream de voz |

| `WS` | `/media-stream` | Gestiona audio y eventos de Twilio |

| `GET` | `/test-llm-tts` | Prueba la generacion textual y segmentacion TTS (requiere `ENABLE_DEBUG_ENDPOINTS=true`) |

| `POST` | `/test-stt` | Prueba la transcripcion STT (requiere `ENABLE_DEBUG_ENDPOINTS=true`) |

| `GET` | `/list-models` | Lista modelos accesibles desde OpenAI (requiere `ENABLE_DEBUG_ENDPOINTS=true`) |

| `GET` | `/` | Health check basico |

  

## Detalle de endpoints

  

### `POST /voice`

  

- Entrada: no requiere body especifico para devolver el TwiML base.

- Salida: XML TwiML con un `<Connect><Stream>` hacia el WebSocket del servicio.

- Respuesta esperada: `200 OK` con `application/xml`.

  

### `WS /media-stream`

  

- Entrada: eventos de Twilio en formato JSON.

- Salida: eventos JSON hacia Twilio como `media`, `mark` y `clear`.

- Responsabilidad: coordinar toda la sesion de audio de una llamada.

  

Ejemplo simplificado de evento `media` entrante:

  

```json

{

  "event": "media",

  "streamSid": "MZXXXXXXXXXXXXXXXXXXXXXXXX",

  "media": {

    "payload": "<audio-base64>"

  }

}

```

  

### `GET /test-llm-tts`

  

- Query param: `q` obligatorio.

- Proposito: validar respuesta textual del LLM y cantidad de segmentos TTS.

- Respuesta: JSON con `input`, `reply`, `tts_segments` y `tts_ready`.

  

### `POST /test-stt`

  

- Entrada: no especificada; usa audio dummy interno.

- Proposito: validar disponibilidad del flujo STT.

- Respuesta: JSON con `text`, `segments`, `language`, `stt_ready` y `model`.

  

### `GET /list-models`

  

- Entrada: no requiere parametros.

- Proposito: listar modelos visibles mediante la API de OpenAI.

- Respuesta: JSON con `models` o `error`.

  

### `GET /`

  

- Entrada: no requiere parametros.

- Proposito: verificar disponibilidad del servicio.

- Respuesta: JSON con estado basico.

  

## Formato estandar de respuestas

  

No existe un contrato de respuesta unificado para todos los endpoints. El sistema usa:

  

- XML para `/voice`;

- JSON para endpoints de prueba y diagnostico;

- mensajes JSON de eventos para el WebSocket de Twilio.

  

## Manejo de errores

  

| Area | Estrategia |

| --- | --- |

| STT | Captura de `HTTPError`, `URLError` y timeouts |

| LLM | Captura generica con mensaje de fallback |

| TTS | Captura de errores HTTP/red y envio de evento interno `error` |

| WebSocket | Captura de `WebSocketDisconnect` y limpieza de sesion |

  

## Codigos de estado o respuestas relevantes

  

| Contexto | Respuesta esperada |

| --- | --- |

| `/` | `200 OK` |

| `/voice` | `200 OK` |

| `/test-llm-tts` | `200 OK` o error controlado si falta OpenAI |

| `/test-stt` | `200 OK` o fallo si falta Deepgram |

| `/list-models` | `200 OK` con `error` en payload si falla la consulta |

  

# 8. Autenticacion y seguridad

  

## Metodo de autenticacion

  

No existe autenticacion de usuarios en el backend actual.

  

## Autorizacion por roles o permisos

  

No existe un esquema de roles o permisos.

  

## Proteccion de rutas

  

Las rutas expuestas no tienen proteccion aplicativa. La seguridad depende principalmente del control de acceso a las credenciales y de la exposicion publica del servicio.

  

## Manejo de sesiones o tokens

  

- Twilio mantiene la sesion de llamada y stream.

- OpenAI y Deepgram se autentican mediante API keys en variables de entorno.

- El backend no genera tokens propios ni maneja sesiones autenticadas de usuario.

  

## Buenas practicas implementadas

  

- uso de variables de entorno para credenciales;

- aislamiento de estado por llamada;

- cierre y cancelacion de tareas al finalizar una sesion;

- restriccion del formato TTS para compatibilidad con Twilio.

  

## Riesgos o pendientes de seguridad

  

| Riesgo | Estado |

| --- | --- |

| Credenciales en archivos locales | Riesgo alto; deben mantenerse fuera de Git y rotarse si fueron expuestas |

| Validacion de origen de Twilio | No especificado en el codigo actual |

| Proteccion de endpoints diagnosticos | No implementada |

| Persistencia segura de logs | No especificada |

| Rate limiting | No implementado |

  

# 9. Variables de entorno y configuracion

  

## Variables requeridas

  

| Variable | Obligatoria | Proposito | Ejemplo seguro |

| --- | --- | --- | --- |

| `PUBLIC_URL` | Si | URL publica del servicio para construir el WebSocket Twilio | `https://mi-servicio.up.railway.app` |

| `OPENAI_API_KEY` | Si | Credencial para llamadas a OpenAI | `sk-xxxxxxxxxxxxxxxx` |

| `DEEPGRAM_API_KEY` | Si | Credencial para STT y TTS en Deepgram | `dg_XXXXXXXXXXXXXXXX` |

  

## Variables recomendadas

  

| Variable | Default actual | Proposito |

| --- | --- | --- |

| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo conversacional |

| `DEEPGRAM_STT_MODEL` | `nova-3` | Modelo STT |

| `DEEPGRAM_STT_PUNCTUATE` | `true` | Inserta puntuacion en transcript |

| `DEEPGRAM_STT_SMART_FORMAT` | `true` | Mejora el formato del transcript |

| `DEEPGRAM_STT_DETECT_LANGUAGE` | `true` | Habilita deteccion de idioma |

| `DEEPGRAM_STT_LANGUAGE_HINT` | no especificado | Fuerza o sugiere idioma de entrada |

| `DEEPGRAM_TTS_MODEL` | `aura-2-thalia-en` | Voz TTS principal en ingles |

| `DEEPGRAM_TTS_ENCODING` | `mulaw` | Encoding del audio saliente |

| `DEEPGRAM_TTS_SAMPLE_RATE` | `8000` | Sample rate del TTS |

| `DEFAULT_CALL_LANGUAGE` | `en` | Idioma fallback de la sesion |

| `INITIAL_GREETING_ENABLED` | `true` | Activa el saludo inicial |

| `INITIAL_GREETING_TEXT` | texto por defecto | Personaliza el saludo inicial |

  

## Variables operativas de audio y tiempo

  

| Variable | Default actual | Proposito |

| --- | --- | --- |

| `MIN_UTTERANCE_MS` | `180` | Duracion minima aceptada para una utterance |

| `MIN_SPEECH_FRAMES` | `5` | Frames minimos de voz |

| `END_SILENCE_FRAMES` | `18` | Frames de silencio para cierre de turno (540 ms) |

| `SPEECH_START_FRAMES` | `2` | Frames requeridos para detectar inicio |

| `MIN_BARGE_IN_FRAMES` | `12` | Frames minimos para interrupcion |

| `PRE_SPEECH_FRAMES` | `5` | Frames previos retenidos |

| `TRIM_TRAILING_SILENCE_FRAMES` | `6` | Recorte final de silencio |

| `MIN_VOICE_RMS` | `260` | Umbral RMS minimo de voz |

| `BARGE_IN_MIN_RMS` | `900` | Umbral RMS minimo para barge-in |

| `ENABLE_BARGE_IN` | `true` | Activa interrupcion del asistente |

| `ASSISTANT_ECHO_IGNORE_MS` | `1200` | Ventana para ignorar eco inicial |

| `LOG_TWILIO_PLAYBACK` | `false` | Activa logs de playback |

| `TWILIO_OUTBOUND_PACING_MS` | `20` | Pausa entre frames salientes |

| `STT_TIMEOUT_SEC` | `0` | Timeout de STT; `0` implica timeout interno del cliente |

| `LLM_TIMEOUT_SEC` | `5.0` | Timeout maximo de la llamada al LLM |

| `TTS_TIMEOUT_SEC` | `5.0` | Timeout maximo por segmento TTS |

| `MAX_HISTORY_MESSAGES` | `12` | Longitud maxima del historial |

| `MAX_RESPONSE_TOKENS` | `150` | Limite de tokens de respuesta |

| `FILLER_DELAY_MS` | `1200` | Milisegundos antes de emitir el filler de espera |

| `FINAL_TRANSCRIPT_GRACE_MS` | `2000` | Ventana de gracia para diferir finals cortos/incompletos |

| `SHORT_FINAL_MAX_WORDS` | `12` | Umbral de palabras para considerar un final como corto |

| `PARTIAL_TRANSCRIPT_START_CHARS` | `40` | Caracteres minimos para iniciar prefetch desde parciales |

| `PARTIAL_TRANSCRIPT_DEBOUNCE_MS` | `600` | Debounce antes de lanzar prefetch |

| `DEEPGRAM_STT_ENDPOINTING_MS` | `700` | Endpointing de Deepgram realtime en milisegundos |

| `DEEPGRAM_UTTERANCE_END_MS` | `700` | Utterance end timeout de Deepgram |

| `FILLER_TTS_ENABLED` | `true` | Habilita la emision de filler de espera TTS |

| `ENABLE_DEBUG_ENDPOINTS` | `false` | Habilita los endpoints de diagnostico (`/test-llm-tts`, `/test-stt`, `/list-models`) |

| `STT_AUDIO_QUEUE_MAXSIZE` | `100` | Tamano maximo de la cola de audio para STT realtime |

| `TRANSCRIPT_QUEUE_MAXSIZE` | `32` | Tamano maximo de la cola de transcripciones |

| `PLAYBACK_QUEUE_MAXSIZE` | `256` | Tamano maximo de la cola de playback |

| `TEXT_SEGMENT_QUEUE_MAXSIZE` | `16` | Tamano maximo de la cola de segmentos de texto |

| `STREAMING_SEGMENT_MAX_CHARS` | `120` | Caracteres maximos por segmento de streaming TTS |

| `FINAL_RESTART_DELTA_CHARS` | `12` | Delta de caracteres para reiniciar pipeline desde un final nuevo |

| `PARTIAL_PREFETCH_MAX_DELTA_CHARS` | `20` | Delta maximo de caracteres para reutilizar prefetch parcial |

| `STT_RECONNECT_MAX_ATTEMPTS` | `3` | Intentos maximos de reconexion STT realtime |

| `STT_RECONNECT_BASE_DELAY_MS` | `250` | Retardo base para reconexion exponencial STT |

| `STT_RECONNECT_MAX_DELAY_MS` | `2000` | Retardo maximo para reconexion exponencial STT |

| `STT_FAILURE_PROMPT_EN` | `"I'm having trouble hearing you right now."` | Mensaje al usuario cuando falla STT (ingles) |

| `STT_FAILURE_PROMPT_ES` | `"Estoy teniendo problemas para escucharte en este momento."` | Mensaje al usuario cuando falla STT (espanol) |

| `IDLE_SILENCE_TIMEOUT_SEC` | `45` | Segundos de inactividad antes de cerrar la llamada |

  

## Configuracion por ambiente

  

### Desarrollo local

  

- uso de `STT_server/entornoLocal.env`;

- ejecucion local con Uvicorn;

- configuracion manual de llaves y URL publica de pruebas.

  

### Produccion en Railway

  

- configuracion mediante variables del servicio en Railway;

- `PORT` es provisto por Railway y no debe fijarse manualmente;

- `PUBLIC_URL` debe apuntar al dominio publico real del servicio.

  

## Ejemplo seguro de configuracion

  

```dotenv

PUBLIC_URL=https://mi-servicio.up.railway.app

OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

OPENAI_MODEL=gpt-4o-mini

DEEPGRAM_API_KEY=dg_xxxxxxxxxxxxxxxx

DEEPGRAM_STT_MODEL=nova-3

DEEPGRAM_STT_DETECT_LANGUAGE=true

DEEPGRAM_STT_PUNCTUATE=true

DEEPGRAM_STT_SMART_FORMAT=true

DEEPGRAM_TTS_MODEL=aura-2-thalia-en

DEEPGRAM_TTS_ENCODING=mulaw

DEEPGRAM_TTS_SAMPLE_RATE=8000

DEFAULT_CALL_LANGUAGE=en

INITIAL_GREETING_ENABLED=true

INITIAL_GREETING_TEXT=Thank you for calling Cialix Customer Support. My name is Tessa. How can I help you today?

LLM_TIMEOUT_SEC=5.0

TTS_TIMEOUT_SEC=5.0

WEB_CONCURRENCY=1

```

  

# 10. Despliegue y ejecucion

  

## Requisitos para correr el proyecto

  

- Python 3.13 o superior recomendado;

- acceso a credenciales validas de OpenAI y Deepgram;

- una cuenta Twilio con capacidad de Media Streams;

- dominio publico HTTPS accesible desde Twilio;

- entorno virtual Python para desarrollo local.

  

## Instalacion local

  

1. Crear y activar un entorno virtual.

2. Instalar dependencias desde `requirements.txt`.

3. Configurar variables de entorno locales.

4. Verificar que `PUBLIC_URL` apunte a una URL publica valida si se va a probar con Twilio.

  

Ejemplo:

  

```bash

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8080

```

  

## Ejecucion en desarrollo

  

La aplicacion se inicia mediante:

  

```bash

uvicorn main:app --host 0.0.0.0 --port 8080

```

  

## Despliegue en Railway

  

El archivo `railway.toml` define el arranque del servicio con:

  

```toml

[build]

builder = "RAILPACK"

[deploy]

startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT --workers ${WEB_CONCURRENCY:-1}"

```

  

Pasos recomendados:

  

1. Conectar el repositorio a Railway.

2. Configurar variables de entorno obligatorias y recomendadas.

3. Habilitar dominio publico del servicio.

4. Establecer `PUBLIC_URL` con el dominio real asignado por Railway.

5. Desplegar y validar el endpoint `/`.

6. Configurar en Twilio el webhook de voz hacia `POST /voice`.

  

## Despliegue en Azure (App Service con Git) — metodo principal

  

El proyecto se despliega a Azure App Service mediante `git push`:

  

1. Crear una Azure Web App con runtime Python 3.13.

2. Configurar el deployment center con Git local.

3. Agregar el remote de Azure: `git remote add Azure <url>`.

4. Configurar variables de entorno en la Web App.

5. Desplegar con `git push Azure main`.

6. Verificar el endpoint de health check `/`.

  

## Consideraciones para produccion

  

- usar un solo worker inicial si se quiere priorizar estabilidad de sesiones WebSocket;

- no versionar archivos con credenciales reales;

- monitorear tiempos de respuesta de STT, LLM y TTS;

- revisar limites de costo y cuota de OpenAI, Deepgram y Twilio;

- considerar limpieza de dependencias heredadas antes de produccion.

  

# 11. Pruebas y mantenimiento

  

## Pruebas existentes o recomendadas

  

No se identifican pruebas automatizadas versionadas en el repositorio.

  

Se recomiendan al menos las siguientes validaciones:

  

- prueba de health check en `/`;

- prueba de conectividad con OpenAI mediante `/list-models`;

- prueba STT mediante `/test-stt`;

- prueba de respuesta textual con `/test-llm-tts?q=...`;

- prueba end-to-end con una llamada real de Twilio.

  

## Estrategia de validacion

  

| Nivel | Validacion sugerida |

| --- | --- |

| Unitario | Normalizacion de idioma, segmentacion de texto y parsing de respuestas |

| Integracion | Deepgram STT, Deepgram TTS, OpenAI y WebSocket de Twilio |

| End-to-end | Flujo completo de llamada telefonica |

  

## Buenas practicas de mantenimiento

  

- mantener separadas credenciales de configuracion local y de produccion;

- revisar cambios de API en OpenAI y Deepgram;

- documentar cualquier cambio de modelo o parametros de audio;

- controlar el crecimiento de los modulos para evitar complejidad excesiva;

- eliminar dependencias heredadas cuando dejen de ser necesarias.

  

## Puntos criticos a revisar

  

- estabilidad del WebSocket con Twilio;

- cancelacion correcta de tareas al cerrar una llamada;

- consistencia del idioma detectado y del idioma de respuesta;

- latencia total por turno;

- manejo de errores de red con servicios externos.

  

# 12. Estado actual y pendientes

  

## Funcionalidades implementadas

  

- recepcion de llamadas con Twilio Media Streams;

- VAD local con webrtcvad y segmentacion por turnos;

- STT dual: realtime (WebSocket) para parciales y prefetch, batch (HTTP) para finals autoritativos con deteccion de idioma;

- sistema de candidatos con fallback automatico para conexiones STT;

- prefetch silencioso de respuesta LLM desde transcripciones parciales;

- ventana de gracia para diferir finals cortos o linguisticamente incompletos;

- deteccion linguistica de utterances incompletas (marcadores en ingles y espanol);

- LLM con OpenAI GPT-4o-mini con streaming;

- TTS con Deepgram, seleccion automatica de voz por idioma;

- filler de espera configurable por idioma;

- barge-in (interrupcion del asistente por el usuario);

- historial corto por sesion (hasta 12 mensajes);

- saludo inicial configurable;

- endpoints de diagnostico y health check;

- proteccion de endpoints de diagnostico mediante `ENABLE_DEBUG_ENDPOINTS`;

- filtrado de utterances no accionables (saludos, fillers, pronombres aislados);

- monitoreo de inactividad con cierre automatico de llamada tras `IDLE_SILENCE_TIMEOUT_SEC`;

- reconexion automatica de STT realtime con backoff exponencial;

- anuncio de fallo de STT al usuario en su idioma;

- arquitectura modular: adapters, domain, services.

  

## Funcionalidades faltantes o no especificadas

  

- autenticacion y proteccion de endpoints;

- persistencia de conversaciones;

- trazabilidad centralizada de llamadas;

- pruebas automatizadas;

- panel de monitoreo u operacion.

  

## Mejoras futuras sugeridas

  

1. implementar pruebas unitarias e integracion;

2. agregar validacion de origen o firma para Twilio;

3. incorporar observabilidad estructurada;

4. persistir metricas y eventos de llamada;

5. limpiar scripts y dependencias heredadas de STT local si ya no forman parte del roadmap;

6. agregar persistencia de conversaciones en base de datos.

  

## Limitaciones conocidas

  

- el estado se pierde ante reinicios del proceso;

- el sistema depende de varios servicios externos para operar;

- el proyecto no incorpora base de datos;

- no existe proteccion aplicativa sobre endpoints de prueba;

- el repositorio contiene artefactos heredados que no representan necesariamente el flujo principal vigente.

  

# 13. Glosario

  

| Termino | Definicion |

| --- | --- |

| ASGI | Interfaz estandar para aplicaciones Python asincronas |

| Barge-in | Interrupcion de la respuesta del asistente cuando el usuario comienza a hablar |

| Deepgram STT | Servicio de reconocimiento de voz a texto |

| Deepgram TTS | Servicio de sintesis de texto a voz |

| Frame | Unidad corta de audio procesada por el VAD |

| LLM | Modelo de lenguaje usado para generar respuestas |

| Media Stream | Canal bidireccional de audio y eventos de Twilio sobre WebSocket |

| PCM | Representacion lineal sin compresion del audio |

| TTS | Conversion de texto a voz |

| TwiML | XML usado por Twilio para describir el comportamiento de una llamada |

| Utterance | Segmento de audio que representa un turno de habla del usuario |

| VAD | Voice Activity Detection, mecanismo para detectar voz y silencio |