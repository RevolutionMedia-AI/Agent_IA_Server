# 1. Resumen del proyecto

  

## Descripcion general

  

Este proyecto implementa un servidor de voz conversacional en tiempo real orientado a telefonia. La aplicacion expone endpoints HTTP y WebSocket mediante FastAPI para integrarse con Twilio Media Streams, transcribe el audio entrante con Deepgram, genera una respuesta breve mediante OpenAI y sintetiza la salida de voz nuevamente con Deepgram para devolverla a la llamada.

  

El sistema esta disenado para operar como backend de un agente telefonico sin interfaz grafica ni persistencia en base de datos. El estado de cada llamada se mantiene en memoria mientras la sesion permanece activa.

  

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

- VAD con `webrtcvad` para detectar inicio y fin de turno;

- transcripcion STT mediante Deepgram;

- generacion de respuesta mediante OpenAI;

- sintesis TTS mediante Deepgram;

- soporte bilingue limitado a ingles y espanol;

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

| Railway | Plataforma de despliegue |

  

## Herramientas de desarrollo

  

| Herramienta | Uso |

| --- | --- |

| Git | Control de versiones |

| venv | Entorno virtual Python local |

| Railway CLI o panel web | Despliegue y configuracion del servicio |

  

## Dependencias relevantes

  

| Dependencia | Rol en el proyecto |

| --- | --- |

| fastapi | Framework principal del backend |

| uvicorn[standard] | Ejecucion del servidor ASGI |

| python-dotenv | Carga de variables locales |

| webrtcvad | Deteccion de voz por frames |

| openai | Cliente del modelo conversacional |

| twilio | Dependencia declarada para integracion con el ecosistema Twilio |

| setuptools<81 | Restriccion de compatibilidad declarada |

  

Dependencias declaradas en el repositorio como `faster-whisper`, `numpy`, `librosa` y `soundfile` aparecen asociadas a scripts auxiliares o a configuracion heredada y no forman parte del flujo principal actual del servidor telefonico.

  

# 3. Arquitectura general

  

## Enfoque arquitectonico

  

La aplicacion sigue un enfoque de backend orientado a eventos y estado en memoria por sesion. Cada llamada activa se representa mediante una estructura `CallSession` que encapsula buffers, colas, historial y metadatos de reproduccion. El procesamiento del audio se divide en etapas desacopladas coordinadas con `asyncio`.

  

## Relacion entre componentes

  

El sistema se compone de los siguientes bloques funcionales:

  

| Componente | Responsabilidad |

| --- | --- |

| Endpoint `/voice` | Entregar TwiML para iniciar el Media Stream |

| Endpoint `/media-stream` | Gestionar eventos WebSocket de Twilio |

| VAD y endpointing | Detectar inicio y fin de una utterance |

| STT | Transcribir audio PCM con Deepgram |

| LLM | Generar respuesta textual breve |

| TTS | Sintetizar audio de salida con Deepgram |

| Playback | Reenviar audio, marcas y comandos de limpieza a Twilio |

| Gestion de sesion | Mantener estado por llamada |

  

## Estructura de comunicacion

  

### Flujo general

  

1. Twilio invoca `POST /voice`.

2. El backend responde con TwiML que indica abrir un stream WebSocket hacia `/media-stream`.

3. Twilio abre el WebSocket bidireccional y comienza a enviar eventos `connected`, `start` y `media`.

4. El servidor acumula audio en frames PCM lineales y aplica VAD.

5. Cuando detecta fin de turno, agrupa la utterance y la envia a Deepgram STT.

6. El texto transcrito se pasa a OpenAI junto con prompt del sistema e historial reciente.

7. La respuesta textual se divide en segmentos aptos para TTS.

8. Cada segmento se sintetiza con Deepgram TTS en `mulaw` a `8000 Hz`.

9. El backend envia el audio a Twilio mediante eventos `media` y sincroniza el fin de segmento con eventos `mark`.

10. Si el usuario interrumpe y el barge-in esta habilitado, el turno actual puede invalidarse y limpiarse.

  

## Capas o modulos principales

  

| Capa | Implementacion principal | Descripcion |

| --- | --- | --- |

| Entrada HTTP | `voice()` | Inicializa la llamada de voz |

| Entrada WebSocket | `media_stream()` | Orquesta el ciclo de vida del stream |

| Audio/VAD | `handle_incoming_media()` | Convierte y segmenta el audio entrante |

| STT | `transcribe_sync()` y `transcribe_block()` | Envia audio a Deepgram y extrae transcript |

| LLM | `call_llm()` | Consulta el modelo conversacional |

| TTS | `stream_tts_segment()` | Solicita sintesis a Deepgram |

| Playback | `playback_loop()` | Emite audio y marcas a Twilio |

| Sesion | `CallSession` | Aisla el estado de una llamada |

  

## Relacion con frontend y base de datos

  

No existe frontend ni base de datos. Toda la comunicacion del sistema ocurre entre servicios externos y el backend:

  

- Twilio consume los endpoints de voz y WebSocket;

- Deepgram procesa STT y TTS;

- OpenAI responde al contexto conversacional;

- Railway hospeda la aplicacion.

  

# 4. Estructura del proyecto

  

## Organizacion de carpetas

  

```text

.

├── main.py

├── railway.toml

├── requirements.txt

├── DOCUMENTACION_TECNICA.md

├── ConvertLocalSTT/

│   ├── ConvertText.py

│   ├── ConvertText.md

│   ├── RealTimeTranscription.py

│   └── RealTimeTranscription.md

└── STT_server/

    ├── __init__.py

    ├── STT_Server.py

    ├── entornoLocal.env

    └── static/

```

  

## Responsabilidad de cada carpeta o modulo

  

| Ruta | Responsabilidad |

| --- | --- |

| `main.py` | Punto de entrada ASGI para Uvicorn y Railway |

| `railway.toml` | Configuracion de build y arranque en Railway |

| `requirements.txt` | Dependencias del proyecto |

| `STT_server/STT_Server.py` | Implementacion principal del servidor |

| `STT_server/entornoLocal.env` | Configuracion local de desarrollo |

| `STT_server/static/` | Carpeta reservada; uso no especificado en el flujo actual |

| `ConvertLocalSTT/` | Scripts auxiliares o heredados de transcripcion local |

  

## Archivos principales

  

| Archivo | Descripcion |

| --- | --- |

| `main.py` | Importa y expone la instancia `app` |

| `STT_server/STT_Server.py` | Nucleo funcional del sistema |

| `railway.toml` | Define el comando de arranque para despliegue |

| `requirements.txt` | Lista dependencias Python |

| `DOCUMENTACION_TECNICA.md` | Documentacion tecnica del repositorio |

  

## Convenciones de nombres

  

- Las constantes globales se expresan en mayusculas.

- Las rutas HTTP y WebSocket usan nombres breves y semanticos.

- La sesion por llamada se modela con la clase `CallSession`.

- El archivo principal mantiene una estructura funcional centralizada; no hay separacion en paquetes por dominio.

  

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

| `utterance_queue` | `asyncio.Queue` | Cola de utterances pendientes |

| `playback_queue` | `asyncio.Queue` | Cola de audio/eventos de salida |

| `tasks` | `set[asyncio.Task]` | Tareas asociadas a la sesion |

| `pending_marks` | `set[str]` | Marcas pendientes de confirmacion |

| `mark_counter` | `int` | Contador incremental de marcas |

| `assistant_speaking` | `bool` | Indica si el asistente esta reproduciendo audio |

| `assistant_started_at` | `float | None` | Timestamp del inicio de la reproduccion |

| `closed` | `bool` | Indica si la sesion fue cerrada |

  

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

  

Cuando una utterance cierra, el backend envia el bloque de audio resultante a Deepgram STT y obtiene el transcript y el idioma detectado o inferido.

  

### 6.5 Generacion conversacional

  

El texto transcrito se combina con:

  

- un prompt del sistema;

- una instruccion explicita de idioma;

- el historial reciente de la sesion.

  

Con ello se genera una respuesta breve mediante OpenAI.

  

### 6.6 Sintesis TTS y playback

  

La respuesta textual se divide en segmentos y cada uno se sintetiza con Deepgram TTS. El audio resultante se fragmenta y se envia a Twilio con eventos `media`. Al final de cada segmento se emite un `mark` para sincronizar el estado del playback.

  

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

| `GET` | `/test-llm-tts` | Prueba la generacion textual y segmentacion TTS |

| `POST` | `/test-stt` | Prueba la transcripcion STT |

| `GET` | `/list-models` | Lista modelos accesibles desde OpenAI |

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

| `MIN_UTTERANCE_MS` | `240` | Duracion minima aceptada para una utterance |

| `MIN_SPEECH_FRAMES` | `5` | Frames minimos de voz |

| `END_SILENCE_FRAMES` | `6` | Frames de silencio para cierre de turno |

| `SPEECH_START_FRAMES` | `2` | Frames requeridos para detectar inicio |

| `MIN_BARGE_IN_FRAMES` | `6` | Frames minimos para interrupcion |

| `PRE_SPEECH_FRAMES` | `5` | Frames previos retenidos |

| `TRIM_TRAILING_SILENCE_FRAMES` | `4` | Recorte final de silencio |

| `MIN_VOICE_RMS` | `260` | Umbral RMS minimo de voz |

| `BARGE_IN_MIN_RMS` | `700` | Umbral RMS minimo para barge-in |

| `ENABLE_BARGE_IN` | `false` | Activa interrupcion del asistente |

| `ASSISTANT_ECHO_IGNORE_MS` | `1200` | Ventana para ignorar eco inicial |

| `LOG_TWILIO_PLAYBACK` | `true` | Activa logs de playback |

| `TWILIO_OUTBOUND_PACING_MS` | `20` | Pausa entre frames salientes |

| `STT_TIMEOUT_SEC` | `0` | Timeout de STT; `0` implica timeout interno del cliente |

| `LLM_TIMEOUT_SEC` | `5.0` | Timeout maximo de la llamada al LLM |

| `TTS_TIMEOUT_SEC` | `5.0` | Timeout maximo por segmento TTS |

| `MAX_HISTORY_MESSAGES` | `10` | Longitud maxima del historial |

| `MAX_RESPONSE_TOKENS` | `150` | Limite de tokens de respuesta |

  

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

INITIAL_GREETING_TEXT=Good day. My name is Athenas. Please tell me how I can help you today.

LLM_TIMEOUT_SEC=5.0

TTS_TIMEOUT_SEC=5.0

WEB_CONCURRENCY=1

```

  

# 10. Despliegue y ejecucion

  

## Requisitos para correr el proyecto

  

- Python 3.11 o superior recomendado;

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

- controlar el crecimiento del archivo principal `STT_Server.py` para evitar complejidad excesiva;

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

- VAD con segmentacion por turnos;

- STT con Deepgram;

- LLM con OpenAI;

- TTS con Deepgram;

- historial corto por sesion;

- saludo inicial configurable;

- endpoints de diagnostico y health check.

  

## Funcionalidades faltantes o no especificadas

  

- autenticacion y proteccion de endpoints;

- persistencia de conversaciones;

- trazabilidad centralizada de llamadas;

- pruebas automatizadas;

- panel de monitoreo u operacion.

  

## Mejoras futuras sugeridas

  

1. separar el archivo principal en modulos por dominio;

2. implementar pruebas unitarias e integracion;

3. agregar validacion de origen o firma para Twilio;

4. incorporar observabilidad estructurada;

5. persistir metricas y eventos de llamada;

6. limpiar scripts y dependencias heredadas de STT local si ya no forman parte del roadmap.

  

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