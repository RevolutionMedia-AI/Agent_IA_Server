# Conexión Frontend (Netlify) ↔ Backend (Railway)

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         NAVEGADOR                                │
│                    (Usuario final)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  NETLIFY (Frontend)                              RAILWAY (Backend)│
│  https://tu-frontend.netlify.app          https://agentiaserver..up.railway.app
│                                                         │
│  Variables de entorno:                                  │ Variables de entorno:
│  VITE_API_URL=https://...railway.app                     │ PUBLIC_URL=https://...railway.app
│                                                         │ OPENAI_API_KEY=sk-...
│                                                         │ DEEPGRAM_API_KEY=...
│                                                         │ ELEVENLABS_API_KEY=...
└─────────────────────────────────────────────────────────────────┘
```

## ⚠️ Problema: CORS

Cuando tu frontend está en Netlify y tu backend en Railway, el navegador bloquea las peticiones por CORS (Cross-Origin Resource Sharing). 

**Solución ya implementada:** El backend tiene CORS configurado para permitir todas las conexiones:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📋 Variables de Entorno

### Backend (Railway) - OBLIGATORIAS

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `PUBLIC_URL` | URL pública del servidor | `https://agentiaserver-production.up.railway.app` |
| `OPENAI_API_KEY` | API key de OpenAI | `sk-xxxxxxxxxxxx` |
| `DEEPGRAM_API_KEY` | API key de Deepgram (STT) | `xxxxxxxxxxxx` |
| `ELEVENLABS_API_KEY` | API key de ElevenLabs (TTS) | `xxxxxxxxxxxx` |

**Variables opcionales:**
| Variable | Default | Descripción |
|----------|---------|-------------|
| `PORT` | `8080` | Puerto del servidor |
| `OPENAI_MODEL` | `gpt-4o-mini` | Modelo de OpenAI |
| `USE_OPENAI_REALTIME` | `true` | Usar Realtime API |
| `DEFAULT_TTS_PROVIDER` | `elevenlabs` | Proveedor TTS |
| `DEFAULT_CALL_LANGUAGE` | `es` | Idioma por defecto |
| `ENABLE_DEBUG_ENDPOINTS` | `false` | Activar endpoints de debug |

### Frontend (Netlify) - OBLIGATORIA

| Variable | Valor | Descripción |
|----------|-------|-------------|
| `VITE_API_URL` | `https://agentiaserver-production.up.railway.app` | URL del backend |

---

## 🔗 Todos los Endpoints del Backend

### Autenticación (NUEVO)

| Método | Endpoint | Descripción | Body |
|--------|----------|-------------|------|
| `POST` | `/auth/register` | Registrar usuario | `{name, email, password}` |
| `POST` | `/auth/login` | Iniciar sesión | `{email, password}` |
| `GET` | `/auth/me` | Usuario actual | Header: `Authorization: Bearer <token>` |
| `POST` | `/auth/logout` | Cerrar sesión | - |

### Tenants

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/tenants` | Listar todos los tenants |
| `POST` | `/tenants` | Crear nuevo tenant |
| `GET` | `/tenants/{tenant_id}` | Obtener tenant específico |
| `PATCH` | `/tenants/{tenant_id}` | Actualizar tenant |
| `DELETE` | `/tenants/{tenant_id}` | Eliminar tenant |
| `POST` | `/tenants/{tenant_id}/validate-twilio` | Validar credenciales Twilio |
| `POST` | `/tenants/{tenant_id}/configure-webhook` | Configurar webhook |
| `POST` | `/tenants/{tenant_id}/list-numbers` | Listar números Twilio |
| `POST` | `/tenants/{tenant_id}/call` | Realizar llamada saliente |

### Sesiones

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/sessions` | Listar sesiones activas |
| `GET` | `/sessions/{session_key}` | Obtener sesión específica |
| `PATCH` | `/sessions/{session_key}` | Actualizar sesión |

### Configuración

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/config` | Obtener configuración disponible |
| `GET` | `/` | Health check |

### Webhooks (Twilio)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/voice` | Webhook de voz Twilio |

### WebSocket

| Tipo | Endpoint | Descripción |
|------|----------|-------------|
| `WS` | `/media-stream` | Stream de medios para llamadas |

---

## 🚀 Pasos de Despliegue

### 1. Backend → Railway (ya está desplegado)

Actualiza los cambios en Railway:

```bash
cd backend_temp

# Copiar archivos modificados a tu repo de backend
# Luego en tu repo de backend:
git add .
git commit -m "feat: Agregar CORS y endpoints de autenticación"
git push origin main
```

Railway redeployará automáticamente.

### 2. Frontend → Netlify

**Opción A: Netlify CLI**
```bash
cd AgentsAi_Frontend
npm run build

# Desplegar con variables de entorno
netlify deploy --prod --auth YOUR_NETLIFY_TOKEN
```

**Opción B: Netlify Dashboard**
1. Ve a https://app.netlify.com
2. Selecciona tu sitio → Site settings → Environment variables
3. Agrega:
   ```
   VITE_API_URL = https://agentiaserver-production.up.railway.app
   ```
4. Ve a Deploys → Trigger deploy

---

## 📁 Archivos Creados/Modificados

```
Frontend_Agent_IA/
├── AgentsAi_Frontend/
│   ├── .env                              # URL del backend
│   ├── .env.example                      # Template
│   └── src/services/auth.service.js      # Servicio de auth
│
├── backend_temp/                         # Repo clonado (para referencia)
│   └── STT_server/
│       ├── STT_Server.py                 # Modificado: CORS + auth router
│       └── routes/auth.py                 # NUEVO: Endpoints de autenticación
│
└── INSTRUCCIONES_DESPLIEGUE.md           # Este archivo
```

---

## 🧪 Verificar Conexión

1. **Backend:**
   ```
   https://agentiaserver-production.up.railway.app/
   → {"status": "ok", "message": "STT server running"}
   ```

2. **Login:**
   ```bash
   curl -X POST https://agentiaserver-production.up.railway.app/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@test.com","password":"123456"}'
   ```

3. **Frontend:** Después de desplegar en Netlify, abre la consola del navegador y verifica que no haya errores de CORS.

---

## ⚠️ Notas Importantes

1. **CORS es para el navegador**: El backend acepta peticiones de cualquier origen, pero el navegador solo permite si el servidor lo permite. Ya está configurado.

2. **No necesitas proxy**: Con CORS configurado, el frontend puede llamar directamente al backend en Railway.

3. **WebSocket**: Los WebSockets también están sujetos a CORS. El backend los acepta desde cualquier origen.

4. **Producción**: En producción real, cambia `allow_origins=["*"]` por los dominios específicos:
   ```python
   allow_origins=["https://tu-frontend.netlify.app"]
   ```

5. **Almacenamiento de usuarios**: Por ahora se usa un archivo JSON. Para producción, usa una base de datos (PostgreSQL en Railway, por ejemplo).