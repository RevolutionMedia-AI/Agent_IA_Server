import STT_server.services.playback_service as p
import STT_server.config as c
print('ASSISTANT_ECHO_IGNORE_MS=', c.ASSISTANT_ECHO_IGNORE_MS)
print('TWILIO_OUTBOUND_PACING_MS=', getattr(p, 'TWILIO_OUTBOUND_PACING_MS', 'n/a'))
