import importlib
m = importlib.import_module('STT_server.domain.language')
text = "Cialix offers a one-time bottle for $89.99 with free standard shipping. We have a 2-month supply at $58.49 per bottle, $116 total."
print('SANITIZED:', m.sanitize_tts_text(text))
