import importlib

m = importlib.import_module("STT_server.domain.language")
print("HAS_SANITIZE=", hasattr(m, "sanitize_tts_text"))
print("SAN:", m.sanitize_tts_text("Hello — $5!! 😂 Extra^^^"))
