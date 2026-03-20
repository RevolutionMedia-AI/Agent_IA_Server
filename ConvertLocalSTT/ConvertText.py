import json
import base64 
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

model_size ="small.en"

model = WhisperModel("small.en", device="cuda", compute_type="int8_float16")

segments, info = model.transcribe("audio.mp3" , beam_size=5)

print("detected languaje '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
   print(segment.text)

