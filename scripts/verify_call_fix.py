"""ponytail: regression check for the three Twilio-call-silence fixes.

Bug history: call connected, greeting played, then silence because
agent_id never reached the WebSocket start event. Three root causes:

  1. TwiML f-string rendered {stream_params} (a Python list) so Twilio
     dropped the <Parameter> elements.
  2. PUT /settings/api-keys/{provider} returned 422 because the OpenAI
     regex rejected base64 chars in modern project keys.
  3. inworld_tts.py defaulted to inworld-tts-2, which Inworld retired.

This script asserts each fix is in place without importing the full
app stack. Run from Agent_IA_Server/: python scripts/verify_call_fix.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 1. TwiML template ────────────────────────────────────────────────────────
stt = (ROOT / "STT_server" / "STT_Server.py").read_text(encoding="utf-8")
bad = re.findall(r"<Stream[^>]*>\{stream_params\}<", stt)
assert not bad, f"BUG STILL PRESENT: bare {{stream_params}} in TwiML: {bad!r}"
good = re.findall(r"<Stream[^>]*>\{stream_params_str\}<", stt)
assert len(good) == 2, f"expected 2 stream_params_str refs in TwiML, got {len(good)}"
print("OK fix-1: TwiML renders stream_params_str (joined XML), not stream_params (list repr)")

# ── 2. OpenAI regex ──────────────────────────────────────────────────────────
cr = (ROOT / "STT_server" / "services" / "credentials_resolver.py").read_text(encoding="utf-8")
m = re.search(r'name="api_key".*?pattern=r"([^"]+)"', cr, re.DOTALL)
assert m, "could not find openai api_key pattern in credentials_resolver.py"
pattern = m.group(1)
print(f"      pattern: {pattern}")
assert re.search(pattern, "sk-" + "a" * 30), "legacy sk- key rejected"
proj_key = "sk-proj-" + "AbCdEfGh1234567890" * 8 + "+/="
assert re.search(pattern, proj_key), "project base64 key rejected"
assert re.search(pattern, "sk-svcacct-" + "x" * 30), "svcacct key rejected"
assert not re.search(pattern, "pk-not-an-openai-key"), "non-sk key accepted"
assert not re.search(pattern, "sk-short"), "too-short sk- key accepted"
print("OK fix-2: OpenAI regex accepts legacy/project/svcacct base64 keys, rejects garbage")

# ── 3. Inworld default model ────────────────────────────────────────────────
itw = (ROOT / "STT_server" / "adapters" / "inworld_tts.py").read_text(encoding="utf-8")
assert 'DEFAULT_MODEL_ID = "inworld-tts-1.5-mini"' in itw, "DEFAULT_MODEL_ID not updated"
# Old constant must not still be assigned (comments referencing the
# history are fine).
remaining = [
    line for line in itw.splitlines()
    if re.match(r'^\s*DEFAULT_MODEL_ID\s*=\s*[\'"]inworld-tts-2[\'"]', line)
]
assert not remaining, f"old DEFAULT_MODEL_ID = 'inworld-tts-2' still present: {remaining!r}"
print("OK fix-3: inworld DEFAULT_MODEL_ID = inworld-tts-1.5-mini (deprecation gone)")

print("\nALL THREE FIXES VERIFIED.")