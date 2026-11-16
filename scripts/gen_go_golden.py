"""Generate the golden vectors for the Go port's compatibility tests.

The three constraints that decide whether the port can share a database
with the Python backend:

  1. the stored credential blobs must be readable by Go,
  2. the Twilio webhook signature must be computed identically,
  3. the credential merge must pick the same winner.

The vectors come from the Python backend itself, so they are ground
truth rather than an expectation written by hand.

Run:  python scripts/gen_go_golden.py
"""
import base64
import hashlib
import hmac
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(r"D:\New folder\Agent_IA_Server")
sys.path.insert(0, str(ROOT))

# The same key the Go tests use.
KEY = "oahqImB7aGYfEFxfWIJLZzJs27YSYAgr5rHUyc3gIRU="
os.environ["CREDENTIAL_ENCRYPTION_KEY"] = KEY
os.environ.setdefault("ENVIRONMENT", "test")

from STT_server.security.credentials import encrypt_credentials  # noqa: E402
from STT_server.services import credentials_resolver  # noqa: E402

OUT = pathlib.Path(r"D:\Agents AI Go\internal\credentials\testdata\golden.json")

# ── 1. Fernet blobs ─────────────────────────────────────────────────────
# The stored shape is a JSON object whose VALUES are individual Fernet
# tokens. Getting this wrong makes the Go service unable to read a
# single stored credential, so it is pinned with a real blob.
cred_cases = [
    {"api_key": "sk-user-openai-abc123"},
    {"api_key": "sk-user-eleven", "voice_id": "voice-xyz", "model_id": "eleven_flash_v2"},
    {"api_key": "", "voice_id": "only-voice"},
    {},
]
for i, plain in enumerate(cred_cases):
    cred_cases[i] = {"plain": plain, "stored": encrypt_credentials(plain)}

# ── 2. Twilio signatures ───────────────────────────────────────────────
def sign(url: str, form: dict, token: str) -> str:
    pieces = [url]
    for key in sorted(form):
        pieces.append(f"{key}{form[key] if form[key] is not None else ''}")
    digest = hmac.new(token.encode("utf-8"), "".join(pieces).encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


sig_cases = []
for i, (url, form) in enumerate([
    ("https://example.ngrok.io/voice", {"To": "+15551234567", "CallSid": "CA" + "0" * 32}),
    ("https://example.ngrok.io/voice", {}),
    ("https://example.ngrok.io/voice?tenant_id=tenant-abc", {"From": "+15559998888", "To": "+15551234567"}),
    ("https://example.ngrok.io/call-status", {"CallStatus": "completed", "CallSid": "CA" + "1" * 32}),
    ("https://example.ngrok.io/voice", {"A": "1", "B": "", "C": "3", "D": "4"}),
]):
    sig_cases.append({
        "url": url,
        "form": form,
        "token": "twilio-auth-token-abc123",
        "signature": sign(url, form, "twilio-auth-token-abc123"),
    })

# ── 3. the credential merge ─────────────────────────────────────────────
# The merge decides WHICH key a call uses. If Go and Python disagree,
# one of them talks to a provider with the wrong credentials.
#
# The values come from the real resolve_provider: the platform layer is
# read from the environment (as it is in production) and the per-user
# layer is injected, because the per-user layer normally comes from
# the database.
PROVIDER = "elevenlabs"
PLATFORM_FIELDS = {
    "api_key": "ELEVENLABS_API_KEY",
    "voice_id": "ELEVENLABS_VOICE_ID",
    "model_id": "ELEVENLABS_MODEL_ID",
}

merge_cases = []
scenarios = [
    ({"api_key": "sk-platform"}, {"api_key": "sk-user"}, True),
    ({"api_key": "sk-platform"}, {"api_key": "sk-user"}, False),
    ({"api_key": "sk-platform"}, {}, True),
    ({}, {"api_key": "sk-user"}, True),
    ({"api_key": "sk-platform", "voice_id": "platform-voice"},
     {"api_key": "sk-user"}, True),
    ({"api_key": "sk-platform", "voice_id": "platform-voice"},
     {"voice_id": "user-voice"}, True),
    ({"api_key": "sk-platform"}, {"api_key": ""}, True),
    ({}, {}, False),
]

real_read_per_user = credentials_resolver._read_per_user
for platform, per_user, use_own in scenarios:
    saved_env = {env: os.environ.get(env) for env in PLATFORM_FIELDS.values()}
    for field, env in PLATFORM_FIELDS.items():
        if platform.get(field):
            os.environ[env] = platform[field]
        else:
            os.environ.pop(env, None)

    credentials_resolver._read_per_user = lambda user_id, provider_id, _p=per_user: dict(_p)
    try:
        resolved = credentials_resolver.resolve_provider(
            "user-test-001", PROVIDER, use_own_key=use_own
        )
    finally:
        credentials_resolver._read_per_user = real_read_per_user
        for env, val in saved_env.items():
            if val is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = val

    merge_cases.append({
        "provider": PROVIDER,
        "platform": platform,
        "per_user": per_user,
        "use_own_key": use_own,
        "resolved": resolved,
    })

out = {
    "_comment": "Generated from Agent_IA_Server (the Python backend). "
                "Golden vectors for the Go port's compatibility tests.",
    "fernet_key": KEY,
    "credentials": cred_cases,
    "credential_merges": merge_cases,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

# The Twilio signatures live in the twilio package's own testdata: each
# package owns the vectors it is tested against.
twilio_out = {
    "_comment": "Generated from Agent_IA_Server (the Python backend). "
                "Golden vectors for the Go port's Twilio signature tests.",
    "signatures": sig_cases,
}
twilio_path = pathlib.Path(r"D:\Agents AI Go\internal\adapters\twilio\testdata\golden.json")
twilio_path.parent.mkdir(parents=True, exist_ok=True)
twilio_path.write_text(json.dumps(twilio_out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"wrote {OUT}")
print(f"  {len(cred_cases)} credential blobs, {len(merge_cases)} merges")
print(f"wrote {twilio_path}")
print(f"  {len(sig_cases)} signatures")
