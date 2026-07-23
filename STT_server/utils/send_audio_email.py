"""Email a generated TTS audio file as an attachment.

The `audio_path` is supplied by the caller. To prevent the function
from being abused as a local file-disclosure primitive (`audio_path =
"/etc/passwd"`), we resolve the path and require it to live under the
allowlisted AUDIO_OUTPUT_DIR (defaults to the project data dir).

If a caller needs to attach files from outside that dir, they should
either pass an absolute path that's still inside AUDIO_OUTPUT_DIR or
update AUDIO_OUTPUT_DIR to include the new root.
"""
from __future__ import annotations

import os
from email.message import EmailMessage
from pathlib import Path

import smtplib

from STT_server.utils.safe_path import UnsafePathError, safe_join

# Configuración
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_TO = "kevin.escalante@revolutionmedia.ai"

# ponytail: filesystem root for attachments. Resolved against the
# project data dir by default. The function refuses any path that
# resolves outside this dir (file-inclusion guard).
AUDIO_OUTPUT_DIR = Path(
    os.getenv(
        "AUDIO_OUTPUT_DIR",
        str(Path(__file__).resolve().parent.parent / "data" / "tts_audio"),
    )
).resolve()


def _resolve_audio_path(audio_path: str) -> Path:
    """Resolve `audio_path` under AUDIO_OUTPUT_DIR.

    Behaviour:
      * Absolute paths are validated to live under AUDIO_OUTPUT_DIR.
      * Relative paths are joined to AUDIO_OUTPUT_DIR.
      * Anything that escapes AUDIO_OUTPUT_DIR raises UnsafePathError.
    """
    p = Path(audio_path)
    if p.is_absolute():
        # Caller passed a full path — verify it's inside the allowlist.
        target = p.resolve()
        try:
            target.relative_to(AUDIO_OUTPUT_DIR)
        except ValueError as exc:
            raise UnsafePathError(
                f"audio_path {target} is outside AUDIO_OUTPUT_DIR "
                f"({AUDIO_OUTPUT_DIR})"
            ) from exc
        return target
    return Path(safe_join(AUDIO_OUTPUT_DIR, audio_path))


def send_audio_email(audio_path, subject="ElevenLabs TTS Audio", body="Adjunto el archivo de audio generado por ElevenLabs TTS."):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    safe_path = _resolve_audio_path(audio_path)
    with open(safe_path, "rb") as f:
        audio_data = f.read()
        msg.add_attachment(audio_data, maintype="audio", subtype="basic", filename=safe_path.name)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        print(f"Audio enviado a {EMAIL_TO}")
