import smtplib
import os
from email.message import EmailMessage

# Configuración
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_TO = "kevin.escalante@revolutionmedia.ai"


def send_audio_email(audio_path, subject="Rime TTS Audio", body="Adjunto el archivo de audio generado por Rime TTS."):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = EMAIL_TO
    msg.set_content(body)

    with open(audio_path, "rb") as f:
        audio_data = f.read()
        msg.add_attachment(audio_data, maintype="audio", subtype="basic", filename=os.path.basename(audio_path))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        print(f"Audio enviado a {EMAIL_TO}")
