import sys, os
sys.path.insert(0, os.getcwd())
from STT_server.domain.language import sanitize_tts_text

examples = [
    "Hello!!! How much is this? $12.99 -- wow 😊 <script>",
    "¡¡¡Hola!!! ¿Cómo estás??? $$$###***",
    "Order #12345!!! Please confirm.",
    "Of course! Could you please provide your name and order number so I can look up the details of your package?",
]

for e in examples:
    print('ORIG:', e)
    print('SAN:', sanitize_tts_text(e))
    print('-'*40)
