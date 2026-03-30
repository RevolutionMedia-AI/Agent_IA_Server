from STT_server.domain.language import sanitize_tts_text
examples = [
    "Hello!!! How much is this? $12.99 -- wow 😊 <script>",
    "¡¡¡Hola!!! ¿Cómo estás??? $$$###***",
    "Order #12345!!! Please confirm."
]
for e in examples:
    print('ORIG:', e)
    print('SAN:', sanitize_tts_text(e))
    print('-'*40)
