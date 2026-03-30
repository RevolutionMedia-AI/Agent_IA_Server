import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from STT_server.domain.language import sanitize_tts_text

text = (
    "Certainly! Cialix offers a few special purchasing options. "
    "You can buy a one-time bottle for $89.99 with free standard shipping. "
    "If you're interested in a supply, we have a 2-month supply at $58.49 per bottle, totaling $116, which saves you $63 and includes free shipping. "
    "Our 4-month supply is $44.54 per bottle, totaling $178, saving you $136 with free shipping. "
    "Lastly, the 6-month supply is $35.99 per bottle, totaling $215, saving you $240 with free shipping. "
    "We also have a VIP Rush Delivery option for an additional $9.99. "
    "Would you like more details on any of these options?"
)

print('ORIG:', text)
print('\nSANITIZED:\n', sanitize_tts_text(text))
