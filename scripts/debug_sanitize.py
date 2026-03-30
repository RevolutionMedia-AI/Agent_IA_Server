import re
s = 'Cialix offers a one-time bottle for $89.99 with free standard shipping. We have a 2-month supply at $58.49 per bottle, $116 total.'
print('ORIG:', s)
print('FINDALL:', re.findall(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", s))
print('AFTER SUB:', re.sub(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", lambda m: m.group(1)+' dollars', s))
