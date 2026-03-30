import re, unicodedata
s = 'Cialix offers a one-time bottle for $89.99 with free standard shipping. We have a 2-month supply at $58.49 per bottle, $116 total.'
print('ORIG:', s)

# normalize
s1 = unicodedata.normalize('NFKC', s)
print('\nAfter normalize:', s1)

# remove control chars
s2 = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s1)
print('\nAfter remove control:', s2)

# smart punctuation map
replacements = {"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-", "…": "...", "•": "-", "·": "-"}
for k, v in replacements.items():
    s2 = s2.replace(k, v)
print('\nAfter smart punct:', s2)

# expand dollars
def _expand_dollar(m: re.Match) -> str:
    amt = m.group(1)
    amt = amt.replace(',', '')
    try:
        value = float(amt)
    except Exception:
        return amt + ' dollars'
    dollars = int(value)
    cents = int(round((value - dollars) * 100))
    if cents:
        return f"{dollars} dollars and {cents} cents"
    return f"{dollars} dollars"

s3 = re.sub(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", _expand_dollar, s2)
print('\nAfter expand dollars:', s3)

s4 = s3.replace('€', ' euros ').replace('£', ' pounds ').replace('¥', ' yen ')
print('\nAfter currency replaces:', s4)

# angle bracket removal
s5 = re.sub(r"<[^>]+>", ' ', s4)
print('\nAfter angle tag removal:', s5)

# allowed filter
def _allowed(ch: str) -> bool:
    if ch.isspace():
        return True
    cat = unicodedata.category(ch)
    if cat[0] in ('L', 'N'):
        return True
    if ch in ".,!?-:;'\"()":
        return True
    return False

s6 = ''.join(ch for ch in s5 if _allowed(ch))
print('\nAfter allowed filter:', s6)

# collapse repeated punctuation
s7 = re.sub(r'([!?.]){2,}', r'\1', s6)
s7 = re.sub(r'-{2,}', '-', s7)
print('\nAfter collapse punctuation:', s7)

# normalize whitespace
s8 = re.sub(r'\s+', ' ', s7).strip()
print('\nAfter normalize whitespace:', s8)

# done
print('\nFINAL:', s8)
