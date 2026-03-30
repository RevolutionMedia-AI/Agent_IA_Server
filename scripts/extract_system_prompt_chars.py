import re, sys
from pathlib import Path
p = Path('STT_server/domain/language.py')
text = p.read_text(encoding='utf-8')
lines = text.splitlines()
start = None
for i, line in enumerate(lines):
    if 'SYSTEM_PROMPT' in line and '=' in line and '(' in line:
        start = i
        break
if start is None:
    print('ERROR: SYSTEM_PROMPT not found', file=sys.stderr)
    sys.exit(2)
end = None
for j in range(start, len(lines)):
    if lines[j].strip() == ')':
        end = j
        break
if end is None:
    print('ERROR: closing parenthesis not found', file=sys.stderr)
    sys.exit(3)
block = '\n'.join(lines[start:end+1])
# extract double-quoted string literals inside the block
strs = re.findall(r'"([^\"]*)"', block, flags=re.DOTALL)
full = ''.join(strs)
# collect special characters: non-alnum and not whitespace
specials = sorted(set(ch for ch in full if not ch.isalnum() and not ch.isspace()))
print(''.join(specials))
for ch in specials:
    code = ord(ch)
    # printable label
    label = ch
    if ch == ' ':
        label = '<SPACE>'
    if ch == '\n':
        label = '<NL>'
    print(f"U+{code:04X}\t{label}")
