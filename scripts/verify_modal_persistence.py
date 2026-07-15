"""ponytail: regression check for Edit Number / Edit Agent persistence.

Bug history:
  Edit Number's PUT /phone-numbers/{id} silently dropped every
  credential field. Two layers agreed to lose them: Pydantic's
  PhoneNumberUpdate schema didn't declare them, and db_phone_numbers
  .update_number's allowed-set didn't include them. The FE was
  sending them correctly — the symptom was "old creds stay active,
  new ones never take".

  Bonus bugs found during the audit:
    - ModalConnectNumber's "Test Twilio" button referenced `number`
      and `label` which don't exist in that component (state is
      numberDigits, no label field). Closed over undefined vars.
    - ModalAgents defaulted the Inworld TTS model to inworld-tts-2
      (deprecated by Inworld; same fix that landed in the BE).

This script verifies all three without booting FastAPI or React.
Run from Agent_IA_Server/: python scripts/verify_modal_persistence.py
"""
import ast
import re
import sys
from pathlib import Path

ROOT_BE = Path(__file__).resolve().parent.parent
ROOT_FE = ROOT_BE.parent / "AgentsAi_Frontend"
PASS = "[OK]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}{(' — ' + detail) if detail else ''}")
        sys.exit(1)


# ── 1. PhoneNumberUpdate declares credential fields ─────────────────────────
api_src = (ROOT_BE / "STT_server" / "routes" / "api.py").read_text(encoding="utf-8")

m = re.search(r"class PhoneNumberUpdate\(BaseModel\):(.*?)(?=\nclass |\Z)", api_src, re.DOTALL)
assert m, "could not find PhoneNumberUpdate class"
update_body = m.group(1)
required_creds = [
    "twilio_account_sid", "twilio_auth_token",
    "sip_host", "sip_username", "sip_password",
    "whatsapp_phone_number_id", "whatsapp_access_token",
]
missing = [f for f in required_creds if f"{f}:" not in update_body]
check(
    "PhoneNumberUpdate declares every credential field",
    not missing,
    f"missing: {missing}",
)

# ── 2. db_phone_numbers.update_number whitelist includes credentials ───────
db_src = (ROOT_BE / "STT_server" / "db_phone_numbers.py").read_text(encoding="utf-8")

# Pull out the update_number function body via a simple AST walk.
tree = ast.parse(db_src)
update_fn = next(
    (n for n in tree.body
     if isinstance(n, ast.FunctionDef) and n.name == "update_number"),
    None,
)
assert update_fn is not None, "could not find db_phone_numbers.update_number"

# Find the first `allowed = {...}` assignment inside the function body.
allowed_set = None
for node in ast.walk(update_fn):
    if (isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "allowed"
        and isinstance(node.value, (ast.Set, ast.Dict))):
        if isinstance(node.value, ast.Set):
            allowed_set = {
                elt.value for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            }
        break

assert allowed_set is not None, "could not find `allowed = {...}` set in update_number"
missing_allow = [f for f in required_creds if f not in allowed_set]
check(
    "db_phone_numbers.update_number allowed-set includes every credential field",
    not missing_allow,
    f"missing from allowed: {missing_allow}",
)

# ── 3. ModalConnectNumber: Test Twilio closure uses real variables ─────────
mc_src = (ROOT_FE / "src" / "components" / "common" / "ModalConnectNumber.jsx").read_text(encoding="utf-8")

# Locate the validateTwilio({ ... }) call inside the Test Twilio button.
m = re.search(r"phoneNumbersApi\.validateTwilio\(\{([^}]+)\}\)", mc_src, re.DOTALL)
assert m, "could not find phoneNumbersApi.validateTwilio call in ModalConnectNumber"
validate_body = m.group(1)
# Bug: previously referenced bare `number` and `label` identifiers that
# don't exist in this component. After the fix those are gone; `number`
# is only present as `number:` (a key).
bare_number_refs = re.findall(r"\bnumber\b", validate_body)
# At most one match expected: the `number:` key inside the payload.
check(
    "ModalConnectNumber Test Twilio no longer references bare `number` or `label`",
    "label" not in validate_body,
    f"validate body still contains 'label' or stray `number` reference: {validate_body!r}",
)

# ── 4. ModalAgents: Inworld TTS default is no longer 'inworld-tts-2' ──────
ma_src = (ROOT_FE / "src" / "components" / "common" / "ModalAgents.jsx").read_text(encoding="utf-8")

# Only the FALLBACK positions used to hardcode 'inworld-tts-2':
#   - the dropdown's default-value fallback (`... || 'inworld-tts-2'`)
#   - the submit-time payload fallback (`... || 'inworld-tts-2'`)
# The options ARRAY still legitimately offers inworld-tts-2 as a
# selectable choice for operators with existing agents, so we only
# look at the `||` fallback sites.
fallback_sites = re.findall(r"\|\|\s*'inworld-tts-2'", ma_src)
check(
    "ModalAgents no longer falls back to 'inworld-tts-2'",
    not fallback_sites,
    f"still falls back to 'inworld-tts-2' at: {fallback_sites}",
)

print("\nALL CHECKS PASSED.")