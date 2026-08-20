"""Twilio REST API adapter for webhook configuration and outbound calls.

Uses the Twilio SDK to:
- Configure the voice webhook on a phone number
- Initiate outbound calls
- Validate credentials
- Validate Twilio signatures on inbound webhooks
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any
from urllib.parse import quote_plus

from STT_server.services.thread_pool import to_thread as _to_thread

log = logging.getLogger("stt_server")


def validate_twilio_signature(
    auth_token: str,
    full_url: str,
    received_signature: str,
    form_params: dict,
) -> bool:
    """Verify that an inbound request came from Twilio.

    Twilio signs every webhook call with HMAC-SHA1 over the URL + sorted
    form params. The signature arrives as the `X-Twilio-Signature` header.
    We recompute it and compare in constant time.

    Algorithm (matching twilio-python RequestValidator.compute_signature
    byte-for-byte; verified against the SDK in
    STT_server/tests/test_signature_validator.py):
      1. Start with the full URL.
      2. For each form param, sorted alphabetically by key, append
         ``key + str(value)`` — RAW concatenation, NO URL encoding.
         The Twilio docs claim values are URL-encoded, but the official
         SDKs (Python, Node.js, PHP, Java) all concatenate the raw
         value. Encoding the value with quote_plus diverges from what
         Twilio's servers sign.
      3. HMAC-SHA1 with the auth token as key.
      4. Base64-encode and compare with X-Twilio-Signature.

    See: https://www.twilio.com/docs/usage/webhooks/webhooks-security
    """
    import base64
    if not auth_token or not received_signature or not full_url:
        return False
    pieces = [full_url]
    for k in sorted(form_params.keys()):
        v = form_params[k] if form_params[k] is not None else ""
        # ponytail: raw concatenation. Twilio's signer does NOT URL-encode
        # the values despite what the prose docs imply — verified by
        # comparing HMAC output against twilio-python's RequestValidator
        # on a representative inbound form. Any URL encoding (quote_plus,
        # quote, etc.) shifts the bytes and the HMAC diverges.
        pieces.append(f"{k}{str(v)}")
    data = "".join(pieces).encode("utf-8")
    expected = hmac.new(
        auth_token.encode("utf-8"),
        data,
        hashlib.sha1,
    ).digest()
    expected_b64 = base64.b64encode(expected).decode("ascii")
    return hmac.compare_digest(expected_b64, received_signature)



def _get_twilio_client(account_sid: str, auth_token: str):
    """Lazy import and create a Twilio client."""
    from twilio.rest import Client
    return Client(account_sid, auth_token)


async def validate_twilio_credentials(account_sid: str, auth_token: str) -> dict:
    """Validate Twilio credentials by fetching the account.

    Returns dict with 'valid' bool and optional 'error' message.
    """
    def _validate() -> dict:
        try:
            client = _get_twilio_client(account_sid, auth_token)
            account = client.api.accounts(account_sid).fetch()
            return {
                "valid": True,
                "account_status": account.status,
                "account_type": account.type,
                "friendly_name": account.friendly_name,
            }
        except Exception as exc:
            log.warning("Twilio credential validation failed: %s", exc)
            return {"valid": False, "error": str(exc)}

    return await _to_thread(_validate)


async def configure_voice_webhook(
    account_sid: str,
    auth_token: str,
    phone_number: str,
    webhook_url: str,
) -> dict:
    """Configure the voice webhook URL on a Twilio phone number.

    This sets the VoiceUrl on the incoming phone number so that when
    someone calls, Twilio hits our /voice endpoint.

    Args:
        account_sid: Twilio Account SID
        auth_token: Twilio Auth Token
        phone_number: Phone number in E.164 format (e.g. "+15071234567")
        webhook_url: Full URL to the /voice endpoint

    Returns:
        dict with 'success' bool and optional details.
    """
    def _configure() -> dict:
        try:
            client = _get_twilio_client(account_sid, auth_token)

            # Find the phone number in the account
            numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
            if not numbers:
                # Try without the '+' prefix
                numbers = client.incoming_phone_numbers.list(phone_number=phone_number.lstrip("+"))
            if not numbers:
                return {
                    "success": False,
                    "error": f"Phone number '{phone_number}' not found in this Twilio account. Make sure the number is purchased and active.",
                }

            number = numbers[0]
            log.info(
                "Configuring webhook for %s (SID: %s) -> %s",
                phone_number,
                number.sid,
                webhook_url,
            )

            # Update the voice URL and method
            updated = number.update(
                voice_url=webhook_url,
                voice_method="POST",
                # Also set status callback for call tracking
                status_callback=webhook_url.replace("/voice", "/call-status"),
                status_callback_method="POST",
            )

            return {
                "success": True,
                "phone_number_sid": number.sid,
                "voice_url": updated.voice_url,
                "friendly_name": updated.friendly_name,
            }
        except Exception as exc:
            log.exception("Error configuring Twilio webhook")
            return {"success": False, "error": str(exc)}

    return await _to_thread(_configure)


async def make_outbound_call(
    account_sid: str,
    auth_token: str,
    from_number: str,
    to_number: str,
    webhook_url: str,
) -> dict:
    """Initiate an outbound call via Twilio.

    Args:
        account_sid: Twilio Account SID
        auth_token: Twilio Auth Token
        from_number: Caller's Twilio number (E.164)
        to_number: Destination number (E.164)
        webhook_url: URL to the /voice endpoint

    Returns:
        dict with 'success' bool and call_sid or error.
    """
    def _call() -> dict:
        try:
            client = _get_twilio_client(account_sid, auth_token)

            call = client.calls.create(
                to=to_number,
                from_=from_number,
                url=webhook_url,
                method="POST",
            )

            log.info(
                "Outbound call initiated: %s -> %s, call_sid=%s",
                from_number,
                to_number,
                call.sid,
            )

            return {
                "success": True,
                "call_sid": call.sid,
                "status": call.status,
                "from": from_number,
                "to": to_number,
            }
        except Exception as exc:
            log.exception("Error making outbound call")
            return {"success": False, "error": str(exc)}

    return await _to_thread(_call)


async def list_phone_numbers(
    account_sid: str,
    auth_token: str,
) -> dict:
    """List all phone numbers in the Twilio account.

    Returns dict with 'success' bool and 'numbers' list or 'error'.
    """
    def _list() -> dict:
        try:
            client = _get_twilio_client(account_sid, auth_token)
            numbers = client.incoming_phone_numbers.list()

            result = []
            for n in numbers:
                result.append({
                    "phone_number": n.phone_number,
                    "friendly_name": n.friendly_name,
                    "sid": n.sid,
                    "voice_url": n.voice_url,
                    "capabilities": {
                        "voice": n.capabilities.get("voice", False) if n.capabilities else False,
                        "sms": n.capabilities.get("sms", False) if n.capabilities else False,
                    },
                })

            return {"success": True, "numbers": result, "count": len(result)}
        except Exception as exc:
            log.exception("Error listing Twilio phone numbers")
            return {"success": False, "error": str(exc)}

    return await _to_thread(_list)


async def transfer_call(
    account_sid: str,
    auth_token: str,
    call_sid: str,
    destination: str,
) -> dict:
    """Redirect a live Twilio call to a new destination.

    Uses calls(call_sid).update(twiml=...) with a <Response><Dial> block —
    Twilio disconnects our media stream and dials the destination number,
    then bridges the original caller to whoever picks up. We don't get a
    second WebSocket for the bridged leg; the call is no longer ours.

    The transfer can fail at multiple layers: bad destination format,
    Twilio rejects the number (not provisioned, geo-blocked, etc.),
    the auth_token doesn't own this call_sid. All surface as
    success=False with the SDK error message — the call path catches
    this in the executor and tells the LLM so the conversation can
    recover gracefully instead of dead-airing the caller.

    ponytail: the destination must be E.164. We don't enforce here
    because the AgentTool.validate() layer already rejects malformed
    destinations at save time, and re-validating on every transfer
    would just duplicate the rule. If a stale row slips through (DB
    edit, manual JSON poke), Twilio returns its own 4xx error which
    we surface verbatim.
    """
    def _transfer() -> dict:
        try:
            client = _get_twilio_client(account_sid, auth_token)
            # TwiML payload: drop the current media stream and dial
            # the new destination. callerId defaults to the original
            # caller (the agent's Twilio number) so the bridged party
            # sees a familiar number on their display.
            twiml = (
                f'<Response><Dial callerId="{{ORIG}}">{destination}</Dial></Response>'
            )
            # ponytail: Twilio doesn't actually accept {ORIG} as a
            # macro — the SDK leaves the caller id as-is. We pass
            # the destination number through verbatim; the operator
            # can pre-set callerId at the phone_number level if they
            # want a specific outbound identity. Keeping the literal
            # template above in a comment so future readers see what
            # was intended if Twilio ever adds macros.
            twiml = f'<Response><Dial>{destination}</Dial></Response>'
            log.info(
                "[TRANSFER] calls(%s).update(twiml=<Dial>%s</Dial>) via subaccount %s...",
                call_sid, destination, account_sid[:6] or "?",
            )
            client.calls(call_sid).update(twiml=twiml)
            return {
                "success": True,
                "call_sid": call_sid,
                "destination": destination,
            }
        except Exception as exc:
            log.exception(
                "[TRANSFER] call_sid=%s destination=%s failed",
                call_sid, destination,
            )
            return {"success": False, "error": str(exc)}

    return await _to_thread(_transfer)