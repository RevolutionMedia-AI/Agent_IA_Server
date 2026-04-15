"""Twilio REST API adapter for webhook configuration and outbound calls.

Uses the Twilio SDK to:
- Configure the voice webhook on a phone number
- Initiate outbound calls
- Validate credentials
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("stt_server")


def _get_twilio_client(account_sid: str, auth_token: str):
    """Lazy import and create a Twilio client."""
    from twilio.rest import Client
    return Client(account_sid, auth_token)


async def validate_twilio_credentials(account_sid: str, auth_token: str) -> dict:
    """Validate Twilio credentials by fetching the account.

    Returns dict with 'valid' bool and optional 'error' message.
    """
    import asyncio

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

    return await asyncio.to_thread(_validate)


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
    import asyncio

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

    return await asyncio.to_thread(_configure)


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
    import asyncio

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

    return await asyncio.to_thread(_call)


async def list_phone_numbers(
    account_sid: str,
    auth_token: str,
) -> dict:
    """List all phone numbers in the Twilio account.

    Returns dict with 'success' bool and 'numbers' list or 'error'.
    """
    import asyncio

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

    return await asyncio.to_thread(_list)