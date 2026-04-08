import logging
import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict

logger = logging.getLogger(__name__)

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def _post_message(account_sid: str, auth_token: str, from_: str, to: str, body: str) -> dict:
    """
    Send a WhatsApp message via Twilio REST API using requests.
    No Twilio SDK — no httpx — no proxies kwarg — no version conflicts.
    Returns the parsed JSON response or raises on failure.
    """
    url = TWILIO_API.format(sid=account_sid)
    resp = requests.post(
        url,
        auth=(account_sid, auth_token),
        data={"From": from_, "To": to, "Body": body},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Twilio {resp.status_code}: {resp.text[:300]}")
    return resp.json()


class WhatsAppSender:
    """Send price forecasts via WhatsApp using Twilio REST API (no SDK)."""

    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

    @property
    def available(self) -> bool:
        return bool(self.account_sid and self.auth_token)

    @staticmethod
    def format_forecast_message(prices: List[float]) -> str:
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        lines = [f"DA-LMP FORECAST - {tomorrow}", ""]
        for hour, price in enumerate(prices, 1):
            lines.append(f"Hour {hour:02d}: ${price:.2f}")
        avg = sum(prices) / len(prices)
        lines += ["", f"Avg: ${avg:.2f}  Min: ${min(prices):.2f}  Max: ${max(prices):.2f}"]
        return "\n".join(lines)

    def send(self, to_number: str, prices: List[float]) -> tuple:
        """Send forecast to a single WhatsApp number.
        Returns (success: bool, error: str|None)."""
        if not self.available:
            msg = "Twilio credentials not set"
            logger.warning(msg)
            return False, msg
        if not to_number:
            msg = "No destination number provided"
            logger.warning(msg)
            return False, msg

        # Normalize to E.164: strip whitespace/newlines, strip prefix, ensure +
        raw = to_number.replace("whatsapp:", "").strip()
        raw = "".join(c for c in raw if c.isdigit() or c == "+")
        if not raw.startswith("+"):
            raw = "+" + raw
        to = f"whatsapp:{raw}"
        body = self.format_forecast_message(prices)

        logger.info(f"Sending WhatsApp from={self.from_number} to={to}")

        try:
            result = _post_message(self.account_sid, self.auth_token, self.from_number, to, body)
            logger.info(f"WhatsApp sent to {raw} (SID: {result.get('sid')})")
            return True, None
        except Exception as e:
            logger.error(f"WhatsApp send failed to {raw}: {e}")
            return False, str(e)

    def send_to_stepdad(self, prices: List[float]) -> tuple:
        number = os.getenv("STEPDAD_WHATSAPP", "")
        if not number:
            return False, "STEPDAD_WHATSAPP env var not set"
        return self.send(number, prices)

    def send_to_you(self, prices: List[float], metadata: Dict = None) -> tuple:
        number = os.getenv("YOUR_WHATSAPP", "")
        if not number:
            return True, None  # optional — not an error
        return self.send(number, prices)

    @staticmethod
    def calculate_metadata(prices: List[float]) -> Dict:
        if not prices:
            return {}
        return {
            "avg": sum(prices) / len(prices),
            "peak_hour": prices.index(max(prices)) + 1,
            "peak_price": max(prices),
            "low_hour": prices.index(min(prices)) + 1,
            "low_price": min(prices),
        }
