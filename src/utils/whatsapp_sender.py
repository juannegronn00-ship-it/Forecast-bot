import json
import logging
import os
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)

TWILIO_API = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"

# da_lmp_forecast template (26 variables):
#   {{1}}      = date ("April 10, 2026")
#   {{2}}–{{25}} = hourly prices ("$27.80")
#   {{26}}     = stats summary ("Avg: $X.XX | Min: $X.XX | Max: $X.XX")
DA_LMP_FORECAST_SID = "HXf2497d6558afbb590b2d7e8be125a726"


def _post_template_message(
    account_sid: str,
    auth_token: str,
    from_: str,
    to: str,
    content_sid: str,
    content_variables: dict,
) -> dict:
    url = TWILIO_API.format(sid=account_sid)
    resp = requests.post(
        url,
        auth=(account_sid, auth_token),
        data={
            "From": from_,
            "To": to,
            "ContentSid": content_sid,
            "ContentVariables": json.dumps(content_variables),
        },
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Twilio {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def format_forecast_variables(
    date: str,
    hourly_prices: List[float],
    avg: float,
    min_price: float,
    max_price: float,
) -> dict:
    """Build ContentVariables dict for the 26-variable da_lmp_forecast template."""
    variables = {"1": date}
    for i, price in enumerate(hourly_prices[:24], start=2):
        variables[str(i)] = f"${price:.2f}"
    variables["26"] = f"Avg: ${avg:.2f} | Min: ${min_price:.2f} | Max: ${max_price:.2f}"
    return variables


class WhatsAppSender:
    """Send price forecasts via WhatsApp using Twilio Content Templates (no SDK)."""

    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

    @property
    def available(self) -> bool:
        return bool(self.account_sid and self.auth_token)

    def send(
        self,
        to_number: str,
        date: str,
        hourly_prices: List[float],
        avg: float,
        min_price: float,
        max_price: float,
    ) -> tuple:
        """Send forecast to a single WhatsApp number via Content Template.
        Returns (success: bool, error: str|None)."""
        if not self.available:
            msg = "Twilio credentials not set"
            logger.warning(msg)
            return False, msg
        if not to_number:
            msg = "No destination number provided"
            logger.warning(msg)
            return False, msg

        raw = to_number.replace("whatsapp:", "").strip()
        raw = "".join(c for c in raw if c.isdigit() or c == "+")
        if not raw.startswith("+"):
            raw = "+" + raw
        to = f"whatsapp:{raw}"

        content_variables = format_forecast_variables(date, hourly_prices, avg, min_price, max_price)

        logger.info(f"Sending WhatsApp template from={self.from_number} to={to} date={date}")

        try:
            result = _post_template_message(
                self.account_sid,
                self.auth_token,
                self.from_number,
                to,
                DA_LMP_FORECAST_SID,
                content_variables,
            )
            logger.info(f"WhatsApp sent to {raw} (SID: {result.get('sid')})")
            return True, None
        except Exception as e:
            logger.error(f"WhatsApp send failed to {raw}: {e}")
            return False, str(e)

    def send_to_stepdad(
        self,
        date: str,
        hourly_prices: List[float],
        avg: float,
        min_price: float,
        max_price: float,
    ) -> tuple:
        number = os.getenv("STEPDAD_WHATSAPP", "")
        if not number:
            return False, "STEPDAD_WHATSAPP env var not set"
        return self.send(number, date, hourly_prices, avg, min_price, max_price)

    def send_to_you(
        self,
        date: str,
        hourly_prices: List[float],
        avg: float,
        min_price: float,
        max_price: float,
        metadata: Dict = None,
    ) -> tuple:
        number = os.getenv("YOUR_WHATSAPP", "")
        if not number:
            return True, None  # optional — not an error
        return self.send(number, date, hourly_prices, avg, min_price, max_price)

    @staticmethod
    def calculate_metadata(prices: List[float]) -> Dict:
        if not prices:
            return {}
        return {
            "avg": sum(prices) / len(prices),
            "min_price": min(prices),
            "max_price": max(prices),
            "peak_hour": prices.index(max(prices)) + 1,
            "low_hour": prices.index(min(prices)) + 1,
        }
