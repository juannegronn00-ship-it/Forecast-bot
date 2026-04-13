"""
Telegram delivery for DA-LMP forecasts.
Uses the Bot API sendMessage endpoint — no approval, no sessions, free forever.
"""
import logging
import os
import requests
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org/bot{token}/{method}"

# TOU labels for the message — makes it readable at a glance
_PERIOD = {
    **{h: "🌙 Night" for h in range(1, 7)},
    **{h: "🌅 Morn " for h in range(7, 10)},
    **{h: "☀️  Mid  " for h in range(10, 16)},
    **{h: "🌤  Shldr" for h in range(16, 20)},
    **{h: "🌆 Eve  " for h in range(20, 25)},
}


def _api(token: str, method: str, payload: dict) -> dict:
    url = _API_BASE.format(token=token, method=method)
    resp = requests.post(url, json=payload, timeout=15)
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {data.get('description')} (code {data.get('error_code')})")
    return data


def _escape(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    for ch in r"\_*[]()~`>#+-=|{}.!":
        text = text.replace(ch, f"\\{ch}")
    return text


class TelegramSender:
    """
    Send DA-LMP forecasts via Telegram Bot API.

    Reads from env:
      TELEGRAM_BOT_TOKEN        — bot token from @BotFather
      TELEGRAM_STEPDAD_CHAT_ID  — chat ID of the primary recipient
      TELEGRAM_YOUR_CHAT_ID     — your own chat ID (monitoring)
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.stepdad_chat_id = os.getenv("TELEGRAM_STEPDAD_CHAT_ID", "")
        self.your_chat_id = os.getenv("TELEGRAM_YOUR_CHAT_ID", "")
        self._sent_this_run: set = set()  # dedup guard: track chat_ids sent in this process

    @property
    def available(self) -> bool:
        return bool(self.token)

    # ------------------------------------------------------------------ #
    # Message formatting
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_message(date_str: str, prices: List[float]) -> str:
        """
        Build a clean, readable forecast message.

        Example:
          ⚡ DA-LMP FORECAST · April 13, 2026

          Hr  Period    Price
          ─────────────────────
          01  🌙 Night  $21.00
          ...
          24  🌆 Eve    $27.00

          📊 Avg $32.58 · Min $19.00 · Max $71.00
          📈 Peak: Hour 20 ($71.00)  📉 Low: Hour 2 ($19.00)
        """
        avg = sum(prices) / len(prices)
        mn = min(prices)
        mx = max(prices)
        peak_h = prices.index(mx) + 1
        low_h = prices.index(mn) + 1

        lines = [
            f"⚡ <b>DA-LMP FORECAST · {date_str}</b>",
            "",
            "<pre>Hr  Period     Price</pre>",
            "<pre>─────────────────────</pre>",
        ]

        # Group into periods with a blank separator between them
        prev_period = None
        for h, price in enumerate(prices, 1):
            period = _PERIOD.get(h, "")
            period_key = period.strip()
            if prev_period and period_key != prev_period:
                lines.append("<pre></pre>")
            prev_period = period_key
            marker = " ◀ peak" if h == peak_h else (" ◀ low " if h == low_h else "")
            lines.append(f"<pre>{h:02d}  {period}  ${price:>6.2f}{marker}</pre>")

        lines += [
            "",
            f"📊 <b>Avg</b> ${avg:.2f} · <b>Min</b> ${mn:.2f} · <b>Max</b> ${mx:.2f}",
            f"📈 Peak: Hour {peak_h:02d} (${mx:.2f})  📉 Low: Hour {low_h:02d} (${mn:.2f})",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Send helpers
    # ------------------------------------------------------------------ #
    def _send_once(self, chat_id: str, message: str, label: str) -> tuple:
        """
        Send to a single chat ID, skipping if already sent to this ID in the
        current process (prevents double-sends from retries or misconfiguration).
        Returns (success, error_or_None).
        """
        if not self.available:
            return False, "TELEGRAM_BOT_TOKEN not set"
        if not chat_id:
            return False, f"{label} chat_id not configured"
        if chat_id in self._sent_this_run:
            logger.warning(f"⚠️  Skipping duplicate send to {label} (chat_id={chat_id} already sent this run)")
            return True, None   # treat as success — message already delivered
        try:
            _api(self.token, "sendMessage", {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
            })
            self._sent_this_run.add(chat_id)
            logger.info(f"Telegram sent to {label} (chat_id={chat_id})")
            return True, None
        except Exception as e:
            logger.error(f"Telegram send to {label} ({chat_id}) failed: {e}")
            return False, str(e)

    def send_to_stepdad(self, message: str) -> tuple:
        """Send to TELEGRAM_STEPDAD_CHAT_ID. Returns (success, error_or_None)."""
        return self._send_once(self.stepdad_chat_id, message, "stepdad")

    def send_to_you(self, message: str) -> tuple:
        """Send to TELEGRAM_YOUR_CHAT_ID. Returns (success, error_or_None)."""
        return self._send_once(self.your_chat_id, message, "monitor")

    def send_forecast(self, date_str: str, prices: List[float]) -> Dict:
        """
        Send forecast to stepdad then to monitoring.
        Each recipient receives exactly one message per run — duplicate
        chat IDs or retry calls are silently skipped by _send_once().
        Returns {stepdad_ok, you_ok, errors}.
        """
        message = self.format_message(date_str, prices)

        stepdad_ok, stepdad_err = self.send_to_stepdad(message)
        if stepdad_ok:
            logger.info("✅ Telegram forecast sent to stepdad")
        else:
            logger.error(f"❌ Telegram to stepdad FAILED: {stepdad_err}")

        you_ok, you_err = self.send_to_you(message)
        if you_ok:
            logger.info("✅ Telegram forecast sent to monitoring")
        else:
            logger.warning(f"⚠️  Telegram to monitoring failed: {you_err}")

        errors = [e for e in [
            f"stepdad: {stepdad_err}" if stepdad_err else None,
            f"monitor: {you_err}" if you_err else None,
        ] if e]

        return {"stepdad_ok": stepdad_ok, "you_ok": you_ok, "errors": errors}

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
