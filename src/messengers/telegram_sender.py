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
_SENT_FLAG_DIR = "/tmp"   # Vercel /tmp persists within a warm instance
_REDIS_TTL_SECONDS = 93600  # 26 hours — covers any same-day cold-start retry

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

        # Log the routing configuration on every init so Vercel logs show exactly
        # which chat IDs are wired up (helps diagnose routing bugs).
        logger.info(
            f"TelegramSender init: stepdad_chat_id={self.stepdad_chat_id!r} "
            f"your_chat_id={self.your_chat_id!r} "
            f"token_set={bool(self.token)}"
        )
        if self.stepdad_chat_id == self.your_chat_id and self.stepdad_chat_id:
            logger.warning(
                "⚠️  ROUTING MISMATCH: TELEGRAM_STEPDAD_CHAT_ID == TELEGRAM_YOUR_CHAT_ID "
                f"({self.stepdad_chat_id}) — stepdad and monitor point to the same chat!"
            )

    @property
    def available(self) -> bool:
        return bool(self.token)

    # ------------------------------------------------------------------ #
    # Message formatting
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_message(
        date_str: str,
        prices: List[float],
        signal_summary: str = "",
        peak_driver: str = "",
        risk_flags: str = "",
    ) -> str:
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

          📡 Signal Summary: ...
          ⚠️ Risk: ...
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

        if signal_summary:
            lines += ["", f"📡 <b>Signal Summary:</b> {signal_summary}"]
        if peak_driver:
            lines += [f"📌 <b>Peak Driver:</b> {peak_driver}"]
        if risk_flags:
            lines += [f"⚠️ <b>Risk:</b> {risk_flags}"]

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

    # ------------------------------------------------------------------ #
    # Persistent idempotency (Upstash Redis — survives cold starts)
    # ------------------------------------------------------------------ #
    # Set UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN in Vercel env
    # to enable cross-invocation duplicate protection.  When not set, the
    # bot falls back to the /tmp flag (works within a single warm instance).
    # Get a free Upstash Redis at https://upstash.com — takes ~2 minutes.

    @staticmethod
    def _redis_key(date_str: str) -> str:
        safe = date_str.replace(" ", "_").replace(",", "")
        return f"dalmp_sent:{safe}"

    def _redis_check(self, date_str: str) -> bool:
        """Return True if Upstash Redis shows this date was already sent."""
        url   = os.getenv("UPSTASH_REDIS_REST_URL", "")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        if not (url and token):
            return False
        try:
            resp = requests.get(
                f"{url.rstrip('/')}/get/{self._redis_key(date_str)}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            return resp.json().get("result") is not None
        except Exception as e:
            logger.warning(f"Redis idempotency check failed (non-critical): {e}")
            return False

    def _redis_mark(self, date_str: str) -> None:
        """Write sent flag to Upstash Redis with a 26-hour TTL."""
        url   = os.getenv("UPSTASH_REDIS_REST_URL", "")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")
        if not (url and token):
            return
        try:
            requests.get(
                f"{url.rstrip('/')}/set/{self._redis_key(date_str)}/1/ex/{_REDIS_TTL_SECONDS}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )
            logger.info(f"Redis idempotency flag set for '{date_str}' (TTL {_REDIS_TTL_SECONDS}s)")
        except Exception as e:
            logger.warning(f"Redis idempotency mark failed (non-critical): {e}")

    # ------------------------------------------------------------------ #
    # /tmp flag (fast secondary guard within a warm instance)
    # ------------------------------------------------------------------ #

    def _daily_flag_path(self, date_str: str) -> str:
        """Path of the daily idempotency flag file."""
        safe = date_str.replace(" ", "_").replace(",", "")
        return os.path.join(_SENT_FLAG_DIR, f"telegram_sent_{safe}.flag")

    def _already_sent_today(self, date_str: str) -> bool:
        """
        Return True if forecast for this date was already sent.
        Checks two layers in order:
          1. Upstash Redis  — persistent across cold starts (requires env vars)
          2. /tmp flag file — fast guard within a single warm Lambda instance
        """
        # Layer 1: Redis (survives cold starts — the real fix for Vercel duplicates)
        if self._redis_check(date_str):
            logger.warning(
                f"⚠️  Redis send-guard: forecast for '{date_str}' already sent this day. "
                "Skipping duplicate (cold-start protected)."
            )
            return True
        # Layer 2: /tmp flag (fast path within same warm instance)
        flag = self._daily_flag_path(date_str)
        if os.path.exists(flag):
            logger.warning(
                f"⚠️  /tmp send-guard: forecast for '{date_str}' already sent "
                f"this instance (flag={flag}). Skipping duplicate send."
            )
            return True
        return False

    def _mark_sent_today(self, date_str: str) -> None:
        """Write idempotency flags to both Redis and /tmp."""
        self._redis_mark(date_str)  # persistent — survives cold starts
        try:
            with open(self._daily_flag_path(date_str), "w") as f:
                import datetime as _dt
                f.write(_dt.datetime.utcnow().isoformat())
        except Exception:
            pass  # /tmp write failure is non-fatal

    def send_forecast(
        self,
        date_str: str,
        prices: List[float],
        signal_summary: str = "",
        peak_driver: str = "",
        risk_flags: str = "",
    ) -> Dict:
        """
        Send forecast to stepdad then to monitoring.
        Caller is responsible for the daily dedup guard — this method always sends.
        Returns {stepdad_ok, you_ok, errors}.
        """
        message = self.format_message(
            date_str, prices,
            signal_summary=signal_summary,
            peak_driver=peak_driver,
            risk_flags=risk_flags,
        )

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
