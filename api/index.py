from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler
import sys
import os
import json
import logging
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _authorized(headers) -> bool:
    """Allow all requests when CRON_SECRET is unset (dev/local).
    When set, Vercel injects it automatically — require matching header."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        return True
    return headers.get("Authorization", "") == f"Bearer {secret}"


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if not _authorized(self.headers):
            self._respond(401, {"error": "Unauthorized"})
            return

        logger.info("=" * 60)
        logger.info(f"CRON TRIGGERED at {datetime.utcnow().isoformat()} UTC  pid={os.getpid()}")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        logger.info(f"Generating forecast for: {tomorrow}")
        logger.info("=" * 60)

        try:
            from src.main import ForecastBot

            bot = ForecastBot()

            # Step 1: run the forecast pipeline (returns prices, does NOT send WhatsApp)
            final_prices = bot.run_forecast()

            if not final_prices or len(final_prices) != 24:
                msg = f"Pipeline returned {len(final_prices) if final_prices else 0} prices (expected 24)"
                logger.error(f"❌ {msg}")
                self._respond(500, {"success": False, "error": msg})
                return

            avg = sum(final_prices) / len(final_prices)
            logger.info(f"Pipeline complete | avg=${avg:.2f} | range=${min(final_prices):.2f}–${max(final_prices):.2f}")

            # Step 2: send via Telegram
            logger.info(f"📤 SEND START at {datetime.utcnow().isoformat()} UTC")
            tg_result = bot.send_telegram(final_prices)
            logger.info(f"📤 SEND END   at {datetime.utcnow().isoformat()} UTC")
            stepdad_ok = tg_result.get("stepdad_ok", False)
            you_ok = tg_result.get("you_ok", False)
            tg_errors = tg_result.get("errors", [])

            if stepdad_ok:
                logger.info(f"✅ Telegram delivered to stepdad for {tomorrow}")
            else:
                logger.error(f"❌ Telegram to stepdad FAILED. Errors: {tg_errors}")

            self._respond(200, {
                "success": True,
                "forecast_date": tomorrow,
                "telegram": {
                    "stepdad": stepdad_ok,
                    "monitor": you_ok,
                    "errors": tg_errors,
                },
                "forecast": {
                    "hours": final_prices,
                    "avg": round(avg, 2),
                    "min": round(min(final_prices), 2),
                    "max": round(max(final_prices), 2),
                },
            })

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ PIPELINE EXCEPTION: {e}\n{tb}")
            self._respond(500, {"success": False, "error": str(e), "traceback": tb})

    def do_POST(self):
        self.do_GET()

    def _respond(self, status: int, body: dict):
        payload = json.dumps(body, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        # Route BaseHTTPRequestHandler logs through our logger
        logger.info(fmt % args)
