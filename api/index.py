from datetime import datetime, date, timedelta
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
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        return True
    return headers.get("Authorization", "") == f"Bearer {secret}"


def _today_flag_path() -> str:
    return f"/tmp/forecast_sent_{date.today().strftime('%Y-%m-%d')}.flag"


def _already_sent_today() -> bool:
    return os.path.exists(_today_flag_path())


def _mark_sent_today() -> None:
    try:
        with open(_today_flag_path(), "w") as f:
            f.write(datetime.utcnow().isoformat())
    except Exception:
        pass


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if not _authorized(self.headers):
            self._respond(401, {"error": "Unauthorized"})
            return

        # Guard: check BEFORE running any pipeline work
        if _already_sent_today():
            flag = _today_flag_path()
            logger.info(f"⏭  Already sent today ({flag}) — returning 200 without re-running.")
            self._respond(200, {"success": True, "skipped": True, "reason": "already_sent_today"})
            return

        logger.info("=" * 60)
        logger.info(f"CRON TRIGGERED at {datetime.utcnow().isoformat()} UTC  pid={os.getpid()}")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        logger.info(f"Generating forecast for: {tomorrow}")
        logger.info("=" * 60)

        try:
            from src.main import ForecastBot

            bot = ForecastBot()

            final_prices = bot.run_forecast()

            if not final_prices or len(final_prices) != 24:
                msg = f"Pipeline returned {len(final_prices) if final_prices else 0} prices (expected 24)"
                logger.error(f"❌ {msg}")
                self._respond(500, {"success": False, "error": msg})
                return

            avg = sum(final_prices) / len(final_prices)
            logger.info(f"Pipeline complete | avg=${avg:.2f} | range=${min(final_prices):.2f}–${max(final_prices):.2f}")

            logger.info(f"📤 SEND START at {datetime.utcnow().isoformat()} UTC")
            tg_result = bot.send_telegram(final_prices)
            logger.info(f"📤 SEND END   at {datetime.utcnow().isoformat()} UTC")
            stepdad_ok = tg_result.get("stepdad_ok", False)
            you_ok = tg_result.get("you_ok", False)
            tg_errors = tg_result.get("errors", [])

            # Mark sent here too — covers the API path (send_telegram marks it
            # in the local-run path; both paths write the same flag file)
            if stepdad_ok:
                _mark_sent_today()
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
