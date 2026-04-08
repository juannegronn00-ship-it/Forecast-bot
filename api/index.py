from http.server import BaseHTTPRequestHandler
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _authorized(headers) -> bool:
    """Allow all requests when CRON_SECRET is unset (dev/local).
    When set (Vercel injects it automatically), require the matching header."""
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        return True
    return headers.get("Authorization", "") == f"Bearer {secret}"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not _authorized(self.headers):
            self._respond(401, {"error": "Unauthorized"})
            return

        try:
            from src.main import ForecastBot
            from src.utils.whatsapp_sender import WhatsAppSender

            bot = ForecastBot()
            final_prices = bot.run_forecast()

            if not final_prices or len(final_prices) != 24:
                self._respond(500, {"success": False, "error": "Pipeline returned no prices"})
                return

            # Attempt WhatsApp send and report the outcome explicitly
            sender = WhatsAppSender()
            stepdad_ok, stepdad_err = sender.send_to_stepdad(final_prices)
            your_ok, your_err = sender.send_to_you(final_prices)

            whatsapp = {"stepdad": stepdad_ok, "monitor": your_ok}
            if stepdad_err:
                whatsapp["stepdad_error"] = stepdad_err
            if your_err:
                whatsapp["monitor_error"] = your_err

            self._respond(200, {
                "success": True,
                "whatsapp": whatsapp,
                "forecast": {
                    "hours": final_prices,
                    "avg": round(sum(final_prices) / 24, 2),
                    "min": round(min(final_prices), 2),
                    "max": round(max(final_prices), 2),
                },
            })

        except Exception as e:
            self._respond(500, {"success": False, "error": str(e)})

    def do_POST(self):
        self.do_GET()

    def _respond(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)
