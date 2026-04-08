from http.server import BaseHTTPRequestHandler
import sys
import os
import json

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from src.main import ForecastBot
            bot = ForecastBot()
            final_prices = bot.run_forecast()

            if final_prices and len(final_prices) == 24:
                status = 200
                body = json.dumps({
                    "success": True,
                    "hours": final_prices,
                    "avg": round(sum(final_prices) / 24, 2),
                    "min": round(min(final_prices), 2),
                    "max": round(max(final_prices), 2),
                }).encode()
                content_type = "application/json"
            else:
                status = 500
                body = json.dumps({"success": False, "error": "Pipeline returned no prices"}).encode()
                content_type = "application/json"

        except Exception as e:
            status = 500
            body = json.dumps({"success": False, "error": str(e)}).encode()
            content_type = "application/json"

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.do_GET()
