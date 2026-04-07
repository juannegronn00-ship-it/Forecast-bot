from http.server import BaseHTTPRequestHandler
import sys
import os

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            from src.main import ForecastBot
            bot = ForecastBot()
            success = bot.run_forecast()
            status = 200 if success else 500
            body = b"Forecast sent successfully" if success else b"Forecast failed"
        except Exception as e:
            status = 500
            body = f"Error: {e}".encode()

        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.do_GET()
