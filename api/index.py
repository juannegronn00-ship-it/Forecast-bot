from http.server import BaseHTTPRequestHandler
import sys
import os
import json

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _authorized(headers) -> bool:
    """Allow if no CRON_SECRET is set (dev/testing), or if the header matches."""
    secret = os.getenv('CRON_SECRET', '')
    if not secret:
        return True
    auth = headers.get('Authorization', '')
    return auth == f'Bearer {secret}'


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not _authorized(self.headers):
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Unauthorized'}).encode())
            return

        try:
            from src.main import ForecastBot
            bot = ForecastBot()
            final_prices = bot.run_forecast()

            if final_prices and len(final_prices) == 24:
                status = 200
                body = json.dumps({
                    'success': True,
                    'hours': final_prices,
                    'avg': round(sum(final_prices) / 24, 2),
                    'min': round(min(final_prices), 2),
                    'max': round(max(final_prices), 2),
                }).encode()
            else:
                status = 500
                body = json.dumps({'success': False, 'error': 'Pipeline returned no prices'}).encode()

        except Exception as e:
            status = 500
            body = json.dumps({'success': False, 'error': str(e)}).encode()

        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.do_GET()
