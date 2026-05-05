"""
DA-LMP base price fetcher.

Primary:  fantods.co AJAX endpoint (40-day rolling mean by hour)
Fallback: MISO DA ExAnte LMP CSV for ILLINOIS.HUB (today's cleared DA prices)

The fantods server is unstable — the MISO CSV fallback provides real
market-cleared data which is actually a better base for forecasting.
"""
import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime, timedelta, date

logger = logging.getLogger(__name__)

_MISO_CSV_BASE = "https://docs.misoenergy.org/marketreports"
_MISO_HEADERS  = {"User-Agent": "forecast-bot/2.0 (contact@example.com)"}


class FantodsScraperr:
    """Scrape DA-LMP prices from fantods, falling back to MISO DA LMP CSV."""

    AJAX_URL = "http://nyc3.fantods.co/socobufalo/miso/miso_node_history.php"

    @staticmethod
    def scrape_data():
        """
        Fetch DA-LMP prices.  Tries fantods first (40-day mean), then falls
        back to today's MISO DA ExAnte LMP for ILLINOIS.HUB.

        Returns:
            {
                'success': bool,
                'prices_by_hour': [24 hourly prices],
                'load_forecast': [],
                'wind_forecast': [],
                'source': str,
            }
        """
        # ── 1. Try fantods ────────────────────────────────────────────────
        result = FantodsScraperr._try_fantods()
        if result["success"]:
            return result

        # ── 2. Fall back to MISO DA LMP CSV ──────────────────────────────
        logger.warning("Fantods unavailable — trying MISO DA ExAnte LMP CSV")
        result = FantodsScraperr._try_miso_csv()
        if result["success"]:
            return result

        return {"success": False, "error": "All base price sources failed",
                "prices_by_hour": [], "load_forecast": [], "wind_forecast": []}

    # ------------------------------------------------------------------ #

    @staticmethod
    def _try_fantods():
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = f"{tomorrow.year}-{tomorrow.month}-{tomorrow.day}"
        tomorrow_label = f"{tomorrow.month}/{tomorrow.day}"
        try:
            response = requests.get(
                FantodsScraperr.AJAX_URL,
                params={
                    "date": date_str, "node": "ENERGY", "price": "dalmp",
                    "color": "same", "filter": "", "filterNo": "", "dow": "all",
                },
                timeout=12,
            )
            response.raise_for_status()
            html = json.loads(response.text)
            soup = BeautifulSoup(html, "html.parser")

            thead = soup.find("thead")
            if not thead:
                return {"success": False, "error": "No thead"}

            date_headers = [th.text.strip() for th in thead.find_all("tr")[-1].find_all("th")]

            if tomorrow_label in date_headers:
                col_idx = date_headers.index(tomorrow_label)
            elif "M" in date_headers:
                col_idx = date_headers.index("M")
            else:
                return {"success": False, "error": "No usable column"}

            prices = []
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if not cells or len(cells) <= col_idx:
                    continue
                try:
                    prices.append(float(cells[col_idx].text.strip()))
                except (ValueError, AttributeError):
                    pass

            if len(prices) < 24:
                return {"success": False, "error": f"Only {len(prices)} prices"}

            logger.info(f"Fantods OK: {len(prices)} prices avg=${sum(prices)/len(prices):.2f}")
            return {"success": True, "prices_by_hour": prices[:24],
                    "load_forecast": [], "wind_forecast": [], "source": "fantods"}

        except Exception as e:
            logger.warning(f"Fantods failed: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _try_miso_csv():
        """
        Fetch today's MISO DA ExAnte LMP CSV and extract ILLINOIS.HUB prices.
        These are actual market-cleared DA prices — a solid base for tomorrow's forecast.
        """
        today_str = date.today().strftime("%Y%m%d")
        url = f"{_MISO_CSV_BASE}/{today_str}_da_exante_lmp.csv"
        try:
            resp = requests.get(url, headers=_MISO_HEADERS, timeout=15)
            resp.raise_for_status()

            prices = _parse_illinois_hub(resp.text)
            if len(prices) != 24:
                return {"success": False, "error": f"Illinois Hub parse returned {len(prices)} prices"}

            avg = sum(prices) / 24
            logger.info(f"MISO DA LMP CSV OK: ILLINOIS.HUB avg=${avg:.2f} max=${max(prices):.2f}")
            return {
                "success": True,
                "prices_by_hour": prices,
                "load_forecast": [],
                "wind_forecast": [],
                "source": "miso_da_csv",
            }
        except Exception as e:
            logger.warning(f"MISO DA CSV failed: {e}")
            return {"success": False, "error": str(e)}


def _parse_illinois_hub(csv_text: str):
    """Extract 24 hourly prices for ILLINOIS.HUB from the MISO DA ExAnte CSV."""
    for line in csv_text.split("\n"):
        if line.startswith("ILLINOIS.HUB,Hub,LMP"):
            parts = line.split(",")
            prices = []
            for v in parts[3:27]:
                v = v.strip()
                if v:
                    try:
                        prices.append(float(v))
                    except ValueError:
                        pass
            if len(prices) == 24:
                return prices
    return []
