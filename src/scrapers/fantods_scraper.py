import requests
from bs4 import BeautifulSoup
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FantodsScraperr:
    """Scrape DA-LMP data from fantods website via its internal AJAX API"""

    AJAX_URL = "http://nyc3.fantods.co/socobufalo/miso/miso_node_history.php"

    @staticmethod
    def scrape_data():
        """
        Fetch DA-LMP prices from the fantods AJAX endpoint.

        The site shows a table of hourly prices for the past ~40 days.
        The last column ('M') is the rolling mean across those days, used
        as the base price when tomorrow's DA prices aren't posted yet (before
        ~4 PM the prior day).

        Returns:
            {
                'success': bool,
                'prices_by_hour': [24 hourly prices],
                'load_forecast': [],   # not available from this source
                'wind_forecast': [],   # not available from this source
            }
        """
        tomorrow = datetime.now() + timedelta(days=1)
        date_str = f"{tomorrow.year}-{tomorrow.month}-{tomorrow.day}"
        tomorrow_label = f"{tomorrow.month}/{tomorrow.day}"

        try:
            response = requests.get(
                FantodsScraperr.AJAX_URL,
                params={
                    "date": date_str,
                    "node": "ENERGY",
                    "price": "dalmp",
                    "color": "same",
                    "filter": "",
                    "filterNo": "",
                    "dow": "all",
                },
                timeout=15,
            )
            response.raise_for_status()

            # The response is a JSON-encoded HTML string
            html = json.loads(response.text)
            soup = BeautifulSoup(html, "html.parser")

            # Find the column index for tomorrow (or fall back to the mean 'M' column)
            thead = soup.find("thead")
            if not thead:
                return {"success": False, "error": "No thead in fantods response"}

            date_headers = [th.text.strip() for th in thead.find_all("tr")[-1].find_all("th")]

            if tomorrow_label in date_headers:
                col_idx = date_headers.index(tomorrow_label)
                logger.info(f"Using tomorrow's column ({tomorrow_label}) at index {col_idx}")
            elif "M" in date_headers:
                col_idx = date_headers.index("M")
                logger.info(f"Tomorrow not posted yet; using mean 'M' column at index {col_idx}")
            else:
                return {"success": False, "error": f"Neither {tomorrow_label} nor M column found"}

            # Parse each data row (rows that start with a <td>)
            prices = []
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                if len(cells) > col_idx:
                    try:
                        price = float(cells[col_idx].text.strip())
                        prices.append(price)
                    except (ValueError, AttributeError):
                        pass

            if len(prices) != 24:
                logger.warning(f"Expected 24 prices, got {len(prices)}")
                if not prices:
                    return {"success": False, "error": "No price data parsed"}

            logger.info(
                f"Fantods scrape OK: {len(prices)} prices, avg ${sum(prices)/len(prices):.2f}"
            )
            return {
                "success": True,
                "prices_by_hour": prices[:24],
                "load_forecast": [],
                "wind_forecast": [],
            }

        except Exception as e:
            logger.error(f"Error scraping fantods: {e}")
            return {"success": False, "error": str(e)}
