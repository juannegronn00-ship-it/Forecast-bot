import requests
import csv
from io import StringIO
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MISOScraper:
    """Fetch load and wind forecast data from MISO official market reports."""

    # MISO publishes daily market report CSVs at docs.misoenergy.org/marketreports/
    # DA load forecast: {YYYYMMDD}_da_load_f.csv
    # Wind actual/forecast: {YYYYMMDD}_wind_actual_movers.csv
    BASE_URL = "https://docs.misoenergy.org/marketreports"

    @staticmethod
    def _build_urls(date: datetime):
        date_str = date.strftime("%Y%m%d")
        return {
            "load": [
                f"{MISOScraper.BASE_URL}/{date_str}_da_load_f.csv",
                f"https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/{date_str}_da_load_f.csv",
            ],
            "wind": [
                f"{MISOScraper.BASE_URL}/{date_str}_wind_actual_movers.csv",
                f"https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/{date_str}_wind_actual_movers.csv",
            ],
        }

    @staticmethod
    def _fetch_csv(urls: list) -> list:
        """Try each URL in order; return parsed rows from the first that succeeds."""
        for url in urls:
            try:
                logger.debug(f"Trying MISO URL: {url}")
                resp = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                reader = csv.DictReader(StringIO(resp.text))
                rows = list(reader)
                if rows:
                    logger.info(f"MISO CSV fetched: {url} ({len(rows)} rows)")
                    return rows
            except Exception as e:
                logger.debug(f"  Failed ({url}): {e}")
        return []

    @staticmethod
    def _extract_numeric_column(rows: list, keywords: list) -> list:
        """
        Search column headers for any keyword match (case-insensitive) and
        extract numeric values.  Returns up to 24 values.
        """
        if not rows:
            return []
        headers = list(rows[0].keys())
        target_col = None
        for kw in keywords:
            for h in headers:
                if kw.lower() in h.lower():
                    target_col = h
                    break
            if target_col:
                break

        if not target_col:
            logger.debug(f"No column matched keywords {keywords} in {headers[:8]}")
            return []

        values = []
        for row in rows:
            raw = row.get(target_col, "").strip()
            try:
                values.append(float(raw.replace(",", "")))
            except (ValueError, AttributeError):
                pass
        return values[:24]

    @staticmethod
    def fetch_data():
        """
        Fetch MISO DA load and wind forecasts for today (as proxy for tomorrow's shape).

        Returns:
            {
                'success': bool,
                'load_forecast': [up to 24 hourly values in GW],
                'wind_forecast': [up to 24 hourly values in GW],
            }
        """
        today = datetime.now()
        urls = MISOScraper._build_urls(today)

        load_values = []
        wind_values = []

        # --- Load forecast ---
        load_rows = MISOScraper._fetch_csv(urls["load"])
        if load_rows:
            load_values = MISOScraper._extract_numeric_column(
                load_rows, ["load", "demand", "mw", "forecast"]
            )
            if load_values:
                # MISO reports are in MW; convert to GW for consistency
                if max(load_values) > 1000:
                    load_values = [v / 1000 for v in load_values]
                logger.info(f"MISO load forecast: {len(load_values)} hours, avg={sum(load_values)/len(load_values):.1f} GW")

        # --- Wind forecast ---
        wind_rows = MISOScraper._fetch_csv(urls["wind"])
        if wind_rows:
            wind_values = MISOScraper._extract_numeric_column(
                wind_rows, ["wind", "generation", "forecast", "mw"]
            )
            if wind_values:
                if max(wind_values) > 100:
                    wind_values = [v / 1000 for v in wind_values]
                logger.info(f"MISO wind forecast: {len(wind_values)} hours, avg={sum(wind_values)/len(wind_values):.1f} GW")

        success = bool(load_values or wind_values)
        if not success:
            logger.warning("MISO: could not fetch load or wind data from any URL")

        return {
            "success": success,
            "load_forecast": load_values[:24],
            "wind_forecast": wind_values[:24],
        }
