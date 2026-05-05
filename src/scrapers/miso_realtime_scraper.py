"""
MISO real-time LMP proxy via today's DA ExAnte LMP CSV.

MISO publishes today's DA clearing prices at docs.misoenergy.org.
We extract ILLINOIS.HUB (the primary MISO price signal for Illinois load zone)
and use the current hour's cleared DA price as a real-time proxy.

Trend is derived by comparing the current hour to the prior two hours —
if the DA curve is rising into the current hour, RT is typically following it.
"""
import logging
import requests
from datetime import date, datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE_URL  = "https://docs.misoenergy.org/marketreports"
_HEADERS   = {"User-Agent": "forecast-bot/2.0 (contact@example.com)"}
_HUB_PREF  = ["ILLINOIS.HUB", "INDIANA.HUB", "MINN.HUB"]   # preference order


class MISORealtimeScraper:
    """
    Fetch MISO 'RT LMP' proxy from today's DA ExAnte LMP CSV.

    Returns the Illinois Hub cleared DA price for the current hour as
    rt_lmp_current, plus a trend derived from the DA price curve shape.
    """

    def fetch_data(self) -> Dict:
        """
        Returns:
          {
            'success': bool,
            'rt_lmp_current': float | None,
            'rt_lmp_trend': 'rising' | 'falling' | 'flat',
            'illinois_hub_24h': [24 floats] | [],   # full day for context
          }
        """
        try:
            today_str = date.today().strftime("%Y%m%d")
            url = f"{_BASE_URL}/{today_str}_da_exante_lmp.csv"
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            resp.raise_for_status()

            hub_prices = _parse_hub_lmp(resp.text)
            if not hub_prices:
                logger.warning("MISO DA CSV: no hub node prices found")
                return _fail()

            # Current hour (1-indexed, capped to data length)
            current_hour = min(datetime.now().hour + 1, len(hub_prices))
            rt_lmp = hub_prices[current_hour - 1]

            # Trend: compare current hour to hour-2 (2 hours back on the DA curve)
            trend = _trend(hub_prices, current_hour)

            avg = sum(hub_prices) / len(hub_prices)
            logger.info(
                f"MISO DA LMP (Illinois Hub): hr{current_hour}=${rt_lmp:.2f}/MWh "
                f"trend={trend}  day_avg=${avg:.2f}"
            )
            return {
                "success":           True,
                "rt_lmp_current":    round(rt_lmp, 2),
                "rt_lmp_trend":      trend,
                "illinois_hub_24h":  [round(p, 2) for p in hub_prices],
            }

        except Exception as e:
            logger.warning(f"MISO RT proxy failed: {e}")
            return _fail()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_hub_lmp(csv_text: str) -> List[float]:
    """
    Parse the MISO DA ExAnte LMP CSV.
    Format: Node,Type,Value,HE 1,HE 2,...,HE 24
    Returns 24-hour prices for the best available hub.
    """
    for target in _HUB_PREF:
        for line in csv_text.split("\n"):
            if line.startswith(f"{target},Hub,LMP"):
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


def _trend(prices: List[float], current_hour: int) -> str:
    """Rising/falling/flat based on the DA price curve around the current hour."""
    if current_hour < 3 or len(prices) < current_hour:
        return "flat"
    now   = prices[current_hour - 1]
    prev2 = prices[current_hour - 3]   # 2 hours ago
    delta = now - prev2
    if delta > 2.5:
        return "rising"
    if delta < -2.5:
        return "falling"
    return "flat"


def _fail() -> Dict:
    return {"success": False, "rt_lmp_current": None, "rt_lmp_trend": "flat", "illinois_hub_24h": []}
