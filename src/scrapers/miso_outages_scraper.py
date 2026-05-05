"""
MISO unplanned outages scraper.
Sums MW of unplanned outages and maps to an alert level.

MISO total installed capacity is ~180 GW.  Thresholds:
  normal   < 25,000 MW offline  (routine maintenance window)
  elevated  25,000–40,000 MW    (tighter supply margin)
  high     > 40,000 MW          (material supply impact, spike risk)
"""
import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)

_OUTAGES_URL = (
    "https://api.misoenergy.org/MISORTWDDataBroker/DataBrokerServices.asmx"
    "?messageType=getUnplannedOutages&returnType=json"
)
_HEADERS = {"User-Agent": "forecast-bot/2.0 (contact@example.com)"}

_ELEVATED_MW = 25_000
_HIGH_MW     = 40_000


class MISOOutagesScraper:
    """Fetch MISO unplanned outage total and compute a market alert level."""

    def fetch_data(self) -> Dict:
        """
        Returns:
          {
            'success': bool,
            'outage_mw': float,
            'alert_level': 'normal' | 'elevated' | 'high',
          }
        """
        try:
            resp = requests.get(_OUTAGES_URL, headers=_HEADERS, timeout=15)
            resp.raise_for_status()
            raw = resp.json()

            outage_mw = _sum_outage_mw(raw)
            level = _alert_level(outage_mw)
            logger.info(f"MISO outages: {outage_mw:,.0f} MW offline | alert_level={level}")
            return {
                "success": True,
                "outage_mw": round(outage_mw),
                "alert_level": level,
            }

        except Exception as e:
            logger.warning(f"MISO outages scrape failed: {e}")
            return {"success": False, "outage_mw": 0, "alert_level": "normal"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sum_outage_mw(data) -> float:
    """Walk the API response structure and sum all MW values."""
    items: List[dict] = []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("OutageInfo", "Outages", "outages", "items", "data", "rows", "Rows"):
            val = data.get(key)
            if isinstance(val, list):
                items = val
                break
            if isinstance(val, dict):
                for k2 in ("Outages", "outages", "items", "data", "Rows", "Row"):
                    v2 = val.get(k2)
                    if isinstance(v2, list):
                        items = v2
                        break
                if items:
                    break

    total = 0.0
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("MW", "mw", "Mw", "OutageMW", "outageMW", "capacity", "Capacity", "megawatts"):
            val = item.get(key)
            if val is not None:
                try:
                    total += float(val)
                    break
                except (ValueError, TypeError):
                    pass

    return total


def _alert_level(mw: float) -> str:
    if mw >= _HIGH_MW:
        return "high"
    if mw >= _ELEVATED_MW:
        return "elevated"
    return "normal"
