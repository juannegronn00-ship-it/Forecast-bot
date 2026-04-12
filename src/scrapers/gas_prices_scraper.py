"""
Natural gas price data for MISO DA-LMP forecasting.

Henry Hub is the NYMEX settlement hub. Its spot price directly determines
the short-run marginal cost of gas-fired generation — typically the marginal
generator in MISO during most hours. When gas is cheap, LMPs stay low;
when gas spikes, LMPs follow.

Primary: EIA Open Data API v2 (free key at https://www.eia.gov/opendata/)
  Set EIA_API_KEY in your .env to enable.
Fallback: Reasoned estimate based on recent seasonal norms.
"""
import logging
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Henry Hub series in EIA v2 API
_EIA_SERIES = "NG.RNGWHHD.D"   # Henry Hub Natural Gas Spot Price, $/MMBtu, daily

# Seasonal norms ($/MMBtu) — used as fallback when EIA API unavailable
# These are approximate rolling averages based on historical patterns
_SEASONAL_DEFAULTS = {
    1: 3.20,   # Jan — winter heating season
    2: 3.00,   # Feb
    3: 2.60,   # Mar — shoulder season begins
    4: 2.20,   # Apr — low shoulder
    5: 2.30,   # May
    6: 2.80,   # Jun — cooling season begins
    7: 3.10,   # Jul — summer peak
    8: 3.00,   # Aug
    9: 2.70,   # Sep — shoulder
    10: 2.60,  # Oct
    11: 3.00,  # Nov — winter approaches
    12: 3.40,  # Dec
}


class GasPricesScraper:
    """Fetch Henry Hub natural gas spot price — key MISO LMP driver."""

    def __init__(self):
        self._eia_key = os.getenv("EIA_API_KEY", "")

    def fetch_data(self) -> Dict:
        """
        Returns:
        {
          'success': bool,
          'source': str,
          'price': float,          # $/MMBtu Henry Hub spot
          'price_level': str,      # 'very_low' | 'low' | 'moderate' | 'elevated' | 'high' | 'spike'
          'trading_signal': str,   # one-line trader interpretation
          'lmp_impact_pct': float, # estimated % impact on MISO LMP vs baseline
        }
        """
        if self._eia_key:
            result = self._fetch_eia()
            if result["success"]:
                return result

        return self._seasonal_estimate()

    # ------------------------------------------------------------------ #
    # EIA Open Data API v2
    # ------------------------------------------------------------------ #
    def _fetch_eia(self) -> Dict:
        try:
            resp = requests.get(
                "https://api.eia.gov/v2/natural-gas/pri/fut/data/",
                params={
                    "api_key": self._eia_key,
                    "frequency": "daily",
                    "data[0]": "value",
                    "facets[series][]": _EIA_SERIES,
                    "sort[0][column]": "period",
                    "sort[0][direction]": "desc",
                    "length": 5,
                },
                timeout=12,
            )
            resp.raise_for_status()
            records = resp.json().get("response", {}).get("data", [])

            # Find the most recent record with a valid price
            for rec in records:
                raw = rec.get("value")
                if raw is not None:
                    try:
                        price = float(raw)
                        period = rec.get("period", "unknown")
                        result = _build_result(price, f"eia ({period})")
                        logger.info(f"EIA gas price: ${price:.2f}/MMBtu ({period})")
                        return result
                    except (ValueError, TypeError):
                        continue

            logger.warning("EIA returned no valid price records")
            return {"success": False, "error": "no valid EIA records"}

        except Exception as e:
            logger.warning(f"EIA API failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Seasonal estimate fallback
    # ------------------------------------------------------------------ #
    def _seasonal_estimate(self) -> Dict:
        month = datetime.now().month
        price = _SEASONAL_DEFAULTS.get(month, 2.50)
        result = _build_result(price, "seasonal_estimate")
        logger.info(f"Gas price (seasonal estimate, no EIA key): ${price:.2f}/MMBtu")
        return result


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _build_result(price: float, source: str) -> Dict:
    level, signal, lmp_impact = _classify_price(price)
    return {
        "success": True,
        "source": source,
        "price": price,
        "price_level": level,
        "trading_signal": signal,
        "lmp_impact_pct": lmp_impact,
    }


def _classify_price(price: float) -> tuple:
    """Return (level, trading_signal, lmp_impact_pct)."""
    # Each $1/MMBtu change in gas ≈ ~$8-10/MWh change in LMP (heat rate ~8 MMBtu/MWh)
    # MISO avg heat rate for marginal unit ~9.5 MMBtu/MWh for combined cycle
    if price < 1.50:
        return (
            "very_low",
            f"${price:.2f}/MMBtu — extremely cheap gas → floors LMP near variable cost, expect very low prices",
            -12.0,
        )
    elif price < 2.20:
        return (
            "low",
            f"${price:.2f}/MMBtu — low gas prices → low thermal dispatch cost → modest LMP",
            -5.0,
        )
    elif price < 3.00:
        return (
            "moderate",
            f"${price:.2f}/MMBtu — moderate gas → normal market conditions, no unusual price pressure",
            0.0,
        )
    elif price < 4.50:
        return (
            "elevated",
            f"${price:.2f}/MMBtu — elevated gas → thermal cost pressure → LMP ~${(price - 2.6) * 9:.0f}/MWh above baseline",
            +4.0,
        )
    elif price < 7.00:
        return (
            "high",
            f"${price:.2f}/MMBtu — high gas → significant cost pass-through → LMP uplift ~${(price - 2.6) * 9:.0f}/MWh",
            +10.0,
        )
    else:
        return (
            "spike",
            f"${price:.2f}/MMBtu — gas price SPIKE → expect materially higher LMP across all hours",
            +20.0,
        )
