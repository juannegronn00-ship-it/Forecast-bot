"""
Natural gas price data for MISO DA-LMP forecasting.

Henry Hub is the NYMEX settlement hub. Its spot price directly determines
the short-run marginal cost of gas-fired generation — typically the marginal
generator in MISO during most hours. When gas is cheap, LMPs stay low;
when gas spikes, LMPs follow.

Primary: Yahoo Finance NG=F — NYMEX natural gas front-month futures (free, no key, current)
Secondary: EIA Open Data API v2 monthly summary (official; may lag 1-2 months)
  Set EIA_API_KEY in your .env to enable.
Fallback: Reasoned estimate based on recent seasonal norms.
"""
import logging
import os
import requests
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

# EIA v2 endpoint — monthly Henry Hub / city-gate spot price
# N9010LA3 = Louisiana Natural Gas Residential Price (monthly, official proxy)
# We use the most recent available monthly record as a lagged signal.
_EIA_SERIES   = "N9010LA3"
_EIA_ENDPOINT = "https://api.eia.gov/v2/natural-gas/pri/sum/data/"

# Seasonal norms ($/MMBtu) — fallback when all live sources fail
_SEASONAL_DEFAULTS = {
    1: 3.20,
    2: 3.00,
    3: 2.60,
    4: 2.20,
    5: 2.30,
    6: 2.80,
    7: 3.10,
    8: 3.00,
    9: 2.70,
    10: 2.60,
    11: 3.00,
    12: 3.40,
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
          'price_level': str,
          'trading_signal': str,
          'lmp_impact_pct': float,
        }
        """
        # 1. Yahoo Finance — NYMEX NG front-month futures (Henry Hub proxy)
        result = self._fetch_yahoo()
        if result["success"]:
            return result

        # 2. EIA API — monthly official data (may lag 1-2 months)
        if self._eia_key:
            result = self._fetch_eia()
            if result["success"]:
                return result

        # 3. Seasonal estimate
        return self._seasonal_estimate()

    # ------------------------------------------------------------------ #
    # Yahoo Finance — NG=F (NYMEX natural gas front-month, Henry Hub)
    # ------------------------------------------------------------------ #
    def _fetch_yahoo(self) -> Dict:
        try:
            import yfinance as yf
            ticker = yf.Ticker("NG=F")
            price = ticker.fast_info.last_price
            if price and price > 0:
                result = _build_result(round(float(price), 3), "yahoo_finance (NG=F)")
                logger.info(f"Yahoo Finance NG=F gas price: ${price:.2f}/MMBtu")
                return result
            return {"success": False, "error": "no price from Yahoo Finance"}
        except Exception as e:
            logger.warning(f"Yahoo Finance failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # EIA Open Data API v2 — monthly summary
    # ------------------------------------------------------------------ #
    def _fetch_eia(self) -> Dict:
        try:
            resp = requests.get(
                _EIA_ENDPOINT,
                params={
                    "api_key":              self._eia_key,
                    "frequency":            "monthly",
                    "data[0]":              "value",
                    "facets[series][]":     _EIA_SERIES,
                    "sort[0][column]":      "period",
                    "sort[0][direction]":   "desc",
                    "length":               5,
                },
                timeout=12,
            )
            resp.raise_for_status()
            records = resp.json().get("response", {}).get("data", [])

            for rec in records:
                raw = rec.get("value")
                if raw is not None:
                    try:
                        # EIA prices for residential are in $/MCF — convert to $/MMBtu
                        # 1 MCF ≈ 1.02 MMBtu for pipeline-quality gas
                        price_mcf = float(raw)
                        price = price_mcf / 1.02
                        period = rec.get("period", "unknown")
                        result = _build_result(round(price, 3), f"eia_monthly ({period})")
                        logger.info(f"EIA monthly gas price: ${price:.2f}/MMBtu ({period})")
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
        logger.info(f"Gas price (seasonal estimate): ${price:.2f}/MMBtu")
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
