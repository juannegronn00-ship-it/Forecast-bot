"""
PJM price correlation signal for MISO DA-LMP forecasting.

Why PJM matters for MISO:
  MISO and PJM are electrically interconnected through high-voltage transmission
  interfaces (notably the MISO-PJM interface at ~6-8 GW capacity).  When PJM
  prices spike — due to cold snaps, high demand, or generation outages — energy
  flows from MISO into PJM, tightening MISO supply and pushing MISO prices up.
  The correlation is strongest in hours 16-24 (peak demand overlap) and during
  weather events affecting both regions simultaneously.

Data sources (in priority order):
  1. EIA API v2 — daily average PJM wholesale price (free key at eia.gov/opendata)
  2. PJM public data file — day-ahead LMP export (no key needed, best-effort scrape)
  3. Seasonal estimate — typical PJM-MISO spread for the current month

Typical PJM-MISO spring relationship:
  PJM Western Hub typically trades $5-15/MWh above MISO Indiana Hub in spring.
  When PJM > $60/MWh, interface flows tighten and MISO correlation strengthens.
"""
import logging
import os
import requests
from datetime import date, timedelta
from typing import Dict, List, Optional

_MISO_CSV_BASE = "https://docs.misoenergy.org/marketreports"
_MISO_HEADERS  = {"User-Agent": "forecast-bot/2.0 (contact@example.com)"}

logger = logging.getLogger(__name__)

# EIA series ID for PJM day-ahead price (Western Hub)
_EIA_PJM_SERIES = "EBA.PJM-ALL.D.H"
_EIA_BASE = "https://api.eia.gov/v2"

# Seasonal PJM estimates ($/MWh) when no live data available
# Based on typical PJM annual price patterns
_SEASONAL_PJM = {
    1: 58.0,   # Jan: high winter heating demand
    2: 55.0,   # Feb: winter, sometimes volatile
    3: 38.0,   # Mar: spring transition
    4: 36.0,   # Apr: mild spring shoulder
    5: 38.0,   # May: spring, some cooling load
    6: 52.0,   # Jun: summer AC ramp
    7: 62.0,   # Jul: peak summer
    8: 58.0,   # Aug: late summer
    9: 42.0,   # Sep: fall shoulder
    10: 38.0,  # Oct: fall
    11: 46.0,  # Nov: early winter
    12: 55.0,  # Dec: winter
}

# Typical MISO average prices by month (for ratio calculation)
_SEASONAL_MISO = {
    1: 42.0,  2: 40.0,  3: 30.0,  4: 28.0,
    5: 30.0,  6: 40.0,  7: 50.0,  8: 48.0,
    9: 34.0,  10: 30.0, 11: 36.0, 12: 44.0,
}

# Spread threshold above which PJM-MISO correlation becomes material
_MATERIAL_SPREAD_THRESHOLD = 8.0   # $/MWh

# Typical PJM hourly price shape (multipliers relative to daily average).
# Derived from PJM historical load/price patterns — used when only a daily
# average is available to reconstruct a realistic 24-hour shape.
_PJM_SHAPE = [
    0.82, 0.79, 0.77, 0.76, 0.77, 0.80,   # hrs  1–6:  night valley
    0.88, 0.97, 1.05, 1.08, 1.09, 1.07,   # hrs  7–12: morning ramp + midday
    1.06, 1.07, 1.09, 1.13, 1.18, 1.21,   # hrs 13–18: afternoon / evening peak
    1.17, 1.14, 1.09, 1.03, 0.95, 0.87,   # hrs 19–24: evening wind-down
]


def _shape_from_daily(avg_price: float) -> List[float]:
    """Distribute a daily-average PJM price into a realistic 24-hour shaped array."""
    return [round(avg_price * m, 2) for m in _PJM_SHAPE]


class PJMScraper:
    """
    Fetch PJM day-ahead price and compute its correlation signal for MISO.

    Returns a trading_signal dict compatible with the FantodsOptimizer context.
    """

    def __init__(self):
        self._eia_key = os.getenv("EIA_API_KEY", "")

    def fetch_data(self) -> Dict:
        """
        Try sources in order. Returns:
        {
          'success': bool,
          'source': str,
          'pjm_price': float,          # $/MWh DA average
          'miso_estimate': float,       # seasonal MISO baseline for comparison
          'spread': float,              # PJM - MISO estimate ($/MWh)
          'correlation_pct': float,     # estimated MISO price impact (±%)
          'trading_signal': str,
        }
        """
        # Try EIA first (most reliable when key is set)
        if self._eia_key:
            result = self._fetch_eia()
            if result["success"]:
                return result

        # Try MISO Indiana Hub from the DA LMP CSV as a PJM proxy
        # Indiana Hub sits on the MISO-PJM border and trades $4-8 below PJM Western Hub
        result = self._fetch_indiana_hub_proxy()
        if result["success"]:
            return result

        # Try PJM public data
        result = self._fetch_pjm_public()
        if result["success"]:
            return result

        # Fall back to seasonal estimate
        return self._seasonal_estimate()

    # ------------------------------------------------------------------ #
    # EIA API v2
    # ------------------------------------------------------------------ #
    def _fetch_eia(self) -> Dict:
        """EIA API — PJM hourly wholesale DA price (falls back to daily avg)."""
        try:
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            today_str = date.today().strftime("%Y-%m-%d")

            # Try hourly DA price endpoint first (gives 24-hour shape)
            resp = requests.get(
                f"{_EIA_BASE}/electricity/rto/region-data/data/",
                params={
                    "api_key": self._eia_key,
                    "data[0]": "value",
                    "facets[respondent][]": "PJM",
                    "facets[type][]": "DF",   # day-ahead demand forecast (proxy for price shape)
                    "start": yesterday,
                    "end": today_str,
                    "sort[0][column]": "period",
                    "sort[0][direction]": "asc",
                    "length": "24",
                },
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json().get("response", {}).get("data", [])

            if items and len(items) >= 12:
                # Build hourly price array from DA price data
                hourly = [None] * 24
                for item in items:
                    try:
                        period = item.get("period", "")   # e.g. "2025-04-30T14"
                        hour = int(period.split("T")[1]) if "T" in period else None
                        val  = float(item.get("value", 0))
                        if hour is not None and 0 <= hour < 24:
                            hourly[hour] = val
                    except Exception:
                        pass
                filled = [h for h in hourly if h is not None]
                if len(filled) >= 12:
                    avg_price = sum(filled) / len(filled)
                    # Fill any gaps with shape-interpolated value
                    shaped = _shape_from_daily(avg_price)
                    hourly_prices = [h if h is not None else shaped[i] for i, h in enumerate(hourly)]
                    logger.info(f"EIA: {len(filled)} hourly PJM values fetched")
                    return self._build_result("EIA API (hourly)", avg_price, hourly_prices=hourly_prices)

            # Fall back to daily-average endpoint
            resp2 = requests.get(
                f"{_EIA_BASE}/electricity/rto/daily-region-data/data/",
                params={
                    "api_key": self._eia_key,
                    "data[0]": "value",
                    "facets[respondent][]": "PJM",
                    "facets[type][]": "D",
                    "start": yesterday,
                    "end": today_str,
                    "sort[0][column]": "period",
                    "sort[0][direction]": "desc",
                    "length": "1",
                },
                timeout=10,
            )
            resp2.raise_for_status()
            data = resp2.json().get("response", {}).get("data", [])
            if not data:
                return {"success": False, "error": "EIA: no PJM data returned"}

            pjm_price = float(data[0]["value"])
            return self._build_result("EIA API (daily avg)", pjm_price)

        except Exception as e:
            logger.warning(f"PJM EIA fetch failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # PJM public data (no key needed)
    # ------------------------------------------------------------------ #
    def _fetch_pjm_public(self) -> Dict:
        """
        PJM DataMiner2 API — DA LMP for Western Hub, hourly.
        Returns per-hour prices (24 values) when available.
        """
        try:
            yesterday = date.today() - timedelta(days=1)
            date_str  = yesterday.strftime("%Y-%m-%d")

            resp = requests.get(
                "https://dataminer2.pjm.com/feed/da_hrl_lmps/definition",
                params={
                    "startRow": "1",
                    "isActiveOnly": "false",
                    "pnode_id": "33092371",   # WESTERN HUB
                    "datetime_beginning_ept": f"{date_str} 00:00",
                    "datetime_ending_ept": f"{date_str} 23:59",
                },
                timeout=10,
                headers={"User-Agent": "forecast-bot/2.0"},
            )
            if resp.status_code != 200:
                return {"success": False, "error": f"PJM DataMiner: HTTP {resp.status_code}"}

            items = resp.json().get("items", [])
            if not items:
                return {"success": False, "error": "PJM DataMiner: no items"}

            # Extract per-hour prices from DataMiner items
            # Each item has datetime_beginning_ept like "2025-04-30 14:00"
            hourly = [None] * 24
            for item in items:
                try:
                    dt_str = item.get("datetime_beginning_ept", "")
                    hour   = int(dt_str.split(" ")[1].split(":")[0]) if dt_str else None
                    price  = float(item.get("total_lmp_da") or item.get("total_lmp_rt") or 0)
                    if hour is not None and 0 <= hour < 24 and price:
                        hourly[hour] = price
                except Exception:
                    pass

            filled = [h for h in hourly if h is not None]
            if not filled:
                return {"success": False, "error": "PJM DataMiner: no price values"}

            avg_price     = sum(filled) / len(filled)
            shaped        = _shape_from_daily(avg_price)
            hourly_prices = [h if h is not None else shaped[i] for i, h in enumerate(hourly)]

            n_hourly = len(filled)
            logger.info(f"PJM DataMiner: {n_hourly}/24 hourly DA prices fetched for {date_str}")
            return self._build_result(f"PJM DataMiner ({n_hourly}h)", avg_price, hourly_prices=hourly_prices)

        except Exception as e:
            logger.warning(f"PJM public data fetch failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # MISO Indiana Hub proxy for PJM Western Hub (no auth needed)
    # ------------------------------------------------------------------ #
    def _fetch_indiana_hub_proxy(self) -> Dict:
        """
        INDIANA.HUB sits directly on the MISO-PJM interface and cleared DA prices
        track PJM Western Hub closely.  The typical Indiana Hub → PJM Western Hub
        spread is +$4 to +$8/MWh due to PJM internal transmission constraints.

        Source: MISO DA ExAnte LMP CSV (publicly available, no key required).
        """
        try:
            today_str = date.today().strftime("%Y%m%d")
            url = f"{_MISO_CSV_BASE}/{today_str}_da_exante_lmp.csv"
            resp = requests.get(url, headers=_MISO_HEADERS, timeout=15)
            resp.raise_for_status()

            indiana_prices = _parse_node_lmp(resp.text, "INDIANA.HUB")
            if len(indiana_prices) != 24:
                logger.warning(f"Indiana Hub parse returned {len(indiana_prices)} prices")
                return {"success": False, "error": "Indiana Hub not found"}

            # PJM Western Hub ≈ Indiana Hub + $6/MWh interface spread (typical)
            _INDIANA_PJM_SPREAD = 6.0
            pjm_hourly = [round(p + _INDIANA_PJM_SPREAD, 2) for p in indiana_prices]
            pjm_avg    = sum(pjm_hourly) / 24

            logger.info(
                f"PJM proxy (Indiana Hub + ${_INDIANA_PJM_SPREAD:.0f} spread): "
                f"avg=${pjm_avg:.2f}  max=${max(pjm_hourly):.2f}"
            )
            return self._build_result("MISO Indiana Hub proxy", pjm_avg, hourly_prices=pjm_hourly)

        except Exception as e:
            logger.warning(f"Indiana Hub PJM proxy failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Seasonal fallback
    # ------------------------------------------------------------------ #
    def _seasonal_estimate(self) -> Dict:
        month   = date.today().month
        pjm_est = _SEASONAL_PJM.get(month, 40.0)
        result  = self._build_result("seasonal_estimate", pjm_est)
        result["source"] = "seasonal_estimate"
        logger.info(f"PJM seasonal estimate for month {month}: ${pjm_est:.2f}/MWh (shaped into 24h)")
        return result

    # ------------------------------------------------------------------ #
    # Signal computation
    # ------------------------------------------------------------------ #
    def _build_result(self, source: str, pjm_price: float, hourly_prices: List[float] = None) -> Dict:
        month = date.today().month
        miso_est = _SEASONAL_MISO.get(month, 30.0)
        spread = pjm_price - miso_est

        # PJM-MISO correlation signal:
        # When spread is small (PJM ≈ MISO): minimal pull on MISO prices
        # When spread widens: interface flows tighten MISO supply → upward pressure
        # Effect is stronger for hours 16-24 (peak demand overlap)
        if spread >= 20:
            corr_pct = 0.08   # very high PJM: strong upward pull on MISO
            signal_txt = f"PJM ${pjm_price:.0f} >> MISO est ${miso_est:.0f} (spread ${spread:.0f}) → STRONG upward pull on MISO interface hrs 16-24"
        elif spread >= 12:
            corr_pct = 0.05
            signal_txt = f"PJM ${pjm_price:.0f} > MISO est ${miso_est:.0f} (spread ${spread:.0f}) → moderate upward pressure on MISO peaks"
        elif spread >= _MATERIAL_SPREAD_THRESHOLD:
            corr_pct = 0.025
            signal_txt = f"PJM ${pjm_price:.0f} modestly above MISO est ${miso_est:.0f} (spread ${spread:.0f}) → slight upward MISO pressure"
        elif spread >= 0:
            corr_pct = 0.0
            signal_txt = f"PJM ${pjm_price:.0f} near MISO est ${miso_est:.0f} → neutral, no material interface pull"
        else:
            corr_pct = max(-0.03, spread * 0.003)   # negative spread → very slight downward
            signal_txt = f"PJM ${pjm_price:.0f} below MISO est ${miso_est:.0f} → slightly negative interface signal"

        logger.info(f"PJM ({source}): ${pjm_price:.2f}/MWh | MISO est: ${miso_est:.2f} | spread: ${spread:.1f} | corr: {corr_pct:+.1%}")

        return {
            "success":         True,
            "source":          source,
            "pjm_price":       round(pjm_price, 2),
            "miso_estimate":   miso_est,
            "spread":          round(spread, 2),
            "correlation_pct": corr_pct,
            "trading_signal":  signal_txt,
            # 24-hour shaped array — used by Claude prompt for per-hour PJM context
            "hourly_prices":   [round(p, 2) for p in (hourly_prices or _shape_from_daily(pjm_price))],
        }


def _parse_node_lmp(csv_text: str, node_name: str) -> List[float]:
    """Parse 24-hour LMP prices for a specific node from MISO DA ExAnte LMP CSV."""
    prefix = f"{node_name},Hub,LMP"
    for line in csv_text.split("\n"):
        if line.startswith(prefix):
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
