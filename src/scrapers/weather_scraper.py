"""
Weather data for MISO DA-LMP forecasting.

Uses NOAA weather.gov (free, no key needed) as primary source.
Falls back to OpenWeatherMap if OPENWEATHER_API_KEY is set and NOAA fails.

Chicago (41.88°N, 87.63°W) is used as the representative MISO load center —
Illinois is the largest state by MISO load and Chicago weather correlates well
with system-wide heating/cooling demand.
"""
import logging
import os
import requests
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Chicago — largest MISO load center
_LAT = 41.88
_LON = -87.63
_CITY = "Chicago, IL"
_NOAA_OFFICE = "LOT"   # NWS Chicago forecast office


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def _ms_to_mph(ms: float) -> float:
    return ms * 2.237


class WeatherScraper:
    """Hourly 24-hour weather forecast for the MISO load center."""

    def __init__(self):
        self._owm_key = os.getenv("OPENWEATHER_API_KEY", "")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fetch_data(self) -> Dict:
        """
        Returns:
        {
          'success': bool,
          'source': str,
          'hourly': [24 × {hour, temp_f, wind_mph, cloud_pct, humidity_pct}],
          'summary': str,          # one-line human description
          'trading_signal': str,   # trader interpretation
        }
        """
        # NOAA first — free, no key, good coverage
        result = self._fetch_noaa()
        if result["success"]:
            return result

        # Fall back to OpenWeatherMap if key is set
        if self._owm_key:
            result = self._fetch_owm()
            if result["success"]:
                return result

        logger.warning("All weather sources failed — no weather data available")
        return {"success": False, "hourly": [], "error": "all sources failed"}

    # ------------------------------------------------------------------ #
    # NOAA weather.gov (free, no API key)
    # ------------------------------------------------------------------ #
    def _fetch_noaa(self) -> Dict:
        try:
            headers = {"User-Agent": "forecast-bot/2.0 (contact@example.com)"}

            # Step 1: resolve grid point
            pts = requests.get(
                f"https://api.weather.gov/points/{_LAT},{_LON}",
                headers=headers, timeout=10,
            )
            pts.raise_for_status()
            props = pts.json()["properties"]
            hourly_url = props["forecastHourly"]

            # Step 2: hourly forecast
            fcast = requests.get(hourly_url, headers=headers, timeout=12)
            fcast.raise_for_status()
            periods = fcast.json()["properties"]["periods"][:24]

            hourly = []
            for i, p in enumerate(periods):
                temp_f = float(p["temperature"])
                # windSpeed can be "10 mph" or "10 to 15 mph" — take first number
                wind_raw = p.get("windSpeed", "10 mph").split()[0]
                wind_mph = float(wind_raw) if wind_raw.isdigit() or wind_raw.replace(".", "").isdigit() else 10.0
                humidity = (p.get("relativeHumidity") or {}).get("value", 60)
                hourly.append({
                    "hour": i + 1,
                    "temp_f": round(temp_f, 1),
                    "wind_mph": round(wind_mph, 1),
                    "cloud_pct": 50,    # NOAA basic forecast doesn't include cloud %
                    "humidity_pct": int(humidity),
                })

            summary, signal = _interpret_weather(hourly)
            logger.info(f"NOAA weather OK: {summary}")
            return {
                "success": True,
                "source": "noaa",
                "hourly": hourly,
                "summary": summary,
                "trading_signal": signal,
            }
        except Exception as e:
            logger.warning(f"NOAA weather failed: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # OpenWeatherMap (requires OPENWEATHER_API_KEY)
    # ------------------------------------------------------------------ #
    def _fetch_owm(self) -> Dict:
        try:
            resp = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={
                    "lat": _LAT, "lon": _LON,
                    "appid": self._owm_key,
                    "units": "imperial",
                    "cnt": 8,          # 8 × 3h slots = 24h
                },
                timeout=10,
            )
            resp.raise_for_status()
            slots = resp.json().get("list", [])[:8]

            hourly = []
            for i, slot in enumerate(slots):
                temp_f = slot["main"]["temp"]
                wind_mph = slot["wind"]["speed"]
                cloud_pct = slot.get("clouds", {}).get("all", 50)
                humidity = slot["main"]["humidity"]
                # Each slot covers 3 hours; expand to 3 individual hours
                base_hour = i * 3
                for offset in range(3):
                    h = base_hour + offset + 1
                    if h > 24:
                        break
                    hourly.append({
                        "hour": h,
                        "temp_f": round(temp_f, 1),
                        "wind_mph": round(wind_mph, 1),
                        "cloud_pct": cloud_pct,
                        "humidity_pct": humidity,
                    })

            summary, signal = _interpret_weather(hourly)
            logger.info(f"OpenWeatherMap OK: {summary}")
            return {
                "success": True,
                "source": "openweathermap",
                "hourly": hourly[:24],
                "summary": summary,
                "trading_signal": signal,
            }
        except Exception as e:
            logger.warning(f"OpenWeatherMap failed: {e}")
            return {"success": False, "error": str(e)}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _interpret_weather(hourly: List[Dict]) -> tuple:
    """Return (summary_str, trading_signal_str)."""
    if not hourly:
        return "No weather data", "Unknown weather impact"

    avg_temp = sum(h["temp_f"] for h in hourly) / len(hourly)
    avg_wind = sum(h["wind_mph"] for h in hourly) / len(hourly)
    avg_cloud = sum(h["cloud_pct"] for h in hourly) / len(hourly)
    min_temp = min(h["temp_f"] for h in hourly)
    max_temp = max(h["temp_f"] for h in hourly)

    summary = (
        f"Avg {avg_temp:.0f}°F (range {min_temp:.0f}–{max_temp:.0f}°F) | "
        f"Wind {avg_wind:.0f} mph | Cloud {avg_cloud:.0f}%"
    )

    # Heating/cooling demand signal
    signals = []
    if avg_temp < 32:
        signals.append("FREEZING → high heating demand → elevated prices")
    elif avg_temp < 45:
        signals.append("COLD → moderate-high heating demand → above-average prices")
    elif avg_temp < 60:
        signals.append("COOL → low heating demand → near-normal prices")
    elif avg_temp < 75:
        signals.append("MILD → minimal HVAC demand → lowest prices of the year")
    elif avg_temp < 88:
        signals.append("WARM → moderate cooling demand → afternoon prices elevated")
    else:
        signals.append("HOT → high cooling demand → significant price uplift")

    # Wind generation signal
    if avg_wind > 20:
        signals.append("HIGH WIND → strong renewable generation → downward price pressure")
    elif avg_wind > 12:
        signals.append("MODERATE WIND → normal renewable contribution")
    else:
        signals.append("LOW WIND → less renewable generation → more thermal dispatch needed")

    # Solar signal (cloud cover)
    if avg_cloud > 75:
        signals.append("OVERCAST → minimal solar → slightly higher midday prices")
    elif avg_cloud < 25:
        signals.append("CLEAR → full solar → downward midday price pressure")

    return summary, " | ".join(signals)


def heating_degree_hours(hourly: List[Dict], base_f: float = 65.0) -> float:
    """Heating degree-hours vs 65°F base (standard utility calculation)."""
    return sum(max(0, base_f - h["temp_f"]) for h in hourly)


def cooling_degree_hours(hourly: List[Dict], base_f: float = 65.0) -> float:
    """Cooling degree-hours vs 65°F base."""
    return sum(max(0, h["temp_f"] - base_f) for h in hourly)
