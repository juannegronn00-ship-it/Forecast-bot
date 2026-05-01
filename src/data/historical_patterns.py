"""
Historical price patterns from the fantods rolling price table.

The fantods AJAX endpoint exposes ~40 days of hourly DA-LMP history for the
MISO ENERGY node.  We parse every date column (not just tomorrow or the mean)
to build same-weekday price profiles — the most direct comparable for any
given forecast day.

Example use:
  patterns = HistoricalPatterns()
  patterns.load()
  profile = patterns.get_weekday_profile(target_weekday=0)  # 0=Monday
  anchor = patterns.get_same_weekday_avg(hour=20, weekday=0)
"""
import logging
import json
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_AJAX_URL = "http://nyc3.fantods.co/socobufalo/miso/miso_node_history.php"

# Day-of-week names for logging
_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class HistoricalPatterns:
    """
    Parses the full fantods price table and provides same-weekday statistics.

    After calling load(), attributes are available:
      .by_weekday: {weekday_int: {hour_int: [prices]}}   # weekday 0=Mon
      .by_date:    {date_obj: [24_prices]}
      .loaded:     bool
    """

    def __init__(self):
        self.by_weekday: Dict[int, Dict[int, List[float]]] = {d: {} for d in range(7)}
        self.by_date: Dict[date, List[float]] = {}
        self.loaded = False

    def load(self, reference_date: Optional[date] = None) -> bool:
        """
        Fetch and parse the full ~40-day price table from fantods.
        Returns True on success.
        """
        ref = reference_date or date.today()
        try:
            resp = requests.get(
                _AJAX_URL,
                params={
                    "date": f"{ref.year}-{ref.month}-{ref.day}",
                    "node": "ENERGY",
                    "price": "dalmp",
                    "color": "same",
                    "filter": "",
                    "filterNo": "",
                    "dow": "all",
                },
                timeout=15,
            )
            resp.raise_for_status()
            html = json.loads(resp.text)
            soup = BeautifulSoup(html, "html.parser")

            # --- Parse column headers ---
            thead = soup.find("thead")
            if not thead:
                logger.warning("Historical: no thead found")
                return False

            headers = [th.text.strip() for th in thead.find_all("tr")[-1].find_all("th")]

            # Map column index → actual date object (skip non-date headers like "H" and "M")
            col_date_map: Dict[int, date] = {}
            for col_idx, header in enumerate(headers):
                if "/" not in header:
                    continue
                parts = header.split("/")
                if len(parts) != 2:
                    continue
                try:
                    m, d = int(parts[0]), int(parts[1])
                    actual_date = _infer_date(m, d, ref)
                    if actual_date:
                        col_date_map[col_idx] = actual_date
                except ValueError:
                    continue

            if not col_date_map:
                logger.warning("Historical: could not parse any date columns")
                return False

            # --- Parse data rows (1 row per hour) ---
            # Build {col_idx: [prices for hours 1-24]}
            col_prices: Dict[int, List[float]] = {c: [] for c in col_date_map}

            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                for col_idx in col_date_map:
                    if len(cells) <= col_idx:
                        continue
                    raw = cells[col_idx].text.strip()
                    try:
                        col_prices[col_idx].append(float(raw))
                    except (ValueError, AttributeError):
                        pass

            # --- Store results ---
            for col_idx, dt in col_date_map.items():
                prices = col_prices[col_idx]
                if len(prices) < 24:
                    continue
                self.by_date[dt] = prices[:24]
                wd = dt.weekday()
                for hr_idx, price in enumerate(prices[:24]):
                    hr = hr_idx + 1
                    if hr not in self.by_weekday[wd]:
                        self.by_weekday[wd][hr] = []
                    self.by_weekday[wd][hr].append(price)

            n_days = len(self.by_date)
            n_weekdays = sum(1 for dt in self.by_date if dt.weekday() < 5)
            logger.info(f"Historical patterns loaded: {n_days} days ({n_weekdays} weekdays)")
            self.loaded = True
            return True

        except Exception as e:
            logger.warning(f"Historical patterns load failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    # Query API
    # ------------------------------------------------------------------ #
    def get_same_weekday_avg(self, hour: int, weekday: int) -> Optional[float]:
        """Average price for this (hour, weekday) from last ~5 occurrences."""
        prices = self.by_weekday.get(weekday, {}).get(hour, [])
        if not prices:
            return None
        recent = prices[-5:]   # cap at last 5 occurrences (~5 weeks)
        return round(sum(recent) / len(recent), 2)

    def get_weekday_profile(self, weekday: int) -> List[float]:
        """
        Average 24-hour price profile for a given weekday.
        Returns list of 24 values (hour 1..24), or empty list if no data.
        """
        profile = []
        for hr in range(1, 25):
            avg = self.get_same_weekday_avg(hr, weekday)
            if avg is None:
                return []
            profile.append(avg)
        return profile

    def get_last_same_weekday(self, weekday: int) -> Optional[List[float]]:
        """Return the most recent date's prices for this weekday."""
        candidates = sorted(
            [dt for dt, _ in self.by_date.items() if dt.weekday() == weekday],
            reverse=True,
        )
        if not candidates:
            return None
        return self.by_date[candidates[0]]

    def get_weekday_stats(self, weekday: int) -> dict:
        """
        Return per-hour statistics for a given weekday (last 5 occurrences).
        Returns {1: {avg, min, max, std, n, recent}, ...} for hours 1-24.
        Used to give Claude historical range/variance, not just averages.
        """
        stats = {}
        for hr in range(1, 25):
            prices = self.by_weekday.get(weekday, {}).get(hr, [])
            if not prices:
                stats[hr] = None
                continue
            recent = prices[-5:]
            avg = sum(recent) / len(recent)
            mn = min(recent)
            mx = max(recent)
            var = sum((p - avg) ** 2 for p in recent) / len(recent)
            std = var ** 0.5
            stats[hr] = {
                "avg": round(avg, 2),
                "min": round(mn, 2),
                "max": round(mx, 2),
                "std": round(std, 2),
                "n": len(recent),
                "recent": [round(p, 2) for p in recent],
            }
        return stats

    def get_last_n_same_weekday(self, weekday: int, n: int = 3) -> list:
        """Return the last N date profiles for a given weekday, newest first."""
        candidates = sorted(
            [dt for dt in self.by_date if dt.weekday() == weekday],
            reverse=True,
        )
        return [(dt, self.by_date[dt]) for dt in candidates[:n]]

    def summary_for_date(self, target_date: date) -> str:
        """Human-readable summary of historical same-weekday data."""
        wd = target_date.weekday()
        profile = self.get_weekday_profile(wd)
        if not profile:
            return "No historical same-weekday data available"

        night_avg = sum(profile[0:6]) / 6
        morning_avg = sum(profile[6:10]) / 4
        midday_avg = sum(profile[10:16]) / 6
        evening_avg = sum(profile[19:24]) / 5
        overall_avg = sum(profile) / 24

        dow_name = _DOW[wd]
        n_samples = len(self.by_weekday.get(wd, {}).get(12, []))

        return (
            f"Historical {dow_name}s (n={n_samples} weeks): "
            f"night avg ${night_avg:.1f} | "
            f"morning avg ${morning_avg:.1f} | "
            f"midday avg ${midday_avg:.1f} | "
            f"evening avg ${evening_avg:.1f} | "
            f"daily avg ${overall_avg:.1f}"
        )


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #
def _infer_date(month: int, day: int, reference: date) -> Optional[date]:
    """
    Convert a bare M/D to a full date by searching backwards from reference.
    The fantods table covers the past ~45 days so we look back at most 50 days.
    """
    for delta in range(0, 50):
        candidate = reference - timedelta(days=delta)
        if candidate.month == month and candidate.day == day:
            return candidate
    return None
