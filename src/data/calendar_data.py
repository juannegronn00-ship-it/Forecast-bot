"""
Calendar context for MISO DA-LMP forecasting.

Energy demand is profoundly shaped by the calendar:
  - Weekends: 8-12% lower load than comparable weekdays
  - Holidays: similar to or lower than weekends
  - Daylight hours: longer days → more solar generation → lower midday prices
  - Season: determines baseline HVAC demand level
"""
import math
from datetime import date, timedelta
from typing import Optional


# US federal holidays (fixed and floating) for 2025–2027
# Format: (month, day) for fixed; callable for floating
_FIXED_HOLIDAYS = {
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (11, 11),  # Veterans Day
    (12, 25),  # Christmas Day
}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of weekday (0=Mon) in month/year."""
    first = date(year, month, 1)
    diff = (weekday - first.weekday()) % 7
    first_occ = first + timedelta(days=diff)
    return first_occ + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of weekday in month/year."""
    # Start from last day and go backward
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    diff = (last.weekday() - weekday) % 7
    return last - timedelta(days=diff)


def _floating_holidays(year: int) -> set:
    return {
        _nth_weekday(year, 1, 0, 3),   # MLK Day — 3rd Monday in Jan
        _nth_weekday(year, 2, 0, 3),   # Presidents Day — 3rd Monday in Feb
        _nth_weekday(year, 5, 0, 4),   # Memorial Day — last Monday in May (use 4th as approx)
        _last_weekday(year, 5, 0),     # Memorial Day — correct last Monday in May
        _nth_weekday(year, 9, 0, 1),   # Labor Day — 1st Monday in Sep
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving — 4th Thursday in Nov
    }


def is_holiday(target: date) -> bool:
    """Return True if target is a US federal holiday."""
    if (target.month, target.day) in _FIXED_HOLIDAYS:
        return True
    return target in _floating_holidays(target.year)


def get_season(target: date) -> str:
    """Return meteorological season for Northern Hemisphere."""
    m = target.month
    if m in (12, 1, 2):
        return "winter"
    elif m in (3, 4, 5):
        return "spring"
    elif m in (6, 7, 8):
        return "summer"
    else:
        return "fall"


def daylight_hours(target: date, lat: float = 41.88) -> float:
    """
    Approximate daylight hours using standard solar declination formula.
    Default latitude: Chicago (41.88°N), representative MISO load center.
    """
    day_of_year = target.timetuple().tm_yday
    # Solar declination angle (degrees)
    decl = 23.45 * math.sin(math.radians(360 / 365 * (day_of_year - 81)))
    # Hour angle at sunrise/sunset
    lat_r = math.radians(lat)
    decl_r = math.radians(decl)
    cos_ha = -math.tan(lat_r) * math.tan(decl_r)
    cos_ha = max(-1.0, min(1.0, cos_ha))  # clamp
    ha = math.degrees(math.acos(cos_ha))
    return round(2 * ha / 15, 1)


def solar_generation_signal(target: date) -> str:
    """Qualitative solar impact based on season and time of year."""
    dl = daylight_hours(target)
    if dl >= 15:
        return f"{dl}h daylight — long summer days, strong solar midday generation → downward midday price pressure"
    elif dl >= 13:
        return f"{dl}h daylight — spring/fall moderate solar contribution"
    else:
        return f"{dl}h daylight — short winter days, minimal solar generation"


def demand_profile_label(target: date) -> str:
    """Human-readable demand profile label for a date."""
    parts = []

    wd = target.weekday()
    if wd >= 5:
        parts.append("WEEKEND (-10% typical demand vs weekday)")
    else:
        parts.append(f"WEEKDAY ({['Mon','Tue','Wed','Thu','Fri'][wd]})")

    if is_holiday(target):
        parts.append("HOLIDAY (-15% typical demand)")

    season = get_season(target)
    season_notes = {
        "winter": "WINTER (peak heating season)",
        "spring": "SPRING shoulder (low HVAC demand)",
        "summer": "SUMMER (peak cooling season)",
        "fall": "FALL shoulder (low HVAC demand)",
    }
    parts.append(season_notes[season])

    return " | ".join(parts)


def load_adjustment_factor(target: date) -> float:
    """
    Estimated multiplicative adjustment to base load based on calendar.
    1.0 = normal weekday. Used to calibrate expected demand level.
    """
    factor = 1.0

    if target.weekday() >= 5:
        factor *= 0.90   # weekend
    if is_holiday(target):
        factor *= 0.88   # holiday

    season = get_season(target)
    if season == "spring":
        factor *= 0.95   # spring shoulder
    elif season == "fall":
        factor *= 0.97
    elif season == "summer":
        factor *= 1.05   # summer cooling
    elif season == "winter":
        factor *= 1.08   # winter heating

    return round(factor, 3)
