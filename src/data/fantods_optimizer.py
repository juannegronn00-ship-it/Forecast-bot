"""
Data-driven DA-LMP price optimizer — zero API cost, signal-based reasoning.

For each hour, the optimizer reasons through actual scraped data:
  1. What anchor price does history suggest? (same-weekday average)
  2. How does today's gas price shift generation costs vs that baseline?
  3. How does today's temperature create HVAC demand at this specific hour?
  4. How does the load forecast compare to the daily average at this hour?
  5. How does wind generation suppress or elevate prices via merit-order?
  6. What does the calendar (weekday, holiday) tell us about demand pattern?

Each signal is computed from actual data values — no fixed period percentages.
The reasoning is logged so every price adjustment is auditable.
"""
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TOU period map (self-contained, no circular imports)
# ---------------------------------------------------------------------------
_TOU = {
    "night":        list(range(1, 7)),    # hrs  1- 6  midnight–5 AM
    "morning_ramp": list(range(7, 10)),   # hrs  7- 9  6–8 AM
    "midday":       list(range(10, 16)),  # hrs 10-15  9 AM–2 PM
    "shoulder":     list(range(16, 20)),  # hrs 16-19  3–6 PM
    "evening_peak": list(range(20, 25)),  # hrs 20-24  7–11 PM
}


def _period(hour: int) -> str:
    for p, hrs in _TOU.items():
        if hour in hrs:
            return p
    return "midday"


# ---------------------------------------------------------------------------
# Signal constants — all grounded in MISO market mechanics
# ---------------------------------------------------------------------------

# Gas price baseline ($/MMBtu) for MISO spring shoulder season.
# Prices above this add generation cost pressure; below eases it.
_GAS_BASELINE = 2.75

# Per-$1 LMP impact from gas price deviation.
# Gas-fired peakers are often marginal in MISO → each $1 gas ≈ +3.5% LMP.
_GAS_LMP_FACTOR = 0.035

# Temperature baseline (°F) — HVAC is neutral at this temp.
_TEMP_BASELINE = 65.0

# Per-°F HVAC sensitivity by period and direction.
# Higher values = that period is more responsive to heating/cooling demand.
# Based on MISO load-temp regression for Illinois load zone.
_HEAT_SENS = {  # per °F below 65
    "night":        0.0025,  # baseboard/furnace overnight
    "morning_ramp": 0.0060,  # morning HVAC ramp-up is sharpest
    "midday":       0.0035,  # furnaces cycling, some commercial heat
    "shoulder":     0.0040,  # residential heating picks back up
    "evening_peak": 0.0065,  # evening heat demand is the peak
}
_COOL_SENS = {  # per °F above 65
    "night":        0.0010,  # overnight AC runs in heatwaves
    "morning_ramp": 0.0020,  # pre-cooling before peak
    "midday":       0.0065,  # solar heat + AC surge — most sensitive period
    "shoulder":     0.0075,  # hottest hours of day, max AC load
    "evening_peak": 0.0045,  # heat retained in buildings, AC continues
}

# Load signal thresholds.
# How much does % deviation from daily avg translate to price pressure?
# MISO pricing is non-linear near capacity: small load increases near peak
# cause disproportionate price spikes.
_LOAD_BREAKPOINTS = [
    (1.15, 0.08),   # ≥115% avg → +8% price pressure (near-scarcity)
    (1.10, 0.05),   # ≥110% avg → +5%
    (1.05, 0.02),   # ≥105% avg → +2%
    (0.95, 0.00),   # 95-105% avg → neutral
    (0.88, -0.02),  # 88-95% avg → -2%
    (0.00, -0.04),  # <88% avg → -4% (very low demand)
]

# Wind merit-order signal (MISO system wind, GW)
# MISO nameplate wind capacity ≈ 24 GW. At high penetration, wind displaces
# gas peakers (marginal cost $35-55/MWh), directly lowering LMP.
# Effect is strongest in hours when thermal peakers would otherwise be on the margin.
# Stepdad calibration: weight wind more heavily — it's a dominant MISO price driver.
_WIND_BREAKPOINTS = [
    (22, -0.090),   # ≥22 GW: near-max penetration → peakers almost fully displaced
    (18, -0.065),   # ≥18 GW: very high wind
    (14, -0.035),   # ≥14 GW: high wind, meaningful merit-order depression
    (10, -0.010),   # 10-14 GW: moderate, slight downward
    (7,   0.010),   # 7-10 GW: normal range, small upward
    (4,   0.030),   # 4-7 GW: below normal, gas filling gap
    (0,   0.055),   # <4 GW: low wind, high thermal dispatch, significant uplift
]

# Calendar adjustment ONLY applied when no historical data (history already
# encodes the weekday pattern). Applied as a last-resort correction.
_WEEKDAY_ADJ = {
    0:  0.025,   # Monday: load ramp from weekend, industrial restart
    1:  0.010,   # Tuesday: full week
    2:  0.005,   # Wednesday
    3:  0.005,   # Thursday
    4: -0.015,   # Friday: early shutdowns, afternoon load softens
    5: -0.080,   # Saturday: low commercial/industrial
    6: -0.100,   # Sunday: lowest load of week
}
_HOLIDAY_ADJ = -0.090   # holidays are even lower than Sunday

# Hard price bounds to prevent nonsensical outputs
_PRICE_FLOOR = 5.0
_PRICE_CEILING = 300.0


# ---------------------------------------------------------------------------
# Signal computation helpers
# ---------------------------------------------------------------------------

def _gas_signal(gas_price: Optional[float]) -> Tuple[float, str]:
    """
    Compute price adjustment from Henry Hub gas price.
    Returns (pct_adj, explanation).
    """
    if gas_price is None:
        return 0.0, "gas=unknown → no adjustment"
    deviation = gas_price - _GAS_BASELINE
    pct = deviation * _GAS_LMP_FACTOR
    pct = max(-0.10, min(0.12, pct))  # clamp at ±10-12%
    if abs(pct) < 0.002:
        direction = "neutral"
    elif pct > 0:
        direction = f"above baseline by ${deviation:.2f}/MMBtu → upward cost pressure"
    else:
        direction = f"below baseline by ${abs(deviation):.2f}/MMBtu → eases generation costs"
    return pct, f"gas ${gas_price:.2f}/MMBtu ({direction}) → {pct:+.1%}"


def _temp_signal(temp_f: Optional[float], period: str) -> Tuple[float, str]:
    """
    Compute HVAC demand adjustment for a specific hour's temperature and period.
    Returns (pct_adj, explanation).
    """
    if temp_f is None:
        return 0.0, "temp=unknown → no adjustment"
    dev = temp_f - _TEMP_BASELINE
    if dev < 0:  # heating demand
        pct = abs(dev) * _HEAT_SENS.get(period, 0.004)
        label = f"{temp_f:.0f}°F ({abs(dev):.0f}° below comfort → heating demand)"
    elif dev > 0:  # cooling demand
        pct = dev * _COOL_SENS.get(period, 0.003)
        label = f"{temp_f:.0f}°F ({dev:.0f}° above comfort → cooling/AC demand)"
    else:
        pct = 0.0
        label = f"{temp_f:.0f}°F (at comfort baseline → no HVAC demand)"
    pct = max(-0.10, min(0.18, pct))
    return pct, f"temp={label} → {pct:+.1%}"


def _load_signal(load_gw: Optional[float], avg_load_gw: float) -> Tuple[float, str]:
    """
    Compute price adjustment from load vs daily average.
    Returns (pct_adj, explanation).
    """
    if not load_gw or not avg_load_gw:
        return 0.0, "load=unknown → no adjustment"
    ratio = load_gw / avg_load_gw
    pct = _LOAD_BREAKPOINTS[-1][1]  # default: lowest tier
    for threshold, adj in _LOAD_BREAKPOINTS:
        if ratio >= threshold:
            pct = adj
            break
    pct_label = f"{load_gw:.0f}GW = {ratio:.0%} of daily avg {avg_load_gw:.0f}GW"
    return pct, f"load={pct_label} → {pct:+.1%}"


def _wind_signal(wind_gw: Optional[float]) -> Tuple[float, str]:
    """
    Compute merit-order price adjustment from MISO system wind.
    Returns (pct_adj, explanation).
    """
    if wind_gw is None:
        return 0.0, "wind=unknown → no adjustment"
    pct = _WIND_BREAKPOINTS[-1][1]  # default: lowest tier
    for threshold, adj in _WIND_BREAKPOINTS:
        if wind_gw >= threshold:
            pct = adj
            break
    penetration = wind_gw / 24.0 * 100  # % of MISO nameplate capacity
    return pct, f"wind={wind_gw:.1f}GW ({penetration:.0f}% nameplate) → {pct:+.1%} merit-order"


def _pjm_signal(corr_pct: float, period: str) -> Tuple[float, str]:
    """
    Apply PJM-MISO price correlation signal.
    Effect is stronger for peak hours (shoulder/evening) when interface flows are highest.
    Returns (pct_adj, explanation).
    """
    if corr_pct == 0.0:
        return 0.0, "PJM=neutral"
    # Weight by period: PJM pull is mostly a peak-hours effect
    period_weight = {
        "night":        0.20,   # minimal interface trading overnight
        "morning_ramp": 0.50,   # some morning ramp effect
        "midday":       0.70,   # moderate
        "shoulder":     1.00,   # full effect during afternoon peak
        "evening_peak": 1.00,   # full effect during evening peak
    }.get(period, 0.70)
    adj = corr_pct * period_weight
    return adj, f"PJM corr={corr_pct:+.1%} × {period_weight:.0%} period_weight → {adj:+.1%}"


def _calendar_signal(weekday: int, is_holiday: bool) -> Tuple[float, str]:
    """
    Compute calendar adjustment. Only used when no historical anchor is available.
    Returns (pct_adj, explanation).
    """
    if is_holiday:
        pct = _HOLIDAY_ADJ
        label = "holiday"
    else:
        pct = _WEEKDAY_ADJ.get(weekday, 0.0)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        label = days[weekday] if weekday < 7 else "unknown"
    return pct, f"calendar={label} (no history → applying DOW adjustment) → {pct:+.1%}"


# ---------------------------------------------------------------------------
# Per-hour reasoning engine
# ---------------------------------------------------------------------------

def _reason_hour(
    hour: int,
    base_price: float,
    hist_price: Optional[float],      # same-weekday historical average for this hour
    load_gw: Optional[float],
    avg_load_gw: float,
    wind_gw: Optional[float],
    temp_f: Optional[float],          # temperature at this specific hour
    gas_price: Optional[float],
    weekday: int,
    is_holiday: bool,
    pjm_corr_pct: float = 0.0,        # PJM-MISO correlation signal
) -> Tuple[float, str]:
    """
    Reason through all signals for one hour and produce a forecast price.

    Logic:
      - If historical same-weekday data exists: use it as the primary anchor.
        History already encodes weekday pattern, seasonal shape, and market level.
        Signals then adjust for HOW this specific day differs from that history.
      - If no history: use base price as anchor, apply all signals including calendar.

    Returns (forecast_price, reasoning_narrative).
    """
    per = _period(hour)
    has_history = hist_price is not None and hist_price > 0

    if has_history:
        anchor = hist_price
        anchor_label = f"hist ${hist_price:.2f} (same-weekday avg)"
    else:
        anchor = base_price
        anchor_label = f"base ${base_price:.2f} (40-day mean)"

    # Compute each signal
    gas_pct,  gas_note  = _gas_signal(gas_price)
    temp_pct, temp_note = _temp_signal(temp_f, per)
    load_pct, load_note = _load_signal(load_gw, avg_load_gw)
    wind_pct, wind_note = _wind_signal(wind_gw)
    pjm_pct,  pjm_note  = _pjm_signal(pjm_corr_pct, per)

    # Calendar only when no history (history already captures weekday pattern)
    cal_pct, cal_note = (0.0, "calendar=skipped (history encodes weekday)") if has_history \
        else _calendar_signal(weekday, is_holiday)

    total_pct = gas_pct + temp_pct + load_pct + wind_pct + pjm_pct + cal_pct
    price = anchor * (1.0 + total_pct)
    price = max(_PRICE_FLOOR, min(_PRICE_CEILING, price))

    reasoning = (
        f"hr{hour:02d}[{per}] anchor={anchor_label} | "
        f"{gas_note} | {temp_note} | {load_note} | {wind_note} | {pjm_note}"
        + (f" | {cal_note}" if not has_history else "")
        + f" | total={total_pct:+.1%} → ${price:.2f}"
    )
    return round(price, 2), reasoning


# ---------------------------------------------------------------------------
# Public optimizer class
# ---------------------------------------------------------------------------

class FantodsOptimizer:
    """
    Signal-based DA-LMP price optimizer.
    No fixed multipliers — every adjustment is computed from actual scraped data.

    Usage:
        result = FantodsOptimizer().optimize(
            base_prices, load_forecast, wind_forecast, trader_context
        )
        # result['optimized_prices'] → List[float], 24 hours
        # result['reasoning']        → List[str], one line per hour
    """

    def optimize(
        self,
        base_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float],
        trader_context: Optional[Dict] = None,
    ) -> Dict:
        ctx = trader_context or {}

        # ── Extract inputs ───────────────────────────────────────────────
        hist_profile  = ctx.get("history_profile", [])
        weather       = ctx.get("weather", {})
        gas_data      = ctx.get("gas", {})
        weekday_int   = ctx.get("weekday_int", 0)
        is_holiday    = ctx.get("is_holiday", False)

        hourly_weather = weather.get("hourly", []) if weather.get("success") else []
        gas_price = gas_data.get("price") if gas_data.get("success") else None
        pjm_data = ctx.get("pjm", {})
        pjm_corr_pct = pjm_data.get("correlation_pct", 0.0) if pjm_data.get("success") else 0.0

        avg_load = sum(load_forecast) / 24 if load_forecast else 100.0

        # Build per-hour lookups
        hist_by_hour: Dict[int, float] = {}
        if hist_profile and len(hist_profile) == 24:
            for h, p in enumerate(hist_profile, 1):
                hist_by_hour[h] = p

        temp_by_hour: Dict[int, float] = {}
        for hw in hourly_weather:
            h = hw.get("hour")
            if h:
                temp_by_hour[h] = hw.get("temp_f")

        wind_by_hour: Dict[int, float] = {}
        for h, w in enumerate(wind_forecast, 1):
            wind_by_hour[h] = w

        # ── Reason through each hour ─────────────────────────────────────
        optimized_prices: List[float] = []
        reasoning_log: List[str] = []

        for h in range(1, 25):
            price, reason = _reason_hour(
                hour         = h,
                base_price   = base_prices[h - 1],
                hist_price   = hist_by_hour.get(h),
                load_gw      = load_forecast[h - 1] if h <= len(load_forecast) else None,
                avg_load_gw  = avg_load,
                wind_gw      = wind_by_hour.get(h),
                temp_f       = temp_by_hour.get(h),
                gas_price    = gas_price,
                weekday      = weekday_int,
                is_holiday   = is_holiday,
                pjm_corr_pct = pjm_corr_pct,
            )
            optimized_prices.append(price)
            reasoning_log.append(reason)
            logger.debug(reason)

        # ── Log period-level summary ─────────────────────────────────────
        logger.info("── FantodsOptimizer (signal-based reasoning) ────────────────")
        if gas_price:
            logger.info(f"  Gas: ${gas_price:.2f}/MMBtu  "
                        f"({'above' if gas_price > _GAS_BASELINE else 'below'} ${_GAS_BASELINE} baseline)")
        if pjm_data.get("success"):
            logger.info(f"  PJM: ${pjm_data.get('pjm_price', 0):.2f}/MWh  "
                        f"spread=${pjm_data.get('spread', 0):.1f}  "
                        f"corr={pjm_corr_pct:+.1%} (peak hours)")
        if hourly_weather:
            temps = [hw.get("temp_f", 65) for hw in hourly_weather]
            logger.info(f"  Temp: {min(temps):.0f}–{max(temps):.0f}°F  "
                        f"(avg {sum(temps)/len(temps):.0f}°F)")
        logger.info(f"  Wind: avg {sum(wind_forecast)/24:.1f} GW system")
        logger.info(f"  History: {'✅ same-weekday anchor loaded' if hist_by_hour else '⚠️ no history — using base prices'}")
        logger.info(f"  Load: avg {avg_load:.0f} GW")

        for pname, phours in _TOU.items():
            b_avg = sum(base_prices[h - 1] for h in phours) / len(phours)
            o_avg = sum(optimized_prices[h - 1] for h in phours) / len(phours)
            h_avg = (sum(hist_by_hour.get(h, 0) for h in phours) / len(phours)) if hist_by_hour else None
            hist_str = f"  hist=${h_avg:.2f}" if h_avg else ""
            logger.info(
                f"  {pname:>13s} hrs {phours[0]:02d}-{phours[-1]:02d}: "
                f"base=${b_avg:.2f}{hist_str}  → optimized=${o_avg:.2f}  "
                f"(Δ={o_avg-b_avg:+.2f}, {(o_avg-b_avg)/b_avg*100:+.1f}%)"
            )

        # Evening/midday ratio for quality check
        mid_avg = sum(optimized_prices[h - 1] for h in range(10, 16)) / 6
        eve_avg = sum(optimized_prices[h - 1] for h in range(20, 25)) / 5
        if mid_avg > 0:
            logger.info(f"  Evening/midday ratio: {eve_avg/mid_avg:.2f}x ({(eve_avg/mid_avg-1)*100:.0f}% above midday)")

        logger.info("─────────────────────────────────────────────────────────────")

        return {
            "success": True,
            "optimized_prices": optimized_prices,
            "reasoning": reasoning_log,
            "adjustments": {
                str(h): round((optimized_prices[h - 1] - base_prices[h - 1]) / base_prices[h - 1], 4)
                for h in range(1, 25)
            },
        }
