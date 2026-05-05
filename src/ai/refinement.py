"""
DA-LMP forecast refinement using Claude Opus (deep reasoning) + Gemini (validation).

Claude approach:
  Extended thinking enabled — Claude reasons like a veteran trader for 10,000 tokens
  before producing output.  Anchors to same-weekday historical prices, then adjusts
  for how TODAY's conditions (temp, wind, gas, PJM spread) deviate from that baseline.

  Output: absolute prices per hour.
  Post-processing: absolute seasonal bounds guard against hallucination only
  (NOT relative to the 40-day base, which is systematically too low).

Gemini approach:
  Cross-validates Claude prices against historical range and MISO season norms.
"""
import logging
import json
import math
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
from anthropic import Anthropic
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TOU periods
# ---------------------------------------------------------------------------
_TOU = {
    "night":        list(range(1, 7)),
    "morning_ramp": list(range(7, 10)),
    "midday":       list(range(10, 16)),
    "shoulder":     list(range(16, 20)),
    "evening_peak": list(range(20, 25)),
}

def _period(hour: int) -> str:
    for p, hrs in _TOU.items():
        if hour in hrs:
            return p
    return "midday"

# ---------------------------------------------------------------------------
# Absolute seasonal bounds ($/MWh) — MISO spring/fall shoulder
# ---------------------------------------------------------------------------
# These are WIDE seasonal guardrails, NOT tight caps.
# They prevent hallucinated prices (e.g. $500 in April) while allowing
# Claude to output real-market values like $85 shoulder on a hot Thursday.
#
# OLD approach: base × ±65% → max shoulder = $28 × 1.65 = $46.20  ← WRONG
# NEW approach: absolute $8–$160 shoulder → Claude can reach $55-85  ← CORRECT
# ---------------------------------------------------------------------------
_ABS_BOUNDS = {
    "night":        (4.0,   80.0),
    "morning_ramp": (6.0,  110.0),
    "midday":       (4.0,  110.0),
    "shoulder":     (8.0,  160.0),
    "evening_peak": (8.0,  150.0),
}
_PRICE_FLOOR   = 4.0
_PRICE_CEILING = 300.0


def _estimate_solar_24h(month: int, cloud_pct_24h: List[float]) -> List[int]:
    """
    Estimate MISO solar generation (MW) for each of 24 hours.

    Peak installed capacity by month reflects seasonal installed base + daylight.
    Intraday factor is a bell curve centred at solar noon (hour 13).
    Cloud cover linearly reduces output (100% cloud → ~15% of clear-sky).
    """
    peak_mw = {
        1: 8000, 2: 9000,  3: 12000, 4: 14000,
        5: 15000, 6: 16000, 7: 16000, 8: 15000,
        9: 13000, 10: 11000, 11: 8000, 12: 7000,
    }.get(month, 12000)

    # Intraday factors: index 0 = hour 1 (midnight-1am)
    day_factors = [
        0,    0,    0,    0,    0,    0,        # hrs 1-6  night
        0.05, 0.22, 0.45, 0.67,                 # hrs 7-10 morning ramp
        0.84, 0.94, 1.00, 0.96,                 # hrs 11-14 peak
        0.84, 0.62, 0.38, 0.18,                 # hrs 15-18 afternoon decline
        0.04, 0,    0,    0,    0,    0,         # hrs 19-24 dusk/night
    ]

    result = []
    for h in range(24):
        cloud = cloud_pct_24h[h] if h < len(cloud_pct_24h) else 50
        cloud_factor = 1.0 - (cloud / 100.0) * 0.85
        mw = peak_mw * day_factors[h] * cloud_factor
        result.append(round(mw))
    return result


def _sanity_clamp(hour: int, proposed: float, base: float, hist: float = 0.0) -> Tuple[float, bool]:
    """
    Guard against physically impossible prices only.

    Uses absolute seasonal bounds as the primary constraint.
    Adds a loose relative guard (0.30x–2.80x of best anchor) to catch
    extreme outliers without suppressing real-market moves.

    hist = same-weekday historical average for this hour (preferred anchor).
    base = 40-day rolling mean (fallback anchor if no history).
    """
    p = _period(hour)
    abs_lo, abs_hi = _ABS_BOUNDS[p]

    # Best anchor: recent same-weekday history beats the all-days 40-day mean
    anchor = hist if (hist and hist > 4.0) else base
    rel_lo = max(abs_lo, anchor * 0.30)
    rel_hi = min(abs_hi, anchor * 2.80)

    clamped = max(rel_lo, min(rel_hi, proposed))
    return round(clamped, 2), abs(clamped - proposed) > 0.01


# ---------------------------------------------------------------------------
# Prompt builder helpers
# ---------------------------------------------------------------------------

def _weather_block(weather: Dict, hourly_w: List[Dict]) -> str:
    if not weather.get("success") or not hourly_w:
        return "WEATHER: unavailable — assume typical spring shoulder conditions"
    temps  = [h.get("temp_f", 65) for h in hourly_w]
    winds  = [h.get("wind_mph", 10) for h in hourly_w]
    avg_t  = sum(temps) / len(temps)
    min_t, max_t = min(temps), max(temps)
    avg_w  = sum(winds) / len(winds)
    signal = weather.get("trading_signal", "")

    # Build period-level temp + wind table
    rows = []
    for pname, hrs in _TOU.items():
        p_temps = [temps[h-1] for h in hrs if h <= len(temps)]
        p_winds = [winds[h-1] for h in hrs if h <= len(winds)]
        if p_temps:
            at = sum(p_temps)/len(p_temps)
            aw = sum(p_winds)/len(p_winds) if p_winds else avg_w
            rows.append(f"  {pname:>13s} (hrs {hrs[0]:02d}-{hrs[-1]:02d}): "
                        f"temp avg {at:.0f}°F  wind avg {aw:.0f} mph")

    return (
        f"WEATHER (Chicago — MISO load center):\n"
        f"  Overnight low: {min_t:.0f}°F  |  Afternoon high: {max_t:.0f}°F  |  Day avg: {avg_t:.0f}°F\n"
        f"  Wind: avg {avg_w:.0f} mph day\n"
        f"  Trading signal: {signal}\n"
        f"  Per-period breakdown:\n" + "\n".join(rows)
    )


def _gas_block(gas: Dict) -> str:
    if not gas.get("success"):
        return "NATURAL GAS: unavailable — assume $2.20/MMBtu April norm"
    price    = gas.get("price", 2.20)
    baseline = 2.20   # April shoulder baseline
    dev      = price - baseline
    heat_rate = 9.5   # MMBtu/MWh for combined-cycle (typical MISO marginal unit)
    lmp_impact = dev * heat_rate
    direction = (
        f"${dev:+.2f}/MMBtu vs ${baseline} April baseline → "
        f"~${lmp_impact:+.2f}/MWh on all LMPs (heat rate {heat_rate} MMBtu/MWh)"
    )
    return (
        f"NATURAL GAS (Henry Hub):\n"
        f"  Current price: ${price:.2f}/MMBtu\n"
        f"  {direction}\n"
        f"  Signal: {gas.get('trading_signal', '')}"
    )


def _pjm_block(pjm: Dict) -> str:
    if not pjm.get("success"):
        return (
            "NEIGHBOR MARKET (PJM): data unavailable\n"
            "  Assume typical April PJM-MISO spread of ~$8/MWh.\n"
            "  At $8 spread, modest upward pull on MISO shoulder/evening hours."
        )
    price    = pjm.get("pjm_price", 0)
    miso_est = pjm.get("miso_estimate", 28)
    spread   = pjm.get("spread", 0)
    signal   = pjm.get("trading_signal", "")
    hourly   = pjm.get("hourly_prices", [])

    if spread >= 15:
        impact = "STRONG pull — generators prefer PJM, MISO supply tightens hrs 16-24. Add $4-8 to shoulder/evening vs no-spread scenario."
    elif spread >= 8:
        impact = "MODERATE pull — add $2-4 to shoulder/evening hrs vs no-spread scenario."
    elif spread >= 3:
        impact = "SLIGHT pull — minimal MISO supply impact, maybe +$1-2 on evening peak."
    else:
        impact = "No material interface effect on MISO prices."

    hourly_str = ""
    if hourly and len(hourly) >= 24:
        rows = [f"hr{h+1:02d}=${hourly[h]:.1f}" for h in range(11, 24)]
        hourly_str = (
            f"\n  Per-hour PJM DA prices (hrs 12-24 — use as per-hour MISO pull signal):\n"
            f"  {' | '.join(rows)}"
        )

    return (
        f"NEIGHBOR MARKET (PJM Western Hub):\n"
        f"  PJM DA price: ${price:.2f}/MWh\n"
        f"  Typical April MISO baseline: ~${miso_est:.2f}/MWh\n"
        f"  PJM-MISO spread: ${spread:.2f}/MWh\n"
        f"  Interface capacity: ~6-8 GW bidirectional\n"
        f"  Market impact: {impact}\n"
        f"  Signal: {signal}"
        f"{hourly_str}"
    )


def _history_block(
    hist_profile: List[float],
    hist_summary: str,
    base_prices: List[float],
    dow: str,
    weekday_stats: Dict,
) -> str:
    """
    Build history section. Shows per-period avg/min/max from recent same-weekday
    occurrences — gives Claude both the central tendency AND the range to reason within.
    """
    if not hist_profile or len(hist_profile) < 24:
        return (
            f"HISTORICAL {dow.upper()} PATTERNS: not available\n"
            f"  Warning: without same-weekday history, prices will be less accurate.\n"
            f"  Fall back to 40-day base prices as reference."
        )

    hist_avg  = sum(hist_profile) / 24
    base_avg  = sum(base_prices) / 24
    lines = [
        f"HISTORICAL {dow.upper()} PATTERNS (your PRIMARY calibration anchor):",
        f"  {hist_summary}",
        f"  Historical {dow} avg ${hist_avg:.2f}/MWh vs 40-day all-day base avg ${base_avg:.2f}/MWh",
        f"  → The base underestimates because it averages all days; {dow}s are more predictive.",
        f"",
        f"  Per-period stats (last 5 {dow}s): avg  |  min – max  |  std",
    ]

    for pname, hrs in _TOU.items():
        h_vals = [hist_profile[h-1] for h in hrs]
        h_avg  = sum(h_vals) / len(h_vals)
        b_avg  = sum(base_prices[h-1] for h in hrs) / len(hrs)

        # Use per-hour stats from weekday_stats if available
        if weekday_stats:
            period_mins = [weekday_stats.get(h, {}) or {} for h in hrs]
            all_mins = [s.get("min") for s in period_mins if s.get("min") is not None]
            all_maxs = [s.get("max") for s in period_mins if s.get("max") is not None]
            all_stds = [s.get("std") for s in period_mins if s.get("std") is not None]
            p_min = min(all_mins) if all_mins else h_avg
            p_max = max(all_maxs) if all_maxs else h_avg
            p_std = sum(all_stds) / len(all_stds) if all_stds else 0.0
            range_str = f"${h_avg:.2f}  |  ${p_min:.2f} – ${p_max:.2f}  |  ±${p_std:.2f}"
        else:
            range_str = f"${h_avg:.2f}  (no per-hour range data)"

        lines.append(
            f"  {pname:>13s} hrs {hrs[0]:02d}-{hrs[-1]:02d}: {range_str}"
            f"  (base avg ${b_avg:.2f})"
        )

    # Show the most recent 3 same-weekday hourly profiles if available
    if weekday_stats:
        # Find hours 17, 18, 20 as representative peak hours
        peak_hrs = [17, 18, 20]
        recent_rows = []
        for hr in peak_hrs:
            s = weekday_stats.get(hr)
            if s and s.get("recent"):
                recent_rows.append(
                    f"  hr{hr:02d}: last {s['n']} {dow}s = {s['recent']}  (avg ${s['avg']:.2f})"
                )
        if recent_rows:
            lines.append("")
            lines.append(f"  Recent {dow} actuals at key peak hours:")
            lines.extend(recent_rows)

    return "\n".join(lines)


def _adjustment_framework_block(gas: Dict, pjm: Dict, hourly_w: List[Dict], load_forecast: List[float]) -> str:
    """
    Explicit quantitative adjustment framework for Claude to reason with.
    Grounded in MISO market mechanics.
    """
    gas_price = gas.get("price", 2.20) if gas.get("success") else 2.20
    gas_dev   = gas_price - 2.20
    gas_lmp   = gas_dev * 9.5  # $/MWh impact

    spread = pjm.get("spread", 0) if pjm.get("success") else 8.0

    temps     = [h.get("temp_f", 65) for h in hourly_w] if hourly_w else []
    avg_temp  = sum(temps) / len(temps) if temps else 65.0
    temp_dev  = avg_temp - 65.0

    avg_load  = sum(load_forecast) / 24 if load_forecast else 100.0

    return f"""QUANTITATIVE ADJUSTMENT FRAMEWORK:
Use these calibrated market mechanics when adjusting from historical {{}}-day averages:

1. GAS COST PASS-THROUGH (affects all hours equally):
   Gas at ${gas_price:.2f}/MMBtu vs $2.20 baseline → {gas_lmp:+.2f}/MWh on all LMPs
   Heat rate ~9.5 MMBtu/MWh for combined cycle (MISO marginal unit)

2. TEMPERATURE / HVAC DEMAND (period-sensitive):
   Today avg {avg_temp:.0f}°F (deviation {temp_dev:+.0f}°F from 65°F comfort):
   - Cooling demand (>65°F): shoulder/evening most sensitive (+$2-4/MWh per 5°F)
   - Heating demand (<65°F): morning ramp + evening peak most sensitive (+$1-3/MWh per 5°F)
   - Midday: solar partially offsets cooling demand in spring

3. PJM SPREAD EFFECT (shoulder/evening hours primarily):
   Spread ${spread:.0f}/MWh → {"strong +$4-8/MWh" if spread>=15 else "moderate +$2-4/MWh" if spread>=8 else "slight +$1-2/MWh" if spread>=3 else "neutral"} on MISO hrs 16-24
   Night hours: minimal interface effect (overnight transmission slack)

4. LOAD SHAPE EFFECT:
   Avg system load {avg_load:.0f} GW. Non-linear: near-peak hours cause disproportionate price spikes.
   Hours near system peak (typically hrs 17-19): most price-sensitive to any load surprise.

5. WIND (merit-order suppression):
   Each additional 5 GW wind above 10 GW baseline: ~-3% to -5% LMP impact
   Low wind (<7 GW): thermal peakers needed → +$3-6/MWh uplift, especially off-peak hours"""


def _hour_table(base_prices, hist_profile, load_forecast, wind_forecast, hourly_w, weekday_stats=None, pjm_hourly=None) -> str:
    rows = []
    avg_load = sum(load_forecast) / 24 if load_forecast else 100.0
    for h in range(24):
        hr    = h + 1
        base  = base_prices[h]
        load  = load_forecast[h] if h < len(load_forecast) else avg_load
        wind  = wind_forecast[h] if h < len(wind_forecast) else 0
        temp  = f"{hourly_w[h].get('temp_f','?'):.0f}°" if h < len(hourly_w) else "  ?"
        pct   = load / avg_load * 100 if avg_load else 100

        # Historical: show avg and range if available
        if hist_profile and len(hist_profile) > h:
            h_avg = hist_profile[h]
            if weekday_stats and weekday_stats.get(hr):
                s = weekday_stats[hr]
                hist_str = f"hist=${h_avg:.0f}(${s['min']:.0f}-${s['max']:.0f})"
            else:
                hist_str = f"hist=${h_avg:.2f}"
        else:
            hist_str = "hist=n/a         "

        pjm_str = f"  pjm=${pjm_hourly[h]:.1f}" if (pjm_hourly and h < len(pjm_hourly)) else ""

        rows.append(
            f"  hr{hr:02d}[{_period(hr):>13s}]  "
            f"base=${base:5.1f}  {hist_str}  "
            f"load={load:5.0f}GW({pct:3.0f}%)  wind={wind:4.1f}GW  temp={temp}"
            f"{pjm_str}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main refiner class
# ---------------------------------------------------------------------------

class AIRefiner:
    """Claude Opus deep reasoning + Gemini validation for DA-LMP forecasting."""

    def __init__(self):
        self.claude_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini_enabled = True
        else:
            self._gemini_enabled = False
            logger.warning("GEMINI_API_KEY not set — Gemini validation disabled")

    # ------------------------------------------------------------------ #
    # Claude deep reasoning forecast
    # ------------------------------------------------------------------ #
    def claude_refine(
        self,
        base_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float],
        target_date: Optional[str] = None,
        trader_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Claude Opus with extended thinking analyzes all market signals and
        reasons to absolute prices.  Same-weekday history is the primary anchor;
        today's specific conditions (gas, temp, wind, PJM) drive deviations.
        """
        ctx = trader_context or {}
        try:
            # ── Parse date context ────────────────────────────────────
            dow, month_name, is_weekend = "Unknown", "Unknown", False
            if target_date:
                try:
                    dt        = datetime.strptime(target_date, "%B %d, %Y")
                    dow       = dt.strftime("%A")
                    is_weekend = dt.weekday() >= 5
                    month_name = dt.strftime("%B")
                except Exception:
                    pass

            avg_load = sum(load_forecast) / 24 if load_forecast else 100.0
            avg_wind = sum(wind_forecast) / 24 if wind_forecast else 10.0
            base_avg = sum(base_prices) / 24

            weather       = ctx.get("weather", {})
            hourly_w      = weather.get("hourly", []) if weather.get("success") else []
            gas           = ctx.get("gas", {})
            pjm           = ctx.get("pjm", {})
            pjm_hourly    = pjm.get("hourly_prices", []) if pjm.get("success") else []
            hist_profile  = ctx.get("history_profile", [])
            hist_summary  = ctx.get("history_summary", "")
            weekday_stats = ctx.get("weekday_stats", {})
            cal           = ctx.get("calendar", "")
            dl            = ctx.get("daylight_hrs", 0)
            lf            = ctx.get("load_factor", 1.0)

            # Build the adjustment framework with the actual day name
            adj_framework = _adjustment_framework_block(gas, pjm, hourly_w, load_forecast)
            adj_framework = adj_framework.replace("historical {}-day averages", f"historical {dow} averages")

            # ── Build prompt ──────────────────────────────────────────
            prompt = f"""You are an expert MISO energy market analyst and trader.
Your task: produce the most accurate possible 24-hour DA-LMP price forecast for tomorrow.
Think carefully — you have extended reasoning available, use it fully.

━━━ FORECAST TARGET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Date     : {target_date or "tomorrow"} ({dow})
Calendar : {cal}
Day type : {'WEEKEND — commercial/industrial load 20-30% lower than weekday' if is_weekend else 'WEEKDAY — full commercial/industrial load'}
Daylight : {dl:.1f}h  |  System avg load forecast: {avg_load:.0f} GW  |  System avg wind: {avg_wind:.1f} GW
Load factor vs seasonal norm: {lf:.3f}

━━━ PRIMARY CALIBRATION: SAME-WEEKDAY HISTORY ━━━━━━━━━━━━━━━━━━━━━━━━
This is your most important anchor. Recent {dow}s cleared at these prices.
Start here, then adjust for how TODAY specifically differs from a typical {dow}.

{_history_block(hist_profile, hist_summary, base_prices, dow, weekday_stats)}

━━━ TODAY'S MARKET CONDITIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{_weather_block(weather, hourly_w)}

{_gas_block(gas)}

{_pjm_block(pjm)}

━━━ QUANTITATIVE ADJUSTMENT FRAMEWORK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{adj_framework}

━━━ HOUR-BY-HOUR REFERENCE TABLE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(base = 40-day all-day mean | hist = same-{dow} avg [min–max] | pjm = PJM Western Hub DA price)
PJM column is critical for hrs 12–24: when PJM is elevated relative to MISO baseline,
interface flows tighten MISO supply — effect strongest at shoulder/evening peak hours.
{_hour_table(base_prices, hist_profile, load_forecast, wind_forecast, hourly_w, weekday_stats, pjm_hourly=pjm_hourly)}

━━━ SOLAR GENERATION NOTE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
April MISO solar: ~4-8 GW peak at hrs 11-15 (clears before sunset ~hr 19).
Solar suppresses midday prices, creates steep ramp at 16:00-18:00 as solar drops.
This is the "duck curve" — prices dip midday, then spike sharply as solar exits.

━━━ REASONING APPROACH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each period, reason through:

1. ANCHOR: What did same-{dow}s actually clear at? (use history above as your floor/ceiling range)
2. GAS ADJUSTMENT: How does today's gas price shift costs vs historical?
3. TEMPERATURE: Is today warmer/cooler than a typical {dow}? By how much?
4. WIND: Is today's wind high or low vs normal? How does it affect thermal dispatch?
5. PJM: How does today's interface spread compare to typical?
6. LOAD SHAPE: Where are the load peaks and valleys in today's forecast?
7. SYNTHESIS: What price range does all this point to for each hour?

KEY INSIGHT: The 40-day base (avg ${base_avg:.2f}) is NOT your target.
Same-{dow} history is your target. Adjust from THAT for today's specific conditions.

SPECIAL FOCUS — HOURS 12–24 (where accuracy is most critical):
  • Use the pjm= column in the hour table as a per-hour MISO pull signal.
  • Hrs 12–15: solar suppresses midday — BUT if PJM is already rising, the duck
    curve dip may be shallower. Cross-check pjm column vs hist to size the dip.
  • Hrs 16–19 (shoulder): PJM interface effect is strongest here. PJM hourly
    prices in this window are your best single predictor of MISO shoulder prices.
    Do NOT flatten this period — real MISO shoulders show $5-20 ramps within 16-19.
  • Hrs 20–24: MISO follows PJM evening wind-down with ~1h lag. Use the pjm=
    column to gauge how fast PJM drops and mirror that shape in MISO.

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with valid JSON (no markdown fences, no extra text outside JSON):
{{
  "reasoning": {{
    "anchor_assessment":  "<How do recent {dow}s compare to the 40-day base? What price level do they suggest overall?>",
    "todays_deviations":  "<How does today's temp/wind/gas/PJM deviate from a typical {dow}? Net up or down vs history?>",
    "night_1_6":          "<Reasoning for night hours: gas cost floor, wind overnight, heating demand if cold>",
    "morning_7_9":        "<Reasoning: morning HVAC ramp, load ramp speed, gas cost>",
    "midday_10_15":       "<Reasoning: solar suppression (4-8 GW clears around hr 11-15), HVAC demand, duck curve dip. Check pjm column: if PJM is rising hrs 12-15, MISO midday depression may be shallower than usual>",
    "shoulder_16_19":     "<KEY PERIOD — reason explicitly: PJM pull magnitude (use per-hour PJM prices from table), AC load at peak temp, solar drop ramp, load level, wind. PJM hrs 16-19 price is the single strongest signal for MISO shoulder>",
    "evening_20_24":      "<Reasoning: use PJM hrs 20-24 from table — how fast does PJM drop? MISO follows with ~1h lag. Residual AC, gas as marginal unit, wind overnight ramp>"
  }},
  "prices": {{
    "1":  <$/MWh>, "2":  <$/MWh>, "3":  <$/MWh>, "4":  <$/MWh>,
    "5":  <$/MWh>, "6":  <$/MWh>, "7":  <$/MWh>, "8":  <$/MWh>,
    "9":  <$/MWh>, "10": <$/MWh>, "11": <$/MWh>, "12": <$/MWh>,
    "13": <$/MWh>, "14": <$/MWh>, "15": <$/MWh>, "16": <$/MWh>,
    "17": <$/MWh>, "18": <$/MWh>, "19": <$/MWh>, "20": <$/MWh>,
    "21": <$/MWh>, "22": <$/MWh>, "23": <$/MWh>, "24": <$/MWh>
  }}
}}"""

            # ── Call Claude Opus with extended thinking ───────────────
            logger.info("Calling claude-opus-4-6 with extended thinking (budget=10000 tokens)...")
            try:
                response = self.claude_client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=16000,
                    thinking={"type": "enabled", "budget_tokens": 10000},
                    messages=[{"role": "user", "content": prompt}],
                )
                # Extended thinking response has thinking + text blocks
                raw = next(
                    (block.text for block in response.content if block.type == "text"),
                    "",
                )
                if not raw:
                    raise ValueError("No text block in extended-thinking response")
                thinking_tokens = sum(
                    getattr(block, "thinking", "").__len__()
                    for block in response.content
                    if block.type == "thinking"
                )
                logger.info(f"Extended thinking: ~{thinking_tokens} thinking chars used")
            except Exception as thinking_err:
                logger.warning(f"Extended thinking failed ({thinking_err}), falling back to standard Opus call")
                response = self.claude_client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=6000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text if response.content else ""

            if not raw or not raw.strip():
                raise ValueError("Claude returned empty response")

            # ── Parse response ────────────────────────────────────────
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            json_match = re.search(r'\{[\s\S]*\}', clean)
            if not json_match:
                raise ValueError(f"No JSON in response (first 200: {clean[:200]!r})")
            data  = json.loads(json_match.group(0))

            prices_raw    = data.get("prices", {})
            reasoning_map = data.get("reasoning", {})
            refined_prices = []
            clamped_count  = 0

            for hour in range(1, 25):
                base     = base_prices[hour - 1]
                hist     = hist_profile[hour - 1] if hist_profile and len(hist_profile) >= hour else 0.0
                proposed = float(prices_raw.get(str(hour), base))
                clamped, was_clamped = _sanity_clamp(hour, proposed, base, hist)
                if was_clamped:
                    clamped_count += 1
                    logger.warning(
                        f"  Sanity clamp hr{hour:02d}: Claude ${proposed:.2f} → ${clamped:.2f} "
                        f"(base ${base:.2f}, hist ${hist:.2f}, period {_period(hour)})"
                    )
                refined_prices.append(clamped)

            # Log period reasoning
            logger.info("Claude period reasoning:")
            for pkey, ptext in reasoning_map.items():
                logger.info(f"  [{pkey}] {ptext[:120]}")

            # Log period-level summary
            for pname, hrs in _TOU.items():
                b_avg = sum(base_prices[h-1] for h in hrs) / len(hrs)
                h_avg = (sum(hist_profile[h-1] for h in hrs) / len(hrs)) if hist_profile else None
                r_avg = sum(refined_prices[h-1] for h in hrs) / len(hrs)
                hist_str = f" | hist_avg=${h_avg:.2f}" if h_avg else ""
                logger.info(
                    f"  {pname:>13s}: base=${b_avg:.2f}{hist_str} → Claude=${r_avg:.2f} "
                    f"(Δ_base={r_avg-b_avg:+.2f})"
                )

            return {
                "success":          True,
                "refined_prices":   refined_prices,
                "reasoning":        str(reasoning_map),
                "period_reasoning": reasoning_map,
                "clamped_hours":    clamped_count,
            }

        except Exception as e:
            logger.error(f"Claude refinement failed: {e}")
            return {
                "success":        False,
                "refined_prices": list(base_prices),
                "error":          str(e),
            }

    # ------------------------------------------------------------------ #
    # Signal-driven forecast (trader's framework — new primary method)
    # ------------------------------------------------------------------ #
    def claude_refine_with_signals(
        self,
        base_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float],
        target_date: Optional[str] = None,
        trader_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Signal-driven DA-LMP forecast using the professional trader's framework.

        Price ≈ Demand − Renewables + Outages + Congestion + Gas_floor + PJM_pull

        Does NOT start from a historical base and adjust percentages.
        Instead reasons from fundamentals for each hour.

        Returns same shape as claude_refine() plus:
          signal_summary, peak_driver, risk_flags
        """
        ctx = trader_context or {}
        try:
            # ── Date context ─────────────────────────────────────────────
            dow, is_holiday_str = "Unknown", "No"
            tomorrow_month = date.today().month
            if target_date:
                try:
                    dt = datetime.strptime(target_date, "%B %d, %Y")
                    dow = dt.strftime("%A")
                    tomorrow_month = dt.month
                except Exception:
                    pass

            is_holiday = ctx.get("is_holiday", False)
            is_holiday_str = "Yes" if is_holiday else "No"

            # ── Extract signals ──────────────────────────────────────────
            weather      = ctx.get("weather", {})
            gas          = ctx.get("gas", {})
            pjm          = ctx.get("pjm", {})
            rt_lmp_data  = ctx.get("rt_lmp", {})
            outages_data = ctx.get("outages", {})

            # Weather arrays (24h)
            temps_f    = weather.get("temps_f", []) if weather.get("success") else []
            wind_mph   = weather.get("wind_mph", []) if weather.get("success") else []
            cloud_pct  = weather.get("cloud_pct", []) if weather.get("success") else []

            # Pad to 24 values if short
            def _pad(lst, default):
                return (lst + [default] * 24)[:24]

            temps_f   = _pad(temps_f, 65)
            wind_mph  = _pad(wind_mph, 10)
            cloud_pct = _pad(cloud_pct, 50)

            # Load & wind (MW — MISO reports in GW sometimes, keep as-is)
            avg_load  = sum(load_forecast) / 24 if load_forecast else 100.0
            avg_wind  = sum(wind_forecast) / 24 if wind_forecast else 10.0
            load_vs_avg = (
                f"ABOVE average by {(max(load_forecast)/avg_load - 1)*100:.0f}% at peak"
                if max(load_forecast) > avg_load * 1.08 else
                "Near or below average"
            )

            # Solar estimate (MW by hour)
            solar_est = _estimate_solar_24h(tomorrow_month, cloud_pct)
            total_renewable = [
                round((wind_forecast[h] if h < len(wind_forecast) else avg_wind) + solar_est[h])
                for h in range(24)
            ]

            # Gas
            gas_price = gas.get("price", 2.20) if gas.get("success") else 2.20
            gas_implied = round(gas_price * 9, 2)

            # RT LMP
            rt_lmp_current = rt_lmp_data.get("rt_lmp_current")
            rt_trend       = rt_lmp_data.get("rt_lmp_trend", "flat")
            if rt_lmp_current is None:
                rt_lmp_str = "unavailable"
            else:
                rt_lmp_str = f"{rt_lmp_current:.2f}"

            # PJM
            pjm_hourly = pjm.get("hourly_prices", []) if pjm.get("success") else []
            if len(pjm_hourly) < 24:
                from src.scrapers.pjm_scraper import _shape_from_daily, _SEASONAL_PJM
                pjm_avg = pjm.get("pjm_price", _SEASONAL_PJM.get(tomorrow_month, 40.0))
                pjm_hourly = _shape_from_daily(pjm_avg)

            # Outages
            outage_mw   = outages_data.get("outage_mw", 0) if outages_data.get("success") else 0
            alert_level = outages_data.get("alert_level", "normal") if outages_data.get("success") else "unknown (data unavailable)"

            # Format lists as compact strings for prompt
            def _fmt_list(lst, fmt=".0f"):
                return "[" + ", ".join(format(v, fmt) for v in lst) + "]"

            # ── Build prompt ─────────────────────────────────────────────
            prompt = f"""You are a professional MISO energy market analyst using the same methodology as experienced DA-LMP traders.

YOUR FRAMEWORK (use this exact reasoning order):

STEP 1 - WEATHER CHECK:
Tomorrow's hourly temperatures (°F): {_fmt_list(temps_f)}
Tomorrow's wind speed (mph): {_fmt_list(wind_mph, '.1f')}
Tomorrow's cloud cover (%): {_fmt_list(cloud_pct)}
→ High temps (>85°F) = AC load spike. Low temps (<30°F) = heating load spike. Either drives prices up.
→ High wind (>15mph sustained) = strong renewable output = price suppression.
→ High cloud cover = solar generation impaired = less supply = prices slightly up midday.

STEP 2 - LOAD FORECAST:
MISO expected load tomorrow (MW by hour): {_fmt_list(load_forecast)}
Average load: {avg_load:.0f} MW
Is tomorrow's load above or below average? {load_vs_avg}
→ Load above average = upward price pressure. Load below = downward pressure.

STEP 3 - WIND + SOLAR GENERATION:
MISO wind generation forecast (MW by hour): {_fmt_list(wind_forecast)}
Estimated solar output (MW, based on cloud cover + season): {_fmt_list(solar_est)}
Total renewable forecast by hour: {_fmt_list(total_renewable)}
→ High renewable output = more supply = lower prices. Particularly impacts hours 10-16 (solar peak) and whenever wind is strong.

STEP 4 - OUTAGES + GRID STRESS:
Reported unplanned outages (MW offline): {outage_mw:,}
Grid alert level: {alert_level}
→ Every 1,000 MW of outages in a tight market = roughly $3-8/MWh upward pressure during peak hours.
→ If alert_level is 'elevated' or 'high', add spike risk premium to hours 17-20.

STEP 5 - REAL-TIME PRICE MOMENTUM:
Current real-time LMP: ${rt_lmp_str}/MWh
RT price trend (last 3 hours): {rt_trend}
→ Rising RT prices = market tightening = DA prices likely to follow up.
→ Falling RT prices = oversupply = DA prices at risk of coming in lower.

STEP 6 - NEIGHBOR MARKET (PJM):
PJM Western Hub DA-LMP by hour: {_fmt_list(pjm_hourly, '.1f')}
MISO typically trades at a $2-6/MWh discount to PJM due to interface limits.
→ If PJM is high (>$60), MISO will be pulled up toward it.
→ If PJM is low (<$35), there is no upward pull from the neighbor market.

STEP 7 - NATURAL GAS:
Henry Hub gas price: ${gas_price:.2f}/MMBtu
MISO gas heat rate: ~9,000 BTU/kWh
Gas-implied energy cost: ${gas_implied:.2f}/MWh
→ This is the floor for gas-fired generation marginal cost. Prices rarely stay below this during peak hours for extended periods.

STEP 8 - DAY-OF-WEEK + CALENDAR:
Tomorrow is: {dow}
Is tomorrow a holiday? {is_holiday_str}
→ Monday-Friday: normal commercial + industrial load
→ Saturday: -10 to -15% load vs weekday
→ Sunday: -15 to -20% load vs weekday
→ Holiday: treat like Sunday or lower

NOW APPLY THE TRADER'S FORMULA FOR EACH HOUR:
Price ≈ Demand_factor − Renewable_factor + Outage_factor + Congestion_factor + Gas_floor_factor + PJM_pull_factor

DO NOT start from a historical base price and adjust it. START FROM THE SIGNALS.

For each of the 24 hours, reason through:
- What is demand doing this hour? (load forecast, temperature, day-of-week)
- What are renewables doing? (wind MW, solar MW)
- Any outage risk this hour?
- What is PJM at this hour? How much does MISO track it?
- Is the gas floor relevant? (typically yes for hours 7-22)
- What is the net price?

PAY SPECIAL ATTENTION TO:
- Hours 1-6 (Night): Low demand, gas floor sets minimum, wind matters most
- Hours 7-9 (Morning ramp): Load accelerates, watch temperature
- Hours 10-15 (Midday): Solar peak, renewable suppression window
- Hours 16-20 (Evening peak): HIGHEST RISK HOURS. Demand peaks, solar falls off, wind often drops at sunset. This is where prices spike. Do NOT underestimate.
- Hours 21-24 (Late evening): Load drops, transition down

RESPOND ONLY WITH THIS JSON (no markdown fences, no text outside the JSON):
{{
  "forecast": {{
    "1": 45.20,
    "2": 43.80,
    "3": 42.10,
    "4": 41.50,
    "5": 42.00,
    "6": 44.30,
    "7": 48.00,
    "8": 52.00,
    "9": 55.00,
    "10": 50.00,
    "11": 46.00,
    "12": 44.00,
    "13": 43.00,
    "14": 44.00,
    "15": 46.00,
    "16": 52.00,
    "17": 62.00,
    "18": 68.00,
    "19": 64.00,
    "20": 58.00,
    "21": 52.00,
    "22": 48.00,
    "23": 44.00,
    "24": 40.00
  }},
  "signal_summary": "2-3 sentence summary of the dominant signals driving tomorrow's prices",
  "peak_driver": "what is driving the peak hour price",
  "risk_flags": "any spike or crash risks worth flagging to the trader"
}}

All prices in $/MWh. Be specific. Do not hedge. Pick the price you believe is correct.
Forecast date: {target_date or "tomorrow"} ({dow})"""

            # ── Call Claude with extended thinking ────────────────────────
            logger.info("Calling claude-opus-4-6 (signal-driven, extended thinking=10000)...")
            raw = ""
            try:
                response = self.claude_client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=16000,
                    thinking={"type": "enabled", "budget_tokens": 10000},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = next(
                    (block.text for block in response.content if block.type == "text"),
                    "",
                )
                if not raw:
                    raise ValueError("No text block in extended-thinking response")
                logger.info(f"Extended thinking response: {len(raw)} chars")
            except Exception as think_err:
                logger.warning(f"Extended thinking failed ({think_err}), retrying standard...")
                response = self.claude_client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=6000,
                    messages=[{"role": "user", "content": prompt}],
                )
                if response.content:
                    raw = response.content[0].text
                logger.info(f"Standard call response: {len(raw)} chars | stop={response.stop_reason}")

            if not raw or not raw.strip():
                raise ValueError(f"Claude returned empty response (stop={getattr(response, 'stop_reason', '?')})")

            # ── Parse — strip markdown fences and find the JSON object ────
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            # Find the outermost JSON object in case Claude added prose before/after
            json_match = re.search(r'\{[\s\S]*\}', clean)
            if not json_match:
                raise ValueError(f"No JSON object found in response (first 200: {clean[:200]!r})")
            data  = json.loads(json_match.group(0))

            prices_raw     = data.get("forecast", data.get("prices", {}))
            signal_summary = data.get("signal_summary", "")
            peak_driver    = data.get("peak_driver", "")
            risk_flags     = data.get("risk_flags", "")

            refined_prices = []
            clamped_count  = 0
            for hour in range(1, 25):
                base     = base_prices[hour - 1]
                proposed = float(prices_raw.get(str(hour), base))
                clamped, was_clamped = _sanity_clamp(hour, proposed, base)
                if was_clamped:
                    clamped_count += 1
                    logger.warning(
                        f"  Clamp hr{hour:02d}: ${proposed:.2f} → ${clamped:.2f} "
                        f"(period={_period(hour)})"
                    )
                refined_prices.append(clamped)

            logger.info(f"Signal-driven forecast: clamped={clamped_count} hours")
            logger.info(f"Signal summary: {signal_summary[:120]}")
            logger.info(f"Risk flags: {risk_flags[:120]}")

            return {
                "success":        True,
                "refined_prices": refined_prices,
                "signal_summary": signal_summary,
                "peak_driver":    peak_driver,
                "risk_flags":     risk_flags,
                "clamped_hours":  clamped_count,
            }

        except Exception as e:
            logger.error(f"claude_refine_with_signals failed: {e}")
            return {
                "success":        False,
                "refined_prices": list(base_prices),
                "signal_summary": "",
                "peak_driver":    "",
                "risk_flags":     "",
                "error":          str(e),
            }

    # ------------------------------------------------------------------ #
    # Gemini validation
    # ------------------------------------------------------------------ #
    def gemini_validate(
        self,
        claude_prices: List[float],
        base_prices:   List[float],
        load_forecast: List[float],
        wind_forecast: List[float],
    ) -> Dict:
        """
        Cross-check Claude's prices against absolute seasonal thresholds.
        Validation is calibrated to NOT flag valid above-base prices — the base
        is known to underestimate, so deviations from it are expected.
        """
        if not self._gemini_enabled:
            return {"success": False, "validation_passed": True, "error": "Gemini disabled"}

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")

            deviations = [
                round((claude_prices[i] - base_prices[i]) / base_prices[i] * 100, 2)
                if base_prices[i] else 0.0
                for i in range(24)
            ]

            # Only auto-flag prices outside absolute seasonal bounds
            auto_flagged = []
            for h in range(24):
                p     = _period(h + 1)
                lo, hi = _ABS_BOUNDS.get(p, (_PRICE_FLOOR, _PRICE_CEILING))
                if claude_prices[h] < lo or claude_prices[h] > hi:
                    auto_flagged.append(h + 1)

            prompt = f"""Validate these MISO DA-LMP spring shoulder season price forecasts.

IMPORTANT: The 40-day base prices UNDERESTIMATE real prices (include weekends, all conditions).
Prices above base by 50-100% are NORMAL for shoulder/evening hours on a weekday.
Flag ONLY prices that are physically implausible for MISO spring:

Base prices (40-day mean, 24h): {[round(p, 2) for p in base_prices]}
Claude-reasoned prices (24h)  : {[round(p, 2) for p in claude_prices]}
% deviation from base         : {deviations}

MISO spring ABSOLUTE suspicious thresholds (flag if clearly outside):
  Night  hrs 1-6  : below $4 or above $80 → suspicious
  Morning hrs 7-9 : below $6 or above $110 → suspicious
  Midday hrs 10-15: below $4 or above $110 → suspicious
  Shoulder 16-19  : below $8 or above $160 → suspicious
  Evening  20-24  : below $8 or above $150 → suspicious

Auto-flagged (outside absolute bounds): {auto_flagged}

Respond ONLY with valid JSON:
{{
  "validation_passed": true or false,
  "concerns": "brief summary or empty string",
  "flagged_hours": [list of hour numbers with genuinely suspicious prices],
  "recommended_rollbacks": {{"<hour>": <corrected_price>}}
}}"""

            resp  = model.generate_content(prompt)
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip(), flags=re.MULTILINE)
            data  = json.loads(clean)

            all_flagged = sorted(set(auto_flagged + data.get("flagged_hours", [])))
            logger.info(
                f"Gemini: passed={data.get('validation_passed', True)} "
                f"| flagged={all_flagged}"
            )
            if data.get("concerns"):
                logger.info(f"Gemini concerns: {data['concerns']}")

            return {
                "success":               True,
                "validation_passed":     data.get("validation_passed", True),
                "concerns":              data.get("concerns", ""),
                "flagged_hours":         all_flagged,
                "recommended_rollbacks": data.get("recommended_rollbacks", {}),
            }

        except Exception as e:
            logger.warning(f"Gemini validation failed (non-critical): {e}")
            return {"success": False, "validation_passed": True, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Merge
    # ------------------------------------------------------------------ #
    @staticmethod
    def merge_results(
        base_prices:   List[float],
        claude_prices: List[float],
        gemini_result: Dict,
    ) -> List[float]:
        """
        Gemini approved → use Claude prices.
        Gemini flagged → apply rollbacks for specific hours, blend the rest.
        """
        final = [round(p, 2) for p in claude_prices]

        if not gemini_result.get("validation_passed", True):
            logger.warning("Gemini flagged concerns — applying surgical corrections")
            rollbacks = gemini_result.get("recommended_rollbacks", {})
            flagged   = gemini_result.get("flagged_hours", [])

            for h in flagged:
                idx = h - 1
                if not (0 <= idx < 24):
                    continue
                if str(h) in rollbacks:
                    corrected = round(float(rollbacks[str(h)]), 2)
                    logger.info(f"  hr{h:02d}: Gemini rollback ${claude_prices[idx]:.2f} → ${corrected:.2f}")
                    final[idx] = corrected
                else:
                    blended = round((base_prices[idx] + claude_prices[idx]) / 2.0, 2)
                    logger.info(f"  hr{h:02d}: blend ${claude_prices[idx]:.2f}+${base_prices[idx]:.2f} → ${blended:.2f}")
                    final[idx] = blended
        else:
            logger.info("Gemini approved — using Claude-reasoned prices")

        return final
