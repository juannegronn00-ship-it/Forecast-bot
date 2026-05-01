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
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
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

    if spread >= 15:
        impact = "STRONG pull — generators prefer PJM, MISO supply tightens hrs 16-24. Add $4-8 to shoulder/evening vs no-spread scenario."
    elif spread >= 8:
        impact = "MODERATE pull — add $2-4 to shoulder/evening hrs vs no-spread scenario."
    elif spread >= 3:
        impact = "SLIGHT pull — minimal MISO supply impact, maybe +$1-2 on evening peak."
    else:
        impact = "No material interface effect on MISO prices."

    return (
        f"NEIGHBOR MARKET (PJM Western Hub):\n"
        f"  PJM DA price: ${price:.2f}/MWh\n"
        f"  Typical April MISO baseline: ~${miso_est:.2f}/MWh\n"
        f"  PJM-MISO spread: ${spread:.2f}/MWh\n"
        f"  Interface capacity: ~6-8 GW bidirectional\n"
        f"  Market impact: {impact}\n"
        f"  Signal: {signal}"
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


def _hour_table(base_prices, hist_profile, load_forecast, wind_forecast, hourly_w, weekday_stats=None) -> str:
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

        rows.append(
            f"  hr{hr:02d}[{_period(hr):>13s}]  "
            f"base=${base:5.1f}  {hist_str}  "
            f"load={load:5.0f}GW({pct:3.0f}%)  wind={wind:4.1f}GW  temp={temp}"
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
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text

            # ── Parse response ────────────────────────────────────────
            clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            data  = json.loads(clean)

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
