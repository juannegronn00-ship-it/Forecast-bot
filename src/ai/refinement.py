import logging
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from anthropic import Anthropic
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Time-of-use period definitions  (hour 1-24, where hour 1 = midnight-1am)
# --------------------------------------------------------------------------- #
TOU_PERIODS = {
    "night":        list(range(1, 7)),    # hours  1-6   (midnight–5 AM)
    "morning_ramp": list(range(7, 10)),   # hours  7-9   (6–8 AM)
    "midday":       list(range(10, 16)),  # hours 10-15  (9 AM–2 PM)
    "shoulder":     list(range(16, 20)),  # hours 16-19  (3–6 PM)
    "evening_peak": list(range(20, 25)),  # hours 20-24  (7–11 PM)
}

# Maximum absolute adjustment allowed per period (decimal fraction of base price)
# These caps exist because morning was $2 off (ok) but later hours were $15-20 off (not ok).
MAX_CAPS = {
    "night":        0.01,   # ±1 %
    "morning_ramp": 0.02,   # ±2 %
    "midday":       0.02,   # ±2 %
    "shoulder":     0.02,   # ±2 %
    "evening_peak": 0.03,   # ±3 %
}


def get_period(hour: int) -> str:
    for period, hours in TOU_PERIODS.items():
        if hour in hours:
            return period
    return "midday"


class AIRefiner:
    """Refine DA-LMP forecasts using Claude (expert trading logic) + Gemini (validation)."""

    def __init__(self):
        self.claude_client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini_enabled = True
        else:
            self._gemini_enabled = False
            logger.warning("GEMINI_API_KEY not set — Gemini validation disabled")

    def claude_refine(
        self,
        base_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float],
        target_date: Optional[str] = None,
    ) -> Dict:
        """
        Use Claude with expert MISO trading methodology to fine-tune prices.

        Key principle: base prices are already grounded in ~40 days of historical
        data from fantods.  Claude's job is FINE-TUNING based on the load/wind
        shape, not wholesale repricing.  Adjustments are hard-capped per period.
        """
        try:
            avg_load = sum(load_forecast) / len(load_forecast) if load_forecast else 100.0
            avg_wind = sum(wind_forecast) / len(wind_forecast) if wind_forecast else 10.0

            # Day-of-week context
            dow = "Unknown"
            is_weekend = False
            month_name = "Unknown"
            if target_date:
                try:
                    dt = datetime.strptime(target_date, "%B %d, %Y")
                    dow = dt.strftime("%A")
                    is_weekend = dt.weekday() >= 5
                    month_name = dt.strftime("%B")
                except Exception:
                    pass

            # Per-hour context table
            rows = []
            for h in range(24):
                hr = h + 1
                period = get_period(hr)
                cap = MAX_CAPS[period]
                load_pct = (load_forecast[h] / avg_load * 100) if avg_load else 100
                rows.append(
                    f"  hr{hr:02d} [{period:>13s}] "
                    f"base=${base_prices[h]:6.2f}  "
                    f"load={load_forecast[h]:6.0f}GW({load_pct:3.0f}%)  "
                    f"wind={wind_forecast[h]:5.1f}GW  "
                    f"max_adj=±{cap*100:.0f}%"
                )

            prompt = f"""You are a veteran MISO energy market trader with decades of DA-LMP forecasting experience.

FORECAST TARGET
  Date   : {target_date or 'tomorrow'} ({dow})
  Season : {month_name} — spring shoulder season (low heating AND low cooling load vs. winter/summer)
  Weekend: {'YES → lower industrial/commercial load, expect softer prices overall' if is_weekend else 'NO → normal weekday load profile'}
  Avg system load : {avg_load:.0f} GW
  Avg wind output : {avg_wind:.1f} GW

HOURLY DATA (base prices = ~40-day rolling historical mean from fantods)
{chr(10).join(rows)}

YOUR JOB — FINE-TUNE, DO NOT REPRICE
The base prices already reflect history.  Errors of $15-20/MWh have occurred when
adjustments were too aggressive.  Apply these trading rules with discipline:

RULE 1 — DEMAND SIGNAL (per hour vs. daily avg load)
  Load > 110 % of avg  →  small positive adj  (demand pressure)
  Load  90–110 % of avg →  near-zero adj       (balanced)
  Load < 90 % of avg   →  small negative adj  (surplus supply)

RULE 2 — WIND SIGNAL (merit-order suppression)
  Wind > 15 GW  →  slight negative adj  (cheap wind displaces gas)
  Wind  8–15 GW →  near-zero
  Wind <  8 GW  →  slight positive adj  (more thermal dispatch)

RULE 3 — SPRING DISCOUNT
  April/May shoulder season = lower overall demand.  When in doubt, lean slightly
  negative.  Do NOT push prices upward without clear load/wind justification.

RULE 4 — WEEKEND DISCOUNT (if weekend)
  Add −0.5 % to every hour on top of other adjustments.

RULE 5 — NIGHT FLOOR (hours 1-6)
  Night prices rarely exceed $28 in MISO spring.  Clamp night adjustments near zero
  unless load is unusually high.

RULE 6 — CAPS ARE HARD LIMITS
  The max_adj column shows the absolute cap for each period.
  NEVER exceed it.  A +5 % adjustment on a $40 base = $2 error (acceptable).
  A +15 % adjustment = $6 error (unacceptable).  Stay within caps.

RESPOND ONLY with valid JSON (no markdown fences, no extra text):
{{
  "adjustments": {{
    "1":  <float in [-{MAX_CAPS['night']}, +{MAX_CAPS['night']}]>,
    "2":  <float>,
    "3":  <float>,
    "4":  <float>,
    "5":  <float>,
    "6":  <float>,
    "7":  <float in [-{MAX_CAPS['morning_ramp']}, +{MAX_CAPS['morning_ramp']}]>,
    "8":  <float>,
    "9":  <float>,
    "10": <float in [-{MAX_CAPS['midday']}, +{MAX_CAPS['midday']}]>,
    "11": <float>,
    "12": <float>,
    "13": <float>,
    "14": <float>,
    "15": <float>,
    "16": <float in [-{MAX_CAPS['shoulder']}, +{MAX_CAPS['shoulder']}]>,
    "17": <float>,
    "18": <float>,
    "19": <float>,
    "20": <float in [-{MAX_CAPS['evening_peak']}, +{MAX_CAPS['evening_peak']}]>,
    "21": <float>,
    "22": <float>,
    "23": <float>,
    "24": <float>
  }},
  "reasoning": "2-3 sentences covering load shape, wind signal, and any seasonal or weekend factors you applied"
}}"""

            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1400,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text
            clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
            data = json.loads(clean)

            adjustments = data.get('adjustments', {})
            refined_prices = []
            clamped_count = 0

            for hour in range(1, 25):
                base = base_prices[hour - 1]
                raw_adj = float(adjustments.get(str(hour), 0.0))
                period = get_period(hour)
                cap = MAX_CAPS[period]
                clamped_adj = max(-cap, min(cap, raw_adj))
                if abs(clamped_adj - raw_adj) > 0.0001:
                    clamped_count += 1
                    logger.debug(f"Hour {hour}: clamped {raw_adj:+.4f} → {clamped_adj:+.4f}")
                refined = base * (1.0 + clamped_adj)
                refined_prices.append(round(refined, 2))

            avg_adj_pct = (sum(float(adjustments.get(str(h), 0)) for h in range(1, 25)) / 24) * 100
            logger.info(f"Claude refinement OK | avg_raw_adj={avg_adj_pct:+.2f}% | clamped={clamped_count} hours")
            logger.info(f"Reasoning: {data.get('reasoning', '')}")

            return {
                'success': True,
                'refined_prices': refined_prices,
                'reasoning': data.get('reasoning', ''),
                'adjustments': adjustments,
                'clamped_hours': clamped_count,
            }

        except Exception as e:
            logger.error(f"Claude refinement failed: {e}")
            return {
                'success': False,
                'refined_prices': list(base_prices),
                'error': str(e),
            }

    def gemini_validate(
        self,
        claude_prices: List[float],
        base_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float],
    ) -> Dict:
        """
        Cross-check Claude's prices against base prices.
        Auto-flag any hour where Claude deviated >5 % from the base.
        Gemini decides whether to approve, rollback, or blend flagged hours.
        """
        if not self._gemini_enabled:
            return {'success': False, 'validation_passed': True, 'error': 'Gemini disabled (no key)'}

        try:
            model = genai.GenerativeModel('gemini-2.0-flash')

            deviations = []
            for i in range(24):
                pct = (claude_prices[i] - base_prices[i]) / base_prices[i] * 100 if base_prices[i] else 0.0
                deviations.append(round(pct, 2))

            auto_flagged = [h + 1 for h, d in enumerate(deviations) if abs(d) > 5.0]

            prompt = f"""Validate these MISO DA-LMP spring-season price forecasts.

Base prices (historical mean, 24h): {[round(p, 2) for p in base_prices]}
Claude-adjusted prices (24h)      : {[round(p, 2) for p in claude_prices]}
Deviation from base (%):           {deviations}
Load forecast (GW, 24h)           : {[round(l, 1) for l in load_forecast]}
Wind forecast (GW, 24h)           : {[round(w, 1) for w in wind_forecast]}
Auto-flagged hours (>5% deviation): {auto_flagged}

Validation rules:
- Any hour deviating >5% from base is suspicious unless load/wind strongly justifies it
- Night hours (1-6) above $30 are suspicious in spring
- Evening hours (20-24) above $65 are suspicious in spring shoulder season
- If a price seems wrong, suggest a specific corrected value

Respond ONLY with valid JSON (no markdown):
{{
  "validation_passed": true or false,
  "concerns": "brief summary",
  "flagged_hours": [list of hour numbers],
  "recommended_rollbacks": {{"<hour>": <corrected_price>}}
}}"""

            response = model.generate_content(prompt)
            clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.text.strip(), flags=re.MULTILINE)
            data = json.loads(clean)

            all_flagged = sorted(set(auto_flagged + data.get('flagged_hours', [])))
            logger.info(f"Gemini validation: passed={data.get('validation_passed', True)} | flagged={all_flagged}")
            if data.get('concerns'):
                logger.info(f"Gemini concerns: {data['concerns']}")

            return {
                'success': True,
                'validation_passed': data.get('validation_passed', True),
                'concerns': data.get('concerns', ''),
                'flagged_hours': all_flagged,
                'recommended_rollbacks': data.get('recommended_rollbacks', {}),
            }

        except Exception as e:
            logger.warning(f"Gemini validation failed (non-critical): {e}")
            return {'success': False, 'validation_passed': True, 'error': str(e)}

    @staticmethod
    def merge_results(
        base_prices: List[float],
        claude_prices: List[float],
        gemini_validation: Dict,
    ) -> List[float]:
        """
        Merge strategy:
        - If Gemini approves → use Claude prices as-is
        - If Gemini flags concerns → apply rollbacks for specific flagged hours,
          blend base+claude for any remaining flagged hours
        """
        final = [round(p, 2) for p in claude_prices]

        if not gemini_validation.get('validation_passed', True):
            logger.warning("Gemini flagged concerns — applying surgical corrections")
            rollbacks = gemini_validation.get('recommended_rollbacks', {})
            flagged = gemini_validation.get('flagged_hours', [])

            for h in flagged:
                idx = h - 1
                if not (0 <= idx < 24):
                    continue
                if str(h) in rollbacks:
                    corrected = round(float(rollbacks[str(h)]), 2)
                    logger.info(f"  Hour {h}: Gemini rollback ${claude_prices[idx]:.2f} → ${corrected:.2f}")
                    final[idx] = corrected
                else:
                    blended = round((base_prices[idx] + claude_prices[idx]) / 2.0, 2)
                    logger.info(f"  Hour {h}: blended ${claude_prices[idx]:.2f}+${base_prices[idx]:.2f} → ${blended:.2f}")
                    final[idx] = blended
        else:
            logger.info("Gemini approved — using Claude-refined prices")

        return final
