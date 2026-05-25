import logging
import os
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

# ─── Daily send guard ────────────────────────────────────────────────────────
def _today_flag_path() -> str:
    return f"/tmp/forecast_sent_{date.today().strftime('%Y-%m-%d')}.flag"

def _already_sent_today() -> bool:
    return os.path.exists(_today_flag_path())

def _mark_sent_today() -> None:
    try:
        with open(_today_flag_path(), "w") as f:
            f.write(datetime.utcnow().isoformat())
    except Exception:
        pass

from src.scrapers.fantods_scraper import FantodsScraperr
from src.scrapers.miso_scraper import MISOScraper
from src.scrapers.gas_prices_scraper import GasPricesScraper
from src.scrapers.pjm_scraper import PJMScraper
from src.scrapers.historical_scraper import HistoricalScraper
from src.utils.similar_day_matcher import SimilarDayMatcher
from src.utils.load_knowledge import get_load_context_for_claude
from src.ai.refinement import AIRefiner
from src.messengers.telegram_sender import TelegramSender
from src.db import supabase_client as db
from src.actuals_logger import run_daily_actuals_check, get_bias_corrections
from src.weekly_digest import is_monday, send_weekly_digest

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/forecast_bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Synthetic fallbacks ─────────────────────────────────────────────────────
SYNTHETIC_LOAD = [
    88, 85, 83, 82, 83, 86, 92, 100, 105, 107, 108, 108,
    107, 106, 106, 107, 110, 114, 116, 114, 111, 106, 99, 93,
]
SYNTHETIC_WIND = [
    12, 13, 14, 14, 13, 12, 10, 9, 8, 8, 9, 10,
    11, 12, 12, 11, 10, 9, 9, 10, 11, 12, 12, 12,
]
SYNTHETIC_PRICES = [
    22, 21, 20, 20, 21, 23, 28, 35, 38, 36, 34, 33,
    32, 31, 32, 34, 37, 42, 45, 43, 40, 36, 30, 25,
]


class ForecastBot:
    """
    DA-LMP forecasting pipeline using similar-day matching + Claude light adjustment.

    Steps:
      1. MISO          — tomorrow's load + wind forecast
      2. HistoricalScraper — last 30 days of load/wind/prices
      3. SimilarDayMatcher — find top 3 comparable days (weekday/weekend enforced)
      4. Weighted base — 50/30/20 weighted average of those 3 days' prices
      5. PJM           — Western Hub price signal
      6. Gas           — Henry Hub spot price
      7. LoadKnowledge — hour-by-hour load physics context
      8. Claude        — light adjustment only (±15% max per hour)
      9. Telegram      — send forecast + comparison summary to Jason
    """

    def __init__(self):
        required = {"CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY")}
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {missing}")

        self.fantods         = FantodsScraperr()
        self.miso            = MISOScraper()
        self.gas             = GasPricesScraper()
        self.pjm             = PJMScraper()
        self.hist_scraper    = HistoricalScraper()
        self.similar_matcher = SimilarDayMatcher()
        self.refiner         = AIRefiner()
        self.sender          = TelegramSender()
        self._ai_signals: dict = {}
        # scraper_health tracks success/fallback status for the DEGRADED DATA warning
        self._scraper_health: dict = {}

    # ────────────────────────────────────────────────────────────────────
    # Core pipeline
    # ────────────────────────────────────────────────────────────────────
    def run_forecast(self) -> list:
        """Execute the 9-step similar-day pipeline and return 24 final prices."""
        logger.info("=" * 65)
        logger.info("STARTING DA-LMP FORECAST PIPELINE (SIMILAR-DAY MATCHING)")
        tomorrow_str  = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        tomorrow_date = date.today() + timedelta(days=1)
        day_type      = "weekday" if tomorrow_date.weekday() < 5 else "weekend"
        logger.info(f"Forecasting for: {tomorrow_str} ({tomorrow_date.strftime('%A')} — {day_type})")
        logger.info("=" * 65)

        # ── STEP 1: Tomorrow's load + wind forecast ───────────────────
        logger.info("\n[1/9] MISO — tomorrow's load + wind forecast...")
        miso_result   = self.miso.fetch_data()
        load_forecast = miso_result.get("load_forecast", []) if miso_result["success"] else []
        wind_forecast = miso_result.get("wind_forecast", []) if miso_result["success"] else []

        if miso_result["success"]:
            logger.info(f"✅ MISO: load={len(load_forecast)}h, wind={len(wind_forecast)}h")
            self._scraper_health["miso"] = True
        else:
            logger.warning("⚠️  MISO failed — using synthetic shapes")
            self._scraper_health["miso"] = False

        if len(load_forecast) < 24:
            load_forecast = list(SYNTHETIC_LOAD)
        if len(wind_forecast) < 24:
            wind_forecast = list(SYNTHETIC_WIND)
        load_forecast = load_forecast[:24]
        wind_forecast = wind_forecast[:24]
        logger.info(
            f"   Load: avg={sum(load_forecast)/24:.1f} GW, "
            f"range={min(load_forecast):.0f}–{max(load_forecast):.0f} GW"
        )
        logger.info(
            f"   Wind: avg={sum(wind_forecast)/24:.1f} GW, "
            f"range={min(wind_forecast):.0f}–{max(wind_forecast):.0f} GW"
        )

        # Fetch fantods prices now — used as fallback if no similar days found
        fantods_result = self.fantods.scrape_data()
        if fantods_result["success"]:
            fantods_prices = fantods_result["prices_by_hour"][:24]
            while len(fantods_prices) < 24:
                fantods_prices.append(fantods_prices[-1] if fantods_prices else 30.0)
            logger.info(f"   Fantods fallback ready: avg=${sum(fantods_prices)/24:.2f}")
        else:
            fantods_prices = list(SYNTHETIC_PRICES)
            logger.warning("   Fantods also failed — synthetic prices will be used as fallback")

        # ── STEP 2: 30 days of historical data ───────────────────────
        logger.info("\n[2/9] Historical scraper — last 30 days of load/wind/prices...")
        try:
            historical_data = self.hist_scraper.get_historical_days(days_back=30)
            logger.info(f"✅ Retrieved {len(historical_data)} historical days")
        except Exception as e:
            logger.warning(f"⚠️  Historical scraper failed: {e} — will skip similar-day matching")
            historical_data = []

        # ── STEP 3: Find top 3 similar days ──────────────────────────
        logger.info(f"\n[3/9] Similar-day matcher — finding top 3 comparable {day_type}s...")
        try:
            similar_days = self.similar_matcher.find_similar_days(
                load_forecast, wind_forecast, historical_data, tomorrow_date, n_similar=3
            )
            logger.info(f"✅ Found {len(similar_days)} similar days (weekday/weekend enforced)")
            for i, day in enumerate(similar_days):
                score_pct = max(0, round((1 - day['similarity_score']) * 100, 1))
                logger.info(
                    f"   #{i+1}: {day['date'].strftime('%a %b %d')} "
                    f"({day['days_ago']}d ago) — {score_pct}% similar"
                )
        except Exception as e:
            logger.warning(f"⚠️  Similar-day matching failed: {e}")
            similar_days = []

        # ── STEP 4: Weighted base forecast ───────────────────────────
        logger.info("\n[4/9] Computing weighted base forecast (50/30/20)...")
        if similar_days:
            try:
                base_prices = self.similar_matcher.compute_weighted_forecast(similar_days)
                comparison_summary = self.similar_matcher.format_comparison_summary(
                    similar_days, tomorrow_date
                )
                logger.info(f"✅ Similar-day base: avg=${sum(base_prices)/24:.2f}")
                logger.info(f"   {comparison_summary}")
            except Exception as e:
                logger.warning(f"⚠️  Weighted forecast failed: {e} — using fantods prices")
                base_prices        = fantods_prices
                comparison_summary = f"Similar-day weighting failed — using 40-day rolling mean"
        else:
            logger.warning(
                f"⚠️  No similar {day_type}s found in last 30 days — "
                "falling back to fantods prices"
            )
            base_prices        = fantods_prices
            comparison_summary = (
                f"No similar {day_type} days found in last 30 days — "
                "base is 40-day rolling mean"
            )

        while len(base_prices) < 24:
            base_prices.append(base_prices[-1] if base_prices else 30.0)
        base_prices = base_prices[:24]

        # ── STEP 5: PJM prices ────────────────────────────────────────
        logger.info("\n[5/9] PJM — Western Hub price signal...")
        pjm_result = self.pjm.fetch_data()
        if pjm_result.get("success"):
            logger.info(f"✅ PJM ({pjm_result['source']}): {pjm_result['trading_signal']}")
            self._scraper_health["pjm"] = True
        else:
            logger.warning("⚠️  PJM fetch failed — no interface signal")
            self._scraper_health["pjm"] = False

        # ── STEP 6: Gas prices ────────────────────────────────────────
        logger.info("\n[6/9] Gas — Henry Hub spot price...")
        gas_result = self.gas.fetch_data()
        if gas_result.get("success"):
            logger.info(f"✅ Gas ({gas_result['source']}): {gas_result['trading_signal']}")
            self._scraper_health["gas"] = True
        else:
            logger.warning("⚠️  Gas price fetch failed")
            self._scraper_health["gas"] = False

        # ── STEP 7: Load knowledge context ───────────────────────────
        logger.info("\n[7/9] Building load knowledge context...")
        tomorrow_dt  = datetime.combine(tomorrow_date, datetime.min.time())
        load_context = get_load_context_for_claude(tomorrow_dt)
        logger.info("✅ Load context built")

        # ── STEP 7.5: Bias corrections from historical errors ─────────
        logger.info("\n[7.5/9] Loading bias corrections from Supabase...")
        try:
            bias_corrections = get_bias_corrections(lookback_days=14)
            if bias_corrections:
                logger.info(f"✅ Bias corrections for {len(bias_corrections)} hours loaded")
            else:
                logger.info("   No significant bias corrections needed yet")
        except Exception as e:
            logger.warning(f"⚠️  Bias correction load failed (non-fatal): {e}")
            bias_corrections = {}

        # ── STEP 8: Claude signal-weighted adjustment ─────────────────
        logger.info("\n[8/9] Claude — signal-weighted adjustment (period-sensitive clamps)...")
        claude_result = self.refiner.claude_light_adjust(
            base_prices      = base_prices,
            similar_days     = similar_days,
            tomorrow_load    = load_forecast,
            tomorrow_wind    = wind_forecast,
            pjm_result       = pjm_result,
            gas_result       = gas_result,
            load_context     = load_context,
            target_date      = tomorrow_str,
            tomorrow_date    = tomorrow_date,
            bias_corrections = bias_corrections,
        )

        if claude_result["success"]:
            refined_prices = claude_result["refined_prices"]
            clamped        = claude_result.get("clamped_hours", 0)
            logger.info(f"✅ Claude adjustment complete | clamped={clamped} hours")
            diffs = [round(refined_prices[i] - base_prices[i], 2) for i in range(24)]
            logger.info(f"   Deltas vs base: {diffs}")
            self._ai_signals = {
                "signal_summary":     claude_result.get("signal_summary", ""),
                "peak_driver":        claude_result.get("peak_driver", ""),
                "risk_flags":         claude_result.get("risk_flags", ""),
                "comparison_summary": comparison_summary,
                "confidence":         claude_result.get("confidence", {}),
                "market_bias":        claude_result.get("market_bias", "NEUTRAL"),
                "market_bias_reason": claude_result.get("market_bias_reason", ""),
            }
        else:
            logger.warning(
                f"⚠️  Claude failed: {claude_result.get('error')} — using similar-day base"
            )
            refined_prices    = list(base_prices)
            self._ai_signals  = {
                "comparison_summary": comparison_summary,
                "confidence":         {h: "LOW" for h in range(1, 25)},
                "market_bias":        "NEUTRAL",
                "market_bias_reason": "",
            }

        # Optional Gemini validation (non-fatal)
        logger.info("\n[8.5/9] Gemini validation (optional)...")
        gemini_result = self.refiner.gemini_validate(
            refined_prices, base_prices, load_forecast, wind_forecast
        )
        if gemini_result["success"]:
            logger.info(
                f"✅ Gemini: passed={gemini_result['validation_passed']} "
                f"| flagged={gemini_result.get('flagged_hours', [])}"
            )
        else:
            logger.warning(f"⚠️  Gemini (non-critical): {gemini_result.get('error')}")

        final_prices = self.refiner.merge_results(base_prices, refined_prices, gemini_result)

        logger.info(f"\n✅ PIPELINE COMPLETE for {tomorrow_str}")
        logger.info(f"   Range: ${min(final_prices):.2f} – ${max(final_prices):.2f}")
        logger.info(f"   Avg:   ${sum(final_prices)/24:.2f}")
        logger.info(f"   Final: {[round(p, 2) for p in final_prices]}")
        logger.info("=" * 65 + "\n")

        # ── Log scraper health to Supabase ────────────────────────────
        self._log_scraper_health()

        # ── Persist forecast to Supabase (for actuals comparison tomorrow) ─
        self._save_forecast_to_supabase(tomorrow_date, final_prices, base_prices)

        return final_prices

    def _log_scraper_health(self) -> None:
        """Write each scraper's success/failure to the scraper_health table."""
        today = date.today().isoformat()
        rows = []
        for scraper_name, success in self._scraper_health.items():
            rows.append({
                "run_date":     today,
                "scraper_name": scraper_name,
                "success":      success,
                "fallback_used": not success,
            })
        if rows:
            db.insert_many("scraper_health", rows)
            failed = [n for n, s in self._scraper_health.items() if not s]
            if failed:
                logger.info(f"Scraper health logged — failed scrapers: {failed}")

    def _save_forecast_to_supabase(
        self, forecast_date, final_prices: list, base_prices: list
    ) -> None:
        """Upsert today's forecast to Supabase so actuals can be compared tomorrow."""
        signals = self._ai_signals
        conf    = signals.get("confidence", {})
        ok = db.upsert("forecasts", {
            "forecast_date":      forecast_date.isoformat(),
            "prices":             {str(h+1): round(p, 4) for h, p in enumerate(final_prices)},
            "confidence":         {str(k): v for k, v in conf.items()},
            "market_bias":        signals.get("market_bias", "NEUTRAL"),
            "market_bias_reason": signals.get("market_bias_reason", ""),
            "signal_summary":     signals.get("signal_summary", ""),
            "peak_driver":        signals.get("peak_driver", ""),
            "risk_flags":         signals.get("risk_flags", ""),
            "base_prices":        {str(h+1): round(p, 4) for h, p in enumerate(base_prices)},
            "scraper_health":     self._scraper_health,
        })
        if ok:
            logger.info(f"✅ Forecast saved to Supabase for {forecast_date}")
        else:
            logger.warning("⚠️  Forecast save to Supabase failed (non-fatal)")

    # ────────────────────────────────────────────────────────────────────
    # Telegram delivery
    # ────────────────────────────────────────────────────────────────────
    def send_telegram(self, final_prices: list) -> dict:
        """Send forecast via Telegram. Returns {stepdad_ok, you_ok, errors}."""
        if not self.sender.available:
            logger.warning("⚠️  TELEGRAM_BOT_TOKEN not configured — skipping send")
            return {"stepdad_ok": False, "you_ok": False, "errors": ["TELEGRAM_BOT_TOKEN not set"]}

        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        signals  = self._ai_signals

        # Build degraded-scrapers list for the warning banner
        failed_scrapers = [
            name for name, ok in self._scraper_health.items() if not ok
        ]
        degraded = failed_scrapers if len(failed_scrapers) >= 2 else None

        result = self.sender.send_forecast(
            tomorrow,
            final_prices,
            confidence          = signals.get("confidence"),
            market_bias         = signals.get("market_bias", ""),
            market_bias_reason  = signals.get("market_bias_reason", ""),
            signal_summary      = signals.get("signal_summary", ""),
            peak_driver         = signals.get("peak_driver", ""),
            risk_flags          = signals.get("risk_flags", ""),
            comparison_summary  = signals.get("comparison_summary", ""),
            degraded_scrapers   = degraded,
        )
        if result.get("stepdad_ok"):
            _mark_sent_today()
        return result

    # ────────────────────────────────────────────────────────────────────
    # Local testing entry point
    # ────────────────────────────────────────────────────────────────────
    def run_once(self) -> list:
        """Run pipeline + send (used for local testing only).

        Uses tomorrow's date as the idempotency key (the DA forecast target)
        so a rogue off-hours run on the same UTC calendar day as the scheduled
        cron cannot block it.

        Raises RuntimeError if the Supabase dedup check fails — a broken DB
        must not silently fall through and cause a send (fails closed).
        """
        # DA forecast target date — always tomorrow (same key as api/index.py)
        target = (date.today() + timedelta(days=1)).isoformat()

        # Primary guard: Supabase sent_forecasts (shared across processes/instances).
        # already_sent_for() RAISES on config/network error — do not swallow it.
        try:
            if db.already_sent_for(target):
                logger.info(f"⏭  Forecast for {target} already sent — Supabase guard triggered.")
                return []
        except RuntimeError as e:
            logger.error(f"Supabase dedup check failed (failing closed): {e}")
            raise

        # Secondary guard: /tmp flag file (local dev only)
        if _already_sent_today():
            logger.info(f"⏭  Already sent today ({_today_flag_path()}) — skipping.")
            return []

        prices = self.run_forecast()
        if prices:
            self.send_telegram(prices)
        return prices


def main():
    logger.info("DA-LMP FORECAST BOT STARTING")
    logger.info(f"Time: {datetime.now()}")

    if _already_sent_today():
        logger.info(f"⏭  Already sent today ({_today_flag_path()}) — exiting.")
        return True

    missing = [v for v in ["CLAUDE_API_KEY"] if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        return False

    bot    = ForecastBot()
    prices = bot.run_once()

    if prices:
        logger.info("--- FORECAST SUCCESSFUL ---")
    else:
        logger.error("Forecast failed")
        return False


if __name__ == "__main__":
    main()
