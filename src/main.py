import logging
import os
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

from src.scrapers.fantods_scraper import FantodsScraperr
from src.scrapers.miso_scraper import MISOScraper
from src.scrapers.weather_scraper import WeatherScraper
from src.scrapers.gas_prices_scraper import GasPricesScraper
from src.scrapers.pjm_scraper import PJMScraper
from src.data.historical_patterns import HistoricalPatterns
from src.data.calendar_data import (
    demand_profile_label,
    daylight_hours,
    load_adjustment_factor,
    solar_generation_signal,
)
from src.utils.matcher import HourMatcher
from src.ai.refinement import AIRefiner
from src.data.fantods_optimizer import FantodsOptimizer
from src.messengers.telegram_sender import TelegramSender

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

# ─── Synthetic fallbacks (used when all scrapers fail) ──────────────────────
# Typical MISO spring load shape (GW)
SYNTHETIC_LOAD = [
    88, 85, 83, 82, 83, 86, 92, 100, 105, 107, 108, 108,
    107, 106, 106, 107, 110, 114, 116, 114, 111, 106, 99, 93,
]
# Typical MISO spring wind generation shape (GW)
SYNTHETIC_WIND = [
    12, 13, 14, 14, 13, 12, 10, 9, 8, 8, 9, 10,
    11, 12, 12, 11, 10, 9, 9, 10, 11, 12, 12, 12,
]
# Typical MISO spring DA-LMP shape ($/MWh)
SYNTHETIC_PRICES = [
    22, 21, 20, 20, 21, 23, 28, 35, 38, 36, 34, 33,
    32, 31, 32, 34, 37, 42, 45, 43, 40, 36, 30, 25,
]


class ForecastBot:
    """
    Orchestrates the full DA-LMP forecasting pipeline.

    Steps:
      1. Fantods  — base DA-LMP prices (40-day rolling mean)
      2. MISO     — load/wind system forecasts
      3. Weather  — Chicago temp, wind, cloud cover (NOAA primary)
      4. Gas      — Henry Hub spot price (EIA API or seasonal estimate)
      5. History  — same-weekday patterns from fantods price history
      6. Claude   — expert trading refinement with full context
      7. Gemini   — validation + surgical rollbacks
      8. Merge    — final 24-hour price array
    """

    def __init__(self):
        required = {"CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY")}
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {missing}")

        self.fantods = FantodsScraperr()
        self.miso = MISOScraper()
        self.weather = WeatherScraper()
        self.gas = GasPricesScraper()
        self.pjm = PJMScraper()
        self.history = HistoricalPatterns()
        self.matcher = HourMatcher()
        self.optimizer = FantodsOptimizer()
        self.refiner = AIRefiner()
        self.sender = TelegramSender()

    # ────────────────────────────────────────────────────────────────────
    # Core pipeline  (returns prices, does NOT send WhatsApp)
    # ────────────────────────────────────────────────────────────────────
    def run_forecast(self) -> list:
        """Execute forecast and return 24 final prices."""
        logger.info("=" * 65)
        logger.info("STARTING DA-LMP FORECAST PIPELINE")
        tomorrow_str = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        tomorrow_date = date.today() + timedelta(days=1)
        logger.info(f"Forecasting for: {tomorrow_str}")
        logger.info("=" * 65)

        # ── STEP 1: Fantods base prices ──────────────────────────────────
        logger.info("\n[1/10] Fantods — base DA-LMP prices...")
        fantods_result = self.fantods.scrape_data()

        if fantods_result["success"]:
            base_prices = fantods_result["prices_by_hour"]
            logger.info(f"✅ Fantods: {len(base_prices)} prices, avg=${sum(base_prices)/len(base_prices):.2f}")
        else:
            logger.warning(f"⚠️  Fantods failed: {fantods_result.get('error')} — using synthetic")
            base_prices = list(SYNTHETIC_PRICES)

        while len(base_prices) < 24:
            base_prices.append(base_prices[-1] if base_prices else 30.0)
        base_prices = base_prices[:24]

        # ── STEP 2: MISO load/wind forecasts ────────────────────────────
        logger.info("\n[2/10] MISO — system load/wind forecasts...")
        miso_result = self.miso.fetch_data()

        if miso_result["success"]:
            load_forecast = miso_result.get("load_forecast", [])
            wind_forecast = miso_result.get("wind_forecast", [])
            logger.info(f"✅ MISO: load={len(load_forecast)}h, wind={len(wind_forecast)}h")
        else:
            logger.warning("⚠️  MISO failed — using synthetic shapes")
            load_forecast, wind_forecast = [], []

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

        # ── STEP 3: Weather ──────────────────────────────────────────────
        logger.info("\n[3/10] Weather — Chicago MISO load center...")
        weather_result = self.weather.fetch_data()
        if weather_result.get("success"):
            logger.info(f"✅ Weather ({weather_result['source']}): {weather_result['summary']}")
            logger.info(f"   Signal: {weather_result['trading_signal']}")
        else:
            logger.warning("⚠️  Weather fetch failed — no weather context for Claude")

        # ── STEP 4: Gas prices ───────────────────────────────────────────
        logger.info("\n[4/10] Gas prices — Henry Hub...")
        gas_result = self.gas.fetch_data()
        if gas_result.get("success"):
            logger.info(f"✅ Gas ({gas_result['source']}): {gas_result['trading_signal']}")
        else:
            logger.warning("⚠️  Gas price fetch failed")

        # ── STEP 5: PJM market correlation ──────────────────────────────
        logger.info("\n[5/10] PJM market — interface correlation signal...")
        pjm_result = self.pjm.fetch_data()
        if pjm_result.get("success"):
            logger.info(f"✅ PJM ({pjm_result['source']}): {pjm_result['trading_signal']}")
        else:
            logger.warning("⚠️  PJM fetch failed — no interface signal")

        # ── STEP 6: Historical same-weekday patterns ─────────────────────
        logger.info("\n[6/10] Historical patterns — same weekday analysis...")
        self.history.load()
        hist_summary  = ""
        hist_profile  = []
        weekday_stats = {}
        if self.history.loaded:
            hist_summary  = self.history.summary_for_date(tomorrow_date)
            hist_profile  = self.history.get_weekday_profile(tomorrow_date.weekday())
            weekday_stats = self.history.get_weekday_stats(tomorrow_date.weekday())
            n_hrs = sum(1 for s in weekday_stats.values() if s)
            logger.info(f"✅ History: {hist_summary}")
            logger.info(f"   Per-hour stats available: {n_hrs}/24 hours")
        else:
            logger.warning("⚠️  Historical patterns unavailable")

        # ── STEP 6: Calendar context ─────────────────────────────────────
        logger.info("\n[7/10] Calendar context...")
        cal_label = demand_profile_label(tomorrow_date)
        dl_hours = daylight_hours(tomorrow_date)
        lf = load_adjustment_factor(tomorrow_date)
        solar_signal = solar_generation_signal(tomorrow_date)
        logger.info(f"   {cal_label}")
        logger.info(f"   {solar_signal}  |  load_factor={lf:.3f}")

        # Check holiday flag for optimizer
        try:
            from src.data.calendar_data import is_holiday
            holiday_flag = is_holiday(tomorrow_date)
        except Exception:
            holiday_flag = False

        # ── STEP 7: Fantods data-driven optimization (zero API cost) ────
        logger.info("\n[8/10] Fantods optimizer — data-driven shape correction...")

        trader_context = {
            "weather": weather_result,
            "gas": gas_result,
            "pjm": pjm_result,
            "history_summary": hist_summary,
            "history_profile": hist_profile,
            "weekday_stats": weekday_stats,
            "calendar": cal_label,
            "daylight_hrs": dl_hours,
            "load_factor": lf,
            "weekday_int": tomorrow_date.weekday(),
            "is_holiday": holiday_flag,
        }

        opt_result = self.optimizer.optimize(
            base_prices,
            load_forecast,
            wind_forecast,
            trader_context=trader_context,
        )

        if opt_result["success"]:
            optimized_prices = opt_result["optimized_prices"]
            opt_diffs = [round(optimized_prices[i] - base_prices[i], 2) for i in range(24)]
            logger.info(f"✅ Optimizer complete | deltas vs base: {opt_diffs}")
        else:
            logger.warning("⚠️  Optimizer failed — using raw base prices")
            optimized_prices = list(base_prices)

        # ── STEP 9: Claude pure-reasoning forecast ──────────────────────
        # Claude sees raw base prices as REFERENCE only, plus all market signals.
        # It reasons from first principles to absolute prices — no % adjustments.
        # If Claude fails (no credits / error), optimizer output is the fallback.
        logger.info("\n[9/10] Claude pure-reasoning forecast...")

        claude_result = self.refiner.claude_refine(
            base_prices,            # raw 40-day mean as reference baseline
            load_forecast,
            wind_forecast,
            target_date=tomorrow_str,
            trader_context=trader_context,
        )

        if claude_result["success"]:
            refined_prices = claude_result["refined_prices"]
            clamped = claude_result.get("clamped_hours", 0)
            logger.info(f"✅ Claude reasoning complete | sanity_clamped={clamped} hours")
            diffs = [round(refined_prices[i] - base_prices[i], 2) for i in range(24)]
            logger.info(f"   Claude deltas vs base: {diffs}")
        else:
            logger.warning(
                f"⚠️  Claude failed: {claude_result.get('error')} "
                f"— falling back to data-driven optimizer output"
            )
            refined_prices = list(optimized_prices)   # optimizer, not raw base

        # ── STEP 9: Gemini validation ────────────────────────────────────
        logger.info("\n[10/10] Gemini validation...")
        gemini_result = self.refiner.gemini_validate(
            refined_prices,
            base_prices,
            load_forecast,
            wind_forecast,
        )

        if gemini_result["success"]:
            logger.info(
                f"✅ Gemini: passed={gemini_result['validation_passed']} "
                f"| flagged={gemini_result.get('flagged_hours', [])}"
            )
        else:
            logger.warning(f"⚠️  Gemini failed (non-critical): {gemini_result.get('error')}")

        # ── Merge ────────────────────────────────────────────────────────
        final_prices = self.refiner.merge_results(base_prices, refined_prices, gemini_result)

        logger.info(f"\n✅ PIPELINE COMPLETE for {tomorrow_str}")
        logger.info(f"   Range: ${min(final_prices):.2f} – ${max(final_prices):.2f}")
        logger.info(f"   Avg:   ${sum(final_prices)/24:.2f}")
        logger.info(f"   Final: {final_prices}")
        logger.info("=" * 65 + "\n")

        return final_prices

    # ────────────────────────────────────────────────────────────────────
    # Telegram delivery
    # ────────────────────────────────────────────────────────────────────
    def send_telegram(self, final_prices: list) -> dict:
        """Send forecast via Telegram. Returns {stepdad_ok, you_ok, errors}."""
        if not self.sender.available:
            logger.warning("⚠️  TELEGRAM_BOT_TOKEN not configured — skipping send")
            return {"stepdad_ok": False, "you_ok": False, "errors": ["TELEGRAM_BOT_TOKEN not set"]}

        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        return self.sender.send_forecast(tomorrow, final_prices)

    # ────────────────────────────────────────────────────────────────────
    # Local testing entry point
    # ────────────────────────────────────────────────────────────────────
    def run_once(self) -> list:
        """Run pipeline + send (used for local testing only)."""
        prices = self.run_forecast()
        if prices:
            self.send_telegram(prices)
        return prices


def main():
    logger.info("DA-LMP FORECAST BOT STARTING")
    logger.info(f"Time: {datetime.now()}")

    missing = [v for v in ["CLAUDE_API_KEY"] if not os.getenv(v)]
    if missing:
        logger.error(f"Missing required env vars: {missing}")
        return False

    bot = ForecastBot()
    logger.info("\n--- RUNNING FORECAST TEST ---\n")
    prices = bot.run_once()

    if prices:
        logger.info("--- FORECAST SUCCESSFUL ---")
    else:
        logger.error("Forecast failed")
        return False


if __name__ == "__main__":
    main()
