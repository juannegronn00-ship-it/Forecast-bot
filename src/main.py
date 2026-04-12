import logging
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.scrapers.fantods_scraper import FantodsScraperr
from src.scrapers.miso_scraper import MISOScraper
from src.utils.matcher import HourMatcher
from src.ai.refinement import AIRefiner
from src.utils.whatsapp_sender import WhatsAppSender

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

# Typical MISO spring load shape (GW) — used when all scrapers fail
SYNTHETIC_LOAD = [
    88, 85, 83, 82, 83, 86, 92, 100, 105, 107, 108, 108,
    107, 106, 106, 107, 110, 114, 116, 114, 111, 106, 99, 93,
]
# Typical MISO spring wind generation shape (GW)
SYNTHETIC_WIND = [
    12, 13, 14, 14, 13, 12, 10, 9, 8, 8, 9, 10,
    11, 12, 12, 11, 10, 9, 9, 10, 11, 12, 12, 12,
]
# Typical MISO spring DA-LMP shape ($/MWh) — fallback if fantods fails
SYNTHETIC_PRICES = [
    22, 21, 20, 20, 21, 23, 28, 35, 38, 36, 34, 33,
    32, 31, 32, 34, 37, 42, 45, 43, 40, 36, 30, 25,
]


class ForecastBot:
    """Orchestrates the full DA-LMP forecasting pipeline."""

    def __init__(self):
        required = {"CLAUDE_API_KEY": os.getenv("CLAUDE_API_KEY")}
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise EnvironmentError(f"Missing required env vars: {missing}")

        self.fantods = FantodsScraperr()
        self.miso = MISOScraper()
        self.matcher = HourMatcher()
        self.refiner = AIRefiner()
        self.sender = WhatsAppSender()

    # ------------------------------------------------------------------ #
    # Core pipeline  (does NOT send WhatsApp — callers handle that)
    # ------------------------------------------------------------------ #
    def run_forecast(self) -> list:
        """
        Execute forecast pipeline and return 24 final prices.
        WhatsApp delivery is intentionally NOT done here — the caller
        (api/index.py or run_once) is responsible, so we never double-send.

        Steps:
          1. Scrape fantods for DA-LMP base prices
          2. Fetch MISO load/wind forecasts
          3. Claude expert refinement (capped adjustments)
          4. Gemini validation + surgical corrections
          5. Merge and return final prices
        """
        logger.info("=" * 60)
        logger.info("STARTING FORECAST PIPELINE")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        logger.info(f"Forecasting for: {tomorrow}")
        logger.info("=" * 60)

        # ---------------------------------------------------------------- #
        # STEP 1: Fantods — base DA-LMP prices
        # ---------------------------------------------------------------- #
        logger.info("\n[1/5] Scraping fantods for DA-LMP base prices...")
        fantods_result = self.fantods.scrape_data()

        if fantods_result["success"]:
            base_prices = fantods_result["prices_by_hour"]
            logger.info(f"✅ Fantods OK: {len(base_prices)} prices, avg=${sum(base_prices)/len(base_prices):.2f}")
        else:
            logger.warning(f"⚠️  Fantods failed: {fantods_result.get('error')} — using synthetic prices")
            base_prices = list(SYNTHETIC_PRICES)

        # Pad/trim to exactly 24
        while len(base_prices) < 24:
            base_prices.append(base_prices[-1] if base_prices else 30.0)
        base_prices = base_prices[:24]

        # ---------------------------------------------------------------- #
        # STEP 2: MISO — load and wind forecasts
        # ---------------------------------------------------------------- #
        logger.info("\n[2/5] Fetching MISO load/wind forecasts...")
        miso_result = self.miso.fetch_data()

        if miso_result["success"]:
            load_forecast = miso_result.get("load_forecast", [])
            wind_forecast = miso_result.get("wind_forecast", [])
            logger.info(f"✅ MISO OK: load={len(load_forecast)}h, wind={len(wind_forecast)}h")
        else:
            logger.warning("⚠️  MISO fetch failed — using synthetic load/wind shape")
            load_forecast = []
            wind_forecast = []

        # Fill gaps with synthetic shapes
        if len(load_forecast) < 24:
            logger.info("   Using synthetic MISO spring load shape")
            load_forecast = list(SYNTHETIC_LOAD)
        if len(wind_forecast) < 24:
            logger.info("   Using synthetic MISO spring wind shape")
            wind_forecast = list(SYNTHETIC_WIND)

        load_forecast = load_forecast[:24]
        wind_forecast = wind_forecast[:24]

        logger.info(f"   Load avg: {sum(load_forecast)/len(load_forecast):.1f} GW  "
                    f"range: {min(load_forecast):.0f}–{max(load_forecast):.0f} GW")
        logger.info(f"   Wind avg: {sum(wind_forecast)/len(wind_forecast):.1f} GW  "
                    f"range: {min(wind_forecast):.0f}–{max(wind_forecast):.0f} GW")

        # ---------------------------------------------------------------- #
        # STEP 3: Claude expert refinement
        # ---------------------------------------------------------------- #
        logger.info("\n[3/5] Claude expert refinement...")
        claude_result = self.refiner.claude_refine(
            base_prices,
            load_forecast,
            wind_forecast,
            target_date=tomorrow,
        )

        if claude_result["success"]:
            refined_prices = claude_result["refined_prices"]
            clamped = claude_result.get("clamped_hours", 0)
            logger.info(f"✅ Claude OK | clamped_hours={clamped}")
            logger.info(f"   Reasoning: {claude_result.get('reasoning', '')}")
            # Log before/after comparison
            diffs = [round(refined_prices[i] - base_prices[i], 2) for i in range(24)]
            logger.info(f"   Hour-by-hour delta: {diffs}")
        else:
            logger.warning(f"⚠️  Claude failed: {claude_result.get('error')} — using base prices")
            refined_prices = list(base_prices)

        # ---------------------------------------------------------------- #
        # STEP 4: Gemini validation
        # ---------------------------------------------------------------- #
        logger.info("\n[4/5] Gemini validation...")
        gemini_result = self.refiner.gemini_validate(
            refined_prices,
            base_prices,
            load_forecast,
            wind_forecast,
        )

        if gemini_result["success"]:
            passed = gemini_result["validation_passed"]
            flagged = gemini_result.get("flagged_hours", [])
            logger.info(f"✅ Gemini OK | passed={passed} | flagged_hours={flagged}")
        else:
            logger.warning(f"⚠️  Gemini failed (non-critical): {gemini_result.get('error')}")

        # ---------------------------------------------------------------- #
        # STEP 5: Merge
        # ---------------------------------------------------------------- #
        logger.info("\n[5/5] Merging results...")
        final_prices = self.refiner.merge_results(base_prices, refined_prices, gemini_result)

        logger.info(f"✅ Final prices: {len(final_prices)} hours")
        logger.info(f"   Range: ${min(final_prices):.2f} – ${max(final_prices):.2f}")
        logger.info(f"   Avg:   ${sum(final_prices)/len(final_prices):.2f}")
        logger.info(f"   Final: {final_prices}")
        logger.info("\n" + "=" * 60)
        logger.info("✅ FORECAST PIPELINE COMPLETE")
        logger.info("=" * 60 + "\n")

        return final_prices

    # ------------------------------------------------------------------ #
    # WhatsApp delivery (separate from pipeline so callers control it)
    # ------------------------------------------------------------------ #
    def send_whatsapp(self, final_prices: list) -> dict:
        """Send forecast via WhatsApp. Returns {stepdad_ok, you_ok, errors}."""
        if not self.sender.available:
            logger.warning("⚠️  Twilio credentials not configured — skipping WhatsApp send")
            return {"stepdad_ok": False, "you_ok": False, "errors": ["Twilio not configured"]}

        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        meta = self.sender.calculate_metadata(final_prices)

        stepdad_ok, stepdad_err = self.sender.send_to_stepdad(
            tomorrow, final_prices, meta["avg"], meta["min_price"], meta["max_price"]
        )
        if stepdad_ok:
            logger.info("✅ WhatsApp sent to stepdad")
        else:
            logger.error(f"❌ WhatsApp to stepdad FAILED: {stepdad_err}")

        you_ok, you_err = self.sender.send_to_you(
            tomorrow, final_prices, meta["avg"], meta["min_price"], meta["max_price"], meta
        )
        if you_ok:
            logger.info("✅ WhatsApp sent to monitoring number")
        else:
            logger.warning(f"⚠️  WhatsApp to monitoring failed: {you_err}")

        errors = []
        if stepdad_err:
            errors.append(f"stepdad: {stepdad_err}")
        if you_err:
            errors.append(f"monitor: {you_err}")

        return {"stepdad_ok": stepdad_ok, "you_ok": you_ok, "errors": errors}

    # ------------------------------------------------------------------ #
    # Local testing entry point
    # ------------------------------------------------------------------ #
    def run_once(self) -> list:
        """Run pipeline + send (used for local testing only)."""
        prices = self.run_forecast()
        if prices:
            self.send_whatsapp(prices)
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
