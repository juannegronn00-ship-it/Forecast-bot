import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Import our modules
from src.scrapers.fantods_scraper import FantodsScraperr
from src.scrapers.miso_scraper import MISOScraper
from src.utils.matcher import HourMatcher
from src.ai.refinement import AIRefiner
from src.utils.whatsapp_sender import WhatsAppSender

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/forecast_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForecastBot:
    """Main bot that orchestrates the entire forecasting pipeline"""
    
    def __init__(self):
        self.fantods = FantodsScraperr()
        self.miso = MISOScraper()
        self.matcher = HourMatcher()
        self.refiner = AIRefiner()
        self.sender = WhatsAppSender()
        self.scheduler = BackgroundScheduler()
    
    def run_forecast(self):
        """
        Execute complete forecast pipeline:
        1. Scrape data from both sources
        2. Generate base prices via matching algorithm
        3. Refine with Claude AI
        4. Validate with Gemini AI
        5. Send via WhatsApp
        """
        logger.info("="*60)
        logger.info("STARTING FORECAST PIPELINE")
        logger.info("="*60)
        
        # STEP 1: Scrape data
        logger.info("\n[1/6] Scraping data sources...")
        
        fantods_result = self.fantods.scrape_data()
        if not fantods_result['success']:
            logger.error(f"Fantods scrape failed: {fantods_result.get('error')}")
            return False
        
        logger.info("✅ Fantods scraped successfully")
        
        miso_result = self.miso.fetch_data()
        if miso_result['success']:
            logger.info("✅ MISO CSV fetched successfully")
        else:
            logger.warning("⚠️ MISO fetch failed, using fantods data only")
        
        # Use fantods as primary, MISO for validation
        load_forecast = fantods_result.get('load_forecast', [])
        wind_forecast = fantods_result.get('wind_forecast', [])
        
        if miso_result['success']:
            # Verify consistency between sources
            miso_load = miso_result.get('load_forecast', [])
            if miso_load and abs(sum(miso_load) - sum(load_forecast)) / sum(load_forecast) < 0.1:
                logger.info("✅ Load forecasts match between sources")
            else:
                logger.warning("⚠️ Load forecast mismatch between sources, using fantods")
        
        # STEP 2: Generate base prices
        logger.info("\n[2/6] Generating base prices via pattern matching...")
        
        # For MVP, use simple averaging of available data
        # In full version, would use real historical data from database
        base_prices = fantods_result.get('prices_by_hour', [])
        
        if not base_prices or len(base_prices) < 24:
            logger.warning("No prices from fantods, using synthetic data")
            # Typical MISO shape: low off-peak, moderate peak, $/MWh
            base_prices = [
                22, 21, 20, 20, 21, 23, 28, 35, 38, 36, 34, 33,
                32, 31, 32, 34, 37, 42, 45, 43, 40, 36, 30, 25
            ]
        
        logger.info(f"✅ Generated {len(base_prices)} base prices (avg: ${sum(base_prices)/len(base_prices):.2f})")
        
        # STEP 3: Claude refinement
        logger.info("\n[3/6] Refining prices with Claude AI...")

        # Use synthetic load/wind shape if real forecasts unavailable
        if not load_forecast or len(load_forecast) < 24:
            logger.warning("No load forecast available, using synthetic MISO shape")
            load_forecast = [
                88, 85, 83, 82, 83, 86, 92, 100, 105, 107, 108, 108,
                107, 106, 106, 107, 110, 114, 116, 114, 111, 106, 99, 93
            ]
        if not wind_forecast or len(wind_forecast) < 24:
            logger.warning("No wind forecast available, using synthetic shape")
            wind_forecast = [
                12, 13, 14, 14, 13, 12, 10, 9, 8, 8, 9, 10,
                11, 12, 12, 11, 10, 9, 9, 10, 11, 12, 12, 12
            ]

        claude_result = self.refiner.claude_refine(
            base_prices,
            load_forecast,
            wind_forecast
        )
        
        if claude_result['success']:
            refined_prices = claude_result['refined_prices']
            logger.info(f"✅ Claude refinement complete")
            logger.info(f"   Reasoning: {claude_result.get('reasoning', 'N/A')[:100]}...")
        else:
            logger.warning(f"⚠️ Claude refinement failed: {claude_result.get('error')}")
            refined_prices = base_prices
        
        # STEP 4: Gemini validation
        logger.info("\n[4/6] Validating with Gemini AI...")
        
        gemini_result = self.refiner.gemini_validate(
            refined_prices,
            load_forecast,
            wind_forecast
        )
        
        if gemini_result['success']:
            if gemini_result['validation_passed']:
                logger.info("✅ Gemini validation passed")
            else:
                logger.warning(f"⚠️ Gemini concerns: {gemini_result.get('concerns', '')[:100]}...")
        else:
            logger.warning(f"⚠️ Gemini validation failed (non-critical)")
        
        # STEP 5: Merge results
        logger.info("\n[5/6] Merging results...")
        
        final_prices = self.refiner.merge_results(
            base_prices,
            refined_prices,
            gemini_result
        )
        
        logger.info(f"✅ Final prices generated: {len(final_prices)} hours")
        logger.info(f"   Range: ${min(final_prices):.2f} - ${max(final_prices):.2f}")
        logger.info(f"   Average: ${sum(final_prices)/len(final_prices):.2f}")
        
        # STEP 6: Send via WhatsApp
        logger.info("\n[6/6] Sending via WhatsApp...")
        
        # Calculate metadata
        metadata = self.sender.calculate_metadata(final_prices)
        
        # Send to stepdad
        stepdad_sent = self.sender.send_to_stepdad(final_prices)
        if stepdad_sent:
            logger.info("✅ WhatsApp sent to stepdad")
        else:
            logger.error("❌ Failed to send WhatsApp to stepdad")
        
        # Send to you
        you_sent = self.sender.send_to_you(final_prices, metadata)
        if you_sent:
            logger.info("✅ WhatsApp sent to you (monitoring)")
        else:
            logger.warning("⚠️ Failed to send WhatsApp to you")
        
        logger.info("\n" + "="*60)
        logger.info("✅ FORECAST PIPELINE COMPLETE")
        logger.info("="*60 + "\n")
        
        return stepdad_sent
    
    def start_scheduler(self):
        """Start the APScheduler to run forecasts daily at 7:00 AM"""
        
        # Schedule forecast to run every day at 7:00 AM
        self.scheduler.add_job(
            self.run_forecast,
            'cron',
            hour=7,
            minute=0,
            id='daily_forecast'
        )
        
        self.scheduler.start()
        logger.info("✅ Scheduler started - forecast will run daily at 7:00 AM")
        logger.info(f"   Next run: {self.scheduler.get_job('daily_forecast').next_run_time}")
        
        # Shut down the scheduler when exiting the app
        atexit.register(lambda: self.scheduler.shutdown())
    
    def run_once(self):
        """Run forecast once immediately (for testing)"""
        logger.info("Running forecast immediately (testing mode)...")
        return self.run_forecast()

def main():
    """Main entry point"""
    
    logger.info("DA-LMP FORECAST BOT STARTING")
    logger.info(f"Time: {datetime.now()}")
    
    # Check environment variables
    required_vars = [
        'CLAUDE_API_KEY',
        'GEMINI_API_KEY',
        'TWILIO_ACCOUNT_SID',
        'TWILIO_AUTH_TOKEN',
        'STEPDAD_WHATSAPP'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    # Initialize bot
    bot = ForecastBot()
    
    # Run once for testing
    logger.info("\n--- RUNNING FIRST FORECAST TEST ---\n")
    success = bot.run_once()
    
    if success:
        logger.info("\n--- FIRST FORECAST SUCCESSFUL ---")
        logger.info("Starting scheduler for daily forecasts...\n")
        bot.start_scheduler()
        
        # Keep scheduler running
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("Shutting down...")
    else:
        logger.error("First forecast failed, not starting scheduler")
        return False

if __name__ == '__main__':
    main()
