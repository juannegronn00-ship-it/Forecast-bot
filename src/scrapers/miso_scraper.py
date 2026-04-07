import requests
import csv
from io import StringIO
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MISOScraper:
    """Scrape forecast data from MISO official website"""
    
    @staticmethod
    def get_latest_csv_url():
        """
        Construct URL for today's CSV file
        MISO format: YYYYMMDD_sr_la_rg.csv
        """
        today = datetime.now()
        filename = f"{today.strftime('%Y%m%d')}_sr_la_rg.csv"
        
        # Try multiple possible URLs
        urls = [
            f"https://www.misoenergy.org/api/v1/markets/market-reports/{filename}",
            f"https://www.misoenergy.org/markets-and-operations/real-time--market-data/market-reports/{filename}",
        ]
        return urls
    
    @staticmethod
    def fetch_data():
        """
        Fetch MISO CSV data for load and wind forecasts
        Returns:
            {
                'success': bool,
                'load_forecast': [24 hourly values],
                'wind_forecast': [24 hourly values]
            }
        """
        urls = MISOScraper.get_latest_csv_url()
        
        for url in urls:
            try:
                logger.debug(f"Attempting to fetch MISO CSV from: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Parse CSV data
                csv_reader = csv.DictReader(StringIO(response.text))
                
                load_forecast = []
                wind_forecast = []
                
                for row in csv_reader:
                    # Extract load forecast
                    for key in row:
                        if 'load' in key.lower() and row[key]:
                            try:
                                load_forecast.append(float(row[key]))
                                break
                            except:
                                pass
                    
                    # Extract wind forecast
                    for key in row:
                        if 'wind' in key.lower() and row[key]:
                            try:
                                wind_forecast.append(float(row[key]))
                                break
                            except:
                                pass
                
                if load_forecast or wind_forecast:
                    logger.info(f"Successfully fetched MISO data: load={len(load_forecast)}, wind={len(wind_forecast)}")
                    return {
                        'success': True,
                        'load_forecast': load_forecast[:24],
                        'wind_forecast': wind_forecast[:24]
                    }
            
            except Exception as e:
                logger.debug(f"Failed to fetch from {url}: {str(e)}")
                continue
        
        logger.warning("Could not fetch MISO CSV from any URL")
        return {
            'success': False,
            'error': 'All MISO URLs failed'
        }
