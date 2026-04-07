import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class HourMatcher:
    """Match similar historical hours based on load and wind conditions"""
    
    @staticmethod
    def calculate_similarity(
        target_load: float,
        target_wind: float,
        hist_load: float,
        hist_wind: float
    ) -> float:
        """
        Calculate similarity score between target and historical hour
        Lower score = better match (0 is perfect match)
        
        Args:
            target_load: Tomorrow's load forecast
            target_wind: Tomorrow's wind forecast
            hist_load: Historical load value
            hist_wind: Historical wind value
        
        Returns:
            Similarity score (0-1, lower is better)
        """
        if target_load == 0 or hist_load == 0:
            return float('inf')  # Can't match if either is 0
        
        # Calculate percentage difference
        load_diff = abs(target_load - hist_load) / target_load
        wind_diff = abs(target_wind - hist_wind) / (target_wind + 0.1)  # Add 0.1 to avoid division by zero
        
        # Weighted similarity (load is more important than wind)
        similarity = (0.7 * load_diff) + (0.3 * wind_diff)
        
        return similarity
    
    @staticmethod
    def find_similar_hour(
        target_load: float,
        target_wind: float,
        historical_load: List[float],
        historical_wind: List[float],
        historical_prices: List[float],
        hour: int
    ) -> float:
        """
        Find the most similar historical hour and return its price
        
        Args:
            target_load: Tomorrow's load for this hour
            target_wind: Tomorrow's wind for this hour
            historical_load: List of historical load values for this hour (last 20 days)
            historical_wind: List of historical wind values for this hour (last 20 days)
            historical_prices: List of historical prices for this hour (last 20 days)
            hour: Hour number (1-24)
        
        Returns:
            Price of most similar historical hour
        """
        if not historical_prices or not historical_load or not historical_wind:
            logger.warning(f"Hour {hour}: Missing historical data")
            return 0.0
        
        best_similarity = float('inf')
        best_price = historical_prices[0]
        
        # Find best matching historical hour
        for i in range(len(historical_prices)):
            if i >= len(historical_load) or i >= len(historical_wind):
                break
            
            similarity = HourMatcher.calculate_similarity(
                target_load,
                target_wind,
                historical_load[i],
                historical_wind[i]
            )
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_price = historical_prices[i]
        
        logger.debug(f"Hour {hour}: Matched with similarity={best_similarity:.3f}, price=${best_price:.2f}")
        return best_price
    
    @staticmethod
    def generate_base_forecast(
        load_forecast: List[float],
        wind_forecast: List[float],
        historical_data: Dict
    ) -> List[float]:
        """
        Generate 24-hour base price forecast using similar day matching
        
        Args:
            load_forecast: Tomorrow's hourly load forecast (24 values)
            wind_forecast: Tomorrow's hourly wind forecast (24 values)
            historical_data: {hour: {load: [...], wind: [...], prices: [...]}}
        
        Returns:
            List of 24 hourly prices
        """
        if len(load_forecast) < 24 or len(wind_forecast) < 24:
            logger.error(f"Incomplete forecast data: load={len(load_forecast)}, wind={len(wind_forecast)}")
            return [0.0] * 24
        
        base_prices = []
        
        for hour in range(1, 25):
            target_load = load_forecast[hour - 1]
            target_wind = wind_forecast[hour - 1]
            
            # Get historical data for this hour
            if hour in historical_data:
                hist = historical_data[hour]
                price = HourMatcher.find_similar_hour(
                    target_load,
                    target_wind,
                    hist.get('load', []),
                    hist.get('wind', []),
                    hist.get('prices', []),
                    hour
                )
            else:
                logger.warning(f"No historical data for hour {hour}")
                price = 0.0
            
            base_prices.append(price)
        
        logger.info(f"Generated base prices: avg=${sum(base_prices)/len(base_prices):.2f}")
        return base_prices
