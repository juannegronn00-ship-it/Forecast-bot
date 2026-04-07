import logging
from typing import List, Dict
from anthropic import Anthropic
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

class AIRefiner:
    """Use Claude and Gemini to refine price forecasts"""
    
    def __init__(self):
        self.claude_client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    
    def claude_refine(
        self,
        base_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float]
    ) -> Dict:
        """
        Use Claude to refine prices based on load and wind patterns
        
        Args:
            base_prices: Algorithm-generated prices
            load_forecast: Tomorrow's load forecast
            wind_forecast: Tomorrow's wind forecast
        
        Returns:
            {
                'refined_prices': [24 adjusted prices],
                'reasoning': str,
                'adjustments': {hour: adjustment_percent}
            }
        """
        try:
            # Create analysis prompt
            prompt = f"""
You are an energy market analyst. Given tomorrow's load and wind forecasts, and base DA-LMP prices from pattern matching, suggest price adjustments.

Base prices (24 hours): {[round(p, 2) for p in base_prices]}
Load forecast (MW): {[round(l, 1) for l in load_forecast]}
Wind forecast (mph): {[round(w, 1) for w in wind_forecast]}

For each hour, consider:
- Higher load + lower wind = higher prices
- Lower load + higher wind = lower prices
- Peak hours (7-23) typically have higher prices

Respond ONLY with JSON format:
{{
    "adjustments": {{
        "1": 0.0,
        "2": 0.0,
        ...
        "24": 0.0
    }},
    "reasoning": "Brief explanation of patterns observed"
}}

Each adjustment is a percentage (e.g., 0.05 means +5%, -0.03 means -3%)
"""
            
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text

            # Strip markdown code fences if present, then parse JSON
            import json
            import re
            clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
            data = json.loads(clean)
            
            # Apply adjustments to base prices
            refined_prices = []
            adjustments = data.get('adjustments', {})
            
            for hour in range(1, 25):
                base = base_prices[hour - 1]
                adjustment = float(adjustments.get(str(hour), 0))
                refined = base * (1 + adjustment)
                refined_prices.append(refined)
            
            logger.info(f"Claude refinement complete: avg adjustment={(sum(float(adjustments.get(str(h), 0)) for h in range(1, 25))/24)*100:.2f}%")
            
            return {
                'success': True,
                'refined_prices': refined_prices,
                'reasoning': data.get('reasoning', ''),
                'adjustments': adjustments
            }
        
        except Exception as e:
            logger.error(f"Claude refinement failed: {str(e)}")
            return {
                'success': False,
                'refined_prices': base_prices,
                'error': str(e)
            }
    
    def gemini_validate(
        self,
        claude_prices: List[float],
        load_forecast: List[float],
        wind_forecast: List[float]
    ) -> Dict:
        """
        Use Gemini to validate Claude's adjustments
        
        Args:
            claude_prices: Claude-refined prices
            load_forecast: Tomorrow's load forecast
            wind_forecast: Tomorrow's wind forecast
        
        Returns:
            {
                'validation_passed': bool,
                'concerns': str,
                'suggested_adjustments': {hour: price}
            }
        """
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
You are validating energy price forecasts. Check if these Claude-adjusted prices make sense given market conditions.

Claude prices: {[round(p, 2) for p in claude_prices]}
Load forecast: {[round(l, 1) for l in load_forecast]}
Wind forecast: {[round(w, 1) for w in wind_forecast]}

Are there any issues? Any hours that seem too high/low? Respond with:
{{
    "validation_passed": true/false,
    "concerns": "List any issues found",
    "flagged_hours": [list of hours with potential issues]
}}
"""
            
            response = model.generate_content(prompt)
            response_text = response.text

            import json
            import re
            clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', response_text.strip(), flags=re.MULTILINE)
            data = json.loads(clean)
            
            logger.info(f"Gemini validation: passed={data.get('validation_passed', True)}")
            
            return {
                'success': True,
                'validation_passed': data.get('validation_passed', True),
                'concerns': data.get('concerns', ''),
                'flagged_hours': data.get('flagged_hours', [])
            }
        
        except Exception as e:
            logger.warning(f"Gemini validation failed (non-critical): {str(e)}")
            return {
                'success': False,
                'validation_passed': True,
                'error': str(e)
            }
    
    @staticmethod
    def merge_results(
        base_prices: List[float],
        claude_prices: List[float],
        gemini_validation: Dict
    ) -> List[float]:
        """
        Merge base, Claude, and Gemini results into final prices
        
        Returns:
            Final 24-hour price forecast
        """
        if gemini_validation.get('validation_passed', True):
            # Claude prices validated, use them
            logger.info("Using Claude-refined prices (Gemini approved)")
            return [round(p, 2) for p in claude_prices]
        else:
            # Validation concerns, average with base
            logger.warning("Gemini flagged concerns, using average of base and Claude")
            final = []
            for i in range(24):
                avg = (base_prices[i] + claude_prices[i]) / 2
                final.append(round(avg, 2))
            return final
