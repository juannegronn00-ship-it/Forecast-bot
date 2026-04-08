from typing import List, Dict
import logging
from typing import List
from twilio.rest import Client
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WhatsAppSender:
    """Send price forecasts via WhatsApp using Twilio"""
    
    def __init__(self):
        self.account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_number = os.getenv('TWILIO_WHATSAPP_NUMBER', 'whatsapp:+14155238886')
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.account_sid or not self.auth_token:
                raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")
            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    @property
    def available(self) -> bool:
        return bool(self.account_sid and self.auth_token)
    
    @staticmethod
    def format_forecast_message(prices: List[float]) -> str:
        """
        Format 24 prices into WhatsApp message
        
        Args:
            prices: List of 24 hourly prices
        
        Returns:
            Formatted message string
        """
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')
        
        message = f"🤖 DA-LMP FORECAST - Tomorrow\n{tomorrow}\n\n"
        
        for hour in range(1, 25):
            price = prices[hour - 1]
            message += f"Hour {hour}: ${price:.2f}\n"
        
        return message
    
    def send_to_stepdad(self, prices: List[float]) -> bool:
        """
        Send forecast to stepdad via WhatsApp
        
        Args:
            prices: List of 24 hourly prices
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            stepdad_number = os.getenv('STEPDAD_WHATSAPP')
            if not stepdad_number:
                logger.error("STEPDAD_WHATSAPP not set in .env")
                return False
            
            message_body = self.format_forecast_message(prices)
            
            message = self.client.messages.create(
                from_=self.twilio_number,
                to=f"whatsapp:{stepdad_number}",
                body=message_body
            )
            
            logger.info(f"WhatsApp sent to stepdad (SID: {message.sid})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to stepdad: {str(e)}")
            return False
    
    def send_to_you(self, prices: List[float], metadata: Dict = None) -> bool:
        """
        Send forecast to you for monitoring
        
        Args:
            prices: List of 24 hourly prices
            metadata: Optional additional info (avg price, peak hour, etc.)
        
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            your_number = os.getenv('YOUR_WHATSAPP')
            if not your_number:
                logger.debug("YOUR_WHATSAPP not set, skipping self-notification")
                return True
            
            message_body = self.format_forecast_message(prices)
            
            # Add metadata if provided
            if metadata:
                message_body += f"\n📊 Stats:\n"
                if 'avg' in metadata:
                    message_body += f"Average: ${metadata['avg']:.2f}\n"
                if 'peak_hour' in metadata:
                    message_body += f"Peak hour: Hour {metadata['peak_hour']} (${metadata['peak_price']:.2f})\n"
                if 'low_hour' in metadata:
                    message_body += f"Low hour: Hour {metadata['low_hour']} (${metadata['low_price']:.2f})\n"
            
            message = self.client.messages.create(
                from_=self.twilio_number,
                to=f"whatsapp:{your_number}",
                body=message_body
            )
            
            logger.info(f"WhatsApp sent to you (SID: {message.sid})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to send WhatsApp to you: {str(e)}")
            return False
    
    @staticmethod
    def calculate_metadata(prices: List[float]) -> Dict:
        """Calculate helpful statistics about the forecast"""
        if not prices:
            return {}
        
        return {
            'avg': sum(prices) / len(prices),
            'peak_hour': prices.index(max(prices)) + 1,
            'peak_price': max(prices),
            'low_hour': prices.index(min(prices)) + 1,
            'low_price': min(prices),
            'range': max(prices) - min(prices)
        }
