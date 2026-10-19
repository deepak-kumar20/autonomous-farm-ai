import random
from typing import Dict
from ..config.settings import EVENT_PROBABILITIES

class PriceShockSystem:
    """Manages market price shocks."""
    
    def __init__(self):
        self.shock_probability = EVENT_PROBABILITIES["price_shock"]
        self.shock_forecast = "LOW"
        self.is_shock_active = False

    def update(self):
        """Updates price shock forecasts and events."""
        roll = random.random()
        
        if roll < self.shock_probability:
            self.is_shock_active = True
            self.shock_forecast = "HIGH"
        elif roll < self.shock_probability * 3:
            self.is_shock_active = False
            self.shock_forecast = "HIGH"
        elif roll < self.shock_probability * 6:
            self.is_shock_active = False
            self.shock_forecast = "MEDIUM"
        else:
            self.is_shock_active = False
            self.shock_forecast = "LOW"
            
    def apply_effects(self, market):
        """Applies price shocks to the market."""
        if self.is_shock_active:
            # Massive drop or spike in all prices
            shock_factor = random.choice([0.5, 2.0])
            for crop_name in market.current_prices:
                market.current_prices[crop_name] *= shock_factor
