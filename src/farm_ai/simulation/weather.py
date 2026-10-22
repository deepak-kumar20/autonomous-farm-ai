"""Drought (weather) view over a pre-generated scenario.

The original implementation rolled the global `random` module every day and derived the
"forecast" label from the same roll as the event. Risk and occurrence now come from the
scenario: `risk(day)` is the configured probability, `occurs(day)` the realised event.
"""
from ..framework.scenario import risk_label


class WeatherSystem:
    def __init__(self, scenario=None):
        self.scenario = scenario

    def risk(self, day: int) -> float:
        return self.scenario.drought_risk[day - 1]

    def occurs(self, day: int) -> bool:
        return self.scenario.drought(day)

    def label(self, day: int) -> str:
        return risk_label(self.risk(day))
