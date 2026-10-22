"""Pest-outbreak view over a pre-generated scenario.

`risk(day)` is the outbreak probability; `occurs(day)` whether an outbreak actually happens.
When it happens, the environment destroys the planted field selected by the scenario's
uniform draw `pest_target_u[day]` (index = floor(u × number of planted fields), fields in id order).
"""
from ..framework.scenario import risk_label


class PestSystem:
    def __init__(self, scenario=None):
        self.scenario = scenario

    def risk(self, day: int) -> float:
        return self.scenario.pest_risk[day - 1]

    def occurs(self, day: int) -> bool:
        return self.scenario.pest(day)

    def label(self, day: int) -> str:
        return risk_label(self.risk(day))
