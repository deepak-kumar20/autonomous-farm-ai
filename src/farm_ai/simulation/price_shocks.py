"""Price-shock view over a pre-generated scenario.

Shocks are already folded into the scenario's reference price path (all crop prices × 0.5 or × 2
on a shock day); this class only exposes the risk and the realised event for display and logging.
"""
from ..framework.scenario import risk_label


class PriceShockSystem:
    def __init__(self, scenario=None):
        self.scenario = scenario

    def risk(self, day: int) -> float:
        return self.scenario.shock_risk[day - 1]

    def occurs(self, day: int) -> bool:
        return self.scenario.shock_event[day - 1]

    def factor(self, day: int) -> float:
        return self.scenario.shock_factor[day - 1]

    def label(self, day: int) -> str:
        return risk_label(self.risk(day))
