"""Crop market.

Two market models are supported (selected by `SimulationConfig.market_model`):

``exogenous`` (default, the original behaviour)
    The price of crop c on day t is the scenario's reference price R_{c,t}: a ±5 % uniform random
    walk clamped to [0.5, 1.5] × base price, multiplied by 0.5 or 2 on price-shock days. Sales do not
    move prices, so every player facing the same scenario sees identical prices.

``price_impact`` (optional extension)
    The player's own sales depress the price they receive. Each crop carries a supply-pressure
    stock S_{c,t} (units) with S_{c,t+1} = δ · S_{c,t} + q_{c,t}. The effective price is

        P_{c,t} = R_{c,t} · max(m, (1 + S_{c,t} / Q_ref) ** (−γ))

    and a sale of q units earns  q · R_{c,t} · max(m, (1 + (S_{c,t} + q/2) / Q_ref) ** (−γ))
    (midpoint price over the sale). P is positive, finite at zero supply (equals R) and
    non-increasing in S. This is a single-seller price-impact model, not a full supply–demand
    equilibrium: there are no modelled competitors. Market outcomes become action-dependent,
    so a human and an AI replaying the same scenario can legitimately receive different prices.
"""
from typing import Dict, Optional

from ..config.settings import CROP_CONFIG


class Market:
    """Represents the market for crops on the current day."""

    def __init__(self, scenario=None, model: str = "exogenous", impact_params=None):
        self.scenario = scenario
        self.model = model
        self.impact = impact_params
        self.day = 1
        self.supply_pressure: Dict[str, float] = {c: 0.0 for c in CROP_CONFIG}
        # Fallback (no scenario): base prices, used only by legacy unit tests
        self._static = {c: cfg["base_price"] for c, cfg in CROP_CONFIG.items()}

    # ── prices ────────────────────────────────────────────────────────
    def reference_price(self, crop_name: str, day: Optional[int] = None) -> float:
        if crop_name not in CROP_CONFIG:
            raise ValueError(f"Unknown crop: {crop_name}")
        if self.scenario is None:
            return self._static[crop_name]
        return self.scenario.price(crop_name, self.day if day is None else day)

    def _impact_factor(self, pressure: float) -> float:
        if self.model != "price_impact" or self.impact is None:
            return 1.0
        p = self.impact
        return max(p.min_factor, (1.0 + max(pressure, 0.0) / p.reference_supply) ** (-p.elasticity))

    def get_current_price(self, crop_name: str) -> float:
        """Marginal price a seller receives right now (₹ per unit)."""
        return self.reference_price(crop_name) * self._impact_factor(self.supply_pressure[crop_name])

    @property
    def current_prices(self) -> Dict[str, float]:
        return {c: self.get_current_price(c) for c in CROP_CONFIG}

    # ── transactions ──────────────────────────────────────────────────
    def sale_revenue(self, crop_name: str, quantity: int) -> float:
        """Revenue (₹) from selling `quantity` units now, without executing the sale."""
        if quantity <= 0:
            return 0.0
        mid = self.supply_pressure[crop_name] + quantity / 2.0
        return quantity * self.reference_price(crop_name) * self._impact_factor(mid)

    def execute_sale(self, crop_name: str, quantity: int) -> float:
        revenue = self.sale_revenue(crop_name, quantity)
        if self.model == "price_impact":
            self.supply_pressure[crop_name] += quantity
        return revenue

    # ── time ──────────────────────────────────────────────────────────
    def advance_day(self, next_day: int) -> None:
        """Moves the market clock; supply pressure decays (price-impact model only)."""
        if self.model == "price_impact" and self.impact is not None:
            for c in self.supply_pressure:
                self.supply_pressure[c] *= self.impact.decay
        self.day = next_day

    def update_prices(self):
        """Legacy no-op kept for API compatibility: prices now come from the scenario."""
        return None
