"""Run-level configuration for the research framework.

A `SimulationConfig` fixes every rule that is *not* part of the exogenous scenario:
the order in which a day's stages run, the feasibility-filter mode and the market model.
Human and AI runs that are compared against each other must share one config.
"""
from dataclasses import dataclass, asdict, field
from typing import Any, Dict

DECISION_ORDERS = ("LFM", "FML")
FILTER_MODES = ("none", "deterministic", "uncertainty")
THRESHOLD_RULES = ("fixed", "break_even")
MARKET_MODELS = ("exogenous", "price_impact")


@dataclass(frozen=True)
class PriceImpactParams:
    """Parameters of the optional own-sales price-impact market (see docs in market.py)."""

    reference_supply: float = 500.0   # units; supply pressure at which price falls by (1/2)**gamma
    elasticity: float = 0.5           # gamma (dimensionless)
    decay: float = 0.7                # fraction of supply pressure carried to the next day
    min_factor: float = 0.25          # lower bound on the impact multiplier


@dataclass(frozen=True)
class SimulationConfig:
    # "LFM": Loss → Field → Market (exogenous losses resolved at the start of the day, before the decision)
    # "FML": Field → Market → Loss (decision first; losses resolved after growth — the original order)
    decision_order: str = "LFM"
    # "none": no pre-execution filter (infeasible actions fail inside the environment, −50 reward, day passes)
    # "deterministic": structural + nominal-horizon constraints checked before execution
    # "uncertainty": deterministic + P(crop harvest-eligible by day 30) >= threshold
    filter_mode: str = "deterministic"
    market_model: str = "exogenous"
    # Threshold used by the uncertainty filter:
    #   "fixed"      → accept PLANT iff P >= completion_threshold
    #   "break_even" → accept PLANT iff P >= planting_cost / (yield × current price)  (risk-neutral break-even)
    threshold_rule: str = "fixed"
    completion_threshold: float = 0.80
    price_impact: PriceImpactParams = field(default_factory=PriceImpactParams)

    def __post_init__(self):
        if self.decision_order not in DECISION_ORDERS:
            raise ValueError(f"decision_order must be one of {DECISION_ORDERS}")
        if self.filter_mode not in FILTER_MODES:
            raise ValueError(f"filter_mode must be one of {FILTER_MODES}")
        if self.market_model not in MARKET_MODELS:
            raise ValueError(f"market_model must be one of {MARKET_MODELS}")
        if self.threshold_rule not in THRESHOLD_RULES:
            raise ValueError(f"threshold_rule must be one of {THRESHOLD_RULES}")
        if not 0.0 <= self.completion_threshold <= 1.0:
            raise ValueError("completion_threshold must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConfig":
        d = dict(d or {})
        pi = d.pop("price_impact", None)
        cfg = cls(**d)
        if pi:
            cfg = cls(**{**cfg.to_dict(), "price_impact": PriceImpactParams(**pi)})
        return cfg

    def label(self) -> str:
        f = self.filter_mode
        if f == "uncertainty":
            f += f"({'break-even' if self.threshold_rule == 'break_even' else f'tau={self.completion_threshold:g}'})"
        return f"order={self.decision_order}|filter={f}|market={self.market_model}"


#: The configuration treated as "the full framework" in experiments and the UI.
FULL_FRAMEWORK = SimulationConfig()
#: The original simulator's semantics (FML order, no pre-execution filter).
LEGACY = SimulationConfig(decision_order="FML", filter_mode="none")
