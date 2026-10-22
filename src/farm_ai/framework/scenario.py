"""Pre-generated, replayable 30-day environmental scenarios.

A scenario fixes every exogenous quantity *before* the season starts:

* daily **risk** values (probabilities) for drought, pest outbreak and price shock;
* daily **events** (whether a drought / outbreak / shock actually occurs), drawn as
  ``event_t = U_t < risk_t`` from a dedicated uniform stream;
* the uniform number used to pick which planted field an outbreak hits;
* the exogenous reference price path of every crop (random walk + shocks).

Risk and occurrence are separate: a high risk does not imply an event and an event can
occur on a low-risk day. Nothing in a scenario depends on any player's actions, so the same
scenario can be replayed for a human and for any AI agent.

Reproducibility: all draws come from ``numpy.random.SeedSequence(seed)`` spawned into one
independent stream per subsystem. Changing one profile parameter (e.g. drought base rate)
leaves the other subsystems' draws unchanged and re-uses the same drought uniforms
(common random numbers across scenario groups).

These are synthetic simulation conditions, not calibrated weather or pest forecasts.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..config.settings import CROP_CONFIG, SEASON_LENGTH, EVENT_PROBABILITIES

SCENARIO_FORMAT_VERSION = 1
CROPS: List[str] = list(CROP_CONFIG.keys())

# Risk-label thresholds (probability per day). Labels are a display convenience only.
RISK_LABEL_THRESHOLDS = (0.06, 0.12)


def risk_label(p: float) -> str:
    lo, hi = RISK_LABEL_THRESHOLDS
    return "LOW" if p < lo else ("MEDIUM" if p < hi else "HIGH")


@dataclass(frozen=True)
class ScenarioProfile:
    """Generation rules for one scenario group."""

    name: str
    description: str
    drought_base: float
    pest_base: float
    shock_base: float
    # Daily risk = base × exp(σ·z_t − σ²/2), z_t a stationary AR(1) with persistence ρ (mean multiplier ≈ 1).
    risk_volatility: float = 0.6
    risk_persistence: float = 0.8
    price_drift: float = 0.05          # uniform daily drift half-width (±5 %, as in the original Market)
    price_floor: float = 0.5           # × base price, applied before a shock
    price_cap: float = 1.5             # × base price, applied before a shock
    shock_factors: tuple = (0.5, 2.0)  # equiprobable multipliers, as in the original PriceShockSystem
    max_risk: float = 0.9


PROFILES: Dict[str, ScenarioProfile] = {
    "baseline": ScenarioProfile(
        "baseline",
        "Original simulator rates: drought 5 %/day, pests 8 %/day, price shocks 10 %/day (time-varying around these means)",
        EVENT_PROBABILITIES["drought"], EVENT_PROBABILITIES["pest_outbreak"], EVENT_PROBABILITIES["price_shock"],
    ),
    "dry": ScenarioProfile(
        "dry", "Drought-prone season: drought base rate tripled to 15 %/day",
        0.15, EVENT_PROBABILITIES["pest_outbreak"], EVENT_PROBABILITIES["price_shock"],
    ),
    "pest": ScenarioProfile(
        "pest", "Pest-pressure season: outbreak base rate raised to 20 %/day",
        EVENT_PROBABILITIES["drought"], 0.20, EVENT_PROBABILITIES["price_shock"],
    ),
    "combined": ScenarioProfile(
        "combined", "Combined stress: drought 15 %/day and pests 20 %/day",
        0.15, 0.20, EVENT_PROBABILITIES["price_shock"],
    ),
}


@dataclass
class Scenario:
    scenario_id: str
    profile: str
    seed: int
    days: int
    drought_risk: List[float]
    drought_event: List[bool]
    pest_risk: List[float]
    pest_event: List[bool]
    pest_target_u: List[float]
    shock_risk: List[float]
    shock_event: List[bool]
    shock_factor: List[float]          # 1.0 when no shock
    reference_prices: Dict[str, List[float]]
    profile_params: Dict[str, Any] = field(default_factory=dict)
    format_version: int = SCENARIO_FORMAT_VERSION

    # Day-indexed accessors (days are 1-based)
    def drought(self, day: int) -> bool:
        return self.drought_event[day - 1]

    def pest(self, day: int) -> bool:
        return self.pest_event[day - 1]

    def price(self, crop: str, day: int) -> float:
        return self.reference_prices[crop][min(max(day, 1), self.days) - 1]

    def risks(self, day: int) -> Dict[str, float]:
        d = min(max(day, 1), self.days) - 1
        return {"drought": self.drought_risk[d], "pest": self.pest_risk[d], "price_shock": self.shock_risk[d]}

    def risk_labels(self, day: int) -> Dict[str, str]:
        return {k: risk_label(v) for k, v in self.risks(day).items()}

    def public_timeline(self) -> List[Dict[str, Any]]:
        """Ex-ante information shown before play: risks only (no realised events or future prices)."""
        return [
            {
                "day": d + 1,
                "drought_risk": self.drought_risk[d],
                "pest_risk": self.pest_risk[d],
                "price_shock_risk": self.shock_risk[d],
                "drought_label": risk_label(self.drought_risk[d]),
                "pest_label": risk_label(self.pest_risk[d]),
                "price_shock_label": risk_label(self.shock_risk[d]),
            }
            for d in range(self.days)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        return cls(**d)

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.scenario_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> "Scenario":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def exogenous_signature(self) -> tuple:
        """Hashable summary of all exogenous draws (used by tests to prove identical replay)."""
        return (
            tuple(self.drought_event), tuple(self.pest_event), tuple(round(u, 12) for u in self.pest_target_u),
            tuple(self.shock_event), tuple(self.shock_factor),
            tuple(tuple(round(p, 9) for p in self.reference_prices[c]) for c in CROPS),
        )


def make_scenario_id(profile: str, seed: int) -> str:
    return f"S-{profile}-{seed}"


def _risk_path(rng: np.random.Generator, base: float, prof: ScenarioProfile, n: int) -> np.ndarray:
    rho, sigma = prof.risk_persistence, prof.risk_volatility
    z = np.empty(n)
    z[0] = rng.standard_normal()
    eps = rng.standard_normal(n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + math.sqrt(1 - rho * rho) * eps[t]
    return np.clip(base * np.exp(sigma * z - 0.5 * sigma * sigma), 0.0, prof.max_risk)


def generate_scenario(profile: str = "baseline", seed: int = 0, days: int = SEASON_LENGTH) -> Scenario:
    if profile not in PROFILES:
        raise ValueError(f"Unknown scenario profile '{profile}'. Choose from {list(PROFILES)}")
    prof = PROFILES[profile]
    ss = np.random.SeedSequence(int(seed))
    # One independent stream per subsystem (order fixed for reproducibility).
    s_drisk, s_devent, s_prisk, s_pevent, s_ptarget, s_srisk, s_sevent, s_sfactor, s_price = (
        np.random.default_rng(child) for child in ss.spawn(9)
    )

    drought_risk = _risk_path(s_drisk, prof.drought_base, prof, days)
    pest_risk = _risk_path(s_prisk, prof.pest_base, prof, days)
    shock_risk = _risk_path(s_srisk, prof.shock_base, prof, days)
    drought_event = s_devent.random(days) < drought_risk
    pest_event = s_pevent.random(days) < pest_risk
    pest_target_u = s_ptarget.random(days)
    shock_event = s_sevent.random(days) < shock_risk
    factor_u = s_sfactor.random(days)
    lo_f, hi_f = prof.shock_factors
    shock_factor = np.where(shock_event, np.where(factor_u < 0.5, lo_f, hi_f), 1.0)

    noise = s_price.uniform(-prof.price_drift, prof.price_drift, size=(days, len(CROPS)))
    prices: Dict[str, List[float]] = {}
    for j, crop in enumerate(CROPS):
        base = CROP_CONFIG[crop]["base_price"]
        path, prev = [], base
        for t in range(days):
            pre = base if t == 0 else min(max(prev * (1 + noise[t, j]), prof.price_floor * base), prof.price_cap * base)
            cur = pre * shock_factor[t]
            path.append(float(cur))
            prev = cur
        prices[crop] = path

    return Scenario(
        scenario_id=make_scenario_id(profile, seed),
        profile=profile,
        seed=int(seed),
        days=days,
        drought_risk=[float(x) for x in drought_risk],
        drought_event=[bool(x) for x in drought_event],
        pest_risk=[float(x) for x in pest_risk],
        pest_event=[bool(x) for x in pest_event],
        pest_target_u=[float(x) for x in pest_target_u],
        shock_risk=[float(x) for x in shock_risk],
        shock_event=[bool(x) for x in shock_event],
        shock_factor=[float(x) for x in shock_factor],
        reference_prices=prices,
        profile_params=json.loads(json.dumps(asdict(prof))),  # JSON-native (tuples → lists) so save/load round-trips exactly
    )


def scenario_from_id(scenario_id: str) -> Scenario:
    """Regenerates a scenario from an id of the form ``S-<profile>-<seed>``."""
    try:
        _, profile, seed = scenario_id.split("-", 2)
        return generate_scenario(profile, int(seed))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Malformed scenario id '{scenario_id}'") from e
