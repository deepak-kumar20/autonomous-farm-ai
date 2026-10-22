"""Stage 3 — feasibility filter.

Every candidate action is checked *before* execution. A rejected action changes nothing
(no cash, field, inventory or day change); the decision-maker must propose another action.
HOLD is always feasible, so a decision always terminates.

Modes (``SimulationConfig.filter_mode``):

``none``
    Pass-through. Candidates are not checked; an infeasible action reaches the environment,
    fails there (counted as an *invalid action*, −50 reward) and the day still passes. This
    reproduces the original simulator.
``deterministic``
    Structural constraints (cash, field state, maturity, inventory, season not over) plus the
    nominal horizon constraint for planting: a crop planted on day d with growth time g can first
    be harvested on day d + g (no drought), so PLANT is feasible only if d + g ≤ 30.
``uncertainty``
    Everything in ``deterministic``, plus PLANT requires
    P(crop is present and harvest-eligible at a decision point on or before day 30) ≥ τ,
    computed by `completion_probability` from the scenario's *published daily risks* (never
    from realised events).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence

from ..config.settings import CROP_CONFIG, SEASON_LENGTH


@dataclass
class FeasibilityResult:
    accepted: bool
    code: str
    reason: str
    mode: str
    action_id: int
    constraints: Dict[str, Any] = field(default_factory=dict)
    completion_probability: Optional[float] = None
    threshold: Optional[float] = None
    compute_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def completion_probability(
    growth_days: int,
    plant_day: int,
    drought_risk: Sequence[float],
    pest_risk: Sequence[float],
    planted_fields_after: int,
    order: str = "LFM",
    horizon: int = SEASON_LENGTH,
) -> float:
    """Exact probability, under the simulator's event rules, that a crop planted today is present
    and MATURE at some decision point on or before `horizon`.

    Model assumptions (documented approximations):
      * drought on day t occurs with probability drought_risk[t-1] (independent across days) and
        sets a GROWING crop with progress > 0 back by one day;
      * an outbreak on day t occurs with probability pest_risk[t-1] and hits this field with
        probability 1 / planted_fields_after (number of planted fields held constant);
      * growth adds one day of progress per day; a crop matures when progress ≥ growth_days.
    The stage order (LFM or FML) is reproduced exactly.
    """
    q = 1.0 / max(1, planted_fields_after)
    dist: Dict[int, float] = {0: 1.0}   # progress → probability (alive, growing)
    mature_alive = 0.0                   # alive and mature, waiting for the next decision point
    done = 0.0

    def drought(dist_in: Dict[int, float], p: float) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for k, pr in dist_in.items():
            if k > 0:
                out[k - 1] = out.get(k - 1, 0.0) + pr * p
                out[k] = out.get(k, 0.0) + pr * (1 - p)
            else:
                out[k] = out.get(k, 0.0) + pr
        return out

    for t in range(plant_day, horizon + 1):
        pd, pp = drought_risk[t - 1], pest_risk[t - 1]
        if t > plant_day:
            if order == "LFM":
                dist = drought(dist, pd)
                survive = 1.0 - pp * q
                dist = {k: v * survive for k, v in dist.items()}
                mature_alive *= survive
            # decision point of day t
            done += mature_alive
            mature_alive = 0.0
        # growth (stage 6 in LFM; before losses in FML)
        grown: Dict[int, float] = {}
        for k, pr in dist.items():
            if k + 1 >= growth_days:
                mature_alive += pr
            else:
                grown[k + 1] = grown.get(k + 1, 0.0) + pr
        dist = grown
        if order == "FML":
            dist = drought(dist, pd)
            survive = 1.0 - pp * q
            dist = {k: v * survive for k, v in dist.items()}
            mature_alive *= survive
    return min(1.0, max(0.0, done))


class FeasibilityFilter:
    def __init__(self, mode: str = "deterministic", threshold: float = 0.8, threshold_rule: str = "fixed"):
        self.mode = mode
        self.threshold = threshold
        self.threshold_rule = threshold_rule

    def plant_threshold(self, env, crop_name: str) -> float:
        if self.threshold_rule == "break_even":
            cfg = CROP_CONFIG[crop_name]
            value = cfg["yield_amount"] * env.market.get_current_price(crop_name)
            return min(1.0, cfg["planting_cost"] / value) if value > 0 else 1.0
        return self.threshold

    def check(self, env, action_id: int) -> FeasibilityResult:
        t0 = time.perf_counter()
        res = self._check(env, int(action_id))
        res.compute_ms = (time.perf_counter() - t0) * 1000.0
        return res

    def _check(self, env, action_id: int) -> FeasibilityResult:
        mode = self.mode
        if env.season_over:
            return FeasibilityResult(False, "SEASON_OVER", "The 30-day season has ended.", mode, action_id)
        if mode == "none":
            return FeasibilityResult(True, "UNFILTERED", "No feasibility filter: action passed to the environment unchecked.", mode, action_id)

        ok, code, reason, constraints = env.check_structural(action_id)
        if not ok:
            return FeasibilityResult(False, code, reason, mode, action_id, constraints)

        action = env.decode(action_id)
        if action.type.value != "PLANT":
            return FeasibilityResult(True, "OK", "All applicable constraints satisfied.", mode, action_id, constraints)

        g = CROP_CONFIG[action.crop_name]["growth_days"]
        day = env.day
        earliest_harvest = day + g
        constraints.update({"growth_days": g, "earliest_harvest_day": earliest_harvest, "last_day": SEASON_LENGTH})
        if earliest_harvest > SEASON_LENGTH:
            return FeasibilityResult(
                False, "HORIZON",
                f"{action.crop_name} needs {g} days: planted on day {day} it could first be harvested on day "
                f"{earliest_harvest}, after the season ends on day {SEASON_LENGTH}.",
                mode, action_id, constraints,
            )
        if mode == "deterministic":
            return FeasibilityResult(True, "OK", "All applicable constraints satisfied.", mode, action_id, constraints)

        p = env.plant_completion_probability(action.crop_name)
        tau = self.plant_threshold(env, action.crop_name)
        rule = "break-even" if self.threshold_rule == "break_even" else "fixed"
        if p < tau:
            return FeasibilityResult(
                False, "LOW_COMPLETION_PROBABILITY",
                f"Estimated probability that {action.crop_name} is ready to harvest by day {SEASON_LENGTH} is "
                f"{p:.0%}, below the {rule} threshold of {tau:.0%}.",
                mode, action_id, constraints, completion_probability=p, threshold=tau,
            )
        return FeasibilityResult(
            True, "OK", f"Feasible; completion probability {p:.0%} ≥ {rule} threshold {tau:.0%}.",
            mode, action_id, constraints, completion_probability=p, threshold=tau,
        )


def decide(env, ranked_actions: List[int], filt: FeasibilityFilter) -> int:
    """Stage 2 → 3: propose candidates in preference order until one passes the filter.

    Each rejection is recorded on the environment's current-day log. Returns the accepted action id.
    """
    tried = set()
    for a in list(ranked_actions) + [0]:
        if a in tried:
            continue
        tried.add(a)
        res = filt.check(env, a)
        env.record_candidate(res)
        if res.accepted:
            return a
    raise RuntimeError("HOLD was rejected — this should be impossible before the season ends")
