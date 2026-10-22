"""Runs complete 30-day episodes and turns the environment's logs into research records.

`run_episode` is used by the experiment runner and by the Human-vs-AI replay, so every AI run
follows exactly the same code path (scenario → env → propose → filter → execute) as a human run
through the API.

Metric definitions (units in brackets) — see research_outputs/README.md:
  initial_wealth, final_wealth  [₹]  cash + inventory × current price (unharvested crops are worth 0)
  profit                        [₹]  final_wealth − initial_wealth
  total_revenue                 [₹]  Σ sale proceeds;  total_costs [₹] Σ planting costs paid
  harvest_qty                   [units] Σ harvested produce (abstract units, not calibrated to kg)
  crops_planted / harvested / destroyed / abandoned [count]
  completion_rate               harvested / planted (abandoned and destroyed crops count as not completed)
  actions_attempted             [count] actions submitted to the environment (one per day → 30)
  invalid_actions               [count] submitted actions that failed structural checks in the env
  candidates_evaluated          [count] candidate actions checked by the filter (Stage 3)
  feasibility_rejections        [count] candidates rejected by the filter
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..config.settings import SEASON_LENGTH, STARTING_CASH
from ..environment.farm_environment import FarmEnvironment
from .agents import Policy
from .config import SimulationConfig
from .feasibility import FeasibilityFilter, decide
from .scenario import Scenario


def make_filter(config: SimulationConfig) -> FeasibilityFilter:
    return FeasibilityFilter(config.filter_mode, config.completion_threshold, config.threshold_rule)


def run_episode(policy: Policy, scenario: Scenario, config: SimulationConfig) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    env = FarmEnvironment(config=config)
    env.reset(options={"scenario": scenario})
    filt = make_filter(config)
    decision_s = 0.0
    filter_ms = 0.0
    error = None
    try:
        while not env.season_over:
            t0 = time.perf_counter()
            ranked = policy.rank(env)
            action = decide(env, ranked, filt)
            decision_s += time.perf_counter() - t0
            filter_ms += sum(c.compute_ms for c in env._candidates)
            env.step(action)
    except Exception as e:  # recorded, never silently replaced
        error = f"{type(e).__name__}: {e}"
    summary = summarise(env, decision_s=decision_s, filter_ms=filter_ms, error=error)
    return list(env.transition_log), summary, [dict(v) for v in env.crop_ledger.values()]


def summarise(env: FarmEnvironment, decision_s: float = 0.0, filter_ms: float = 0.0, error: Optional[str] = None) -> Dict[str, Any]:
    log = env.transition_log
    ledger = list(env.crop_ledger.values())
    planted = len(ledger)
    harvested = sum(1 for c in ledger if c["outcome"] == "harvested")
    destroyed = sum(1 for c in ledger if c["outcome"] == "destroyed")
    completed_season = env.season_over and len(log) == SEASON_LENGTH
    final_inv = env._inventory_value()
    return {
        "scenario_id": env.scenario.scenario_id,
        "scenario_profile": env.scenario.profile,
        "scenario_seed": env.scenario.seed,
        "decision_order": env.config.decision_order,
        "filter_mode": env.config.filter_mode,
        "market_model": env.config.market_model,
        "threshold_rule": env.config.threshold_rule if env.config.filter_mode == "uncertainty" else None,
        "completion_threshold": env.config.completion_threshold if env.config.filter_mode == "uncertainty" else None,
        "initial_wealth": float(STARTING_CASH),
        "final_cash": float(env.cash),
        "final_inventory_value": float(final_inv),
        "final_wealth": float(env.cash + final_inv),
        "profit": float(env.cash + final_inv - STARTING_CASH),
        "total_revenue": float(sum(r["revenue"] for r in log)),
        "total_costs": float(sum(r["planting_cost"] for r in log)),
        "harvest_qty": int(sum(r["harvest_qty"] for r in log)),
        "units_sold": int(sum(r["sold_qty"] for r in log)),
        "crops_planted": planted,
        "crops_harvested": harvested,
        "crops_destroyed": destroyed,
        "crops_abandoned": planted - harvested - destroyed,
        "destroyed_planting_cost": float(sum(c["planting_cost"] for c in ledger if c["outcome"] == "destroyed")),
        "drought_setback_days": int(sum(c["drought_setbacks"] for c in ledger)),
        "completion_rate": (harvested / planted) if planted else None,
        "actions_attempted": len(log),
        "invalid_actions": int(sum(1 for r in log if not r["action_valid"])),
        "candidates_evaluated": int(sum(r["candidates_evaluated"] for r in log)),
        "feasibility_rejections": int(sum(r["feasibility_rejections"] for r in log)),
        "total_reward": float(sum(r["reward"] for r in log)),
        "decision_time_s": decision_s,
        "filter_time_ms": filter_ms,
        "season_completed": bool(completed_season),
        "error": error,
    }
