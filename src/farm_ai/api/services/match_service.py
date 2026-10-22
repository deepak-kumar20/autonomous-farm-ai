"""Human-vs-AI matches.

A match fixes one saved scenario and one SimulationConfig. The human plays first through the normal
farm endpoints (same environment, same feasibility filter). When the human's season ends the full
episode (daily records, summary, crop ledger) is written to disk. AI runs are then executed with
`framework.episode.run_episode` on a *fresh copy of the scenario loaded from disk* — never on the
human's environment — so the human run cannot leak into or mutate the AI run.

Files: <DATA_DIR>/scenarios/<scenario_id>.json and <DATA_DIR>/matches/<match_id>.json.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from ...config.settings import DATA_DIR, STARTING_CASH, NUM_FIELDS
from ...framework.agents import make_policy
from ...framework.config import SimulationConfig
from ...framework.episode import run_episode, summarise
from ...framework.scenario import Scenario, generate_scenario, scenario_from_id
from .farm_service import FarmSessionManager, to_json_compatible


def scenario_dir() -> str:
    return os.path.join(os.environ.get("FARM_DATA_DIR", DATA_DIR), "scenarios")


def match_dir() -> str:
    return os.path.join(os.environ.get("FARM_DATA_DIR", DATA_DIR), "matches")


def load_or_create_scenario(scenario_id: Optional[str] = None, profile: str = "baseline", seed: Optional[int] = None) -> Scenario:
    if scenario_id:
        path = os.path.join(scenario_dir(), f"{scenario_id}.json")
        sc = Scenario.load(path) if os.path.isfile(path) else scenario_from_id(scenario_id)
    else:
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") % 1_000_000
        sc = generate_scenario(profile, seed)
    sc.save(scenario_dir())
    return sc


def public_scenario(sc: Scenario) -> Dict[str, Any]:
    """Ex-ante view of a scenario: risks only, no realised events or future prices."""
    p = sc.profile_params
    return {
        "scenario_id": sc.scenario_id,
        "profile": sc.profile,
        "seed": sc.seed,
        "days": sc.days,
        "description": p.get("description"),
        "base_rates": {"drought": p.get("drought_base"), "pest": p.get("pest_base"), "price_shock": p.get("shock_base")},
        "price_model": {
            "daily_drift": p.get("price_drift"),
            "floor_x_base": p.get("price_floor"),
            "cap_x_base": p.get("price_cap"),
            "shock_factors": p.get("shock_factors"),
        },
        "starting_prices": {c: v[0] for c, v in sc.reference_prices.items()},
        "timeline": sc.public_timeline(),
        "note": "Simulated conditions for a virtual farm. Values are daily probabilities (risks), not forecasts of real weather and not realised events.",
    }


class MatchStore:
    def __init__(self, farm: FarmSessionManager):
        self.farm = farm
        self._lock = threading.Lock()

    def _path(self, match_id: str) -> str:
        return os.path.join(match_dir(), f"{match_id}.json")

    def save(self, match: Dict[str, Any]) -> None:
        os.makedirs(match_dir(), exist_ok=True)
        with self._lock, open(self._path(match["match_id"]), "w") as f:
            json.dump(to_json_compatible(match), f)

    def load(self, match_id: str) -> Dict[str, Any]:
        path = self._path(match_id)
        if not os.path.isfile(path):
            raise KeyError(f"Match '{match_id}' not found")
        with open(path) as f:
            return json.load(f)

    def list(self) -> List[Dict[str, Any]]:
        if not os.path.isdir(match_dir()):
            return []
        out = []
        for name in sorted(os.listdir(match_dir())):
            if name.endswith(".json"):
                with open(os.path.join(match_dir(), name)) as f:
                    m = json.load(f)
                out.append({k: m.get(k) for k in ("match_id", "scenario_id", "created_at", "player_label", "status")}
                           | {"human_profit": (m.get("human") or {}).get("summary", {}).get("profit"),
                              "ai_runs": len(m.get("ai_runs", []))})
        return out

    def create(self, scenario: Scenario, config: SimulationConfig, player_label: str) -> Dict[str, Any]:
        match_id = "M-" + uuid.uuid4().hex[:10]
        session_id, state, _ = self.farm.create_session(scenario=scenario, config=config, match_id=match_id)
        match = {
            "match_id": match_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "player_label": player_label,
            "status": "human_playing",
            "scenario_id": scenario.scenario_id,
            "config": config.to_dict(),
            "initial_state": {"cash": STARTING_CASH, "fields": NUM_FIELDS, "day": 1, "inventory": state["inventory"]},
            "human": {"session_id": session_id, "records": [], "summary": None, "ledger": []},
            "ai_runs": [],
        }
        self.save(match)
        return match

    def complete_human(self, match_id: str, env) -> Dict[str, Any]:
        match = self.load(match_id)
        if match["status"] != "human_playing":
            return match
        match["human"].update(
            records=list(env.transition_log),
            summary=summarise(env),
            ledger=[dict(v) for v in env.crop_ledger.values()],
        )
        match["status"] = "human_complete"
        self.save(match)
        return match

    def run_ai(self, match_id: str, agent: str, deterministic: bool = True, ppo_path: Optional[str] = None) -> Dict[str, Any]:
        match = self.load(match_id)
        if match["status"] == "human_playing":
            raise ValueError("The human run is not finished yet. Play all 30 days first.")
        scenario = Scenario.load(os.path.join(scenario_dir(), f"{match['scenario_id']}.json"))
        config = SimulationConfig.from_dict(match["config"])
        policy = make_policy(agent, seed=scenario.seed, ppo_path=ppo_path, deterministic=deterministic)
        records, summary, ledger = run_episode(policy, scenario, config)
        run = {
            "run_id": "R-" + uuid.uuid4().hex[:8],
            "agent": agent,
            "agent_info": policy.describe(),
            "deterministic": deterministic,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "records": records,
            "summary": summary,
            "ledger": ledger,
        }
        match["ai_runs"].append(run)
        match["status"] = "compared"
        self.save(match)
        return run
