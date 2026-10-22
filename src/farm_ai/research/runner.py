"""Experiment runner: executes every (agent, scenario, config) episode required by the experiment
config exactly once and writes transparent CSV datasets to research_outputs/data/.

Nothing here invents values: every row comes from an executed episode. Episodes that raise are kept
with their error message (and excluded from statistics by the table generator, which reports the
exclusion). Agents that cannot be constructed (e.g. an incompatible PPO checkpoint) are recorded as
skipped with the reason.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..config.settings import DATA_DIR
from ..framework.agents import make_policy, PPOCheckpointError, check_ppo_checkpoint
from ..framework.config import SimulationConfig
from ..framework.episode import run_episode
from ..framework.scenario import generate_scenario, Scenario

EXPERIMENT_FILES = {
    "A_agent_comparison": "agent_comparison_raw.csv",
    "B_feasibility_ablation": "ablation_results_raw.csv",
    "C_decision_order": "ablation_results_raw.csv",
    "D_environmental_robustness": "environmental_robustness_raw.csv",
    "F_feasibility_filters": "feasibility_results_raw.csv",
    "M_market_model": "market_model_raw.csv",
}


def _round(v):
    return round(v, 4) if isinstance(v, float) else v


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _round(r.get(k)) for k in fieldnames})


def run_id_for(agent: str, scenario_id: str, config_name: str) -> str:
    return "E-" + hashlib.sha1(f"{agent}|{scenario_id}|{config_name}".encode()).hexdigest()[:10]


def run_all(config_path: str, out_dir: str, root: str, n_override: Optional[int] = None, log=print) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = json.load(f)
    n = n_override or cfg["n_scenarios"]
    seeds = list(range(cfg.get("seed_offset", 0), cfg.get("seed_offset", 0) + n))
    configs = {k: SimulationConfig.from_dict(v) for k, v in cfg["configs"].items()}
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    # ── agents ──────────────────────────────────────────────────────────
    status: Dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_scenarios": n, "agents": {}}
    ppo_path = os.path.join(root, cfg["ppo_checkpoint"]) if cfg.get("ppo_checkpoint") else None
    usable_agents = []
    for agent in cfg["agents"]:
        if agent == "ppo":
            try:
                meta = check_ppo_checkpoint(ppo_path)["meta"]
                status["agents"]["ppo"] = {"status": "ok", "checkpoint": cfg["ppo_checkpoint"], "timesteps": meta.get("timesteps")}
                usable_agents.append(agent)
            except PPOCheckpointError as e:
                status["agents"]["ppo"] = {"status": "skipped", "reason": str(e)}
                log(f"[skip] PPO: {e}")
        else:
            status["agents"][agent] = {"status": "ok"}
            usable_agents.append(agent)

    # ── which episodes are needed ───────────────────────────────────────
    needed: Dict[tuple, set] = defaultdict(set)
    for exp_id, exp in cfg["experiments"].items():
        for profile in exp["profiles"]:
            for cname in exp["configs"]:
                for agent in usable_agents:
                    for seed in seeds:
                        needed[(agent, profile, seed, cname)].add(exp_id)

    scenarios: Dict[str, Scenario] = {}
    for (_, profile, seed, _) in needed:
        sid = f"S-{profile}-{seed}"
        if sid not in scenarios:
            scenarios[sid] = generate_scenario(profile, seed)

    summaries, daily, ledger_rows = [], [], []
    policies: Dict[str, Any] = {}
    t_start = time.time()
    keys = sorted(needed, key=lambda k: (k[0], k[1], k[3], k[2]))
    for i, (agent, profile, seed, cname) in enumerate(keys):
        sc = scenarios[f"S-{profile}-{seed}"]
        rid = run_id_for(agent, sc.scenario_id, cname)
        exps = ";".join(sorted(needed[(agent, profile, seed, cname)]))
        # Random agent: seeded per scenario so its choices are reproducible and paired across configs
        if agent == "ppo":
            policy = policies.get("ppo") or make_policy("ppo", ppo_path=ppo_path, deterministic=cfg.get("ppo_deterministic", True))
            policies["ppo"] = policy
        else:
            policy = make_policy(agent, seed=seed)
        records, summary, ledger = run_episode(policy, sc, configs[cname])
        base = {"run_id": rid, "experiments": exps, "agent": agent, "config_name": cname,
                "config_label": configs[cname].label()}
        summaries.append({**base, **summary})
        for r in records:
            daily.append({**base, "scenario_id": sc.scenario_id, **r})
        for c in ledger:
            ledger_rows.append({**base, "scenario_id": sc.scenario_id, "scenario_profile": profile, **c,
                                "harvest_eligible_by_horizon": int(c["harvest_eligible_day"] is not None)})
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(keys)} episodes ({time.time() - t_start:.0f}s)")

    # ── write datasets ──────────────────────────────────────────────────
    write_csv(os.path.join(data_dir, "episode_summary.csv"), summaries)
    write_csv(os.path.join(data_dir, "daily_episode_records.csv"), daily)
    write_csv(os.path.join(data_dir, "crop_ledger.csv"), ledger_rows)
    manifest = []
    for sid, sc in sorted(scenarios.items()):
        manifest.append({
            "scenario_id": sid, "profile": sc.profile, "seed": sc.seed, "days": sc.days,
            "drought_base": sc.profile_params["drought_base"], "pest_base": sc.profile_params["pest_base"],
            "shock_base": sc.profile_params["shock_base"],
            "mean_drought_risk": sum(sc.drought_risk) / sc.days, "mean_pest_risk": sum(sc.pest_risk) / sc.days,
            "mean_shock_risk": sum(sc.shock_risk) / sc.days,
            "drought_events": sum(sc.drought_event), "pest_events": sum(sc.pest_event),
            "price_shocks": sum(sc.shock_event),
        })
    write_csv(os.path.join(data_dir, "scenario_manifest.csv"), manifest)
    with open(os.path.join(data_dir, "scenarios.jsonl"), "w") as f:
        for sid, sc in sorted(scenarios.items()):
            f.write(json.dumps(sc.to_dict()) + "\n")

    by_file: Dict[str, List[Dict]] = defaultdict(list)
    for exp_id, exp in cfg["experiments"].items():
        for s in summaries:
            if exp_id in s["experiments"].split(";"):
                by_file[EXPERIMENT_FILES[exp_id]].append({"experiment_id": exp_id, **s})
    for fname, rows in by_file.items():
        write_csv(os.path.join(data_dir, fname), rows)

    hv = human_vs_ai(data_dir, usable_agents, ppo_path, cfg.get("ppo_deterministic", True), log)

    errors = [s for s in summaries if s["error"]]
    status.update({
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes_run": len(summaries),
        "episodes_successful": len(summaries) - len(errors),
        "episodes_failed": len(errors),
        "failures": [{"run_id": s["run_id"], "error": s["error"]} for s in errors[:50]],
        "episodes_skipped": sum(1 for (a, p, s_, c) in needed if a not in usable_agents),
        "skipped_agents": [a for a in cfg["agents"] if a not in usable_agents],
        "skipped_episodes_planned": len(seeds) * sum(
            len(e["profiles"]) * len(e["configs"]) for e in cfg["experiments"].values()
        ) * len([a for a in cfg["agents"] if a not in usable_agents]),
        "human_vs_ai": hv,
        "runtime_s": round(time.time() - t_start, 1),
    })
    with open(os.path.join(out_dir, "logs", "run_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    return status


def human_vs_ai(data_dir: str, agents: List[str], ppo_path: Optional[str], deterministic: bool, log=print) -> Dict[str, Any]:
    """Experiment E: every finished human match is replayed by each usable agent on the saved scenario."""
    match_files = sorted(glob.glob(os.path.join(os.environ.get("FARM_DATA_DIR", DATA_DIR), "matches", "*.json")))
    rows, daily = [], []
    used = 0
    for path in match_files:
        with open(path) as f:
            m = json.load(f)
        if m.get("status") == "human_playing" or not (m.get("human") or {}).get("summary"):
            continue
        used += 1
        sc_path = os.path.join(os.environ.get("FARM_DATA_DIR", DATA_DIR), "scenarios", f"{m['scenario_id']}.json")
        sc = Scenario.load(sc_path)
        config = SimulationConfig.from_dict(m["config"])
        hs = m["human"]["summary"]
        rows.append({"match_id": m["match_id"], "player_type": "human", "player": m.get("player_label", "human"), **hs})
        for r in m["human"]["records"]:
            daily.append({"match_id": m["match_id"], "player_type": "human", "player": m.get("player_label"), **r})
        for agent in agents:
            policy = make_policy(agent, seed=sc.seed, ppo_path=ppo_path, deterministic=deterministic)
            recs, summ, _ = run_episode(policy, Scenario.load(sc_path), config)
            rows.append({"match_id": m["match_id"], "player_type": "ai", "player": agent, **summ})
            for r in recs:
                daily.append({"match_id": m["match_id"], "player_type": "ai", "player": agent, **r})
    write_csv(os.path.join(data_dir, "human_ai_comparison_raw.csv"), rows,
              None if rows else ["match_id", "player_type", "player", "scenario_id", "profit"])
    write_csv(os.path.join(data_dir, "human_ai_daily_records.csv"), daily,
              None if daily else ["match_id", "player_type", "player", "day", "wealth_end"])
    log(f"Human-vs-AI: {used} completed human match(es) found")
    return {"completed_human_matches": used, "rows": len(rows)}
