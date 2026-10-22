"""PPO training for environment version 2 (scenario-driven, six-stage daily process).

Usage (from project root):
    python scripts/train_ppo.py                      # defaults below
    python scripts/train_ppo.py --timesteps 1000000 --seed 7

What is trained
---------------
* Stable-Baselines3 PPO, MlpPolicy, on `FarmEnvironment(config=SimulationConfig(filter_mode="none"))`:
  the agent sees every action and invalid actions are penalised (−50 ₹ reward) exactly as in the
  original simulator. The feasibility filter is applied only at decision time (action masking by
  ranking), so the *same* checkpoint can be evaluated with and without the filter.
* Training scenarios are generated from the env's seeded RNG with seeds in [1e6, 2^31) and a uniform
  mix of the four scenario profiles. Validation uses seeds 900000–900019 and evaluation in the
  experiments uses seeds 0–999, so training, validation and test scenarios never overlap.
* Rewards are divided by 1,000 during training only (`RewardScale` wrapper) to keep value targets
  O(1); this does not change the optimal policy. Logged profits are always unscaled ₹.

Outputs
-------
* `<out>.zip` checkpoint and `<out>.meta.json` metadata (env version, spaces, timesteps, seeds, config)
* `research_outputs/logs/ppo_training_eval.csv` — validation curve (greedy policy, unscaled ₹)
* `research_outputs/logs/ppo_training_monitor.csv` — per-episode training returns (scaled)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from typing import Dict, List

import gymnasium as gym
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from farm_ai.environment.farm_environment import FarmEnvironment, ENV_VERSION, OBS_SIZE, TOTAL_ACTIONS  # noqa: E402
from farm_ai.framework.config import SimulationConfig  # noqa: E402
from farm_ai.framework.scenario import generate_scenario, PROFILES  # noqa: E402

REWARD_SCALE = 1e-3
VALIDATION_SEEDS = list(range(900_000, 900_020))
TRAIN_CONFIG = SimulationConfig(decision_order="LFM", filter_mode="none")
HYPERPARAMS = dict(
    policy="MlpPolicy",
    policy_kwargs={"net_arch": [128, 128]},
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,
)
N_ENVS = 8


class RewardScale(gym.RewardWrapper):
    def reward(self, reward):
        return reward * REWARD_SCALE


def greedy_validation(model: PPO, filter_mode: str = "none") -> Dict[str, float]:
    """Unscaled ₹ profit and invalid-action rate of the greedy policy on held-out scenarios."""
    from farm_ai.framework.agents import ppo_action_probabilities  # local import avoids a cycle at module load
    from farm_ai.framework.feasibility import FeasibilityFilter, decide

    profits, invalid, steps = [], 0, 0
    filt = FeasibilityFilter(filter_mode)
    for i, seed in enumerate(VALIDATION_SEEDS):
        profile = list(PROFILES)[i % len(PROFILES)]
        env = FarmEnvironment(config=SimulationConfig(decision_order="LFM", filter_mode=filter_mode))
        obs, _ = env.reset(options={"scenario": generate_scenario(profile, seed)})
        while not env.season_over:
            if filter_mode == "none":
                a, _ = model.predict(obs, deterministic=True)
                a = int(a)
            else:
                ranked = list(np.argsort(-ppo_action_probabilities(model, obs), kind="stable"))
                a = decide(env, ranked, filt)
            obs, r, done, _, info = env.step(a)
            invalid += r == -50.0
            steps += 1
        profits.append(info["profit"])
    return {
        "mean_profit": statistics.mean(profits),
        "sd_profit": statistics.stdev(profits),
        "invalid_rate": invalid / steps,
    }


class ValidationCallback(BaseCallback):
    def __init__(self, every: int, rows: List[Dict], verbose=0):
        super().__init__(verbose)
        self.every = every
        self.rows = rows
        self.next_at = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_at:
            self.next_at += self.every
            for fm in ("none", "deterministic"):
                v = greedy_validation(self.model, fm)
                self.rows.append({"timesteps": self.num_timesteps, "filter_mode": fm, **v})
                print(f"[val] t={self.num_timesteps:>8} filter={fm:13s} profit={v['mean_profit']:9.0f} "
                      f"invalid={v['invalid_rate']:.3f}", flush=True)
        return True


def make_env(rank: int, seed: int):
    def _f():
        env = FarmEnvironment(config=TRAIN_CONFIG, profile_mix=list(PROFILES))
        env = Monitor(env)
        env = RewardScale(env)
        env.reset(seed=seed * 1000 + rank + 1_000_000)
        return env
    return _f


def train(timesteps: int, seed: int, out: str, eval_every: int, log_dir: str) -> Dict:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    venv = DummyVecEnv([make_env(i, seed) for i in range(N_ENVS)])
    model = PPO(env=venv, seed=seed, verbose=0, device="cpu", **HYPERPARAMS)
    rows: List[Dict] = []
    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=ValidationCallback(eval_every, rows))
    elapsed = time.time() - t0
    rows.append({"timesteps": model.num_timesteps, "filter_mode": "none", **greedy_validation(model, "none")})
    rows.append({"timesteps": model.num_timesteps, "filter_mode": "deterministic", **greedy_validation(model, "deterministic")})
    model.save(out)

    meta = {
        "env_version": ENV_VERSION,
        "obs_size": OBS_SIZE,
        "n_actions": TOTAL_ACTIONS,
        "algorithm": "PPO (stable-baselines3)",
        "timesteps": int(model.num_timesteps),
        "seed": seed,
        "n_envs": N_ENVS,
        "hyperparams": {k: v for k, v in HYPERPARAMS.items()},
        "reward": "Δwealth (₹) per day, −50 for an invalid action, × 1e-3 during training",
        "train_config": TRAIN_CONFIG.to_dict(),
        "train_profiles": list(PROFILES),
        "train_seed_range": "[1e6, 2^31) drawn by the env RNG",
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
        "final_validation": rows[-2:],
        "train_seconds": round(elapsed, 1),
    }
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(log_dir, "ppo_training_eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timesteps", "filter_mode", "mean_profit", "sd_profit", "invalid_rate"])
        w.writeheader()
        w.writerows(rows)
    # Per-episode training returns from the Monitor wrappers
    with open(os.path.join(log_dir, "ppo_training_monitor.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "episode", "episode_return_rupees", "length"])
        for i, e in enumerate(venv.envs):
            mon = e.env  # RewardScale wraps Monitor, so Monitor records unscaled ₹ returns (incl. −50 penalties)
            for j, (r, l) in enumerate(zip(mon.get_episode_rewards(), mon.get_episode_lengths())):
                w.writerow([i, j, r, l])
    print(f"Saved {out}.zip ({model.num_timesteps} steps, {elapsed:.0f}s)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "models/ppo/farm_ppo_v2"))
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "research_outputs/logs"))
    a = ap.parse_args()
    train(a.timesteps, a.seed, a.out, a.eval_every, a.log_dir)


if __name__ == "__main__":
    main()
