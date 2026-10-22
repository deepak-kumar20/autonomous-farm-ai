"""Decision-makers for the research framework (Stage 2 — candidate generation).

Every policy returns a *preference-ordered list of action ids* from the same Discrete(24) space
the human uses. `framework.feasibility.decide` walks that list until the filter accepts one
(under ``filter_mode="none"`` the first candidate is always passed through unchecked).

* `RandomPolicy`   — the original RandomAgent's candidate set (HOLD, harvest any MATURE field,
                     plant any affordable crop on an EMPTY field, sell any crop in stock), sampled
                     uniformly. Change from the original: SELL always sells the whole stack (the
                     discrete action), instead of a random partial quantity.
* `RuleBasedPolicy`— wraps the original RuleBasedAgent unchanged; its choice is followed by HOLD.
* `PPOPolicy`      — ranks all 24 actions by the trained policy's action probabilities
                     (greedy = most probable first; stochastic = sampled order).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..agents.rule_based_agent import RuleBasedAgent
from ..config.settings import CROP_CONFIG
from ..environment.farm_environment import (
    FarmEnvironment, farm_action_to_int, TOTAL_ACTIONS, OBS_SIZE, ENV_VERSION,
)


class Policy:
    name = "policy"

    def rank(self, env: FarmEnvironment) -> List[int]:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"agent": self.name}


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def candidates(self, env: FarmEnvironment) -> List[int]:
        state = env.get_dict_state()
        cands = [0]
        for f in state["fields"]:
            if f["state"] == "MATURE":
                cands.append(1 + f["id"])
            elif f["state"] == "EMPTY":
                for ci, crop in enumerate(CROP_CONFIG):
                    if state["cash"] >= CROP_CONFIG[crop]["planting_cost"]:
                        cands.append(6 + ci * 5 + f["id"])
        for ci, crop in enumerate(state["inventory"]):
            if state["inventory"][crop] > 0:
                cands.append(21 + ci)
        return cands

    def rank(self, env):
        cands = self.candidates(env)
        order = list(self.rng.permutation(len(cands)))
        return [cands[i] for i in order]


class RuleBasedPolicy(Policy):
    name = "rule_based"

    def __init__(self):
        self.agent = RuleBasedAgent()

    def rank(self, env):
        a = farm_action_to_int(self.agent.select_action(env.get_dict_state()))
        return [a if a is not None else 0, 0]


class PPOCheckpointError(RuntimeError):
    pass


def check_ppo_checkpoint(path: str) -> Dict[str, Any]:
    """Validates that a PPO checkpoint matches this environment before it is used.

    Reads the SB3 zip's saved spaces and the sidecar metadata `<path>.meta.json` (written by the
    training script). Raises PPOCheckpointError with a human-readable reason when incompatible.
    """
    from stable_baselines3 import PPO

    zip_path = path if path.endswith(".zip") else path + ".zip"
    if not os.path.isfile(zip_path):
        raise PPOCheckpointError(f"Checkpoint not found: {zip_path}")
    meta_path = zip_path[:-4] + ".meta.json"
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    if meta.get("env_version") != ENV_VERSION:
        raise PPOCheckpointError(
            f"Checkpoint was trained on environment version {meta.get('env_version', '1 (legacy, no metadata)')}, "
            f"but this simulator is version {ENV_VERSION}. Retrain with scripts/train_ppo.py."
        )
    model = PPO.load(zip_path, device="cpu")
    if tuple(model.observation_space.shape) != (OBS_SIZE,) or model.action_space.n != TOTAL_ACTIONS:
        raise PPOCheckpointError(
            f"Space mismatch: checkpoint obs {model.observation_space.shape}/actions {model.action_space.n}, "
            f"env obs ({OBS_SIZE},)/actions {TOTAL_ACTIONS}."
        )
    return {"model": model, "meta": meta, "path": zip_path}


def ppo_action_probabilities(model, obs: np.ndarray) -> np.ndarray:
    """π(a | obs) for all 24 actions from an SB3 PPO model."""
    import torch

    obs_t, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        return model.policy.get_distribution(obs_t).distribution.probs.cpu().numpy()[0]


class PPOPolicy(Policy):
    name = "ppo"

    def __init__(self, path: str, deterministic: bool = True, seed: int = 0):
        info = check_ppo_checkpoint(path)
        self.model = info["model"]
        self.meta = info["meta"]
        self.path = info["path"]
        self.deterministic = deterministic
        self.rng = np.random.default_rng(seed)

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        return ppo_action_probabilities(self.model, obs)

    def rank(self, env):
        probs = self.action_probabilities(env._get_observation())
        if self.deterministic:
            return list(np.argsort(-probs, kind="stable"))
        # Sample an ordering without replacement (Plackett–Luce from the policy's probabilities)
        p = probs.astype(np.float64) + 1e-12
        keys = self.rng.random(len(p)) ** (1.0 / p)
        return list(np.argsort(-keys))

    def describe(self):
        return {"agent": self.name, "checkpoint": self.path, "deterministic": self.deterministic, **{f"meta_{k}": v for k, v in self.meta.items() if not isinstance(v, (dict, list))}}


def make_policy(agent: str, seed: int = 0, ppo_path: Optional[str] = None, deterministic: bool = True) -> Policy:
    if agent == "random":
        return RandomPolicy(seed)
    if agent == "rule_based":
        return RuleBasedPolicy()
    if agent == "ppo":
        if not ppo_path:
            raise PPOCheckpointError("No PPO checkpoint path configured")
        return PPOPolicy(ppo_path, deterministic=deterministic, seed=seed)
    raise ValueError(f"Unknown agent '{agent}'")
