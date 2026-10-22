"""Entry-point script for training the PPO agent (environment v2).

Run from project root:
    python scripts/train_ppo.py                         # 1,000,000 steps, seed 7 → models/ppo/farm_ppo_v2.zip
    python scripts/train_ppo.py --timesteps 200000 --out models/ppo/test_run
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.training.train_ppo import main  # noqa: E402

if __name__ == "__main__":
    main()
