"""Runs all research experiments and regenerates tables, charts and documents.

Usage (from the project root, inside the virtual environment):
    python scripts/run_experiments.py                 # full run (n_scenarios from research_outputs/experiment_config.json)
    python scripts/run_experiments.py --n 5           # quick smoke test with 5 scenarios per profile
    python scripts/run_experiments.py --outputs-only  # rebuild tables/charts/docs from existing CSVs

Outputs go to research_outputs/ (data/, tables/, charts/, logs/, paper_assets/).
Nothing here runs when the app starts.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from farm_ai.research.runner import run_all  # noqa: E402
from farm_ai.research.outputs import build_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "research_outputs", "experiment_config.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "research_outputs"))
    ap.add_argument("--n", type=int, default=None, help="override n_scenarios per profile")
    ap.add_argument("--outputs-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "logs"), exist_ok=True)
    if not a.outputs_only:
        status = run_all(a.config, a.out, ROOT, n_override=a.n)
        print(f"Episodes: {status['episodes_successful']} ok, {status['episodes_failed']} failed, "
              f"skipped agents: {status['skipped_agents'] or 'none'} ({status['runtime_s']}s)")
    build_all(a.out, a.config, ROOT)
    print(f"Research outputs written to {a.out}")


if __name__ == "__main__":
    main()
