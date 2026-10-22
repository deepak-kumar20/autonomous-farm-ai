import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd  # noqa: E402

from farm_ai.research.runner import run_all  # noqa: E402
from farm_ai.research.outputs import build_all, build_tables  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMALL = {
    "n_scenarios": 2, "seed_offset": 0, "agents": ["random", "rule_based"], "ppo_checkpoint": None,
    "configs": {
        "full": {"decision_order": "LFM", "filter_mode": "deterministic"},
        "no_filter": {"decision_order": "LFM", "filter_mode": "none"},
        "order_fml": {"decision_order": "FML", "filter_mode": "deterministic"},
        "uncertainty_fixed": {"filter_mode": "uncertainty", "completion_threshold": 0.8},
        "uncertainty_break_even": {"filter_mode": "uncertainty", "threshold_rule": "break_even"},
        "price_impact": {"market_model": "price_impact"},
    },
    "experiments": {
        "A_agent_comparison": {"profiles": ["baseline"], "configs": ["full"]},
        "B_feasibility_ablation": {"profiles": ["baseline"], "configs": ["full", "no_filter"]},
        "C_decision_order": {"profiles": ["baseline"], "configs": ["full", "order_fml"]},
        "D_environmental_robustness": {"profiles": ["baseline", "pest"], "configs": ["full"]},
        "F_feasibility_filters": {"profiles": ["baseline"], "configs": ["no_filter", "full", "uncertainty_fixed", "uncertainty_break_even"]},
        "M_market_model": {"profiles": ["baseline"], "configs": ["full", "price_impact"]},
    },
}


class TestResearchOutputs(unittest.TestCase):
    def test_pipeline_small_run(self):
        with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as data:
            os.environ["FARM_DATA_DIR"] = data
            try:
                cfg_path = os.path.join(out, "cfg.json")
                with open(cfg_path, "w") as f:
                    json.dump(SMALL, f)
                status = run_all(cfg_path, out, ROOT, log=lambda *_: None)
                self.assertEqual(status["episodes_failed"], 0)
                build_all(out, cfg_path, ROOT)
            finally:
                os.environ.pop("FARM_DATA_DIR", None)
            for i in range(1, 11):
                names = [f for f in os.listdir(os.path.join(out, "tables")) if f.startswith(f"table_{i}_") and f.endswith(".csv")]
                self.assertEqual(len(names), 1, f"table {i} missing")
                self.assertLessEqual(pd.read_csv(os.path.join(out, "tables", names[0])).shape[1], 10)
            for c in ("chart_1_wealth_over_30_days", "chart_2_agent_profit_comparison", "chart_4_feasibility_ablation"):
                self.assertTrue(os.path.isfile(os.path.join(out, "charts", c + ".png")))
            self.assertFalse(os.path.isfile(os.path.join(out, "charts", "chart_6_human_vs_ai.png")))
            # summary metrics agree with raw daily records
            s = pd.read_csv(os.path.join(out, "data", "episode_summary.csv"))
            d = pd.read_csv(os.path.join(out, "data", "daily_episode_records.csv"))
            agg = d.groupby("run_id").agg(rev=("revenue", "sum"), cost=("planting_cost", "sum"), h=("harvest_qty", "sum"), n=("day", "size"))
            m = s.set_index("run_id").join(agg)
            self.assertTrue(((m["total_revenue"] - m["rev"]).abs() < 0.01).all())
            self.assertTrue(((m["total_costs"] - m["cost"]).abs() < 0.01).all())
            self.assertTrue((m["harvest_qty"] == m["h"]).all())
            self.assertTrue((m["n"] == 30).all())
            t8 = pd.read_csv(os.path.join(out, "tables", "table_8_human_vs_ai.csv"))
            self.assertEqual(t8.iloc[0]["Player type"], "Not run")

    def test_empty_data_reports_not_run(self):
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(os.path.join(out, "data"))
            build_tables(out, SMALL, {})
            t5 = pd.read_csv(os.path.join(out, "tables", "table_5_main_agent_comparison.csv"))
            self.assertEqual(t5.iloc[0]["Status"], "Not run")


if __name__ == "__main__":
    unittest.main()
