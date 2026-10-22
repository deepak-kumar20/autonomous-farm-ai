import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.api.services.farm_service import FarmSessionManager  # noqa: E402
from farm_ai.framework.agents import make_policy, check_ppo_checkpoint, PPOCheckpointError  # noqa: E402
from farm_ai.framework.config import SimulationConfig  # noqa: E402
from farm_ai.framework.episode import run_episode  # noqa: E402
from farm_ai.framework.scenario import generate_scenario  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestEpisode(unittest.TestCase):
    def test_summary_agrees_with_daily_records(self):
        for agent in ("random", "rule_based"):
            for cfg in (SimulationConfig(), SimulationConfig(filter_mode="none", decision_order="FML")):
                records, s, ledger = run_episode(make_policy(agent, seed=4), generate_scenario("combined", 4), cfg)
                self.assertEqual(len(records), 30)
                self.assertIsNone(s["error"])
                self.assertAlmostEqual(s["profit"], s["total_revenue"] - s["total_costs"] + s["final_inventory_value"], places=6)
                self.assertAlmostEqual(s["final_wealth"], records[-1]["wealth_end"], places=6)
                self.assertEqual(s["harvest_qty"], sum(r["harvest_qty"] for r in records))
                self.assertEqual(s["crops_planted"], sum(1 for r in records if r["planted_crop"]))
                self.assertEqual(s["crops_planted"], len(ledger))
                self.assertEqual(s["crops_destroyed"], sum(r["crops_destroyed"] for r in records))
                self.assertEqual(s["invalid_actions"], sum(1 for r in records if not r["action_valid"]))
                # every day starts where the previous ended
                for a, b in zip(records, records[1:]):
                    self.assertAlmostEqual(a["cash_end"], b["cash_start"], places=6)

    def test_human_and_ai_share_validation(self):
        """Replaying an AI episode's actions through the human API path reproduces it exactly."""
        sc = generate_scenario("baseline", 9)
        cfg = SimulationConfig()
        records, summary, _ = run_episode(make_policy("random", seed=9), sc, cfg)
        mgr = FarmSessionManager()
        sid, _, _ = mgr.create_session(scenario=generate_scenario("baseline", 9), config=cfg)
        for r in records:
            accepted, *_ = mgr.execute_action(sid, r["action_id"])
            self.assertTrue(accepted)
        env = mgr.get_env(sid)
        self.assertTrue(env.season_over)
        self.assertAlmostEqual(env.cash, summary["final_cash"], places=6)
        self.assertEqual([r["wealth_end"] for r in env.transition_log], [r["wealth_end"] for r in records])

    def test_rejected_human_action_does_not_advance(self):
        mgr = FarmSessionManager()
        sid, _, _ = mgr.create_session(seed=1)
        accepted, feas, state, *_ = mgr.execute_action(sid, 1)     # harvest empty field 0
        self.assertFalse(accepted)
        self.assertEqual(feas["code"], "FIELD_EMPTY")
        self.assertEqual(state["day"], 1)

    def test_legacy_checkpoint_rejected(self):
        with self.assertRaises(PPOCheckpointError) as ctx:
            check_ppo_checkpoint(os.path.join(ROOT, "models/ppo/farm_ppo_500k"))
        self.assertIn("version", str(ctx.exception))
        with self.assertRaises(PPOCheckpointError):
            check_ppo_checkpoint(os.path.join(ROOT, "models/ppo/does_not_exist"))

    @unittest.skipUnless(os.path.isfile(os.path.join(ROOT, "models/ppo/farm_ppo_v2.zip")), "v2 checkpoint not trained")
    def test_v2_checkpoint_plays_valid_season(self):
        info = check_ppo_checkpoint(os.path.join(ROOT, "models/ppo/farm_ppo_v2"))
        self.assertEqual(info["meta"]["env_version"], 2)
        _, s, _ = run_episode(make_policy("ppo", ppo_path=os.path.join(ROOT, "models/ppo/farm_ppo_v2")),
                              generate_scenario("baseline", 0), SimulationConfig())
        self.assertIsNone(s["error"])
        self.assertEqual(s["invalid_actions"], 0)       # the filter guarantees validity


if __name__ == "__main__":
    unittest.main()
