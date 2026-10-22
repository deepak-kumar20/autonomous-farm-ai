import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.environment.farm_environment import FarmEnvironment, PLANT_OFFSET, HARVEST_OFFSET, SELL_OFFSET  # noqa: E402
from farm_ai.framework.config import SimulationConfig  # noqa: E402
from farm_ai.framework.feasibility import FeasibilityFilter, completion_probability, decide  # noqa: E402
from farm_ai.framework.scenario import generate_scenario  # noqa: E402

RICE_F1 = PLANT_OFFSET + 1 * 5 + 1


def make_env(mode="deterministic"):
    env = FarmEnvironment(config=SimulationConfig(filter_mode=mode))
    env.reset(options={"scenario": generate_scenario("baseline", 3)})
    return env


class TestFeasibility(unittest.TestCase):
    def test_rejection_reasons(self):
        env = make_env()
        f = FeasibilityFilter("deterministic")
        cases = {HARVEST_OFFSET + 0: "FIELD_EMPTY", SELL_OFFSET: "NO_INVENTORY"}
        for a, code in cases.items():
            r = f.check(env, a)
            self.assertFalse(r.accepted)
            self.assertEqual(r.code, code)
            self.assertTrue(r.reason)
        env.cash = 500
        r = f.check(env, RICE_F1)
        self.assertEqual(r.code, "INSUFFICIENT_CASH")
        self.assertEqual(r.constraints["planting_cost"], 1500)
        self.assertTrue(f.check(env, 0).accepted)

    def test_horizon_constraint(self):
        env = make_env()
        f = FeasibilityFilter("deterministic")
        while env.day < 22:
            env.step(0)
        self.assertTrue(f.check(env, RICE_F1).accepted)           # 22 + 8 = 30 ≤ 30
        env.step(0)
        r = f.check(env, RICE_F1)                                  # 23 + 8 = 31 > 30
        self.assertFalse(r.accepted)
        self.assertEqual(r.code, "HORIZON")
        self.assertIsNone(r.completion_probability)                # no probability in deterministic mode
        self.assertTrue(f.check(env, PLANT_OFFSET + 0).accepted)   # Wheat: 23 + 5 = 28

    def test_rejection_does_not_change_state(self):
        env = make_env()
        snap = (env.day, env.cash, dict(env.inventory), [f.state for f in env.fields])
        a = decide(env, [HARVEST_OFFSET + 0, SELL_OFFSET, 0], FeasibilityFilter("deterministic"))
        self.assertEqual(a, 0)
        self.assertEqual(snap, (env.day, env.cash, dict(env.inventory), [f.state for f in env.fields]))
        self.assertEqual(sum(1 for c in env._candidates if not c.accepted), 2)
        env.step(a)
        self.assertEqual(env.transition_log[-1]["feasibility_rejections"], 2)

    def test_completion_probability_matches_monte_carlo(self):
        rng = np.random.default_rng(0)
        for order in ("LFM", "FML"):
            for crop, plant_day in (("Rice", 18), ("Wheat", 20)):
                sc = generate_scenario("baseline", 1)
                sc.drought_risk = [0.3] * 30
                sc.pest_risk = [0.06] * 30
                g = {"Wheat": 5, "Rice": 8}[crop]
                p = completion_probability(g, plant_day, sc.drought_risk, sc.pest_risk, 1, order)
                hits, trials = 0, 3000
                for _ in range(trials):
                    sc.drought_event = list(rng.random(30) < 0.3)
                    sc.pest_event = list(rng.random(30) < 0.06)
                    env = FarmEnvironment(config=SimulationConfig(decision_order=order, filter_mode="none"))
                    env.reset(options={"scenario": sc})
                    while env.day < plant_day:
                        env.step(0)
                    env.step(PLANT_OFFSET + ["Wheat", "Rice", "Corn"].index(crop) * 5)
                    while not env.season_over:
                        env.step(0)
                    hits += env.crop_ledger[1]["harvest_eligible_day"] is not None
                mc = hits / trials
                se = (p * (1 - p) / trials) ** 0.5
                self.assertLess(abs(mc - p), 4 * se + 1e-3, f"{order} {crop}: DP {p:.3f} vs MC {mc:.3f}")

    def test_uncertainty_threshold(self):
        env = make_env("uncertainty")
        strict, lax = FeasibilityFilter("uncertainty", 0.99), FeasibilityFilter("uncertainty", 0.0)
        r = strict.check(env, PLANT_OFFSET)
        self.assertIsNotNone(r.completion_probability)
        self.assertEqual(r.threshold, 0.99)
        self.assertTrue(lax.check(env, PLANT_OFFSET).accepted)
        be = FeasibilityFilter("uncertainty", threshold_rule="break_even")
        rb = be.check(env, PLANT_OFFSET)
        self.assertAlmostEqual(rb.threshold, 1000 / (100 * env.market.get_current_price("Wheat")))


if __name__ == "__main__":
    unittest.main()
