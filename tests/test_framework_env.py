import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.environment.farm_environment import FarmEnvironment, PLANT_OFFSET, HARVEST_OFFSET  # noqa: E402
from farm_ai.framework.config import SimulationConfig  # noqa: E402
from farm_ai.framework.scenario import generate_scenario  # noqa: E402

WHEAT_F0 = PLANT_OFFSET + 0 * 5 + 0   # plant Wheat in field 0


def calm(seed=0):
    """A scenario with no drought, pest or shock events (prices still drift)."""
    sc = generate_scenario("baseline", seed)
    sc.drought_event = [False] * 30
    sc.pest_event = [False] * 30
    return sc


def env_for(sc, order="LFM"):
    env = FarmEnvironment(config=SimulationConfig(decision_order=order, filter_mode="none"))
    env.reset(options={"scenario": sc})
    return env


class TestEnvironment(unittest.TestCase):
    def test_thirty_transitions_then_terminal(self):
        env = env_for(calm())
        steps, done = 0, False
        while not done:
            _, _, done, _, info = env.step(0)
            steps += 1
        self.assertEqual(steps, 30)
        self.assertEqual(len(env.transition_log), 30)
        self.assertEqual([r["day"] for r in env.transition_log], list(range(1, 31)))
        self.assertIn("profit", info)

    def test_step_after_season_raises(self):
        env = env_for(calm())
        for _ in range(30):
            env.step(0)
        cash = env.cash
        with self.assertRaises(ValueError):
            env.step(WHEAT_F0)
        self.assertEqual(env.cash, cash)

    def test_crop_maturity_day_convention(self):
        # Wheat (5 days) planted on day 1 is harvest-eligible on day 6 (= 1 + 5), not before.
        env = env_for(calm())
        env.step(WHEAT_F0)
        for day in range(2, 6):
            self.assertEqual(env.day, day)
            self.assertFalse(env.check_structural(HARVEST_OFFSET + 0)[0], f"harvestable too early on day {day}")
            env.step(0)
        self.assertEqual(env.day, 6)
        self.assertTrue(env.check_structural(HARVEST_OFFSET + 0)[0])
        env.step(HARVEST_OFFSET + 0)
        self.assertEqual(env.inventory["Wheat"], 100)
        self.assertEqual(env.crop_ledger[1]["harvest_eligible_day"], 6)

    def test_invalid_action_changes_nothing_but_day(self):
        env = env_for(calm())
        before = (env.cash, dict(env.inventory), [f.state for f in env.fields])
        _, reward, _, _, info = env.step(HARVEST_OFFSET + 2)   # harvest an empty field
        self.assertEqual(reward, -50.0)
        self.assertFalse(info["day_record"]["action_valid"])
        self.assertEqual((env.cash, dict(env.inventory), [f.state for f in env.fields]), before)
        self.assertEqual(env.day, 2)

    def test_losses_only_on_event_days(self):
        sc = calm()
        sc.pest_risk = [0.9] * 30          # high risk but no outbreak events
        sc.drought_risk = [0.9] * 30
        env = env_for(sc)
        env.step(WHEAT_F0)
        for _ in range(3):
            env.step(0)
        self.assertEqual(env.fields[0].growth_progress, 4)   # no set-backs despite 90 % risk
        self.assertEqual(env.fields[0].state.value, "GROWING")
        # Now an actual drought on day 5 and an outbreak on day 6
        sc2 = calm()
        sc2.drought_event[4] = True
        sc2.pest_event[5] = True
        env2 = env_for(sc2)
        env2.step(WHEAT_F0)
        for _ in range(3):
            env2.step(0)
        # day 5 begins: drought resolved first (LFM) → progress 4 → 3
        self.assertEqual(env2.fields[0].growth_progress, 3)
        env2.step(0)
        self.assertEqual(env2.fields[0].state.value, "EMPTY")      # pest destroyed it at the start of day 6
        self.assertEqual(env2.crop_ledger[1]["outcome"], "destroyed")

    def test_lfm_vs_fml_order(self):
        # Pest outbreak on day 1. LFM resolves it before planting (nothing to hit); FML after planting.
        sc = calm()
        sc.pest_event[0] = True
        lfm, fml = env_for(sc, "LFM"), env_for(sc, "FML")
        lfm.step(WHEAT_F0)
        fml.step(WHEAT_F0)
        self.assertEqual(lfm.fields[0].state.value, "GROWING")
        self.assertEqual(fml.fields[0].state.value, "EMPTY")

    def test_terminal_valuation_uses_day30_price(self):
        env = env_for(calm())
        env.step(WHEAT_F0)
        for _ in range(4):
            env.step(0)
        env.step(HARVEST_OFFSET + 0)
        while env.day < 30:
            env.step(0)
        _, _, done, _, info = env.step(0)
        self.assertTrue(done)
        self.assertAlmostEqual(info["inventory_value"], 100 * env.scenario.price("Wheat", 30), places=6)


if __name__ == "__main__":
    unittest.main()
