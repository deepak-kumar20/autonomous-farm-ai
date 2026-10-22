import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.framework.scenario import generate_scenario, Scenario, scenario_from_id, PROFILES  # noqa: E402


class TestScenario(unittest.TestCase):
    def test_scenario_reproducible(self):
        a, b = generate_scenario("baseline", 123), generate_scenario("baseline", 123)
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertNotEqual(a.exogenous_signature(), generate_scenario("baseline", 124).exogenous_signature())
        self.assertEqual(scenario_from_id(a.scenario_id).to_dict(), a.to_dict())

    def test_saved_scenario_replays_identically(self):
        sc = generate_scenario("combined", 7)
        with tempfile.TemporaryDirectory() as d:
            loaded = Scenario.load(sc.save(d))
        self.assertEqual(loaded.exogenous_signature(), sc.exogenous_signature())
        self.assertEqual(loaded.to_dict(), sc.to_dict())

    def test_risk_distinct_from_event(self):
        # Events are realised draws: there must be risky days without events and (over many seeds)
        # event rates close to the mean risk, never a deterministic function of the risk label.
        events = risk_sum = days_with_risk_no_event = 0
        for seed in range(200):
            sc = generate_scenario("pest", seed)
            events += sum(sc.pest_event)
            risk_sum += sum(sc.pest_risk)
            days_with_risk_no_event += sum(1 for r, e in zip(sc.pest_risk, sc.pest_event) if r > 0.2 and not e)
        self.assertGreater(days_with_risk_no_event, 0)
        self.assertAlmostEqual(events / risk_sum, 1.0, delta=0.08)

    def test_profiles_share_common_random_numbers(self):
        base, pest = generate_scenario("baseline", 5), generate_scenario("pest", 5)
        self.assertEqual(base.reference_prices, pest.reference_prices)
        self.assertEqual(base.drought_event, pest.drought_event)
        self.assertEqual(base.pest_target_u, pest.pest_target_u)
        # A higher pest rate can only add outbreak days given the same uniforms
        self.assertTrue(all(p or not b for b, p in zip(base.pest_event, pest.pest_event)))
        self.assertEqual(set(PROFILES), {"baseline", "dry", "pest", "combined"})


if __name__ == "__main__":
    unittest.main()
