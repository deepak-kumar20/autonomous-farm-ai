import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.environment.farm_environment import FarmEnvironment, PLANT_OFFSET, HARVEST_OFFSET, SELL_OFFSET  # noqa: E402
from farm_ai.environment.actions import Action, ActionType  # noqa: E402
from farm_ai.framework.config import SimulationConfig  # noqa: E402
from farm_ai.framework.scenario import generate_scenario  # noqa: E402
from farm_ai.models.market import Market  # noqa: E402
from farm_ai.framework.config import PriceImpactParams  # noqa: E402


def calm_env(market="exogenous"):
    sc = generate_scenario("baseline", 2)
    sc.drought_event = [False] * 30
    sc.pest_event = [False] * 30
    env = FarmEnvironment(config=SimulationConfig(filter_mode="none", market_model=market))
    env.reset(options={"scenario": sc})
    return env


def grow_and_harvest(env):
    env.step(PLANT_OFFSET)
    for _ in range(4):
        env.step(0)
    env.step(HARVEST_OFFSET)


class TestMarket(unittest.TestCase):
    def test_sell_cannot_go_negative(self):
        env = calm_env()
        grow_and_harvest(env)
        _, r, _, _, info = env.step(Action(ActionType.SELL, crop_name="Wheat", quantity=500))
        self.assertEqual(r, -50.0)
        self.assertEqual(env.inventory["Wheat"], 100)
        env.step(SELL_OFFSET)
        self.assertEqual(env.inventory["Wheat"], 0)
        cash = env.cash
        _, r, _, _, _ = env.step(SELL_OFFSET)
        self.assertEqual((r, env.cash, env.inventory["Wheat"]), (-50.0, cash, 0))

    def test_exogenous_prices_identical_across_players(self):
        a, b = calm_env(), calm_env()
        grow_and_harvest(a)
        a.step(SELL_OFFSET)
        for _ in range(6):
            b.step(0)
        b.step(0)
        self.assertEqual(a.day, b.day)
        self.assertEqual(a.market.current_prices, b.market.current_prices)

    def test_price_impact_depends_on_actions(self):
        seller, holder = calm_env("price_impact"), calm_env("price_impact")
        grow_and_harvest(seller)
        for _ in range(6):
            holder.step(0)
        ref = seller.market.reference_price("Wheat")
        seller.step(SELL_OFFSET)
        holder.step(0)
        self.assertLess(seller.market.get_current_price("Wheat"), holder.market.get_current_price("Wheat"))
        rec = seller.transition_log[-1]
        self.assertLess(rec["revenue"], 100 * ref)
        self.assertGreater(rec["revenue"], 0)

    def test_price_impact_bounds(self):
        m = Market(generate_scenario("baseline", 0), "price_impact", PriceImpactParams())
        r = m.reference_price("Rice")
        self.assertAlmostEqual(m.get_current_price("Rice"), r)            # zero supply → reference price
        m.supply_pressure["Rice"] = 1e9
        self.assertAlmostEqual(m.get_current_price("Rice"), r * PriceImpactParams().min_factor)
        self.assertGreater(m.sale_revenue("Rice", 10), 0)


if __name__ == "__main__":
    unittest.main()
