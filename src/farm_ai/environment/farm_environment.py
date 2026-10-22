import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np

from ..config.settings import STARTING_CASH, NUM_FIELDS, CROP_CONFIG, SEASON_LENGTH
from ..models.field import Field, FieldState
from ..models.crop import Crop
from ..models.market import Market
from .state import FarmState
from .actions import Action, ActionType
from .rewards import RewardCalculator
from ..simulation.season import SeasonManager
from ..simulation.weather import WeatherSystem
from ..simulation.pests import PestSystem
from ..simulation.price_shocks import PriceShockSystem
from ..framework.config import SimulationConfig
from ..framework.scenario import Scenario, generate_scenario, risk_label, PROFILES
from ..framework.feasibility import completion_probability

# ─────────────────────────────────────────────────────────────
# Flat discrete action index layout (total = 24)
#
#  0        → HOLD
#  1–5      → HARVEST field 0,1,2,3,4
#  6–20     → PLANT crop 0,1,2  ×  field 0,1,2,3,4
#             index = 6 + crop_idx*5 + field_idx
#  21–23    → SELL all of crop 0,1,2
# ─────────────────────────────────────────────────────────────
CROPS: List[str] = list(CROP_CONFIG.keys())   # ["Wheat","Rice","Corn"]
NUM_CROPS: int = len(CROPS)
HARVEST_OFFSET = 1
PLANT_OFFSET = HARVEST_OFFSET + NUM_FIELDS        # 6
SELL_OFFSET = PLANT_OFFSET + NUM_CROPS * NUM_FIELDS  # 21
TOTAL_ACTIONS = SELL_OFFSET + NUM_CROPS              # 24

# Environment semantics version. Version 2 = scenario-driven, six-stage daily process.
ENV_VERSION = 2

# ─────────────────────────────────────────────────────────────
# Observation layout (env v2; all floats in [0, 5])
#
#  0       day / SEASON_LENGTH
#  1       cash / STARTING_CASH                         (capped at 5)
#  2       debt / STARTING_CASH                         (always 0 — debt is not modelled)
#  3–7     field state  (EMPTY=0, GROWING=0.5, MATURE=1)
#  8–12    field growth progress / max_growth_days
#  13–17   field crop index (1..3) / NUM_CROPS          (0 = no crop)
#  18–20   inventory[Wheat,Rice,Corn] / 300             (capped at 1)
#  21–23   current price / base price                   (capped at 3)
#  24–26   risk (probability) of the next unresolved loss stage for drought, pest, price shock,
#          divided by 0.3 and capped at 3
# total = 27 features
# ─────────────────────────────────────────────────────────────
OBS_SIZE = 27
OBS_HIGH = 5.0
MAX_INVENTORY = 300.0
RISK_SCALE = 0.3
FORECAST_MAP = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}
FIELD_STATE_MAP = {FieldState.EMPTY: 0.0, FieldState.GROWING: 0.5, FieldState.MATURE: 1.0}


def _int_action_to_farm_action(action_idx: int) -> Action:
    """Converts a flat integer action index to a farm Action object."""
    if not 0 <= action_idx < TOTAL_ACTIONS:
        raise ValueError(f"Action index {action_idx} outside 0..{TOTAL_ACTIONS - 1}")
    if action_idx == 0:
        return Action(ActionType.HOLD)
    if action_idx < PLANT_OFFSET:
        return Action(ActionType.HARVEST, field_id=action_idx - HARVEST_OFFSET)
    if action_idx < SELL_OFFSET:
        idx = action_idx - PLANT_OFFSET
        return Action(ActionType.PLANT, field_id=idx % NUM_FIELDS, crop_name=CROPS[idx // NUM_FIELDS])
    return Action(ActionType.SELL, crop_name=CROPS[action_idx - SELL_OFFSET], quantity=None)


def farm_action_to_int(action: Action) -> Optional[int]:
    """Inverse of `_int_action_to_farm_action` (SELL maps to 'sell all' regardless of quantity)."""
    if action.type == ActionType.HOLD:
        return 0
    if action.type == ActionType.HARVEST and action.field_id is not None:
        return HARVEST_OFFSET + action.field_id
    if action.type == ActionType.PLANT and action.field_id is not None and action.crop_name in CROPS:
        return PLANT_OFFSET + CROPS.index(action.crop_name) * NUM_FIELDS + action.field_id
    if action.type == ActionType.SELL and action.crop_name in CROPS:
        return SELL_OFFSET + CROPS.index(action.crop_name)
    return None


class FarmEnvironment(gym.Env):
    """Scenario-driven 30-day farm simulation (Gymnasium + Stable-Baselines3 compatible).

    One `step` = one simulated day. The stages of a day (see research_outputs/implementation_mapping.md):

      Stage 1  resolve exogenous losses for the day (drought set-back, pest outbreak) from the scenario
      Stage 2  the decision-maker proposes a candidate action          (outside the env)
      Stage 3  the feasibility filter accepts / rejects candidates     (framework.feasibility, outside)
      Stage 4  execute an accepted field action (PLANT / HARVEST)
      Stage 5  execute an accepted market action (SELL)
      Stage 6  grow crops, move the market clock, compute reward, log the transition, advance the day

    With ``decision_order="LFM"`` Stage 1 runs at the start of the day, before the decision.
    With ``decision_order="FML"`` (the original simulator) losses are resolved after growth, at the
    end of the day. Exactly one action is taken per day, so Stages 4 and 5 are mutually exclusive.

    Action space: Discrete(24). Observation: Box(27,) in [0, 5].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        scenario: Optional[Scenario] = None,
        default_profile: str = "baseline",
        profile_mix: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=OBS_HIGH, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(TOTAL_ACTIONS)

        self.config = config or SimulationConfig()
        self._fixed_scenario = scenario
        self.default_profile = default_profile
        self.profile_mix = list(profile_mix) if profile_mix else None
        for p in [default_profile] + (self.profile_mix or []):
            if p not in PROFILES:
                raise ValueError(f"Unknown profile {p}")

        self.scenario: Optional[Scenario] = None
        self.cash: float = STARTING_CASH
        self.debt: float = 0.0
        self.fields: List[Field] = []
        self.inventory: Dict[str, int] = {}
        self.market: Optional[Market] = None
        self.season_manager: Optional[SeasonManager] = None
        self.weather_system: Optional[WeatherSystem] = None
        self.pest_system: Optional[PestSystem] = None
        self.price_shock_system: Optional[PriceShockSystem] = None
        self.transition_log: List[Dict[str, Any]] = []
        self.crop_ledger: Dict[int, Dict[str, Any]] = {}
        self.last_events: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Starts a new season. Scenario priority: options['scenario'] > fixed scenario > generated.

        A generated scenario uses `seed` when given; otherwise a seed is drawn from the env's
        own seeded RNG in the range [1e6, 2^31), which keeps training scenarios disjoint from the
        small evaluation seeds used in experiments.
        """
        super().reset(seed=seed)
        options = options or {}
        scenario = options.get("scenario") or self._fixed_scenario
        if scenario is None:
            profile = options.get("profile")
            if profile is None:
                if self.profile_mix:
                    profile = self.profile_mix[int(self.np_random.integers(len(self.profile_mix)))]
                else:
                    profile = self.default_profile
            s = seed if seed is not None else int(self.np_random.integers(1_000_000, 2**31 - 1))
            scenario = generate_scenario(profile, s)
        self.scenario = scenario

        self.cash = STARTING_CASH
        self.debt = 0.0
        self.fields = [Field(i) for i in range(NUM_FIELDS)]
        self.inventory = {crop: 0 for crop in CROP_CONFIG.keys()}
        self.market = Market(scenario, self.config.market_model, self.config.price_impact)
        self.season_manager = SeasonManager()
        self.weather_system = WeatherSystem(scenario)
        self.pest_system = PestSystem(scenario)
        self.price_shock_system = PriceShockSystem(scenario)
        self.transition_log = []
        self.crop_ledger = {}
        self._next_crop_uid = 1
        self._candidates: List[Any] = []
        self._day_losses: List[Dict[str, Any]] = []
        self.last_events = {}
        self.initial_wealth = self._total_wealth()

        self._begin_day()
        return self._get_observation(), {"scenario_id": scenario.scenario_id, "env_version": ENV_VERSION}

    def step(self, action: Union[int, Action]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Executes one simulated day (Stages 4–6, plus Stage 1 for the next day under LFM).

        Accepts an int index (PPO / API / framework agents) or an `Action` (legacy baseline agents).
        Raises ValueError once the season is over.
        """
        if self.season_over:
            raise ValueError("Season is over: no actions can be taken after day 30.")
        t0 = time.perf_counter()
        day = self.day

        if isinstance(action, (int, np.integer)):
            action_id = int(action)
            farm_action = _int_action_to_farm_action(action_id)
        else:
            farm_action = action
            action_id = farm_action_to_int(farm_action)
        if farm_action.type == ActionType.SELL and farm_action.quantity is None:
            farm_action = Action(ActionType.SELL, crop_name=farm_action.crop_name,
                                 quantity=self.inventory.get(farm_action.crop_name, 0))

        cash_start = self.cash
        wealth_start = self._total_wealth()
        prices_start = dict(self.market.current_prices)
        record: Dict[str, Any] = {
            "day": day,
            "action_id": action_id,
            "action_type": farm_action.type.value,
            "action_label": str(farm_action),
            "field_id": farm_action.field_id,
            "crop": farm_action.crop_name,
            "planted_crop": None,
            "planting_cost": 0.0,
            "harvest_crop": None,
            "harvest_qty": 0,
            "sold_crop": None,
            "sold_qty": 0,
            "revenue": 0.0,
        }

        # Stages 4 / 5 — execute (structural constraints are always enforced here)
        ok, code, reason, _ = self.check_structural_action(farm_action)
        valid_action = ok
        if ok:
            self._execute_action(farm_action, record)
        record["action_valid"] = valid_action
        record["invalid_code"] = None if ok else code
        record["invalid_reason"] = None if ok else reason

        # Stage 6 (+ Stage 1 position depends on order)
        for field in self.fields:
            field.advance_day()
        if self.config.decision_order == "FML":
            self._resolve_losses(day)
        losses = list(self._day_losses)

        self.season_manager.advance_day()
        if not self.season_over:
            self.market.advance_day(self.day)

        curr_wealth = self._total_wealth()
        reward = RewardCalculator.calculate_shaped_reward(
            prev_wealth=wealth_start, curr_wealth=curr_wealth, valid_action=valid_action
        )

        risks = self.scenario.risks(day)
        record.update({
            "candidates_evaluated": len(self._candidates),
            "feasibility_rejections": sum(1 for c in self._candidates if not c.accepted),
            "rejection_codes": ";".join(c.code for c in self._candidates if not c.accepted),
            "drought_risk": risks["drought"],
            "pest_risk": risks["pest"],
            "price_shock_risk": risks["price_shock"],
            "drought_event": self.scenario.drought(day),
            "pest_event": self.scenario.pest(day),
            "price_shock_event": self.scenario.shock_event[day - 1],
            "price_shock_factor": self.scenario.shock_factor[day - 1],
            "pest_destroyed_field": next((l["field_id"] for l in losses if l["type"] == "pest"), None),
            "crops_destroyed": sum(1 for l in losses if l["type"] == "pest"),
            "destroyed_planting_cost": sum(l["planting_cost"] for l in losses if l["type"] == "pest"),
            "drought_setbacks": sum(1 for l in losses if l["type"] == "drought"),
            "cash_start": cash_start,
            "cash_end": self.cash,
            "wealth_start": wealth_start,
            "wealth_end": curr_wealth,
            "inventory_value_end": self._inventory_value(),
            "reward": float(reward),
            "decision_order": self.config.decision_order,
            "filter_mode": self.config.filter_mode,
            "market_model": self.config.market_model,
            "step_ms": (time.perf_counter() - t0) * 1000.0,
        })
        for c in CROPS:
            record[f"price_{c.lower()}"] = prices_start[c]
            record[f"inventory_{c.lower()}"] = self.inventory[c]
        self.transition_log.append(record)

        terminated = self.season_over
        info: Dict[str, Any] = {"day_record": record}
        if terminated:
            self._finalise_ledger()
            inventory_value = self._inventory_value()
            info.update(self.season_manager.evaluate_season(self.cash, inventory_value, STARTING_CASH))
        else:
            self._begin_day()

        return self._get_observation(), float(reward), terminated, False, info

    # ──────────────────────────────────────────────────────────────────
    # Day lifecycle helpers
    # ──────────────────────────────────────────────────────────────────

    @property
    def day(self) -> int:
        return self.season_manager.current_day

    @property
    def season_over(self) -> bool:
        return self.season_manager is None or self.season_manager.is_over

    def _begin_day(self) -> None:
        """Opens the current day: Stage 1 under LFM, then the decision point."""
        self._candidates = []
        self._day_losses = []
        if self.config.decision_order == "LFM":
            self._resolve_losses(self.day)
        # Decision point: record crops that are present and harvest-eligible
        for f in self.fields:
            if f.state == FieldState.MATURE and f.crop_uid in self.crop_ledger:
                led = self.crop_ledger[f.crop_uid]
                if led["harvest_eligible_day"] is None:
                    led["harvest_eligible_day"] = self.day

    def _resolve_losses(self, day: int) -> None:
        """Stage 1: apply the scenario's realised events for `day` (risk alone never causes damage)."""
        events = {"day": day, "drought": False, "pest_field": None, "price_shock": None}
        if self.scenario.drought(day):
            events["drought"] = True
            for f in self.fields:
                if f.state == FieldState.GROWING and f.growth_progress > 0:
                    f.growth_progress -= 1
                    self.crop_ledger[f.crop_uid]["drought_setbacks"] += 1
                    self._day_losses.append({"type": "drought", "field_id": f.field_id, "crop_uid": f.crop_uid, "planting_cost": 0.0})
        if self.scenario.pest(day):
            planted = [f for f in self.fields if f.is_planted]
            if planted:
                target = planted[min(int(self.scenario.pest_target_u[day - 1] * len(planted)), len(planted) - 1)]
                led = self.crop_ledger[target.crop_uid]
                led["destroyed_day"] = day
                led["outcome"] = "destroyed"
                self._day_losses.append({"type": "pest", "field_id": target.field_id, "crop_uid": target.crop_uid,
                                         "planting_cost": led["planting_cost"]})
                events["pest_field"] = target.field_id
                target._reset()
        if self.scenario.shock_event[day - 1]:
            events["price_shock"] = self.scenario.shock_factor[day - 1]
        self.last_events = events

    def _finalise_ledger(self) -> None:
        for led in self.crop_ledger.values():
            if led["outcome"] is None:
                led["outcome"] = "unharvested_mature" if led["harvest_eligible_day"] else "immature_at_end"

    # ──────────────────────────────────────────────────────────────────
    # Feasibility support (used by framework.feasibility)
    # ──────────────────────────────────────────────────────────────────

    def decode(self, action_id: int) -> Action:
        return _int_action_to_farm_action(int(action_id))

    def record_candidate(self, result) -> None:
        """Stage 3 bookkeeping: remember every candidate checked today."""
        self._candidates.append(result)

    def check_structural(self, action_id: int) -> Tuple[bool, str, str, Dict[str, Any]]:
        return self.check_structural_action(self.decode(action_id))

    def check_structural_action(self, action: Action) -> Tuple[bool, str, str, Dict[str, Any]]:
        """Hard constraints every executed action must satisfy. Returns (ok, code, reason, constraint values)."""
        if self.season_over:
            return False, "SEASON_OVER", "The season has ended.", {}
        if action.type == ActionType.HOLD:
            return True, "OK", "Holding is always allowed.", {}
        if action.type == ActionType.PLANT:
            if action.field_id is None or not 0 <= action.field_id < len(self.fields) or action.crop_name not in CROP_CONFIG:
                return False, "BAD_TARGET", "Unknown field or crop.", {}
            f = self.fields[action.field_id]
            cost = CROP_CONFIG[action.crop_name]["planting_cost"]
            c = {"cash": self.cash, "planting_cost": cost, "field_state": f.state.value}
            if f.state != FieldState.EMPTY:
                return False, "FIELD_OCCUPIED", f"Field {action.field_id + 1} already has {f.crop.name if f.crop else 'a crop'}.", c
            if self.cash < cost:
                return False, "INSUFFICIENT_CASH", f"{action.crop_name} seeds cost ₹{cost:,.0f} but only ₹{self.cash:,.0f} is available.", c
            return True, "OK", "OK", c
        if action.type == ActionType.HARVEST:
            if action.field_id is None or not 0 <= action.field_id < len(self.fields):
                return False, "BAD_TARGET", "Unknown field.", {}
            f = self.fields[action.field_id]
            c = {"field_state": f.state.value, "progress": f.growth_progress,
                 "growth_days": f.crop.growth_days if f.crop else None}
            if f.state == FieldState.EMPTY:
                return False, "FIELD_EMPTY", f"Field {action.field_id + 1} is empty — nothing to harvest.", c
            if f.state != FieldState.MATURE:
                return False, "NOT_MATURE", f"{f.crop.name} in field {action.field_id + 1} is not mature ({f.growth_progress}/{f.crop.growth_days} days).", c
            return True, "OK", "OK", c
        if action.type == ActionType.SELL:
            if action.crop_name not in CROP_CONFIG:
                return False, "BAD_TARGET", "Unknown crop.", {}
            have = self.inventory.get(action.crop_name, 0)
            qty = have if action.quantity is None else action.quantity
            c = {"inventory": have, "quantity": qty}
            if have <= 0 or qty <= 0:
                return False, "NO_INVENTORY", f"No {action.crop_name} in storage to sell.", c
            if qty > have:
                return False, "NO_INVENTORY", f"Only {have} {action.crop_name} in storage.", c
            return True, "OK", "OK", c
        return False, "UNSUPPORTED", f"{action.type.value} is not implemented.", {}

    def plant_completion_probability(self, crop_name: str) -> float:
        """Model probability that a crop planted now is harvest-eligible by day 30 (uses published risks only)."""
        planted_after = sum(1 for f in self.fields if f.is_planted) + 1
        return completion_probability(
            CROP_CONFIG[crop_name]["growth_days"], self.day,
            self.scenario.drought_risk, self.scenario.pest_risk, planted_after,
            order=self.config.decision_order,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public helpers
    # ──────────────────────────────────────────────────────────────────

    def get_dict_state(self) -> Dict[str, Any]:
        state = self._get_farm_state().get_state()
        ahead = self._upcoming_risks()
        state["risks"] = ahead
        state["scenario_id"] = self.scenario.scenario_id if self.scenario else None
        state["events"] = dict(self.last_events)
        return state

    def get_actions(self, filt=None) -> List[Dict[str, Any]]:
        """All 24 actions with structural validity and (optionally) the filter's verdict."""
        actions = []
        for idx in range(TOTAL_ACTIONS):
            farm_action = _int_action_to_farm_action(idx)
            ok, code, reason, _ = self.check_structural_action(farm_action)
            act: Dict[str, Any] = {
                "id": idx,
                "type": farm_action.type.value,
                "label": str(farm_action),
                "valid": ok,
                "structurally_valid": ok,
                "reason": None if ok else reason,
                "code": code,
                "completion_probability": None,
            }
            if filt is not None and filt.mode != "none":
                res = filt.check(self, idx)
                act["valid"] = res.accepted
                act["code"] = res.code
                act["reason"] = None if res.accepted else res.reason
                act["completion_probability"] = res.completion_probability
            if farm_action.field_id is not None:
                act["field_id"] = farm_action.field_id
            if farm_action.crop_name is not None:
                act["crop_name"] = farm_action.crop_name
            actions.append(act)
        return actions

    def get_metrics(self) -> Dict[str, Any]:
        inv_val = self._inventory_value()
        wealth = self._total_wealth()
        return {
            "current_day": int(self.season_manager.current_day),
            "cash": float(self.cash),
            "debt": float(self.debt),
            "inventory_value": float(inv_val),
            "total_wealth": float(wealth),
            "profit": float(wealth - STARTING_CASH),
            "survived": bool(self.cash > 0 or wealth > 0),
            "season_over": bool(self.season_manager.is_over),
            "fields_planted": sum(1 for f in self.fields if f.crop is not None),
            "fields_mature": sum(1 for f in self.fields if f.state == FieldState.MATURE),
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _upcoming_risks(self) -> Dict[str, float]:
        """Probabilities for the next loss stage not yet resolved (tomorrow under LFM, today under FML)."""
        if self.season_over:
            return {"drought": 0.0, "pest": 0.0, "price_shock": 0.0}
        d = self.day + 1 if self.config.decision_order == "LFM" else self.day
        if d > SEASON_LENGTH:
            return {"drought": 0.0, "pest": 0.0, "price_shock": 0.0}
        return self.scenario.risks(d)

    def _execute_action(self, action: Action, record: Dict[str, Any]) -> None:
        if action.type == ActionType.PLANT:
            field = self.fields[action.field_id]
            cfg = CROP_CONFIG[action.crop_name]
            field.plant(Crop.from_config(action.crop_name, cfg))
            uid = self._next_crop_uid
            self._next_crop_uid += 1
            field.crop_uid = uid
            self.cash -= cfg["planting_cost"]
            record.update(planted_crop=action.crop_name, planting_cost=cfg["planting_cost"])
            self.crop_ledger[uid] = {
                "crop_uid": uid, "crop": action.crop_name, "field_id": action.field_id,
                "planted_day": self.day, "planting_cost": cfg["planting_cost"],
                "predicted_completion_probability": self.plant_completion_probability_for_new(action.crop_name),
                "harvest_eligible_day": None, "harvested_day": None, "harvest_qty": 0,
                "destroyed_day": None, "drought_setbacks": 0, "outcome": None,
            }
        elif action.type == ActionType.HARVEST:
            field = self.fields[action.field_id]
            crop_name = field.crop.name
            uid = field.crop_uid
            qty = field.harvest()
            self.inventory[crop_name] += qty
            record.update(harvest_crop=crop_name, harvest_qty=qty)
            if uid in self.crop_ledger:
                self.crop_ledger[uid].update(harvested_day=self.day, harvest_qty=qty, outcome="harvested")
        elif action.type == ActionType.SELL:
            qty = action.quantity
            revenue = self.market.execute_sale(action.crop_name, qty)
            self.inventory[action.crop_name] -= qty
            self.cash += revenue
            record.update(sold_crop=action.crop_name, sold_qty=qty, revenue=revenue)

    def plant_completion_probability_for_new(self, crop_name: str) -> float:
        # Called after the field is already marked planted, so do not add one more field.
        planted_after = sum(1 for f in self.fields if f.is_planted)
        return completion_probability(
            CROP_CONFIG[crop_name]["growth_days"], self.day,
            self.scenario.drought_risk, self.scenario.pest_risk, planted_after,
            order=self.config.decision_order,
        )

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0] = min(self.season_manager.current_day, SEASON_LENGTH) / SEASON_LENGTH
        obs[1] = np.clip(self.cash / STARTING_CASH, 0.0, OBS_HIGH)
        obs[2] = np.clip(self.debt / STARTING_CASH, 0.0, 1.0)
        max_growth = max(c["growth_days"] for c in CROP_CONFIG.values())
        for i, field in enumerate(self.fields):
            obs[3 + i] = FIELD_STATE_MAP[field.state]
            obs[8 + i] = min(field.growth_progress / max_growth, 1.0)
            obs[13 + i] = (CROPS.index(field.crop.name) + 1) / NUM_CROPS if field.crop is not None else 0.0
        for j, crop in enumerate(CROPS):
            obs[18 + j] = np.clip(self.inventory[crop] / MAX_INVENTORY, 0.0, 1.0)
            obs[21 + j] = np.clip(self.market.get_current_price(crop) / CROP_CONFIG[crop]["base_price"], 0.0, 3.0)
        r = self._upcoming_risks()
        obs[24] = min(r["drought"] / RISK_SCALE, 3.0)
        obs[25] = min(r["pest"] / RISK_SCALE, 3.0)
        obs[26] = min(r["price_shock"] / RISK_SCALE, 3.0)
        return obs

    def _get_farm_state(self) -> FarmState:
        r = self._upcoming_risks()
        return FarmState(
            day=self.season_manager.current_day,
            cash=self.cash,
            debt=self.debt,
            fields=self.fields,
            inventory=self.inventory.copy(),
            market_prices=self.market.current_prices,
            risk_forecasts={k: risk_label(v) for k, v in r.items()},
        )

    def _inventory_value(self) -> float:
        return sum(qty * self.market.get_current_price(crop) for crop, qty in self.inventory.items())

    def _total_wealth(self) -> float:
        return self.cash + self._inventory_value()

    def _is_action_valid(self, action: Action) -> bool:
        """Backward-compatible structural validity check."""
        return self.check_structural_action(action)[0]
