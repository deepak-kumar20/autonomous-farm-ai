# Audit Report — Autonomous Farm AI

This report records (A) what the repository actually implemented **before** the research-framework work
(git commit `b84afb9`), measured by direct inspection and probe scripts, and (B) what was changed.
Every claim in Part A was verified against the code or by running it; probe output is quoted verbatim.

---

## Part A — Initial implementation (as found)

### A1. Repository layout

| Area | Location | Notes |
|---|---|---|
| Simulator (Gymnasium env) | `src/farm_ai/environment/farm_environment.py` | `Discrete(24)` actions, `Box(27,)` observations |
| Crop / field / market models | `src/farm_ai/models/` | `Crop`, `Field`, `Market` (random-walk prices) |
| Risk systems | `src/farm_ai/simulation/{weather,pests,price_shocks,season}.py` | Use the **global** Python `random` module |
| Agents | `src/farm_ai/agents/` | `RandomAgent`, `RuleBasedAgent`, `PPOAgent` (SB3 2.7.1) |
| API | `src/farm_ai/api/` | FastAPI; in-memory sessions; PPO loaded once at startup |
| Frontend | `frontend/` | React 19 + Vite; game-style UI (7 pages) |
| Checkpoints | `models/ppo/farm_ppo.zip` (50k steps), `farm_ppo_500k.zip` (500k steps) | |
| Training logs | `runs/ppo/PPO_1` (to 50,176 steps), `runs/ppo/PPO_2` (to 500,224 steps) | TensorBoard; git-ignored |
| Experiment scripts | `scripts/run_baselines.py`, `scripts/action_distribution.py`, `src/farm_ai/training/evaluate_ppo.py` | Print to console only; no saved datasets |
| Tests | `tests/` | 35 tests, all passing (`python -m unittest discover -s tests`); `pytest` not installed |

### A2. Daily transition (actual order in `FarmEnvironment.step`)

1. Resolve integer → `Action`; SELL quantity = full inventory of that crop.
2. Execute the action (`_execute_action`); a `ValueError` marks the action invalid (no state change from the action).
3. Advance growth of every field (`Field.advance_day`; GROWING → MATURE when `progress >= growth_days`).
4. `market.update_prices()` (±5 % uniform random walk, clamped to [0.5, 1.5] × base price).
5. `weather.update()`, `pests.update()`, `price_shocks.update()` (each draws one global `random.random()` roll).
6. Apply effects: drought (−1 progress to every planted field with progress > 0, **including MATURE fields**), pest
   (destroys one random planted field), price shock (all prices × 0.5 or × 2, **after** the clamp).
7. Reward = Δ(cash + inventory value), or −50 for an invalid action.
8. Day += 1; `terminated = day > 30`.

So the implemented order was **Field/Market action → Growth → Market drift → Losses (events)**, not Loss → Field → Market.

### A3. Horizon and off-by-one behaviour

* Day starts at 1. Actions are executed on days 1–30; termination occurs after the 30th transition (day becomes 31). ✔
* **Bug:** `step()` keeps executing after termination. Probe:
  `step after termination allowed: reward -1000.0 terminated True cash 9000.0` (a PLANT on "day 31" spent ₹1,000).
* Terminal wealth = cash + inventory × the price **after** the final day's random walk and possible shock, i.e. the
  last action is followed by one extra, untradeable price move.
* A crop planted on day *d* reaches `progress = 1` the same day; it becomes MATURE at the end of day *d + g − 1* and is
  first harvestable on day *d + g* (g = growth days). Crops still in the field at season end have **zero** value.

### A4. Reproducibility — **not reproducible**

`FarmEnvironment.reset(seed)` only seeds Gymnasium's `self.np_random`, which no subsystem uses. Weather, pests, price
drift and price shocks all use the unseeded global `random`. Probe (rule-based agent, seed 42, run twice):

```
1. reproducibility: rule-based, seed=42 twice -> 29759.34 21339.29
```

No seed is ever set for `random` anywhere in `src/`, `scripts/` or `tests/` (`grep random.seed` → no matches).

### A5. Weather, pests and "risk forecasts"

* No temperature, rainfall or other weather variable exists. "Weather" = a binary drought event, p = 0.05/day.
* Pest outbreak p = 0.08/day; destroys one uniformly random planted field (growing or mature).
* Price shock p = 0.10/day; multiplies **all** crop prices by 0.5 or 2.0.
* The "forecast" label is derived from the **same roll** that triggers the event (`roll < p` → event + HIGH;
  `roll < 3p` → HIGH; `roll < 6p` → MEDIUM). It is shown after the event has happened and carries no information about
  the next day. Probe over 200,000 simulated days:

```
3. drought active 0.050; HIGH label 0.151; P(active|HIGH)=0.335; P(HIGH|active)=1.000
   P(drought tomorrow | label today=LOW) = 0.050
   P(drought tomorrow | label today=MEDIUM) = 0.049
   P(drought tomorrow | label today=HIGH) = 0.050
```

  Risk and occurrence were therefore not separated: the label was a noisy same-day disclosure of the event.

### A6. Market

* Prices follow an exogenous random walk; **sales do not affect prices** (no supply–demand mechanism).
* Selling is all-or-nothing per crop at the current price. Inventory cannot go negative (checked).
* Because prices are exogenous, two players facing the same random sequence would see identical prices.

### A7. Feasibility filtering — **not implemented**

The task brief describes "deterministic" and "uncertainty-aware" feasibility filters. No such code exists
(`grep -ri "feasib\|uncertain"` over `src/ scripts/ tests/ frontend/src/` → no matches). The only check is
`_is_action_valid` (cash, empty/mature field, inventory > 0), used to label actions in the API. It has no
time-to-horizon constraint: e.g. planting Rice (8 days) on day 28 is "valid" and silently wastes ₹1,500.

### A8. Agents

| Agent | Actual behaviour | Issues |
|---|---|---|
| Random | Uniform choice over HOLD, harvest-mature, plant-affordable-on-empty, and SELL of a **random partial quantity** | Uses `Action` objects (partial sells) not available to PPO/humans via the 24-action space |
| Rule-based | Harvest → sell all → plant crop with highest `yield × current price − cost` if `growth_days < 30 − day` → hold | Horizon rule is one day conservative; uses current price as forecast |
| PPO | SB3 `PPO`, MlpPolicy [64,64], n_steps 256, unscaled Δwealth reward, −50 invalid penalty, no action masking | See A9 |

### A9. PPO checkpoints — **not validated**

Both checkpoints load, but the greedy (deterministic) policy collapses to a single invalid action. Probe over 20
episodes each:

```
farm_ppo      det=True : mean profit 0    sd 0    invalid-rate 1.000  action mix {'HARVEST': 600}
farm_ppo      det=False: mean profit 1200 sd 6671 invalid-rate 0.622
farm_ppo_500k det=True : mean profit 0    sd 0    invalid-rate 1.000  action mix {'HARVEST': 600}
farm_ppo_500k det=False: mean profit 3772 sd 7412 invalid-rate 0.625
```

The API served the 500k model with `deterministic=true` by default, so the "AI Farmer" in the UI repeatedly issued
"HARVEST in Field 4" on an empty field (−50 reward each day). `docs/ppo_experiment.md` lists 100k and 200k action
distributions for which no checkpoints or logs exist. Training reward (TensorBoard `rollout/ep_rew_mean`) rose from
−3,819 to +2,441 over 500k steps, but that reward includes invalid-action penalties and is not terminal profit.

### A10. Experiment data

Baseline/PPO scripts print per-episode numbers to the console; nothing is saved. There were no episode-level datasets,
tables or charts, and (because of A4) the printed numbers cannot be reproduced.

### A11. Frontend

The UI showed `risk_forecasts` labels as forecasts (see A5 — they are not), and always used the deterministic PPO
policy (see A9). There was no scenario selection, no Human-vs-AI mode and no feasibility feedback; invalid actions
silently consumed a day.

---

## Part B — Changes made

See the end of this file (appended after implementation) and `implementation_mapping.md`.

### B1. Simulation corrections

| Issue (Part A) | Change | Verified by |
|---|---|---|
| A4 unseeded randomness | All exogenous draws pre-generated in a `Scenario` from `numpy.random.SeedSequence(seed)` with one stream per subsystem; the global `random` module is no longer used by the simulator | `test_framework_scenario.test_scenario_reproducible`; rule-based seed 42 now gives the same profit every run |
| A3 actions after day 30 | `FarmEnvironment.step` raises `ValueError` once the season is over; the API returns 400 | `test_framework_env.test_step_after_season_raises` |
| A3 extra post-season price move | Terminal wealth is valued at the day-30 price (no untradeable move after the last action) | `test_framework_env.test_terminal_valuation_uses_day30_price` |
| A5 risk label leaked the event | Risk = configured daily probability from a smooth stochastic process; event = separate Bernoulli draw; labels are derived from the probability only; the observation exposes the next unresolved day's probabilities | `test_framework_scenario.test_risk_distinct_from_event`; `test_framework_env.test_losses_only_on_event_days` |
| A2 order differed from the framework | Six-stage process with selectable order: LFM (default) or FML (original) | `test_framework_env.test_lfm_vs_fml_order` |
| Drought also decremented MATURE crops | Drought affects GROWING crops only (it never changed outcomes, but was inconsistent) | `test_framework_env.test_losses_only_on_event_days` |
| Pest target used global random | Target = ⌊u_t · n_planted⌋ with u_t from the scenario | scenario replay tests |
| A6 market undocumented | Exogenous model documented; optional `price_impact` model added for sensitivity analysis | `test_framework_market.*` |
| No logging | Every transition is recorded (actions, validity, filter rejections, risks, events, prices, cash, inventory, harvest, sales, losses, wealth, reward), with a per-crop ledger | `test_framework_episode.test_summary_agrees_with_daily_records` |

The environment version is now 2. The observation keeps 27 features with the same layout, except that features 24–26
are now probabilities (scaled) instead of leaked labels, and the observation-space bound is corrected to [0, 5].
(The original declared [0, 1] while cash could reach 5.)

### B2. New components

* **Feasibility filter** (`framework/feasibility.py`): modes none / deterministic / uncertainty. Structured results
  carry a code, reason, constraint values, and probability and threshold only when computed. The completion probability
  is an exact DP validated against Monte Carlo simulation of the environment.
* **Agents on one code path** (`framework/agents.py`, `framework/episode.py`): every agent ranks the same 24 actions.
  `decide` applies the filter. PPO checkpoints are checked (environment version and spaces) before use.
* **PPO retrained** (`training/train_ppo.py`): 2,002,944 steps, seed 7, with a validation curve logged to
  `research_outputs/logs/`. Final validation (20 held-out scenarios): greedy without filter ₹18,398 mean profit with
  an 8.8 % invalid-action rate; with the deterministic filter ₹19,342 and 0 % invalid. The legacy checkpoints are kept
  but rejected by the compatibility check.
* **API:** `/api/scenarios` (generate/view ex-ante scenarios), `/api/matches` (Human-vs-AI), `/api/agents`,
  `/api/farm/{id}/scenario` and `/api/farm/{id}/log`. Actions return `accepted` and `feasibility`; rejected actions
  do not advance the day. PPO predict/step use the session's filter.
* **Frontend:** a new Versus page (scenario selection with risk calendar → human season → AI replay → comparison
  table, wealth chart and day-by-day history). Rejection reasons appear as toasts and on seed packets. The Weather page
  shows risks as probabilities with a "simulated" label and the season's risk calendar. New seasons use random
  scenarios.
* **Experiment pipeline:** `scripts/run_experiments.py` → `research/runner.py` (episodes → CSV) →
  `research/outputs.py`, `charts.py`, `diagrams.py` (tables, charts, diagrams, generated README and results summary).

### B3. Tests

The original 35 tests were kept and pass. New test files: `test_framework_scenario.py`, `test_framework_env.py`,
`test_framework_feasibility.py`, `test_framework_market.py`, `test_framework_episode.py`,
`test_framework_matches.py` and `test_research_outputs.py`. See the final run in the completion report.

### B4. Not changed / not implemented

Temperature and rainfall variables, spoilage, credit, partial sales and multi-action days were not added. The
risk-adjusted objective J is not used by any agent. See `limitations.md`.
