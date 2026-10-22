# Implementation mapping

Maps each component of the research framework to the code that implements it and the test that verifies it. Generated from `src/farm_ai/research/mapping.py` (the same data as Table 2).

Status: **Verified** = implemented and covered by the named passing test; **Partially implemented** = only part of the intended behaviour exists; **Not implemented** = absent (documented, not claimed).

## 30-day horizon — Verified

- Intended: Exactly 30 daily decisions; no actions after day 30
- Implementation: SeasonManager + FarmEnvironment.step raises after day 30
- Verification: `test_framework_env.test_thirty_transitions_then_terminal; test_step_after_season_raises`
- Notes: Terminal wealth valued at the day-30 price

## Pre-generated scenario — Verified

- Intended: All exogenous conditions fixed before day 1 and replayable
- Implementation: framework/scenario.py generate_scenario / Scenario.save / load
- Verification: `test_framework_scenario.test_scenario_reproducible; test_saved_scenario_replays_identically`
- Notes: SeedSequence, one stream per subsystem

## Risk vs. event — Verified

- Intended: Risk is a probability; damage only when an event occurs
- Implementation: drought_risk/pest_risk vs drought_event/pest_event; _resolve_losses reads events only
- Verification: `test_framework_scenario.test_risk_distinct_from_event; test_framework_env.test_losses_only_on_event_days`

## Weather model — Partially implemented

- Intended: Daily weather affecting crops
- Implementation: Binary drought event only (−1 growth day)
- Verification: `test_framework_env.test_losses_only_on_event_days`
- Notes: No temperature or rainfall variables exist

## Stage 1 — losses — Verified

- Intended: Resolve exogenous losses first
- Implementation: FarmEnvironment._resolve_losses at start of day (LFM)
- Verification: `test_framework_env.test_lfm_vs_fml_order`
- Notes: Spoilage not modelled

## Stage 2 — candidates — Verified

- Intended: Decision-maker proposes actions
- Implementation: Policy.rank (agents) / UI request (human)
- Verification: `test_framework_episode.test_human_and_ai_share_validation`
- Notes: One action per day

## Stage 3 — feasibility filter — Verified

- Intended: Reject infeasible actions before any state change, with reasons
- Implementation: framework/feasibility.py FeasibilityFilter + decide
- Verification: `test_framework_feasibility.test_rejection_reasons; test_rejection_does_not_change_state`
- Notes: Modes: none / deterministic / uncertainty

## Horizon constraint — Verified

- Intended: Do not plant crops that cannot finish by day 30
- Implementation: PLANT requires day + growth_days ≤ 30
- Verification: `test_framework_feasibility.test_horizon_constraint`

## Uncertainty-aware filter — Verified

- Intended: Use a computed completion probability
- Implementation: Exact DP over drought/pest risks; fixed or break-even threshold
- Verification: `test_framework_feasibility.test_completion_probability_matches_monte_carlo`
- Notes: Assumes constant number of planted fields

## Stage 4 — field actions — Verified

- Intended: Execute PLANT / HARVEST
- Implementation: FarmEnvironment._execute_action
- Verification: `test_framework_env.test_crop_maturity_day_convention`

## Stage 5 — market actions — Verified

- Intended: Execute SELL without negative inventory
- Implementation: Market.execute_sale; structural check
- Verification: `test_framework_market.test_sell_cannot_go_negative`
- Notes: Stages 4 and 5 are exclusive (one action/day)

## Stage 6 — update and log — Verified

- Intended: Grow crops, advance market and day, log transition
- Implementation: FarmEnvironment.step → transition_log, crop_ledger
- Verification: `test_framework_episode.test_summary_agrees_with_daily_records`

## Market model — Verified

- Intended: Coherent prices, inventory, revenue
- Implementation: Exogenous random walk (default); optional own-sales price impact
- Verification: `test_framework_market.test_exogenous_prices_identical_across_players; test_price_impact_depends_on_actions`
- Notes: No competitors modelled

## Human vs AI — Verified

- Intended: Same scenario and start; human first, AI replay
- Implementation: api/services/match_service.py + Versus page
- Verification: `test_framework_matches.test_match_flow_human_then_ai; test_human_run_does_not_mutate_scenario`

## Agent validation — Verified

- Intended: Check PPO checkpoint compatibility before use
- Implementation: framework/agents.check_ppo_checkpoint (env version + spaces)
- Verification: `test_framework_episode.test_legacy_checkpoint_rejected`
- Notes: Legacy v1 checkpoints are rejected

## Experiment pipeline — Verified

- Intended: Reproducible runs, CSV exports, generated tables/charts
- Implementation: research/runner.py, research/outputs.py
- Verification: `test_research_outputs.test_pipeline_small_run; test_empty_data_reports_not_run`

## Risk-adjusted objective — Not implemented

- Intended: J = E[W − λ·Risk − μ·Violations]
- Implementation: Not used: agents optimise Δwealth; evaluation reports mean, SD, CI
- Verification: none
- Notes: Documented in experiment_methods.md

## Spoilage — Not implemented

- Intended: Stored produce decays after shelf life
- Implementation: —
- Verification: none
- Notes: Inventory never spoils

## Key files

| File | Role |
|---|---|
| `src/farm_ai/framework/scenario.py` | Scenario profiles, seeded generation, save/load |
| `src/farm_ai/environment/farm_environment.py` | Six-stage daily process, logging, crop ledger |
| `src/farm_ai/framework/feasibility.py` | Feasibility filter, completion-probability DP, `decide` |
| `src/farm_ai/framework/config.py` | Decision order, filter mode, market model |
| `src/farm_ai/models/market.py` | Exogenous and price-impact markets |
| `src/farm_ai/framework/agents.py` | Random / rule-based / PPO policies, checkpoint check |
| `src/farm_ai/framework/episode.py` | Episode runner and summary metrics |
| `src/farm_ai/api/services/match_service.py` | Human-vs-AI matches |
| `src/farm_ai/research/` | Experiment runner, statistics, tables, charts, diagrams |
| `frontend/src/pages/VersusPage.tsx` | Human-vs-AI user interface |
