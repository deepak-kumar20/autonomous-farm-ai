# Research outputs — Autonomous Farm AI

Everything in this folder is produced by code from executed simulations. Nothing is hand-entered.

## How to run

From the repository root, using the project's virtual environment:

```bash
# (optional) retrain PPO — about 7 minutes on a laptop CPU; writes models/ppo/farm_ppo_v2.zip + .meta.json and logs/ppo_training_*.csv
.venv/bin/python scripts/train_ppo.py --timesteps 2000000 --seed 7

# run all experiments, then build tables, charts, diagrams and summaries
.venv/bin/python scripts/run_experiments.py

# quick smoke test (5 scenarios per profile)
.venv/bin/python scripts/run_experiments.py --n 5

# rebuild tables/charts/docs from the existing CSVs only
.venv/bin/python scripts/run_experiments.py --outputs-only
```

(On Windows use `.venv\Scripts\python`.) Experiments never run when the app starts. Settings: `experiment_config.json`.

## Last run

- Started 2026-10-22T20:51:41, finished 2026-10-22T20:52:01, runtime 18.5 s
- Scenarios per profile: 50 (seeds 0–49)
- Episodes: **2850 successful**, **0 failed**, agents skipped: none
- Agent status: random: ok, rule_based: ok, ppo: ok

| Exp. | Experiment | Status |
|---|---|---|
| A | Main agent comparison | completed |
| B | Feasibility-filter ablation | completed |
| C | Decision-order ablation | completed |
| D | Environmental robustness | completed |
| E | Human vs AI | not run — no completed human session |
| F | Feasibility-filter comparison | completed |
| M | Market-model sensitivity | completed |

## Folder layout

| Path | Content |
|---|---|
| `data/daily_episode_records.csv` | One row per simulated day per episode (actions, validity, rejections, risks, events, prices, cash, inventory, harvest, sales, losses, wealth, reward) |
| `data/episode_summary.csv` | One row per episode (all summary metrics) |
| `data/crop_ledger.csv` | One row per planted crop (planting day, predicted completion probability, outcome) |
| `data/scenario_manifest.csv`, `data/scenarios.jsonl` | Every evaluation scenario (summary + full replayable JSON) |
| `data/*_raw.csv` | Per-experiment subsets of the episode summary (`experiment_id` column) |
| `data/human_ai_comparison_raw.csv`, `data/human_ai_daily_records.csv` | Experiment E (empty until a human match is played) |
| `tables/` | Tables 1–10 (+ supplementary 10b, 11) as CSV and Markdown |
| `charts/` | Charts as 300-dpi PNG + SVG, captions in `charts/README.md` |
| `paper_assets/` | Architecture and workflow diagrams (PNG + SVG) |
| `logs/` | `run_status.json`, PPO training curves |
| `audit_report.md`, `implementation_mapping.md`, `experiment_methods.md`, `results_summary.md`, `limitations.md` | Documentation |

Tables: `table_10_feasibility_performance.csv`, `table_10b_completion_probability_calibration.csv`, `table_11_market_model.csv`, `table_1_simulation_configuration.csv`, `table_2_framework_implementation_mapping.csv`, `table_3_agent_configurations.csv`, `table_4_environmental_scenarios.csv`, `table_5_main_agent_comparison.csv`, `table_6_feasibility_ablation.csv`, `table_7_decision_order_ablation.csv`, `table_8_human_vs_ai.csv`, `table_9_environmental_robustness.csv`

Charts: `chart_10_feasibility_decisions.png`, `chart_11_completion_probability_calibration.png`, `chart_1_wealth_over_30_days.png`, `chart_2_agent_profit_comparison.png`, `chart_3_weather_pest_robustness.png`, `chart_4_feasibility_ablation.png`, `chart_5_decision_order_ablation.png`, `chart_7_ppo_training_curve.png`, `chart_8_market_price_and_sales.png`, `chart_9_crop_losses_by_scenario.png`

## Metric definitions (units)

| Metric | Definition | Unit |
|---|---|---|
| Cash | Money held | ₹ |
| Wealth | Cash + Σ inventory × current price (crops still in the field count 0) | ₹ |
| Final wealth | Wealth after the day-30 action, valued at day-30 prices | ₹ |
| Profit | Final wealth − initial wealth (₹10,000) | ₹ |
| Revenue / Costs | Σ sale proceeds / Σ planting costs, from transaction records | ₹ |
| Harvest quantity | Σ harvested produce | abstract units (not kg) |
| Crops destroyed | Crops removed by pest outbreaks | count |
| Drought set-back days | Growth days lost to droughts | crop-days |
| Completion rate | Crops harvested / crops planted (destroyed and unharvested crops count as not completed) | ratio |
| Actions attempted | Actions submitted to the environment (one per day) | count |
| Invalid-action rate | Invalid executed actions / actions attempted | ratio |
| Feasibility rejection rate | Candidates rejected by the filter / candidates evaluated by the filter | ratio |
| Reward | Δ wealth per day, −50 for an invalid action (PPO training signal) | ₹ |
| Mean, SD, 95 % CI | Sample mean; sample SD (n−1); mean ± t(0.975, n−1)·s/√n | — |
| Brier score | mean (p − y)² for completion predictions | — |
