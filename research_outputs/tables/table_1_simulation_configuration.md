# Table 1 — Simulation configuration

| Parameter | Description | Value | Unit | Configuration source | Notes |
|---|---|---|---|---|---|
| Season length | Days per season | 30 | days | config/settings.py SEASON_LENGTH | Actions on days 1–30; terminal valuation at the day-30 price |
| Decision interval | One action per simulated day | 1 | day | FarmEnvironment.step | No hourly steps |
| Fields | Number of plots | 5 | fields | config/settings.py NUM_FIELDS |  |
| Initial cash | Starting cash = initial wealth | 10,000 | ₹ | config/settings.py STARTING_CASH | No debt or credit modelled |
| Wheat | Seed cost / growth / yield / base price | 1,000 / 5 / 100 / 40 | ₹ / days / units / ₹ per unit | config/settings.py CROP_CONFIG | Units are abstract produce units, not kg |
| Rice | Seed cost / growth / yield / base price | 1,500 / 8 / 150 / 35 | ₹ / days / units / ₹ per unit | config/settings.py CROP_CONFIG | Units are abstract produce units, not kg |
| Corn | Seed cost / growth / yield / base price | 1,200 / 6 / 120 / 30 | ₹ / days / units / ₹ per unit | config/settings.py CROP_CONFIG | Units are abstract produce units, not kg |
| Profile: baseline | Base daily probabilities: drought / pest / price shock | 0.05 / 0.08 / 0.1 | probability per day | framework/scenario.py PROFILES | Daily risk = base × lognormal AR(1) multiplier |
| Profile: dry | Base daily probabilities: drought / pest / price shock | 0.15 / 0.08 / 0.1 | probability per day | framework/scenario.py PROFILES | Daily risk = base × lognormal AR(1) multiplier |
| Profile: pest | Base daily probabilities: drought / pest / price shock | 0.05 / 0.2 / 0.1 | probability per day | framework/scenario.py PROFILES | Daily risk = base × lognormal AR(1) multiplier |
| Profile: combined | Base daily probabilities: drought / pest / price shock | 0.15 / 0.2 / 0.1 | probability per day | framework/scenario.py PROFILES | Daily risk = base × lognormal AR(1) multiplier |
| Risk process | Volatility σ / persistence ρ of daily risk multiplier | 0.6 / 0.8 | — | ScenarioProfile | Mean multiplier ≈ 1; risk capped at 0.9 |
| Risk labels | LOW / MEDIUM / HIGH cut-offs | <0.06 / <0.12 / ≥0.12 | probability per day | framework/scenario.py | Display only |
| Price drift | Uniform daily drift; clamp before shocks | ±5%; [0.5, 1.5] × base | fraction | ScenarioProfile | Original Market rule |
| Price shock | Multiplier on all crops on a shock day | 0.5 or 2 | × | ScenarioProfile | Equiprobable |
| Drought effect | GROWING crops with progress > 0 lose one day | 1 | day | FarmEnvironment._resolve_losses | Mature crops unaffected |
| Pest effect | Destroys one planted field (uniform pick) | 1 | field | FarmEnvironment._resolve_losses | Only on outbreak days |
| Decision order (default) | Stage order within a day | LFM | — | SimulationConfig | LFM = Loss → Field → Market |
| Feasibility filter (default) | Pre-execution filter mode | deterministic | — | SimulationConfig | Ablations: none / uncertainty |
| Uncertainty threshold | Fixed τ, or break-even τ = cost / (yield × price) | 0.8 or break-even | probability | SimulationConfig |  |
| Market model (default) | Price formation | exogenous | — | SimulationConfig / models/market.py | price_impact: Q_ref=500, γ=0.5, δ=0.7 |
| Invalid-action penalty | Reward when an executed action fails | -50 | ₹ (reward units) | environment/rewards.py | Reward = Δ wealth otherwise |
| Evaluation scenarios | Seeds per profile | 50 | scenarios | experiment_config.json | Seeds 0–49 |
