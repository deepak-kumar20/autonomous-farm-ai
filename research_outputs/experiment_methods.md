# Experimental methods

Written to support the *Experimental Setup* section of Paper 2. Symbols map to code; file paths are relative to
the repository root.

## 1. Simulator

A finite-horizon, discrete-time farm simulator (`src/farm_ai/environment/farm_environment.py`, Gymnasium API,
environment version 2). One step is one day; a season has T = 30 decisions. The farm has 5 fields, starts with
₹10,000 cash and no debt, and can grow three crops:

| Crop | Seed cost c (₹) | Growth time g (days) | Yield y (units) | Base price p̄ (₹/unit) |
|---|---|---|---|---|
| Wheat | 1,000 | 5 | 100 | 40 |
| Rice | 1,500 | 8 | 150 | 35 |
| Corn | 1,200 | 6 | 120 | 30 |

Produce units are abstract and not calibrated to kilograms. Parameters are synthetic, inherited from the original
project, and not calibrated to any real region or market.

### State

S_t = (C_t, F_t, G_t, I_t, W_t, P_t, M_t):

| Symbol | Meaning | Code |
|---|---|---|
| C_t | cash (₹) | `env.cash` |
| F_t | field states ∈ {EMPTY, GROWING, MATURE} | `env.fields[i].state` |
| G_t | crop type and growth progress (days) per field | `env.fields[i].crop`, `.growth_progress` |
| I_t | inventory per crop (units) | `env.inventory` |
| W_t | drought risk and realised drought event | `scenario.drought_risk/drought_event` |
| P_t | pest risk and realised outbreak | `scenario.pest_risk/pest_event` |
| M_t | market prices and (price-impact model only) supply pressure | `env.market` |

### Actions

Discrete(24): HOLD; HARVEST field i (5); PLANT crop k on field i (15); SELL all stock of crop k (3). Humans,
rule-based, random and PPO agents all use this space.

### Transition and stage order

S_{t+1} = f(S_t, a_t, E_t), where E_t are day t's exogenous draws from the scenario. Within a day (default
"LFM", Loss → Field → Market):

1. **Losses.** If `drought_event[t]`, every GROWING crop with progress > 0 loses one day. If `pest_event[t]`, the
   planted field with index ⌊u_t · n_planted⌋ (fields in id order) is destroyed, where u_t = `pest_target_u[t]`.
   Risk values never cause damage by themselves.
2. **Candidates.** The decision-maker proposes actions in preference order.
3. **Feasibility filter.** See §3. Rejected candidates change nothing.
4. **Field action.** PLANT pays c and sets progress 0; HARVEST of a MATURE field adds y units to inventory.
5. **Market action.** SELL all units of a crop at the current price.
6. **Update.** Every planted crop gains one day of progress and becomes MATURE when progress ≥ g. The market clock
   moves to t + 1; the reward and the daily record are written; t ← t + 1.

Because exactly one action is taken per day, stages 4 and 5 are mutually exclusive. The ablation order "FML"
(the original simulator's order) runs stage 1 after crop growth at the end of the day, which lets a pest hit a
crop planted that day and lets a crop mature before the day's drought applies.

**Day convention.** A crop planted on day d is first harvestable on day d + g. The season ends after the day-30
action. No action can be taken afterwards (`step` raises).

### Wealth, profit and reward

W_t = C_t + Σ_k I_{k,t} · p_{k,t}, where p_{k,t} is the current (marginal) price. Crops still in the field are worth 0.
Terminal wealth W_T is valued at the day-30 price. Profit = W_T − W_0 with W_0 = ₹10,000. Revenue and costs come
from the transaction log, so Profit = Revenue − Costs + final inventory value (asserted in tests).

The per-day reward used for PPO training is r_t = W_{t+1} − W_t, or −50 when the executed action is invalid.

A risk-adjusted objective J = E[W_T − λ·Risk − μ·Violations] was **not** implemented. Agents maximise expected Δwealth
(PPO) or follow fixed rules. The evaluation reports mean, SD and confidence intervals of profit, plus loss and
violation counts as separate metrics.

## 2. Scenarios (`src/farm_ai/framework/scenario.py`)

A scenario fixes every exogenous quantity before day 1. From `SeedSequence(seed)` it draws nine independent
streams (one per subsystem):

* **Daily risk.** p_t = clip(base · exp(σ z_t − σ²/2), 0, 0.9), where z_t is a stationary AR(1) with ρ = 0.8 and σ = 0.6
  (mean multiplier ≈ 1). This is generated separately for drought, pest and price shock.
* **Events.** e_t = 1[U_t < p_t], i.e. Bernoulli(p_t) given p_t, with U_t from a dedicated uniform stream.
* **Prices.** P_{k,1} = p̄_k. P_{k,t} = clamp(P_{k,t−1}(1 + ε_{k,t}), 0.5p̄_k, 1.5p̄_k) · s_t, with ε ~ U(−0.05, 0.05)
  and s_t ∈ {0.5, 2} (equiprobable) on shock days, otherwise 1.

Four profiles change only the base rates: baseline (drought 0.05, pest 0.08, shock 0.10, the original simulator's
values), dry (drought 0.15), pest (pest 0.20) and combined (both). Because the streams are shared, a given seed
uses identical price paths, outbreak-target draws and event uniforms in every profile (common random numbers). A
higher rate can only add events.

Before play, humans see the risk calendar (daily probabilities), the base rates and the day-1 prices. The realised
events and future prices stay hidden. Under LFM, the observation's risk features are the probabilities for the
next unresolved loss stage (tomorrow).

## 3. Feasibility filter (`src/farm_ai/framework/feasibility.py`)

Feasible(a, S_t) = 1 iff every applicable constraint holds:

* **Structural:** the field is EMPTY and C_t ≥ c for PLANT; MATURE for HARVEST; I_k > 0 for SELL; the season is not over.
* **Deterministic horizon (PLANT):** t + g ≤ 30.
* **Uncertainty-aware (PLANT, mode `uncertainty`):** P̂ ≥ τ. Here P̂ is the probability that the crop is present and
  MATURE at a decision point on or before day 30. It is computed exactly by dynamic programming over the growth
  progress, using the scenario's published risks for each future day. The pest-hit probability per outbreak is
  1/n, with n = planted fields after this planting, held constant (approximation). The threshold τ is fixed (0.8) or
  break-even, τ = c / (y · p_{k,t}), i.e. accept iff the expected harvest value covers the seed cost (risk-neutral).

Candidates are proposed in the decision-maker's preference order until one is accepted. HOLD is always feasible.
Without a filter (`none`), the first candidate is executed directly. If it is infeasible it fails (invalid action,
−50 reward, the day passes), which reproduces the original simulator.

The DP is checked against Monte Carlo simulation of the actual environment (`tests/test_framework_feasibility.py`).

## 4. Market models (`src/farm_ai/models/market.py`)

* **exogenous (default):** price = scenario reference price; sales do not move prices. Paired human/AI runs
  therefore face identical prices.
* **price_impact (sensitivity analysis):** P = R · max(0.25, (1 + S/Q_ref)^−γ). Here S is the player's own decaying
  supply pressure (S ← δS + q; Q_ref = 500 units, γ = 0.5, δ = 0.7), and a sale earns q · R · factor(S + q/2). There are no
  modelled competitors. Outcomes become action-dependent, so paired runs may legitimately face different prices.

## 5. Agents (`src/farm_ai/framework/agents.py`)

* **Random:** uniform over its own candidate set (HOLD, harvest any MATURE field, plant any affordable crop on any
  EMPTY field, sell any crop in stock); seeded with the scenario seed. Its preference order is a random permutation.
* **Rule-based:** harvest → sell all → plant the crop maximising y·p − c if g < 30 − t and the value is positive → hold
  (unchanged from the original project). Preference order: its choice, then HOLD.
* **PPO:** Stable-Baselines3 PPO, MLP 128-128, n_steps 512 × 8 envs, batch 256, 10 epochs, γ 0.99, λ 0.95,
  lr 3·10⁻⁴, clip 0.2, entropy 0.01. It was trained 2,002,944 steps (seed 7) on LFM with no filter and with rewards
  scaled by 10⁻³. Training scenarios were drawn uniformly from the four profiles with seeds ≥ 10⁶; validation used
  seeds 900000–900019. The final checkpoint is used (not the best validation point). At decision time all 24 actions
  are ranked by π(a|s) (greedy), so the same checkpoint is evaluated with and without the filter. Compatibility
  (environment version, observation and action spaces) is checked before use. The two legacy checkpoints fail this
  check.

## 6. Experiments (`research_outputs/experiment_config.json`)

Evaluation scenarios: seeds 0–49 for each profile, disjoint from training and validation seeds. Each (agent,
scenario, configuration) episode runs once and is shared across the experiments that need it.

| Exp. | Profiles | Configurations compared |
|---|---|---|
| A — agent comparison | baseline | full framework (LFM + deterministic filter) |
| B — feasibility ablation | baseline, combined | full vs no filter |
| C — decision order | baseline, combined | LFM vs FML (deterministic filter) |
| D — robustness | all four | full |
| E — human vs AI | saved matches | each finished human match replayed by every agent |
| F — filter modes | all four | none, deterministic, uncertainty (τ = 0.8), uncertainty (break-even) |
| M — market (supplementary) | baseline | exogenous vs price_impact |

## 7. Statistics (`src/farm_ai/research/stats.py`)

* Mean over n valid episodes. Sample SD with n − 1; it is undefined for n < 2.
* 95 % CI: mean ± t_{0.975,n−1} · s/√n, with tabulated t values (interpolated in 1/df beyond df = 30).
* Paired differences d_i = Profit_X(i) − Profit_Y(i) over identical scenarios, summarised with the same CI.
* Invalid-action rate = invalid executed actions / actions attempted.
* Filter rejection rate = rejected candidates / candidates evaluated.
* Completion rate = harvested / planted.
* Calibration: Brier score of P̂ against the observed indicator "harvest-eligible by day 30", compared with the
  climatology score ȳ(1 − ȳ), plus a 10-bin reliability diagram.

No multiple-comparison correction or hypothesis tests are applied. CIs are descriptive.

## 8. Reproducing

```bash
.venv/bin/python scripts/train_ppo.py --timesteps 2000000 --seed 7   # optional, ≈7 min CPU
.venv/bin/python scripts/run_experiments.py                          # ≈20 s for 2,850 episodes
```

Software: Python 3.9.6, Gymnasium 1.1.1, Stable-Baselines3 2.7.1, PyTorch 2.8.0, NumPy 2.0.2, pandas 2.3.3,
Matplotlib 3.9.4.
