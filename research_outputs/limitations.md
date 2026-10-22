# Limitations

Read these before drawing conclusions from the tables and charts.

## Simulator validity

* **Synthetic, uncalibrated parameters.** Crop costs, growth times, yields, prices and event probabilities are
  illustrative values inherited from the project. Nothing has been validated against real agronomic or market data,
  so the results support no claim about real farms.
* **Thin weather model.** "Weather" is a binary drought event that removes one growth day. There is no temperature,
  rainfall, soil moisture or yield sensitivity to weather.
* **Simple pest model.** An outbreak destroys exactly one planted field. There is no partial damage, spread or
  mitigation action.
* **Not modelled:** spoilage, storage limits, costs other than seed, credit or debt, labour, and selling part of a stack.
* **One action per day.** Field and market actions cannot be combined on one day, so stages 4 and 5 of the framework
  never both run in a day.
* **Zero value for crops in the field.** Wealth counts unharvested crops at 0, which produces large mid-season wealth
  dips and makes daily wealth a poor progress measure.

## Framework conformance

* **Order ablation scope.** The ablation (Experiment C) can only move the *loss* stage relative to the decision.
  Field-vs-market order inside a day is undefined with one action per day.
* **Approximate completion probability.** The uncertainty filter's probability assumes the number of planted fields
  stays constant after planting. It also uses published daily risks, not information about realised events. It is
  better than climatology (lower Brier score) but not perfectly calibrated (Table 10b, chart 11).
* **The fixed τ = 0.8 threshold is harmful here.** It rejects most plantings, including those with positive expected
  value. The break-even rule is more sensible but is still a risk-neutral, single-crop criterion; it ignores
  portfolio effects and the opportunity cost of the field.
* **No risk-adjusted objective.** J = E[W − λ·Risk − μ·Violations] is not used by any agent. Risk is reported but
  not optimised.
* **Price-impact market is limited.** It models only the player's own sales (no competitors) and is used only as a
  sensitivity analysis.

## Agents

* **PPO is not tuned and not optimal.** One seed, one hyper-parameter set, 2 M steps. PPO beats Random but is below
  the rule-based agent on baseline scenarios (Table 5). Under stress PPO loses less profit than the other agents and
  overtakes the rule-based agent only in the combined profile (Table 9). PPO also plants fewer crops per season (about
  9.8 vs about 11–12), which exposes fewer crops to outbreaks. That is a plausible mechanism, not evidence of
  risk-aware reasoning.
* **Training/test mismatch.** PPO was trained without the feasibility filter and is evaluated with ranking-based
  masking. A policy trained with masking in the loop (e.g. MaskablePPO) might behave differently.
* **The rule-based agent already enforces a horizon rule.** So the deterministic filter adds nothing for it, and
  Experiment B's effect for that agent is exactly zero by construction.
* **The random agent's candidate set already excludes most structurally invalid actions.** Its "no filter" invalid
  rate is therefore 0; the filter helps it mainly through the horizon constraint.

## Statistics

* 50 scenarios per profile. CIs are descriptive t-intervals and no multiple-comparison correction is applied.
  Several CIs in Experiments C and the PPO part of B include 0 and should be read as "no detectable difference".
* Profit distributions are skewed (episodes where price shocks coincide with large inventories). The t-interval is
  an approximation.
* Scenario groups share random streams (common random numbers). This sharpens paired comparisons but means the
  groups are not independent samples.

## Human-vs-AI

* **No human data yet.** At the time of the last run, no completed human session had been recorded, so Experiment E,
  Table 8 and chart 6 report "Not run". Results from a single session should be presented as a case study only.
* Humans see the full 30-day risk calendar before play. Agents see only the next day's risk in their observation,
  although the uncertainty filter uses the full calendar. Information parity is close but not exact.
* The browser end-to-end test used a scripted player writing to a temporary directory. It is not human data and is
  not included.

## Legacy artefacts

* `models/ppo/farm_ppo.zip` and `farm_ppo_500k.zip`, `runs/ppo/`, and the numbers in `docs/ppo_experiment.md`
  belong to the original (version 1) simulator. They cannot be reproduced (unseeded randomness) and are not used in
  any result here.
