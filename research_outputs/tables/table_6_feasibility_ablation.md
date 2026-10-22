# Table 6 — Feasibility-filter ablation (Experiment B)

| Configuration | Filter enabled | Scenario count | Mean final wealth (₹) | Mean profit (₹) | Invalid-action rate | Mean rejections | Crop completion rate | Decision time (ms/episode) |
|---|---|---|---|---|---|---|---|---|
| Random · baseline · full | Yes (deterministic) | 50 | 27,477 | 17,477 | 0.000 | 11.02 | 71.4% | 0.64 |
| Random · baseline · no filter | No | 50 | 22,258 | 12,258 | 0.000 | 0.00 | 54.8% | 0.42 |
| Rule-based · baseline · full | Yes (deterministic) | 50 | 37,844 | 27,844 | 0.000 | 0.00 | 82.9% | 0.43 |
| Rule-based · baseline · no filter | No | 50 | 37,844 | 27,844 | 0.000 | 0.00 | 82.9% | 0.39 |
| PPO · baseline · full | Yes (deterministic) | 50 | 31,360 | 21,360 | 0.000 | 6.78 | 86.0% | 3.36 |
| PPO · baseline · no filter | No | 50 | 31,063 | 21,063 | 0.060 | 0.00 | 85.7% | 3.17 |
| Random · combined · full | Yes (deterministic) | 50 | 19,346 | 9,346 | 0.000 | 13.72 | 47.1% | 0.62 |
| Random · combined · no filter | No | 50 | 14,295 | 4,295 | 0.000 | 0.00 | 36.9% | 0.44 |
| Rule-based · combined · full | Yes (deterministic) | 50 | 23,098 | 13,098 | 0.000 | 0.00 | 52.9% | 0.44 |
| Rule-based · combined · no filter | No | 50 | 23,098 | 13,098 | 0.000 | 0.00 | 52.9% | 0.40 |
| PPO · combined · full | Yes (deterministic) | 50 | 24,764 | 14,764 | 0.000 | 18.86 | 66.3% | 3.35 |
| PPO · combined · no filter | No | 50 | 24,471 | 14,471 | 0.077 | 0.00 | 66.1% | 3.17 |

*Completion rate = crops harvested / crops planted. Decision time = mean wall-clock time per episode spent in candidate ranking + filtering.*
- Random, baseline: paired profit difference full − no filter = +5,218 [+3,726, +6,711] ₹
- Rule-based, baseline: paired profit difference full − no filter = +0 [+0, +0] ₹
- PPO, baseline: paired profit difference full − no filter = +296 [-365, +958] ₹
- Random, combined: paired profit difference full − no filter = +5,050 [+3,887, +6,214] ₹
- Rule-based, combined: paired profit difference full − no filter = +0 [+0, +0] ₹
- PPO, combined: paired profit difference full − no filter = +293 [-649, +1,235] ₹
