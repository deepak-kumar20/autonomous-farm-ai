# Table 5 — Main agent comparison (Experiment A, baseline scenarios, full framework)

| Agent | Episodes | Mean final wealth (₹) | Mean profit (₹) | Profit SD (₹) | Mean harvest (units) | Mean crops destroyed | Invalid-action rate | Mean feasibility rejections | 95% CI of mean profit (₹) |
|---|---|---|---|---|---|---|---|---|---|
| Random | 50 | 27,477 | 17,477 | 11,844 | 967.60 | 2.18 | 0.000 | 11.02 | [14,111, 20,842] |
| Rule-based | 50 | 37,844 | 27,844 | 15,866 | 1,335.00 | 1.88 | 0.000 | 0.00 | [23,335, 32,353] |
| PPO | 50 | 31,360 | 21,360 | 15,226 | 848.80 | 1.38 | 0.000 | 6.78 | [17,033, 25,686] |

*CI = mean ± t(0.975, n−1)·s/√n. Invalid-action rate = invalid / actions attempted (30 per episode). Rejections = filter rejections per episode.*
- Paired difference Random − Rule-based: -10,367 [-12,791, -7,944] ₹ (mean [95% CI], n = 50 scenarios; random higher in 8.0% of scenarios)
- Paired difference PPO − Random: +3,883 [+702, +7,064] ₹ (mean [95% CI], n = 50 scenarios; ppo higher in 58.0% of scenarios)
- Paired difference PPO − Rule-based: -6,484 [-10,528, -2,441] ₹ (mean [95% CI], n = 50 scenarios; ppo higher in 28.0% of scenarios)
