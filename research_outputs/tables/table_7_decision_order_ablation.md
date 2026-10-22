# Table 7 — Decision-order ablation (Experiment C)

| Configuration | Stage order | Episodes | Mean final wealth (₹) | Mean profit (₹) | Profit SD (₹) | Mean crops destroyed | Invalid-action rate | Difference from full framework (₹) |
|---|---|---|---|---|---|---|---|---|
| Random · baseline | Loss → Field → Market (LFM) | 50 | 27,477 | 17,477 | 11,844 | 2.18 | 0.000 | reference |
| Random · baseline | Field → Market → Loss (FML) | 50 | 27,902 | 17,902 | 13,521 | 2.26 | 0.000 | +425 [-1,814, +2,664] |
| Rule-based · baseline | Loss → Field → Market (LFM) | 50 | 37,844 | 27,844 | 15,866 | 1.88 | 0.000 | reference |
| Rule-based · baseline | Field → Market → Loss (FML) | 50 | 37,018 | 27,018 | 15,814 | 1.92 | 0.000 | -826 [-1,898, +246] |
| PPO · baseline | Loss → Field → Market (LFM) | 50 | 31,360 | 21,360 | 15,226 | 1.38 | 0.000 | reference |
| PPO · baseline | Field → Market → Loss (FML) | 50 | 31,074 | 21,074 | 15,485 | 1.50 | 0.000 | -286 [-859, +288] |
| Random · combined | Loss → Field → Market (LFM) | 50 | 19,346 | 9,346 | 13,746 | 5.66 | 0.000 | reference |
| Random · combined | Field → Market → Loss (FML) | 50 | 17,450 | 7,450 | 11,360 | 5.76 | 0.000 | -1,896 [-4,785, +994] |
| Rule-based · combined | Loss → Field → Market (LFM) | 50 | 23,098 | 13,098 | 14,653 | 5.24 | 0.000 | reference |
| Rule-based · combined | Field → Market → Loss (FML) | 50 | 22,573 | 12,573 | 14,316 | 5.30 | 0.000 | -525 [-2,152, +1,102] |
| PPO · combined | Loss → Field → Market (LFM) | 50 | 24,764 | 14,764 | 13,893 | 3.30 | 0.000 | reference |
| PPO · combined | Field → Market → Loss (FML) | 50 | 23,946 | 13,946 | 13,626 | 3.42 | 0.000 | -818 [-1,897, +260] |

*Only the stage order changes; scenario, agent, filter (deterministic) and market are identical. Difference = paired (FML − LFM) profit, mean [95% CI].*
