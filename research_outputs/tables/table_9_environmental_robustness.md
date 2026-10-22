# Table 9 — Environmental robustness (Experiment D, full framework)

| Scenario | Agent | Episodes | Mean profit (₹) | Profit SD (₹) | Mean final wealth (₹) | Mean harvest (units) | Mean crops destroyed | Invalid-action rate | Difference from baseline (₹) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | Random | 50 | 17,477 | 11,844 | 27,477 | 967.60 | 2.18 | 0.000 | reference |
| baseline | Rule-based | 50 | 27,844 | 15,866 | 37,844 | 1,335.00 | 1.88 | 0.000 | reference |
| baseline | PPO | 50 | 21,360 | 15,226 | 31,360 | 848.80 | 1.38 | 0.000 | reference |
| dry | Random | 50 | 15,728 | 12,831 | 25,728 | 886.20 | 2.18 | 0.000 | -1,748 [-3,635, +139] |
| dry | Rule-based | 50 | 24,516 | 14,658 | 34,516 | 1,210.00 | 1.94 | 0.000 | -3,327 [-5,974, -681] |
| dry | PPO | 50 | 21,392 | 15,325 | 31,392 | 852.40 | 1.40 | 0.000 | +32 [-399, +464] |
| pest | Random | 50 | 9,755 | 12,139 | 19,755 | 763.80 | 5.64 | 0.000 | -7,722 [-9,847, -5,597] |
| pest | Rule-based | 50 | 16,558 | 15,555 | 26,558 | 1,017.00 | 5.14 | 0.000 | -11,286 [-13,956, -8,617] |
| pest | PPO | 50 | 14,726 | 14,480 | 24,726 | 642.80 | 3.26 | 0.000 | -6,633 [-8,236, -5,030] |
| combined | Random | 50 | 9,346 | 13,746 | 19,346 | 699.00 | 5.66 | 0.000 | -8,131 [-10,961, -5,301] |
| combined | Rule-based | 50 | 13,098 | 14,653 | 23,098 | 887.80 | 5.24 | 0.000 | -14,746 [-18,195, -11,296] |
| combined | PPO | 50 | 14,764 | 13,893 | 24,764 | 650.40 | 3.30 | 0.000 | -6,595 [-8,359, -4,831] |

*Difference = paired (stress − baseline) profit for the same agent and seed, mean [95% CI].*
