# Table 10 — Feasibility-filter performance (Experiment F, all agents and profiles pooled)

| Filter mode | Episodes | Actions evaluated | Accepted actions | Rejected actions | Rejection rate | Completion rate | Prediction metric | Mean filter time (ms/episode) |
|---|---|---|---|---|---|---|---|---|
| No feasibility filter | 600 | 18000 | 18000 | 0 | 0.000 | 61.0% | n/a (no probability used) | 0.03 |
| Full framework (LFM + deterministic filter) | 600 | 23162 | 18000 | 5162 | 0.223 | 66.7% | n/a (no probability used) | 0.16 |
| Uncertainty filter (fixed τ = 0.8) | 600 | 139731 | 18000 | 121731 | 0.871 | 80.5% | Brier 0.106 (climatology 0.107, n=938) | 7.17 |
| Uncertainty filter (break-even τ) | 600 | 33312 | 18000 | 15312 | 0.460 | 70.8% | Brier 0.168 (climatology 0.173, n=5616) | 1.38 |

*Rejection rate = rejected candidates / candidates evaluated by the filter (not / days). Completion rate = crops harvested / crops planted. Brier score = mean (p − y)², y = crop was present and harvest-eligible on or before day 30; reported only for modes that use the probability.*
- Random: Full framework (LFM + deterministic filter) − no filter profit = +4,878 [+4,184, +5,571] ₹ (all four profiles pooled)
- Rule-based: Full framework (LFM + deterministic filter) − no filter profit = +0 [+0, +0] ₹ (all four profiles pooled)
- PPO: Full framework (LFM + deterministic filter) − no filter profit = +228 [-118, +575] ₹ (all four profiles pooled)
- Random: Uncertainty filter (fixed τ = 0.8) − no filter profit = -3,984 [-5,462, -2,506] ₹ (all four profiles pooled)
- Rule-based: Uncertainty filter (fixed τ = 0.8) − no filter profit = -19,427 [-21,679, -17,175] ₹ (all four profiles pooled)
- PPO: Uncertainty filter (fixed τ = 0.8) − no filter profit = -11,714 [-13,567, -9,861] ₹ (all four profiles pooled)
- Random: Uncertainty filter (break-even τ) − no filter profit = +5,176 [+4,433, +5,920] ₹ (all four profiles pooled)
- Rule-based: Uncertainty filter (break-even τ) − no filter profit = -1,721 [-2,735, -708] ₹ (all four profiles pooled)
- PPO: Uncertainty filter (break-even τ) − no filter profit = +116 [-294, +527] ₹ (all four profiles pooled)
