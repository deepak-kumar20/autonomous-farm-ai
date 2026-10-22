# Table 3 — Agent configurations

| Agent | Policy type | Training status | Checkpoint or configuration | Action space | Evaluation episodes | Mean profit (₹, full framework) | Invalid-action rate without filter |
|---|---|---|---|---|---|---|---|
| Random | Uniform over own candidate set | None (no learning) | seed = scenario seed | Discrete(24), shared with human | 50 | 17,477 | 0.000 |
| Rule-based | Hand-written priority rules | None (no learning) | agents/rule_based_agent.py | Discrete(24), shared with human | 50 | 27,844 | 0.000 |
| PPO | PPO, MLP 128-128, greedy (argmax) ranking | Trained 2,002,944 steps (seed 7) | models/ppo/farm_ppo_v2.zip | Discrete(24), shared with human | 50 | 21,360 | 0.060 |

*Mean profit from Experiment A (baseline scenarios, full framework). Invalid-action rate = invalid executed actions / actions attempted, from Experiment B's no-filter arm (baseline).*
