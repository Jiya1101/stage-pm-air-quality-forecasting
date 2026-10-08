# Ablation results (real data, test = 2025-09-01 .. 2026-09-20)

| Variant | best ep | val MAE | test MAE | skill +1h | skill +6h | skill +24h | CSI +1h | 95% cov |
|---|---|---|---|---|---|---|---|---|
| A_baseline | 27 | 16.66 | 26.43 | +0.120 | +0.397 | +0.118 | 0.829 | 0.93 |
| G_physics_constraint | 24 | 16.58 | 26.31 | +0.111 | +0.383 | +0.147 | 0.834 | 0.92 |
| H_full_model | 24 | 16.48 | 25.89 | +0.113 | +0.406 | +0.150 | 0.831 | 0.93 |

Persistence MAE per horizon (same observed hours): +1h: 16.02, +6h: 48.75, +24h: 40.47

## How to read this

* Trained on direct hourly CPCB station data (no AQI-derived estimates): train Jan 2022 - Feb 14 2025, validation Feb 21 - Aug 31 2025, test Sep 1 2025 - Sep 20 2026. 30 epochs, hidden size 32, stride 6, seed 0, one run per variant.
* G and H compute the same forecast (H only adds source/regime heads, which have no labels on real data and receive no gradient), yet their test MAE differs by 0.42 and their
  bias on severe hours differs a lot. That difference is training randomness, so gaps of this size between A, G and H are not evidence that one variant is better.
* The physics rule (A -> G) shifts skill from +6 h (+0.397 -> +0.383) to +24 h (+0.118 -> +0.147) with no net change in test MAE, the same pattern seen on the earlier data.
* These numbers are not comparable with `ablation_results.md` / `ablation_results_long.md`: the labels, pollutant inputs and test hours are different data.
* Skill = 1 - MAE_model / MAE_persistence on the same observed hours.
