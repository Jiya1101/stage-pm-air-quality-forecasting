# Ablation results (real data, test = 2025-09-01 .. 2026-09-20)

| Variant | best ep | val MAE | test MAE | skill +1h | skill +6h | skill +24h | CSI +1h | 95% cov |
|---|---|---|---|---|---|---|---|---|
| A_baseline | 10 | 18.52 | 28.69 | +0.050 | +0.170 | +0.071 | 0.830 | 0.94 |
| B_wind_graph | 7 | 18.69 | 28.50 | +0.050 | +0.182 | +0.072 | 0.832 | 0.93 |
| C_blh | 10 | 18.35 | 29.12 | +0.046 | +0.120 | +0.097 | 0.828 | 0.92 |
| D_inversion | 10 | 18.37 | 29.09 | +0.046 | +0.123 | +0.096 | 0.827 | 0.93 |
| E_external_sources | 9 | 18.53 | 28.72 | +0.046 | +0.142 | +0.102 | 0.827 | 0.92 |
| F_transport_lag | 10 | 18.37 | 28.99 | +0.045 | +0.128 | +0.098 | 0.827 | 0.92 |
| G_physics_constraint | 9 | 18.19 | 28.48 | +0.051 | +0.154 | +0.105 | 0.833 | 0.93 |
| H_full_model | 10 | 18.29 | 28.53 | +0.035 | +0.149 | +0.114 | 0.823 | 0.94 |

Persistence MAE per horizon (same observed hours): +1h: 13.85, +6h: 44.55, +24h: 38.70

## How to read this

* One seed, 10 epochs, hidden size 32, stride 6. Most variants were still improving at epoch 10.
* Skill = 1 - MAE_model / MAE_persistence on the same observed hours (persistence = last reading carried forward).
* All eight variants beat persistence at every horizon; the spread between variants is about 2% of test MAE,
  which is not yet distinguishable from seed noise (see `scripts/seed_check.py`).
* Training targets for 2022-2023 are largely AQI-derived PM2.5 estimates; the 2025-2026 validation and test
  targets are direct sensor readings. PM10/NO2/CO/O3 are mostly imputed in training and mostly real in test.
* The 2024 gap (no PM2.5 anywhere) means train and val/test are not adjacent in time.
