# Ablation results (real data, test = 2025-09-01 .. 2026-09-20)

| Variant | best ep | val MAE | test MAE | skill +1h | skill +6h | skill +24h | CSI +1h | 95% cov |
|---|---|---|---|---|---|---|---|---|
| G_physics_constraint | 27 | 17.62 | 28.18 | +0.060 | +0.174 | +0.103 | 0.828 | 0.93 |
| H_full_model | 27 | 17.57 | 27.78 | +0.047 | +0.205 | +0.103 | 0.822 | 0.92 |
| G2_physics_calendar | 29 | 17.69 | 27.80 | +0.057 | +0.199 | +0.104 | 0.826 | 0.94 |
| H2_full_calendar | 24 | 17.77 | 28.00 | +0.061 | +0.172 | +0.117 | 0.823 | 0.95 |

Persistence MAE per horizon (same observed hours): +1h: 13.85, +6h: 44.55, +24h: 38.70

## How to read this

* 30 epochs, hidden size 32, stride 6, one seed; the saved model is the best validation epoch.
* G2 / H2 add calendar inputs (hour, weekday, season, stubble-burning season, Diwali proximity). The calendar change
  improves G's test error by 0.38 but worsens H's by 0.22: opposite signs, about the size of seed noise (~0.3,
  see `scripts/seed_check.py`), so no calendar effect is claimed.
* Longer training is the clearest gain: H 28.53 -> 27.78, G 28.48 -> 28.18 (test MAE vs the 10-epoch runs).
* Skill = 1 - MAE_model / MAE_persistence on the same observed hours.
