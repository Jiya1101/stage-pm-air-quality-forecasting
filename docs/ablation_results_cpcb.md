# Ablation results on direct CPCB data (test = 2025-09-01 .. 2026-09-20)

All eight variants A-H trained on the real CPCB hourly data (no AQI-derived estimates): train Jan 2022 - Feb 14 2025, validation
Feb 21 - Aug 31 2025, test Sep 1 2025 - Sep 20 2026. 30 epochs, hidden size 32, stride 6, seed 0, **one run per variant**.

| Variant | best ep | val MAE | test MAE | skill +1h | skill +6h | skill +24h | CSI +1h | 95% cov |
|---|---|---|---|---|---|---|---|---|
| A_baseline | 27 | 16.66 | 26.43 | +0.120 | +0.397 | +0.118 | 0.829 | 0.93 |
| B_wind_graph | 27 | 16.58 | 26.44 | +0.121 | +0.396 | +0.118 | 0.829 | 0.93 |
| C_blh | 27 | 16.59 | 26.46 | +0.117 | +0.393 | +0.121 | 0.830 | 0.93 |
| D_inversion | 21 | 16.59 | 25.66 | +0.121 | +0.408 | +0.161 | 0.834 | 0.92 |
| E_external_sources | 27 | 16.59 | 26.73 | +0.117 | +0.395 | +0.099 | 0.831 | 0.92 |
| F_transport_lag | 27 | 16.63 | 26.28 | +0.117 | +0.396 | +0.130 | 0.830 | 0.92 |
| G_physics_constraint | 24 | 16.58 | 26.31 | +0.111 | +0.383 | +0.147 | 0.834 | 0.92 |
| H_full_model | 24 | 16.48 | 25.89 | +0.113 | +0.406 | +0.150 | 0.831 | 0.93 |

Persistence (last reading carried forward) MAE on the same observed hours: +1h 16.02, +6h 48.75, +24h 40.47.

Model MAE per horizon (+1h / +6h / +24h), and MAE on the 13 training stations vs the 2 held-out stations:

| Variant | +1h | +6h | +24h | training stations | held-out stations |
|---|---|---|---|---|---|
| A | 14.09 | 29.40 | 35.70 | 26.13 | 28.19 |
| B | 14.08 | 29.46 | 35.69 | 26.13 | 28.30 |
| C | 14.15 | 29.58 | 35.56 | 26.20 | 28.02 |
| D | 14.08 | 28.87 | 33.96 | 25.43 | 27.06 |
| E | 14.15 | 29.51 | 36.46 | 26.43 | 28.58 |
| F | 14.14 | 29.43 | 35.20 | 26.03 | 27.79 |
| G | 14.24 | 30.07 | 34.53 | 26.07 | 27.75 |
| H | 14.21 | 28.98 | 34.39 | 25.62 | 27.52 |

## How to read this

* Validation error is flat across all eight variants (16.48 - 16.66), so only the test year separates them, and there the spread is about 4% (25.66 - 26.73).
* **H1 (stability):** adding boundary layer (B -> C) changes nothing (26.44 -> 26.46). Adding inversion strength (C -> D) is the largest single step (26.46 -> 25.66, +24h skill
  12.1% -> 16.1%) but validation is unchanged and E, F, G, H, which all contain the inversion feature, do not keep the gain consistently. A possible +24h effect, not established.
* **H2 (transport):** adding external sources (D -> E) made things worse (25.66 -> 26.73; +24h skill 16.1% -> 9.9%), and the transport lag (E -> F) won back part of it (26.28).
  Net D -> F is 0.62 worse. Not supported; a weak test, because industry and dust inputs are placeholders.
* **H3 (physics):** F -> G is flat overall (26.28 -> 26.31) with +24h skill 13.0% -> 14.7% and +6h skill 39.6% -> 38.3%: the same trade between horizons seen on the earlier data.
* Run-to-run noise is poorly measured: the baseline H with another seed differs by 0.01 in test MAE, but G and H (same forecast function, different random stream) differ by 0.42 overall and by over 1 at +6h.
  Steps of 0.4 - 1.1 in test error go up and down with no consistent direction. With one run per variant, none of the differences above is a conclusion; three seeds per variant would be needed.
* The held-out stations are consistently about 6-8% worse than the training stations for every variant.
* These numbers are not comparable with `ablation_results.md` / `ablation_results_long.md`: the labels, pollutant inputs and test hours are different data.
* Skill = 1 - MAE_model / MAE_persistence on the same observed hours.
