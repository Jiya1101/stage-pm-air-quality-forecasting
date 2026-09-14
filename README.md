# Air Quality Forecasting — STAGE-PM

Physics-constrained, source-aware, multi-scale dynamic transport graph network for
PM2.5 forecasting in Delhi-NCR.

This implements the **STAGE-PM** architecture (Source → Transport → Atmospheric
stability → Graph → Explainable operator) as scoped in
[`docs/DrPriyaVSuggestions.pdf`](docs) and cross-referenced against the 15
foundational papers summarized in `docs/AirQualityForecastingResearchPapers.pdf`.
The core idea: don't model PM2.5 as station-to-station correlation. Model it as a
**source → transport → stability-controlled dispersion → receptor accumulation**
physical process, with a GNN as the learned implementation of that process, not the
invention itself.

## Why this is not "just another STGNN"

Conventional dynamic-graph PM2.5 models (wind-projected edges, GAT/GRU stacks,
binary BLH cutoffs) are no longer novel on their own — see `docs/` for citations.
The upgrades implemented here:

1. **Continuous Atmospheric Stability Index** `S_t ∈ [0,1]` (learned, from BLH,
   inversion strength, RH, wind speed, solar radiation) replaces the hard
   "BLH < 100m" rule. [`src/aqf/features/stability.py`](src/aqf/features/stability.py)
2. **Dynamic transport lag** `τ(s,i,t) = d(s,i) / U_eff(t)` — pollution from a
   Punjab fire does not teleport to Delhi; it arrives hours later, gated by wind
   speed and direction alignment. [`src/aqf/features/transport.py`](src/aqf/features/transport.py)
3. **Probabilistic transport corridors** — a source influences a *distribution*
   over receptor stations, not one deterministic wind-blows-there edge.
4. **Fire Emission Proxy (FEP)** — FIRMS fire counts are never treated as
   emissions; they're converted to an FRP/confidence/area-weighted activity index.
   [`src/aqf/features/fire_proxy.py`](src/aqf/features/fire_proxy.py)
5. **Tri-layer multi-scale graph**: `G(t) = G_L(t) ⊕ G_R(t) ⊕ G_A(t)` — local
   intra-Delhi graph, regional transboundary transport graph (Punjab/Haryana/
   Rajasthan/W-UP source nodes), and a global atmospheric-state conditioning
   graph. [`src/aqf/graph/builder.py`](src/aqf/graph/builder.py)
6. **Physics-constrained neural operator** — message passing is decomposed into
   decoupled asymmetric **advection** and symmetric **diffusion** operators plus a
   local **reaction** term, instead of generic attention.
   [`src/aqf/models/operators.py`](src/aqf/models/operators.py)
7. **Physics regularization loss**: `L = L_forecast + λ1·L_transport +
   λ2·L_mass + λ3·L_stability` — an ADR-residual penalty, an open-boundary mass
   conservation penalty, and a stability-consistency penalty.
   [`src/aqf/losses/physics.py`](src/aqf/losses/physics.py)
8. **Multi-task output**: point forecasts (+1h/+6h/+24h), exceedance probability,
   a **source contribution head** (traffic/industry/biomass/dust/background, unit
   sum), a calibrated uncertainty interval, and an **event-aware regime
   classifier** (Normal / Inversion / Biomass Burning / Dust).
9. **Ablation experiments A→H** that isolate each hypothesis (wind graph → BLH →
   inversion → external sources → transport lag → physics constraint → full
   model), not just "vs LSTM/GRU/GAT". [`src/aqf/training/ablation.py`](src/aqf/training/ablation.py)

Three testable hypotheses (H1 stability, H2 transport, H3 physics) are documented
alongside the ablation configs — see `src/aqf/training/ablation.py`.

## Status

Data access (CPCB/data.gov.in, CDS API for ERA5, NASA FIRMS) needs credentials this
environment doesn't have. So:

- The **model, physics losses, graph construction, and training/ablation
  pipeline are fully implemented and runnable now** against a physically
  plausible **synthetic simulator** (`src/aqf/data/synthetic.py`) that bakes in
  wind-transported, lagged, stability-gated plumes — this is what lets you verify
  the architecture actually learns transport lag and stability gating before any
  real data is available.
- The **real data-source clients** (`src/aqf/data/cpcb.py`, `era5.py`,
  `firms.py`) are implemented against their real APIs but need you to supply
  credentials (see "Connecting real data" below). Swapping synthetic → real data
  requires no model changes — both produce the same `RawSeries` schema.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py          # builds data/synthetic/
python scripts/train.py --config full               # trains the full STAGE-PM model
python scripts/train.py --ablation-suite            # runs experiments A through H
python scripts/evaluate.py --checkpoint runs/full/best.pt
```

## Connecting real data

| Source | What you need | Where |
|---|---|---|
| CPCB / CAAQMS | `data.gov.in` API key (free registration) | [src/aqf/data/cpcb.py](src/aqf/data/cpcb.py) |
| ERA5 (BLH, wind, inversion, RH, radiation) | Copernicus CDS API key (`~/.cdsapirc`) | [src/aqf/data/era5.py](src/aqf/data/era5.py) |
| NASA FIRMS (VIIRS fire detections) | FIRMS MAP_KEY (free) | [src/aqf/data/firms.py](src/aqf/data/firms.py) |

Set the keys via environment variables (`CPCB_API_KEY`, `CDS_API_KEY`,
`FIRMS_MAP_KEY`) or the `.cdsapirc` file CDS expects, then set
`config.data.source = "real"` instead of `"synthetic"`.

## Project layout

```
src/aqf/
  features/     stability index, transport lag, fire emission proxy
  graph/        station registry, multi-scale graph builder (G_L, G_R, G_A)
  data/         cpcb.py, era5.py, firms.py (real), synthetic.py (simulator), dataset.py (windowing)
  models/       encoders, physics-constrained operator, temporal transformer, multi-task heads, stage_pm.py
  losses/       physics-constrained regularization loss
  training/     train.py, ablation.py (experiments A-H)
  evaluation/   metrics.py (incl. per-regime + spatial holdout), evaluate.py
```
