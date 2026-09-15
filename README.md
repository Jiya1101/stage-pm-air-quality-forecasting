# Air Quality Forecasting — STAGE-PM

Physics-constrained, source-aware, multi-scale dynamic transport graph network for
PM2.5 forecasting in Delhi-NCR.



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

## Implementation status

**Engineering: complete.** Every architectural component Dr. Priya's review
called for is implemented, unit-tested, and verified against real API
responses (not just synthetic data) — the sections below list exactly what
was verified and how.

| Component | Status | Evidence |
|---|---|---|
| STAGE-PM model (S/T/A encoders, physics-constrained graph operator, temporal transformer, multi-task heads) | ✅ Done | `tests/test_pipeline.py` passes; forward pass verified on synthetic + real data |
| Physics-constrained losses (transport residual, mass conservation, stability penalty) | ✅ Done | Gradient flow verified; loss decreases monotonically across training |
| Multi-scale graph `G_L ⊕ G_R ⊕ G_A`, continuous stability gating, dynamic transport lag | ✅ Done | Unit-tested against known geometry (haversine, bearing) |
| Ablation framework (Experiments A→H, testing H1/H2/H3) | ✅ Done, not yet **run** at scale | Code + configs complete; only smoke-tested so far |
| Real data pipeline: OpenAQ + FIRMS + ERA5 + opencity.in, with proper observed-vs-imputed loss masking | ✅ Done, live-verified | See "Connecting real data" below — every source pulled real values and cross-checked for physical plausibility |
| Synthetic simulator (wind-transported, lagged, stability-gated plumes) | ✅ Done | Used to validate the architecture before any real data access existed |
| Evaluation framework (per-horizon, per-severity-bucket, per-regime, spatial holdout) | ✅ Done | Implemented and exercised end-to-end |
| **Full historical training run + ablation suite results** | ⏳ In progress | This is the actual research deliverable — see below |
| `industry_index` / `dust_index` / `traffic_index` features | ⚠️ Placeholder | No free real-data source identified yet for these three secondary features |

**What's genuinely left is the experiment, not the code.** A 2018-2023
historical data pull (OpenAQ + opencity.in-blended PM2.5, real NASA FIRMS
fire history, real ERA5 atmospheric stability data, all 15 Delhi stations)
is running now to feed the first full training run and the A→H ablation
suite that will actually test hypotheses H1 (stability), H2 (transport),
and H3 (physics constraint) — the scientific claims this architecture
exists to validate.

### Real data, verified live (not assumed)

Every real-data source below was pulled and sanity-checked against known
physical ranges before being trusted, and two real, non-obvious bugs were
found and fixed in the process (see git log for details): OpenAQ's Delhi
archive has a platform-wide gap from ~Feb 2018 to Feb 2025 for most
stations (worked around via opencity.in's independent 2017-2023 archive),
and NASA FIRMS' near-real-time product only retains ~2.5 months of history
(worked around by switching to its Standard-Processing product for
historical dates).

- **OpenAQ**: real Delhi station list, live PM2.5/PM10/NO2/CO/O3 + met pulled successfully
- **NASA FIRMS**: real fire detections pulled for peak stubble-burning season (Nov 2019) — Punjab showed mean FRP ~53, correctly dwarfing Haryana/Rajasthan/UP
- **ERA5**: real BLH (115–1516m, mean 526m) and RH (51–89%) pulled for Delhi — both physically correct for the season tested
- **opencity.in**: real hourly AQI 2017-2023 for all 15 stations, zero API key required

### Model, runnable today

The model trains end-to-end right now — on synthetic data (no setup needed)
or on the real 21-day window already validated in this repo's history:

```bash
python scripts/generate_synthetic_data.py && python scripts/train.py --config full
```

## Quickstart

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py          # builds data/synthetic/
python scripts/train.py --config full               # trains the full STAGE-PM model
python scripts/train.py --ablation-suite            # runs experiments A through H
python scripts/evaluate.py --checkpoint runs/full/best.pt
```

## Connecting real data

| Source | Gives us | What you need | Where |
|---|---|---|---|
| **OpenAQ** | Real multi-pollutant concentrations (PM2.5/PM10/NO2/CO/O3 in real ug/m3) + met variables, where it has coverage | Free API key at [explore.openaq.org/register](https://explore.openaq.org/register) | [src/aqf/data/openaq.py](src/aqf/data/openaq.py) |
| **data.opencity.in** | Real hourly AQI 2017-2023, zero gaps, exact match to all 15 `LOCAL_STATIONS` — the historical PM2.5 backbone | **Nothing** — no key, no login | [src/aqf/data/opencity.py](src/aqf/data/opencity.py) |
| ERA5 (BLH, wind, inversion, RH, radiation) | The atmospheric-stability inputs the transport graph depends on | Copernicus CDS API key (`~/.cdsapirc`) | [src/aqf/data/era5.py](src/aqf/data/era5.py) |
| NASA FIRMS | Real VIIRS fire detections for the Fire Emission Proxy | FIRMS MAP_KEY (free) | [src/aqf/data/firms.py](src/aqf/data/firms.py) |
| CPCB / data.gov.in | Live current-snapshot AQI only (not a historical archive — see cpcb.py docstring) | `data.gov.in` API key (via MeriPehchaan SSO) | [src/aqf/data/cpcb.py](src/aqf/data/cpcb.py) |

**Important, discovered by actually pulling the data rather than assuming:**
OpenAQ's Delhi archive has a real, verified platform-wide gap — every
station's original sensor generation stopped within days of Feb 2018, and a
fresh one only started (for literally every station, same exact date) on
2025-02-18. So OpenAQ alone cannot provide 2018-2023 history. The pipeline
therefore blends both sources: **opencity.in supplies the gap-free
2017-2023 PM2.5 backbone** (via `opencity.approximate_pm25_from_aqi()`,
which inverts CPCB's official AQI breakpoint table — a documented estimate,
not a direct concentration reading), and **OpenAQ supplies real
multi-pollutant concentrations** wherever it actually has coverage (notably
the current/recent window, Feb 2025 onward, plus a couple of stations with
partial 2018-2022 bridging). `data/real_pipeline.py::assemble_real_raw_series`
tracks exactly which reading came from which source, and which hours have no
real signal at all, in `local_observed_mask` — training and evaluation only
score genuinely-observed hours (see `losses/physics.py::forecast_loss`).

Copy `.env.example` to `.env` and fill in `OPENAQ_API_KEY`/`CPCB_API_KEY`/
`FIRMS_MAP_KEY` (auto-loaded by every script — see `scripts/_pathfix.py`);
ERA5 is the exception, its key goes in `~/.cdsapirc` instead (that's the only
file `cdsapi` reads). Then set `config.data.source = "real"` and run
`python scripts/fetch_real_data.py --start 2018-01-01 --end 2023-12-31`.

## Project layout

```
src/aqf/
  features/     stability index, transport lag, fire emission proxy
  graph/        station registry, multi-scale graph builder (G_L, G_R, G_A)
  data/         openaq.py, opencity.py, cpcb.py, era5.py, firms.py (real sources), real_pipeline.py (assembly),
                synthetic.py (simulator), dataset.py (windowing), schema.py (shared RawSeries format)
  models/       encoders, physics-constrained operator, temporal transformer, multi-task heads, stage_pm.py
  losses/       physics-constrained regularization loss
  training/     train.py, ablation.py (experiments A-H)
  evaluation/   metrics.py (incl. per-regime + spatial holdout), evaluate.py
```
