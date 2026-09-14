"""Ablation experiments A-H, and the three formal hypotheses they test.

Dr. Priya's explicit instruction: "Do not simply compare your model against
LSTM, GRU and GAT. Your research needs ablation experiments that prove each
scientific hypothesis." This module encodes exactly the ladder she specified:

  A. Baseline          -- historical PM2.5 + meteorology only
  B. + wind graph       -- directional wind-projected local edges
  C. + BLH              -- boundary layer height feeds the stability gate
  D. + inversion         -- inversion strength added to the stability gate
  E. + external sources  -- Punjab/Haryana/Rajasthan/W-UP source nodes activate
  F. + transport lag     -- tau = d / U_eff (source influence is no longer instantaneous)
  G. + physics constraint -- ADR/mass/stability regularization losses activate
  H. Full model          -- everything on (source attribution + regime head too)

Three testable hypotheses this ladder is designed to isolate:

  H1 (stability):  C,D vs B -- incorporating atmospheric stability into dynamic
                    graph construction significantly improves forecasting
                    during shallow-boundary-layer (high S_t) episodes.
  H2 (transport):   F vs E -- explicit source-to-receptor transport pathways
                    and dynamic transport lag improve prediction during
                    regional pollution episodes (evaluate on biomass_burning
                    regime subset, see evaluation/evaluate.py).
  H3 (physics):     G,H vs F -- physics-constrained graph propagation improves
                    generalization (spatial holdout) and extreme-event
                    forecasting vs a purely data-driven STGNN.
"""
from __future__ import annotations

import copy

from aqf.config import AblationFlags, Config

EXPERIMENTS: dict[str, AblationFlags] = {
    "A_baseline": AblationFlags(
        use_wind_graph=False, use_blh=False, use_inversion=False, use_external_sources=False,
        use_transport_lag=False, use_physics_loss=False, use_stability_gate=False,
        use_source_head=False, use_regime_head=False,
    ),
    "B_wind_graph": AblationFlags(
        use_wind_graph=True, use_blh=False, use_inversion=False, use_external_sources=False,
        use_transport_lag=False, use_physics_loss=False, use_stability_gate=False,
        use_source_head=False, use_regime_head=False,
    ),
    "C_blh": AblationFlags(
        use_wind_graph=True, use_blh=True, use_inversion=False, use_external_sources=False,
        use_transport_lag=False, use_physics_loss=False, use_stability_gate=True,
        use_source_head=False, use_regime_head=False,
    ),
    "D_inversion": AblationFlags(
        use_wind_graph=True, use_blh=True, use_inversion=True, use_external_sources=False,
        use_transport_lag=False, use_physics_loss=False, use_stability_gate=True,
        use_source_head=False, use_regime_head=False,
    ),
    "E_external_sources": AblationFlags(
        use_wind_graph=True, use_blh=True, use_inversion=True, use_external_sources=True,
        use_transport_lag=False, use_physics_loss=False, use_stability_gate=True,
        use_source_head=False, use_regime_head=False,
    ),
    "F_transport_lag": AblationFlags(
        use_wind_graph=True, use_blh=True, use_inversion=True, use_external_sources=True,
        use_transport_lag=True, use_physics_loss=False, use_stability_gate=True,
        use_source_head=False, use_regime_head=False,
    ),
    "G_physics_constraint": AblationFlags(
        use_wind_graph=True, use_blh=True, use_inversion=True, use_external_sources=True,
        use_transport_lag=True, use_physics_loss=True, use_stability_gate=True,
        use_source_head=False, use_regime_head=False,
    ),
    "H_full_model": AblationFlags(
        use_wind_graph=True, use_blh=True, use_inversion=True, use_external_sources=True,
        use_transport_lag=True, use_physics_loss=True, use_stability_gate=True,
        use_source_head=True, use_regime_head=True,
    ),
}


def build_experiment_configs(base_cfg: Config) -> dict[str, Config]:
    configs = {}
    for name, flags in EXPERIMENTS.items():
        cfg = copy.deepcopy(base_cfg)
        cfg.name = name
        cfg.ablation = flags
        configs[name] = cfg
    return configs


def run_suite(base_cfg: Config, verbose: bool = True) -> dict[str, dict]:
    from aqf.training.train import run  # local import: avoid circular import at module load time

    results = {}
    for name, cfg in build_experiment_configs(base_cfg).items():
        if verbose:
            print(f"\n=== Running experiment {name} ===")
        results[name] = run(cfg, verbose=verbose)
    return results
