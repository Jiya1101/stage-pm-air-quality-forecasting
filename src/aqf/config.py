"""Central configuration for STAGE-PM.

Plain dataclasses rather than YAML: keeps the project dependency-free and makes
ablation configs (src/aqf/training/ablation.py) easy to express as small
overrides of a single base config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AblationFlags:
    """Toggles for experiments A-H (see training/ablation.py).

    Each flag disables one physical mechanism so its marginal contribution to
    forecast skill can be isolated, per Dr. Priya's requirement to not just
    compare against LSTM/GRU/GAT baselines.
    """

    use_wind_graph: bool = True       # directional wind-projected edges (Exp B+)
    use_blh: bool = True              # boundary-layer height feature (Exp C+)
    use_inversion: bool = True        # thermal inversion strength feature (Exp D+)
    use_external_sources: bool = True  # Punjab/Haryana/Rajasthan/W-UP source nodes (Exp E+)
    use_transport_lag: bool = True    # dynamic transport lag tau = d / U_eff (Exp F+)
    use_physics_loss: bool = True     # ADR + mass + stability regularization (Exp G+)
    use_stability_gate: bool = True   # continuous S_t gating of edges (full model refinement)
    use_source_head: bool = True      # source contribution multi-task head
    use_regime_head: bool = True      # event-aware regime classifier


@dataclass
class DataConfig:
    source: Literal["synthetic", "real"] = "synthetic"
    synthetic_dir: str = "data/synthetic"
    real_dir: str = "data/real"
    start_year: int = 2018
    train_end_year: int = 2021   # 2018-2021 -> train
    val_year: int = 2022         # 2022 -> val
    test_year: int = 2023        # 2023 -> test
    lookback_hours: int = 48
    horizons_hours: tuple = (1, 6, 24)
    exceedance_threshold: float = 200.0  # CPCB "severe" PM2.5 threshold (ug/m3)
    spatial_holdout_frac: float = 0.15   # fraction of local stations held out entirely


@dataclass
class GraphConfig:
    local_distance_sigma_km: float = 12.0     # local diffusion length scale
    regional_distance_sigma_km: float = 220.0  # regional plume spread length scale
    max_wind_speed_ms: float = 20.0
    min_wind_speed_ms: float = 0.3            # floor to avoid tau -> inf
    max_lag_hours: int = 48                    # clip transport lag to lookback window


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    n_operator_layers: int = 3
    n_transformer_layers: int = 2
    n_transformer_heads: int = 4
    dropout: float = 0.1
    n_source_categories: int = 5   # traffic, industry, biomass, dust, background
    n_regimes: int = 4             # Normal, Inversion, Biomass Burning, Dust


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_transport: float = 0.1
    lambda_mass: float = 0.05
    lambda_stability: float = 0.05
    lambda_source: float = 0.2
    lambda_regime: float = 0.1
    lambda_nll: float = 0.05
    grad_clip: float = 5.0
    device: str = "cpu"
    seed: int = 0
    out_dir: str = "runs"


@dataclass
class Config:
    name: str = "full"
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ablation: AblationFlags = field(default_factory=AblationFlags)
