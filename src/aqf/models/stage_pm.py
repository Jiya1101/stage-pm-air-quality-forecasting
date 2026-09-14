"""STAGE-PM: Source -> Transport -> Atmospheric stability -> Graph -> Explainable operator.

Assembles every piece:
  S: SourceEncoder over regional (Punjab/Haryana/Rajasthan/W-UP) node features
  T: dynamic transport lag + probabilistic corridors (features/transport.py, graph/builder.py)
  A: AtmosphericEncoder -> continuous stability index S_t + global context g_A
  G: DynamicGraphBuilder -> A_adv(t), A_diff(t), P(i,s,t) -- the tri-layer G_L/G_R/G_A graph
  E: stacked PhysicsOperatorLayer (decoupled advection/diffusion/reaction) + TemporalTransformer
  Multi-task heads: point forecast + uncertainty, exceedance probability,
     source contribution, event-aware regime.

`forward` also returns the raw ingredients needed for the physics-constrained
regularization loss (A_adv, A_diff, S_t, scalar external injection) so
losses/physics.py never has to recompute graph geometry.
"""
from __future__ import annotations

import torch
from torch import nn

from aqf.config import AblationFlags, GraphConfig, ModelConfig
from aqf.data.dataset import PM25_IDX
from aqf.graph.builder import DynamicGraphBuilder, StaticGraph
from aqf.models.encoders import AtmosphericEncoder, LocalEncoder, SourceEncoder
from aqf.models.heads import ForecastHead, RegimeHead, SourceContributionHead
from aqf.models.operators import PhysicsOperatorLayer, ScalarADRPhysics, lagged_regional_message
from aqf.models.temporal import TemporalTransformer


class StagePM(nn.Module):
    def __init__(
        self,
        n_local_features: int,
        n_regional_features: int,
        n_atmos_features: int,
        n_horizons: int,
        model_cfg: ModelConfig,
        graph_cfg: GraphConfig,
        ablation: AblationFlags,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = model_cfg
        self.graph_cfg = graph_cfg
        self.ablation = ablation
        D = model_cfg.hidden_dim

        self.static_graph = StaticGraph.build(local_sigma_km=graph_cfg.local_distance_sigma_km)
        self.static_tensors = self.static_graph.as_tensors(device=device)
        self.graph_builder = DynamicGraphBuilder(self.static_tensors, graph_cfg, ablation)

        self.source_encoder = SourceEncoder(n_regional_features, D, model_cfg.dropout)
        self.local_encoder = LocalEncoder(n_local_features, D, model_cfg.dropout)
        self.atmos_encoder = AtmosphericEncoder(n_atmos_features, D, model_cfg.dropout)

        self.operator_layers = nn.ModuleList(
            [PhysicsOperatorLayer(D, model_cfg.dropout) for _ in range(model_cfg.n_operator_layers)]
        )
        self.temporal = TemporalTransformer(D, model_cfg.n_transformer_layers, model_cfg.n_transformer_heads, dropout=model_cfg.dropout)

        self.forecast_head = ForecastHead(D, n_horizons)
        self.source_head = SourceContributionHead(D, model_cfg.n_source_categories) if ablation.use_source_head else None
        self.regime_head = RegimeHead(D, model_cfg.n_regimes) if ablation.use_regime_head else None

        self.scalar_physics = ScalarADRPhysics()

    def _masked_atmos(self, atmos_seq: torch.Tensor) -> torch.Tensor:
        """Ablation: zero out BLH / inversion-strength columns when their flag is off, per config.DataConfig ATMOS_FEATURE_COLS order."""
        if self.ablation.use_blh and self.ablation.use_inversion:
            return atmos_seq
        atmos_seq = atmos_seq.clone()
        if not self.ablation.use_blh:
            atmos_seq[..., 0] = 1500.0  # neutral BLH (well-mixed default) so it contributes no trapping signal
        if not self.ablation.use_inversion:
            atmos_seq[..., 1] = 0.0  # neutral inversion strength
        return atmos_seq

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        local_seq = batch["local_seq"]              # (B, L, N_local, F_local)
        regional_seq = batch["regional_seq"]          # (B, L, N_reg, F_reg)
        atmos_seq = self._masked_atmos(batch["atmos_seq"])  # (B, L, F_atmos)
        local_wind_unit = batch["local_wind_unit_seq"]      # (B, L, N_local, 2)
        regional_wind_unit = batch["regional_wind_unit_seq"]  # (B, L, N_reg, 2)
        regional_wind_speed = batch["regional_wind_speed_seq"]  # (B, L, N_reg)

        B, L, N_local, _ = local_seq.shape
        N_reg = regional_seq.shape[2]

        S_t, g_A = self.atmos_encoder(atmos_seq)  # (B, L, 1), (B, L, D)
        S_t_flat = S_t.reshape(B * L)

        h_R = self.source_encoder(regional_seq)   # (B, L, N_reg, D)
        h_L = self.local_encoder(local_seq)        # (B, L, N_local, D)

        A_adv = self.graph_builder.local_adjacency(
            local_wind_unit.reshape(B * L, N_local, 2), S_t_flat
        ).reshape(B, L, N_local, N_local)
        A_diff = self.graph_builder.local_diffusion_adjacency(S_t_flat).reshape(B, L, N_local, N_local)
        transport_p = self.graph_builder.regional_transport_weights(
            regional_wind_unit.reshape(B * L, N_reg, 2), S_t_flat
        ).reshape(B, L, N_reg, N_local)

        if self.ablation.use_transport_lag:
            reg_wind_speed_for_lag = regional_wind_speed
        else:
            # ablate lag: force tau -> 0 by making wind "infinitely fast" (near-instant transport)
            reg_wind_speed_for_lag = torch.full_like(regional_wind_speed, self.graph_cfg.max_wind_speed_ms)

        m_R_embed = lagged_regional_message(
            h_R, transport_p, self.static_tensors.regional_distance_km, reg_wind_speed_for_lag,
            self.graph_cfg.min_wind_speed_ms, self.graph_cfg.max_wind_speed_ms, self.graph_cfg.max_lag_hours,
        )  # (B, L, N_local, D)

        h = h_L
        for layer in self.operator_layers:
            h = layer(h, A_adv, A_diff, m_R_embed, g_A)

        h_seq = h.permute(0, 2, 1, 3)  # (B, N_local, L, D)
        z = self.temporal(h_seq)        # (B, N_local, D)

        out = self.forecast_head(z)
        if self.source_head is not None:
            out["source_contrib"] = self.source_head(z)
        if self.regime_head is not None:
            out["regime_logits"] = self.regime_head(z)

        # ---- physics-loss ingredients (scalar field, evaluated on the observed window) ----
        fep_raw = regional_seq[..., 0:1]  # (B, L, N_reg, 1) fire emission proxy channel
        if self.ablation.use_transport_lag:
            fep_speed_for_lag = regional_wind_speed
        else:
            fep_speed_for_lag = torch.full_like(regional_wind_speed, self.graph_cfg.max_wind_speed_ms)
        external_scalar = lagged_regional_message(
            fep_raw, transport_p, self.static_tensors.regional_distance_km, fep_speed_for_lag,
            self.graph_cfg.min_wind_speed_ms, self.graph_cfg.max_wind_speed_ms, self.graph_cfg.max_lag_hours,
        ).squeeze(-1)  # (B, L, N_local)

        C = local_seq[..., PM25_IDX]  # (B, L, N_local) observed PM2.5
        physics_terms = self.scalar_physics(C, A_adv, A_diff, external_scalar)
        physics_terms["S_t"] = S_t.squeeze(-1)  # (B, L)

        out["_physics"] = physics_terms
        out["_S_t_final"] = S_t[:, -1, 0]  # (B,) stability at t0, useful for diagnostics
        return out
