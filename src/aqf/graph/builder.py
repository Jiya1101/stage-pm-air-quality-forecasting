"""Multi-scale dynamic graph construction: G(t) = G_L(t) (+) G_R(t) (+) G_A(t).

  G_L(t): local intra-Delhi graph. Static distance-decay backbone modulated by
          a dynamic local wind-alignment adjacency (captures street-level
          transport between nearby stations).
  G_R(t): regional transport graph. Directed, wind-aligned, distance-decayed,
          *stability-gated* and *transport-lag-aware* edges from external
          source nodes (Punjab/Haryana/Rajasthan/W-UP) into local receptors.
  G_A(t): not a literal adjacency matrix -- a global atmospheric-state
          embedding (see models/encoders.py::AtmosphericStabilityEncoder) that
          conditions both G_L and G_R through the stability gate G(S_t).

Continuous stability gating (S_t in [0,1], learned -- see features/stability.py):
    A_ij^t = A_ij^{wind,t} * G(S_t) * T_ij^t
As S_t -> 1 (strong inversion / trapping): long-range edges attenuate, local
self-loops amplify. As S_t -> 0 (well-mixed): regional advective edges expand.
This replaces the discontinuous "BLH < 100m -> cut edges" rule Dr. Priya
flagged as reviewer-bait.

All static geometry (distances, direction unit vectors) is precomputed once
via `StaticGraph.build(...)`; only wind/stability-dependent weights are
recomputed per timestep, and that recomputation is done batched & vectorized
in torch inside `DynamicGraphBuilder` so it stays fully differentiable and
cheap even across a long training window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aqf.features.transport import (
    static_direction_matrix,
    static_distance_matrix,
    torch_probabilistic_transport_weights,
)
from aqf.graph.stations import LOCAL_STATIONS, REGIONAL_SOURCES


@dataclass
class StaticGraph:
    """Precomputed, time-invariant geometry for the tri-layer graph."""

    local_lat: np.ndarray
    local_lon: np.ndarray
    regional_lat: np.ndarray
    regional_lon: np.ndarray

    local_distance_km: np.ndarray          # (N_local, N_local)
    local_direction: np.ndarray            # (N_local, N_local, 2) i -> j
    regional_distance_km: np.ndarray       # (N_regional, N_local)
    regional_direction: np.ndarray         # (N_regional, N_local, 2) source -> receptor

    local_static_adjacency: np.ndarray     # (N_local, N_local) exp-decay, zero diagonal, row-normalized

    @property
    def n_local(self) -> int:
        return self.local_lat.shape[0]

    @property
    def n_regional(self) -> int:
        return self.regional_lat.shape[0]

    @classmethod
    def build(cls, local_sigma_km: float = 12.0) -> "StaticGraph":
        local_lat = np.array([s.lat for s in LOCAL_STATIONS])
        local_lon = np.array([s.lon for s in LOCAL_STATIONS])
        regional_lat = np.array([s.lat for s in REGIONAL_SOURCES])
        regional_lon = np.array([s.lon for s in REGIONAL_SOURCES])

        local_d = static_distance_matrix(local_lat, local_lon, local_lat, local_lon)
        local_dir = static_direction_matrix(local_lat, local_lon, local_lat, local_lon)
        regional_d = static_distance_matrix(regional_lat, regional_lon, local_lat, local_lon)
        regional_dir = static_direction_matrix(regional_lat, regional_lon, local_lat, local_lon)

        local_adj = np.exp(-(local_d ** 2) / (2.0 * local_sigma_km ** 2))
        np.fill_diagonal(local_adj, 0.0)
        row_sum = local_adj.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 1e-12, 1.0, row_sum)
        local_adj = local_adj / row_sum

        return cls(
            local_lat=local_lat,
            local_lon=local_lon,
            regional_lat=regional_lat,
            regional_lon=regional_lon,
            local_distance_km=local_d,
            local_direction=local_dir,
            regional_distance_km=regional_d,
            regional_direction=regional_dir,
            local_static_adjacency=local_adj,
        )

    def as_tensors(self, device: str = "cpu") -> "StaticGraphTensors":
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        return StaticGraphTensors(
            local_distance_km=t(self.local_distance_km),
            local_direction=t(self.local_direction),
            local_static_adjacency=t(self.local_static_adjacency),
            regional_distance_km=t(self.regional_distance_km),
            regional_direction=t(self.regional_direction),
        )


@dataclass
class StaticGraphTensors:
    local_distance_km: torch.Tensor
    local_direction: torch.Tensor
    local_static_adjacency: torch.Tensor
    regional_distance_km: torch.Tensor
    regional_direction: torch.Tensor


def stability_gate(S_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """G(S_t) applied to (off-diagonal, self-loop) edge weights.

    off_diag_scale = (1 - S_t): long-range / cross-station edges attenuate as
        the atmosphere becomes more stable/trapping.
    self_loop_scale = (1 + S_t): local self-accumulation amplifies under the
        same conditions (mass that isn't advected away stays put).
    S_t: (B,) or (B,1) in [0,1]. Returns tensors broadcastable to (B,1,1).
    """
    s = S_t.view(S_t.shape[0], 1, 1)
    return (1.0 - s), (1.0 + s)


class DynamicGraphBuilder:
    """Assembles A_L(t) and the regional transport probability P(i,s,t) for a batch of timesteps."""

    def __init__(self, static: StaticGraphTensors, graph_cfg, ablation):
        self.static = static
        self.cfg = graph_cfg
        self.ablation = ablation

    def local_adjacency(
        self,
        local_wind_unit: torch.Tensor,   # (B, N_local, 2)
        S_t: torch.Tensor,               # (B,)
    ) -> torch.Tensor:
        """Dynamic local adjacency A_L(t): (B, N_local, N_local), row i -> j."""
        static_adj = self.static.local_static_adjacency.unsqueeze(0)  # (1, N, N)

        if self.ablation.use_wind_graph:
            direction = self.static.local_direction.unsqueeze(0)  # (1, N, N, 2)
            align = (local_wind_unit.unsqueeze(2) * direction).sum(-1).clamp(min=0.0)  # (B, N, N)
            # Blend directional wind alignment with the static distance backbone so
            # the graph doesn't collapse to zero when wind is exactly calm.
            dyn_adj = static_adj * (0.4 + 0.6 * align)
        else:
            dyn_adj = static_adj.expand(local_wind_unit.shape[0], -1, -1)

        if self.ablation.use_stability_gate:
            off_scale, _self_scale = stability_gate(S_t)
            dyn_adj = dyn_adj * off_scale

        return dyn_adj

    def local_diffusion_adjacency(self, S_t: torch.Tensor) -> torch.Tensor:
        """Symmetric, wind-independent distance-decay diffusion adjacency A_diff(t): (B, N_local, N_local).

        Unlike `local_adjacency` (the directed advection operator), this
        carries no wind directionality -- it represents isotropic turbulent
        mixing between nearby stations, whose *strength* still contracts
        under high atmospheric stability (Dr. Priya's continuous-gating
        requirement) even though its *direction* is not wind-dependent.
        """
        static_adj = self.static.local_static_adjacency.unsqueeze(0).expand(S_t.shape[0], -1, -1)
        if self.ablation.use_stability_gate:
            off_scale, _ = stability_gate(S_t)
            static_adj = static_adj * off_scale
        return static_adj

    def regional_transport_weights(
        self,
        regional_wind_unit: torch.Tensor,  # (B, N_regional, 2)
        S_t: torch.Tensor,                 # (B,)
    ) -> torch.Tensor:
        """P(i, s, t): (B, N_regional, N_local) probabilistic transport corridor."""
        if not self.ablation.use_external_sources:
            b = regional_wind_unit.shape[0]
            return torch.zeros(b, self.static.regional_direction.shape[0], self.static.regional_direction.shape[1],
                                device=regional_wind_unit.device)

        if self.ablation.use_stability_gate:
            off_scale, _ = stability_gate(S_t)  # (B,1,1), regional edges expand as S->0
            gate = off_scale
        else:
            gate = torch.ones(regional_wind_unit.shape[0], 1, 1, device=regional_wind_unit.device)

        return torch_probabilistic_transport_weights(
            wind_unit_vec=regional_wind_unit,
            direction_unit_vec=self.static.regional_direction,
            distance_km=self.static.regional_distance_km,
            stability_gate=gate,
            distance_sigma_km=self.cfg.regional_distance_sigma_km,
        )
