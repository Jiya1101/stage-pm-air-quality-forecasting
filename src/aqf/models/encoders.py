"""S / T / A encoders of STAGE-PM: Source, (support for) Transport, Atmospheric stability.

The "Transport" letter of STAGE-PM is not a separate encoder here -- transport
(wind alignment, distance decay, dynamic lag, probabilistic corridors) is
geometry, computed in graph/builder.py and features/transport.py and consumed
directly by the graph operator (models/operators.py). These three modules are
the *feature* encoders that turn raw per-node measurements into the embeddings
the operator passes messages between.
"""
from __future__ import annotations

import torch
from torch import nn

from aqf.features.stability import StabilityIndex


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SourceEncoder(nn.Module):
    """S: encodes regional source-node features (FEP, wind, industry, dust) into embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = MLPEncoder(in_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, regional_seq: torch.Tensor) -> torch.Tensor:
        """regional_seq: (B, L, N_reg, F_reg) -> (B, L, N_reg, D)"""
        return self.mlp(regional_seq)


class LocalEncoder(nn.Module):
    """Encodes local receptor-station features into the initial node embedding for the operator."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = MLPEncoder(in_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, local_seq: torch.Tensor) -> torch.Tensor:
        """local_seq: (B, L, N_local, F_local) -> (B, L, N_local, D)"""
        return self.mlp(local_seq)


class AtmosphericEncoder(nn.Module):
    """A: continuous stability index S_t plus a global atmospheric-state embedding g_A(t) (the G_A graph layer).

    G_A is not a literal adjacency matrix here -- per the architecture, it
    acts as a global conditioning filter on G_L and G_R (via S_t, consumed by
    graph/builder.py::stability_gate) and is also concatenated into the
    operator's update step so the network can condition on atmospheric state
    beyond just the scalar stability index (e.g. distinguishing a dust-driven
    vs biomass-driven high-S_t episode).
    """

    def __init__(self, atmos_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.stability = StabilityIndex(hidden_dim=16)
        self.context_mlp = MLPEncoder(atmos_dim, hidden_dim, hidden_dim, dropout)

    def forward(self, atmos_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """atmos_seq: (B, L, F_atmos) with first 5 cols = [BLH, inversion, RH, WS, SR].

        Returns (S_t: (B, L, 1), g_A: (B, L, D)).
        """
        S_t = self.stability(atmos_seq[..., :5])
        g_A = self.context_mlp(atmos_seq)
        return S_t, g_A
