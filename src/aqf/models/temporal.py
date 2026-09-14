"""Temporal Transformer stage: turns a per-node embedding sequence into a single context vector per node.

Follows the "Temporal Transformer/TCN" block in Dr. Priya's recommended
architecture (Dynamic Transport Encoder -> Physics-Constrained Graph Neural
Operator -> Temporal Transformer/TCN -> Multi-task output). A standard
learned positional encoding is used since PM2.5 dynamics have strong diurnal
structure the model should be able to key on directly.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int, n_heads: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, L, D) per-node embedding sequences -> (B, N, D) context at the final timestep.

        N and B are folded together to run the transformer once over all
        (batch, node) sequences; a causal mask is used since only past-window
        information should inform the summary at t0.
        """
        B, N, L, D = h.shape
        h = h.reshape(B * N, L, D) + self.pos_embedding[:, :L, :]
        causal_mask = torch.triu(torch.ones(L, L, device=h.device, dtype=torch.bool), diagonal=1)
        out = self.encoder(h, mask=causal_mask)
        out = out.reshape(B, N, L, D)
        return out[:, :, -1, :]  # context at t0 (last observed step)
