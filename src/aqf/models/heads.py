"""Multi-task output heads.

Outputs (per Dr. Priya's spec, "don't only predict PM2.5 = 250, predict why"):
  1-3. PM2.5 point forecasts at +1h/+6h/+24h, each with a calibrated
       uncertainty interval (mean + log-variance -> Gaussian NLL training,
       Sec "Add uncertainty").
  4. exceedance probability P(PM2.5 > threshold) per horizon.
  5. source contribution vector (traffic/industry/biomass/dust/background),
     unit-sum constrained via softmax.
  6. event-aware regime classification (Normal / Inversion / Biomass Burning
     / Dust Advection).
"""
from __future__ import annotations

import torch
from torch import nn


class ForecastHead(nn.Module):
    def __init__(self, hidden_dim: int, n_horizons: int):
        super().__init__()
        self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons))
        self.logvar_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons))
        self.exceed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons))

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = torch.nn.functional.softplus(self.mean_head(z))  # PM2.5 >= 0
        # logvar operates on PM_SCALE-normalized targets (see losses/physics.py::PM_SCALE) -- clamp
        # keeps normalized std in [~0.05, ~7.4], i.e. real-unit std in roughly [5, 740] ug/m3.
        logvar = self.logvar_head(z).clamp(min=-6.0, max=4.0)
        exceed_prob = torch.sigmoid(self.exceed_head(z))
        return {"pm25_mean": mean, "pm25_logvar": logvar, "exceed_prob": exceed_prob}


class SourceContributionHead(nn.Module):
    """Real-time source attribution head (Dr. Priya: "the single most important addition")."""

    def __init__(self, hidden_dim: int, n_categories: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_categories))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(z), dim=-1)  # unit-sum constraint


class RegimeHead(nn.Module):
    """Event-aware pollution regime classifier: Normal / Inversion / Biomass Burning / Dust Advection."""

    def __init__(self, hidden_dim: int, n_regimes: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_regimes))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits (cross-entropy applied outside)
