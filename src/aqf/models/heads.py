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


# ug/m3 per unit of the mean head's raw output, when it predicts a change from the last reading.
DELTA_SCALE = 50.0


class ForecastHead(nn.Module):
    """Point forecast + uncertainty + exceedance.

    With `last` (the most recent PM2.5 reading at each station) the mean is
    predicted as a *change from that reading*: mean = softplus(last + DELTA_SCALE * f(z)).
    The last mean-head layer is zero-initialised, so an untrained model is
    exactly persistence and training has to earn any improvement over it.
    Without this, the network must rediscover "next hour ~ this hour" from
    scratch -- measured on the real 2023 test year, a 4-epoch model that
    predicted absolute PM2.5 was worse than persistence at every horizon
    (+1h MAE 49.4 vs 20.5).
    """

    def __init__(self, hidden_dim: int, n_horizons: int):
        super().__init__()
        self.mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons))
        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)
        self.logvar_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons))
        self.exceed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, n_horizons))

    def forward(self, z: torch.Tensor, last: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        raw = self.mean_head(z)
        if last is not None:
            mean = torch.nn.functional.softplus(last.unsqueeze(-1) + DELTA_SCALE * raw)  # PM2.5 >= 0
        else:
            mean = torch.nn.functional.softplus(raw)  # PM2.5 >= 0
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
