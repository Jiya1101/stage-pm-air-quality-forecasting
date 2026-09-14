"""Continuous Atmospheric Stability Index.

Dr. Priya's critique of the original design: a hard "BLH < 100m -> suppress
long-distance edges" rule invites the immediate reviewer question "why 100m
exactly, is transport really discontinuous there?" -- the answer is no.

Replacement: a continuous, learnable index

    S_t = sigmoid(w^T [BLH_t, dT_inversion_t, RH_t, WS_t, SR_t] + b),  S_t in [0,1]

0 -> strong convective mixing / ventilation. 1 -> strong nocturnal inversion /
surface trapping. This is used to continuously gate graph edges (see
graph/builder.py) instead of a binary switch.

Two entry points are provided:
  - `StabilityIndex` (nn.Module): the learnable version used inside the model,
    trained end-to-end so the network can discover the right combination
    rather than have it hand-tuned.
  - `heuristic_stability_index`: a fixed-weight, non-learned reference
    implementation (useful for feature engineering / sanity plots / synthetic
    data generation, where no gradient is available).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


STABILITY_FEATURES = ["blh_m", "inversion_strength_k", "rh_pct", "wind_speed_ms", "solar_rad_wm2"]


class StabilityIndex(nn.Module):
    """Learnable continuous atmospheric stability index S_t in [0, 1].

    Input features are min-max normalized internally using running fixed
    physically-reasonable ranges so the linear layer sees O(1) inputs
    regardless of raw units.
    """

    # (min, max) physically reasonable ranges for normalization
    _RANGES = {
        "blh_m": (10.0, 3000.0),
        "inversion_strength_k": (-2.0, 10.0),
        "rh_pct": (0.0, 100.0),
        "wind_speed_ms": (0.0, 20.0),
        "solar_rad_wm2": (0.0, 1000.0),
    }

    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # BLH and wind speed should push stability DOWN (more mixing) as they
        # rise; inversion strength and RH should push it UP. We don't hard
        # constrain the MLP but this sign convention is captured in how the
        # raw features are normalized before the MLP: BLH and wind speed are
        # sign-flipped so all five normalized inputs are monotonically
        # "trapping-positive" a priori, giving the optimizer a good start.

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 5) in order STABILITY_FEATURES
        out = []
        for i, key in enumerate(STABILITY_FEATURES):
            lo, hi = self._RANGES[key]
            v = (x[..., i] - lo) / (hi - lo + 1e-8)
            v = v.clamp(0.0, 1.0)
            if key in ("blh_m", "wind_speed_ms"):
                v = 1.0 - v  # low BLH / low wind -> high trapping signal
            out.append(v)
        return torch.stack(out, dim=-1)

    def forward(self, atmos_features: torch.Tensor) -> torch.Tensor:
        """atmos_features: (..., 5) raw [BLH, inversion, RH, WS, SR] -> S_t (..., 1) in [0,1]."""
        x = self._normalize(atmos_features)
        return torch.sigmoid(self.net(x))


def heuristic_stability_index(
    blh_m: np.ndarray,
    inversion_strength_k: np.ndarray,
    rh_pct: np.ndarray,
    wind_speed_ms: np.ndarray,
    solar_rad_wm2: np.ndarray,
) -> np.ndarray:
    """Fixed-weight reference stability index for feature engineering / synthetic ground truth.

    Not used by the trainable model (see StabilityIndex above) -- this exists
    so data-generation and diagnostic code can compute a physically sane S_t
    without needing a trained network.
    """
    blh_term = 1.0 - np.clip(blh_m / 1500.0, 0.0, 1.0)
    inv_term = np.clip(inversion_strength_k / 6.0, 0.0, 1.0)
    rh_term = np.clip(rh_pct / 100.0, 0.0, 1.0)
    wind_term = 1.0 - np.clip(wind_speed_ms / 8.0, 0.0, 1.0)
    solar_term = 1.0 - np.clip(solar_rad_wm2 / 600.0, 0.0, 1.0)
    raw = 0.30 * blh_term + 0.25 * inv_term + 0.15 * rh_term + 0.20 * wind_term + 0.10 * solar_term
    return np.clip(raw, 0.0, 1.0)
