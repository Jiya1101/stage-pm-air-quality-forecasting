"""Frequency-weighted MAE: rare target values get larger weights, derived from the training targets themselves.

Why: PM2.5 targets are heavily right-skewed (a few percent of hours are above 300, most are below 100), and a
plain MAE is minimised near the middle of that distribution, so the forecast sits below the truth on severe days.

Method (label-distribution-smoothing style inverse-frequency weighting, cf. Yang et al., "Delving into Deep
Imbalanced Regression", ICML 2021; the exact form below is ours):

    z      = log1p(y)                           target in log space, so a 100 -> 200 step and a 400 -> 800 step count alike
    p(z)   = kernel density of the training z   histogram on a fine grid, Gaussian kernel, bandwidth from Scott's rule
                                                h = 1.06 * std(z) * n^(-1/5)      (no hand-picked pollution thresholds)
    w0(y)  = p(z(y)) ^ (-alpha)                 inverse frequency, softened by alpha (alpha=1: pure inverse, 0: no weighting)
    w1(y)  = min(w0(y), Q_q(w0 over training targets))     cap at the q-th quantile so the rarest few hours cannot explode
    w(y)   = w1(y) / mean_train(w1)             rescale so the average training weight is exactly 1

    L_MAE = sum_i  w(y_i) * |yhat_i - y_i| * mask_i  /  sum_i mask_i

Because mean_train(w) = 1, the loss keeps the same overall scale as the unweighted MAE, so lambda_nll, the physics
terms and the learning rate keep their meaning. Only the MAE term is weighted; NLL and the exceedance BCE are untouched.
alpha=0.5 and q=0.99 are fixed a priori (TrainConfig.freq_alpha / freq_cap_quantile), not tuned on validation or test.
"""
from __future__ import annotations

import numpy as np
import torch

from aqf.data.dataset import PM25_IDX


def collect_targets(raw, t0_indices: np.ndarray, horizons, held_out_mask: np.ndarray) -> np.ndarray:
    """PM2.5 target values the loss actually sees: observed hours at every horizon, non-held-out stations only."""
    keep = ~held_out_mask
    out = []
    for h in horizons:
        y = raw.local[t0_indices + h][:, keep, PM25_IDX]
        if raw.local_observed_mask is not None:
            obs = raw.local_observed_mask[t0_indices + h][:, keep, PM25_IDX]
        else:
            obs = np.ones_like(y, dtype=bool)
        out.append(y[obs])
    return np.concatenate(out)


class FrequencyWeights:
    def __init__(self, grid_z: np.ndarray, weights: np.ndarray, info: dict):
        self.grid_z = grid_z.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.info = info
        self._gz = torch.from_numpy(self.grid_z)
        self._gw = torch.from_numpy(self.weights)

    @classmethod
    def fit(cls, y: np.ndarray, alpha: float = 0.5, cap_quantile: float = 0.99, n_grid: int = 512) -> "FrequencyWeights":
        y = np.asarray(y, dtype=np.float64)
        z = np.log1p(np.clip(y, 0, None))
        n = len(z)
        bandwidth = 1.06 * z.std() * n ** (-1 / 5)                       # Scott's rule
        lo, hi = z.min() - 3 * bandwidth, z.max() + 3 * bandwidth
        grid = np.linspace(lo, hi, n_grid)
        step = grid[1] - grid[0]
        counts, _ = np.histogram(z, bins=np.append(grid - step / 2, grid[-1] + step / 2))
        half = int(4 * bandwidth / step) + 1
        k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / bandwidth) ** 2)
        density = np.convolve(counts.astype(float), k / k.sum(), mode="same")
        density = density / (density.sum() * step)                       # integrates to 1 over z
        density = np.maximum(density, 1e-12)

        w0_grid = density ** (-alpha)
        w0_train = np.interp(z, grid, w0_grid)
        cap = np.quantile(w0_train, cap_quantile)
        w1_grid = np.minimum(w0_grid, cap)
        scale = np.minimum(w0_train, cap).mean()
        w_grid = w1_grid / scale
        w_train = np.interp(z, grid, w_grid)
        info = {
            "alpha": alpha, "cap_quantile": cap_quantile, "n_targets": int(n), "scott_bandwidth_log1p": float(bandwidth),
            "cap_value_before_rescale": float(cap), "mean_train_weight": float(w_train.mean()),
            "min_weight": float(w_train.min()), "max_weight": float(w_train.max()),
            "weight_by_value": {str(v): float(np.interp(np.log1p(v), grid, w_grid)) for v in (10, 25, 50, 100, 150, 200, 300, 400, 500, 600, 800)},
        }
        return cls(grid, w_grid, info)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        """Weights for target tensor y (any shape), by linear interpolation on the fitted grid."""
        gz, gw = self._gz.to(y.device), self._gw.to(y.device)
        z = torch.log1p(y.clamp(min=0)).clamp(gz[0].item(), gz[-1].item())
        idx = torch.bucketize(z.contiguous(), gz).clamp(1, len(gz) - 1)
        z0, z1, w0, w1 = gz[idx - 1], gz[idx], gw[idx - 1], gw[idx]
        t = (z - z0) / (z1 - z0)
        return w0 + t * (w1 - w0)
