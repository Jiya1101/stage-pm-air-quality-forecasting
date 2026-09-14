"""Windowing dataset: turns RawSeries into (lookback window -> multi-horizon targets) samples.

Splitting strategy follows Dr. Priya's leakage warning explicitly: no random
hourly shuffling. We use

  - a temporal split by calendar year (train / val / test), and
  - a spatial holdout: a fixed subset of local stations whose labels are
    excluded from the training loss entirely, so generalization to a
    never-trained-on station can be measured at evaluation time (see
    evaluation/evaluate.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from aqf.config import DataConfig
from aqf.data.schema import LOCAL_FEATURE_COLS, REGIONAL_FEATURE_COLS, RawSeries

PM25_IDX = LOCAL_FEATURE_COLS.index("pm25")
LOCAL_WIND_U_IDX = LOCAL_FEATURE_COLS.index("wind_u_ms")
LOCAL_WIND_V_IDX = LOCAL_FEATURE_COLS.index("wind_v_ms")
REG_WIND_U_IDX = REGIONAL_FEATURE_COLS.index("wind_u_ms")
REG_WIND_V_IDX = REGIONAL_FEATURE_COLS.index("wind_v_ms")


def _unit(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    vec = np.stack([u, v], axis=-1)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.clip(norm, 1e-6, None)


@dataclass
class SplitIndices:
    train_t0: np.ndarray
    val_t0: np.ndarray
    test_t0: np.ndarray
    held_out_local_mask: np.ndarray  # (N_local,) bool, True = excluded from training loss


def make_split(raw: RawSeries, cfg: DataConfig, stride_hours: int = 6, seed: int = 0) -> SplitIndices:
    years = raw.timestamps.year.to_numpy()
    max_h = max(cfg.horizons_hours)
    lo = cfg.lookback_hours - 1
    hi = len(raw.timestamps) - max_h - 1

    all_t0 = np.arange(lo, hi, stride_hours)
    t0_years = years[all_t0]

    train_t0 = all_t0[t0_years <= cfg.train_end_year]
    val_t0 = all_t0[t0_years == cfg.val_year]
    test_t0 = all_t0[t0_years == cfg.test_year]

    rng = np.random.default_rng(seed)
    n_local = raw.local.shape[1]
    n_held = max(1, int(round(cfg.spatial_holdout_frac * n_local)))
    held_idx = rng.choice(n_local, size=n_held, replace=False)
    held_mask = np.zeros(n_local, dtype=bool)
    held_mask[held_idx] = True

    return SplitIndices(train_t0=train_t0, val_t0=val_t0, test_t0=test_t0, held_out_local_mask=held_mask)


class AQFWindowDataset(Dataset):
    """One sample = a `lookback_hours` window ending at t0, plus targets at t0+h for each horizon h."""

    def __init__(self, raw: RawSeries, cfg: DataConfig, t0_indices: np.ndarray):
        self.raw = raw
        self.cfg = cfg
        self.t0_indices = t0_indices

        self.local_wind_unit = _unit(raw.local[:, :, LOCAL_WIND_U_IDX], raw.local[:, :, LOCAL_WIND_V_IDX])
        self.regional_wind_unit = _unit(raw.regional[:, :, REG_WIND_U_IDX], raw.regional[:, :, REG_WIND_V_IDX])
        self.regional_wind_speed = np.sqrt(
            raw.regional[:, :, REG_WIND_U_IDX] ** 2 + raw.regional[:, :, REG_WIND_V_IDX] ** 2
        )

    def __len__(self) -> int:
        return len(self.t0_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t0 = int(self.t0_indices[idx])
        L = self.cfg.lookback_hours
        lo = t0 - L + 1

        local_seq = self.raw.local[lo:t0 + 1]                # (L, N_local, F_local)
        regional_seq = self.raw.regional[lo:t0 + 1]           # (L, N_reg, F_reg)
        atmos_seq = self.raw.atmos[lo:t0 + 1]                  # (L, F_atmos)
        local_wind_unit_seq = self.local_wind_unit[lo:t0 + 1]  # (L, N_local, 2)
        reg_wind_unit_seq = self.regional_wind_unit[lo:t0 + 1]  # (L, N_reg, 2)
        reg_wind_speed_seq = self.regional_wind_speed[lo:t0 + 1]  # (L, N_reg)

        horizons = self.cfg.horizons_hours
        y = np.stack([self.raw.local[t0 + h, :, PM25_IDX] for h in horizons], axis=-1)  # (N_local, n_horizons)
        y_exceed = (y > self.cfg.exceedance_threshold).astype(np.float32)

        sample = {
            "local_seq": torch.from_numpy(local_seq.astype(np.float32)),
            "regional_seq": torch.from_numpy(regional_seq.astype(np.float32)),
            "atmos_seq": torch.from_numpy(atmos_seq.astype(np.float32)),
            "local_wind_unit_seq": torch.from_numpy(local_wind_unit_seq.astype(np.float32)),
            "regional_wind_unit_seq": torch.from_numpy(reg_wind_unit_seq.astype(np.float32)),
            "regional_wind_speed_seq": torch.from_numpy(reg_wind_speed_seq.astype(np.float32)),
            "y_pm25": torch.from_numpy(y.astype(np.float32)),
            "y_exceed": torch.from_numpy(y_exceed),
            "t0": torch.tensor(t0, dtype=torch.long),
        }

        if self.raw.source_contrib_gt is not None:
            sample["y_source_contrib"] = torch.from_numpy(
                self.raw.source_contrib_gt[t0 + horizons[0]].astype(np.float32)
            )
        if self.raw.regime_gt is not None:
            sample["y_regime"] = torch.from_numpy(self.raw.regime_gt[t0].astype(np.int64))

        return sample
