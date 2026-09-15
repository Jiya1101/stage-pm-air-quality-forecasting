"""Common schema that all data sources (synthetic + real CPCB/ERA5/FIRMS) produce.

Keeping this identical across synthetic and real data is what lets
`config.data.source` be flipped from "synthetic" to "real" with zero changes
to graph construction, the model, or training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCAL_FEATURE_COLS = [
    "pm25", "pm10", "no2", "co", "o3", "temp_c", "rh_pct", "wind_u_ms", "wind_v_ms", "traffic_index",
]
REGIONAL_FEATURE_COLS = [
    "fep", "wind_u_ms", "wind_v_ms", "industry_index", "dust_index",
]
ATMOS_FEATURE_COLS = [
    "blh_m", "inversion_strength_k", "rh_pct", "wind_speed_ms", "solar_rad_wm2", "pressure_hpa", "precip_mm",
]

SOURCE_CATEGORIES = ["traffic", "industry", "biomass", "dust", "background"]
REGIME_CATEGORIES = ["normal", "inversion", "biomass_burning", "dust_advection"]


@dataclass
class RawSeries:
    """Hourly time-aligned arrays for the whole station network.

    Shapes:
      timestamps:        (T,)
      local:              (T, N_local, len(LOCAL_FEATURE_COLS))
      regional:           (T, N_regional, len(REGIONAL_FEATURE_COLS))
      atmos:              (T, len(ATMOS_FEATURE_COLS))
      source_contrib_gt:  (T, N_local, len(SOURCE_CATEGORIES)) or None (synthetic ground truth only)
      regime_gt:          (T, N_local) int categorical or None (synthetic ground truth only)
      local_observed_mask: (T, N_local, len(LOCAL_FEATURE_COLS)) bool or None -- True where `local` holds a
                            genuinely-observed reading rather than an imputed/filled value. Real data only
                            (see data/real_pipeline.py::impute_for_training); None for synthetic data, where
                            every value is "observed" by construction.
    """

    timestamps: pd.DatetimeIndex
    local: np.ndarray
    regional: np.ndarray
    atmos: np.ndarray
    local_ids: list[str]
    regional_ids: list[str]
    source_contrib_gt: np.ndarray | None = None
    regime_gt: np.ndarray | None = None
    local_observed_mask: np.ndarray | None = None

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            timestamps=self.timestamps.values.astype("datetime64[ns]"),
            local=self.local,
            regional=self.regional,
            atmos=self.atmos,
            local_ids=np.array(self.local_ids),
            regional_ids=np.array(self.regional_ids),
            source_contrib_gt=self.source_contrib_gt if self.source_contrib_gt is not None else np.array([]),
            regime_gt=self.regime_gt if self.regime_gt is not None else np.array([]),
            local_observed_mask=self.local_observed_mask if self.local_observed_mask is not None else np.array([]),
        )

    @classmethod
    def load(cls, path: str) -> "RawSeries":
        d = np.load(path, allow_pickle=False)
        sc = d["source_contrib_gt"]
        rg = d["regime_gt"]
        om = d["local_observed_mask"] if "local_observed_mask" in d else np.array([])
        return cls(
            timestamps=pd.DatetimeIndex(d["timestamps"]),
            local=d["local"],
            regional=d["regional"],
            atmos=d["atmos"],
            local_ids=list(d["local_ids"]),
            regional_ids=list(d["regional_ids"]),
            source_contrib_gt=sc if sc.size else None,
            regime_gt=rg if rg.size else None,
            local_observed_mask=om if om.size else None,
        )
