"""Fire Emission Proxy (FEP).

Dr. Priya's correction: NASA FIRMS gives detected thermal anomalies (fire
radiative power, confidence, location), not a direct PM2.5 emission flux.
Calling raw fire *counts* "emission" invites a reviewer objection. Instead we
derive a "Satellite-derived biomass-burning activity index":

    FEP_t = sum_i FRP_i * confidence_i * area_i * temporal_weight_i

aggregated per source region per hour. `area_i` approximates the ground area
represented by one VIIRS detection pixel (~375m nominal, but effective area
grows with distance from nadir / scan angle -- we use the nominal pixel area
as a conservative default, overridable per-detection when FIRMS supplies a
scan/track quality field).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VIIRS_NOMINAL_PIXEL_AREA_KM2 = 0.375 * 0.375

# FIRMS confidence is categorical for VIIRS ("l"/"n"/"h") or numeric for MODIS (0-100).
_CONFIDENCE_MAP = {"l": 0.3, "low": 0.3, "n": 0.7, "nominal": 0.7, "h": 0.95, "high": 0.95}


def _confidence_to_weight(confidence) -> float:
    if isinstance(confidence, (int, float, np.floating, np.integer)):
        return float(np.clip(confidence, 0.0, 100.0) / 100.0)
    return _CONFIDENCE_MAP.get(str(confidence).strip().lower(), 0.5)


def fire_emission_proxy(
    detections: pd.DataFrame,
    region_col: str = "region_id",
    time_col: str = "acq_datetime",
    frp_col: str = "frp_mw",
    confidence_col: str = "confidence",
    half_life_hours: float = 6.0,
) -> pd.DataFrame:
    """Aggregate raw FIRMS detections into an hourly per-region FEP index.

    Parameters
    ----------
    detections: one row per satellite fire detection, with at least
        [region_col, time_col, frp_col, confidence_col]. `time_col` must be
        tz-naive or tz-aware pandas datetimes.
    half_life_hours: exponential decay applied to a detection's contribution
        as it ages within its hour bucket's neighborhood -- approximates a
        fire's continued smoldering/smoke emission after the satellite
        overpass rather than treating it as an instantaneous point emission.

    Returns
    -------
    DataFrame indexed by hourly timestamp with one FEP column per region.
    """
    df = detections.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["hour"] = df[time_col].dt.floor("h")
    df["conf_w"] = df[confidence_col].apply(_confidence_to_weight)
    df["area_km2"] = VIIRS_NOMINAL_PIXEL_AREA_KM2
    df["temporal_weight"] = 1.0  # detections are already binned to their own hour
    df["fep_contrib"] = df[frp_col].astype(float) * df["conf_w"] * df["area_km2"] * df["temporal_weight"]

    hourly = df.groupby(["hour", region_col])["fep_contrib"].sum().unstack(fill_value=0.0)
    hourly = hourly.sort_index()

    # Smooth with an exponential decay kernel to represent continued
    # smoldering emission between discrete satellite overpasses (VIIRS
    # revisits ~1-2x/day), rather than a spiky instantaneous signal.
    if half_life_hours > 0 and len(hourly) > 1:
        full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
        hourly = hourly.reindex(full_index, fill_value=0.0)
        alpha = 1.0 - 0.5 ** (1.0 / half_life_hours)
        hourly = hourly.ewm(alpha=alpha, adjust=False).mean()

    hourly.index.name = "hour"
    return hourly


def fep_from_arrays(frp_mw: np.ndarray, confidence: np.ndarray, area_km2: np.ndarray | None = None) -> float:
    """Vectorized single-region FEP for a batch of simultaneous detections (no time decay)."""
    if area_km2 is None:
        area_km2 = np.full_like(frp_mw, VIIRS_NOMINAL_PIXEL_AREA_KM2, dtype=float)
    conf_w = np.array([_confidence_to_weight(c) for c in confidence])
    return float(np.sum(frp_mw * conf_w * area_km2))
