"""Calendar inputs: time of day, weekday/weekend, season, stubble-burning season, Diwali proximity.

Pure functions of the timestamp, so they need no download and are known for any future hour. Times are shifted
to IST (UTC+5:30) first, because traffic and festival effects follow local time, not UTC.

Diwali dates (main day, IST) were entered by hand -- double-check them before relying on this for a new year:
2022-10-24, 2023-11-12, 2024-11-01, 2025-10-20, 2026-11-08.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CALENDAR_FEATURE_COLS = [
    "hour_sin", "hour_cos", "is_weekend", "doy_sin", "doy_cos", "stubble_season", "diwali_proximity",
]

DIWALI_DATES = ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"]
_DIWALI_DECAY_DAYS = 1.5   # proximity = exp(-|days from Diwali| / 1.5): ~1 on the day, ~0.1 after 3.5 days
_DIWALI_WINDOW_DAYS = 7.0


def calendar_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """(T, len(CALENDAR_FEATURE_COLS)) float32. `timestamps` are naive UTC, as in RawSeries."""
    ist = pd.DatetimeIndex(timestamps) + pd.Timedelta(hours=5, minutes=30)
    hour = ist.hour.to_numpy() + ist.minute.to_numpy() / 60.0
    doy = ist.dayofyear.to_numpy()
    month_day = ist.month.to_numpy() * 100 + ist.day.to_numpy()

    # Days from the nearest Diwali, measured from IST midday of the festival day (the firecracker emissions
    # peak that night and into the next morning).
    centres = pd.DatetimeIndex([pd.Timestamp(d) + pd.Timedelta(hours=12) for d in DIWALI_DATES])
    days = (ist.values[:, None] - centres.values[None, :]) / np.timedelta64(1, "D")
    nearest = days[np.arange(len(ist)), np.abs(days).argmin(axis=1)]
    proximity = np.where(np.abs(nearest) <= _DIWALI_WINDOW_DAYS, np.exp(-np.abs(nearest) / _DIWALI_DECAY_DAYS), 0.0)

    out = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        (ist.dayofweek.to_numpy() >= 5).astype(float),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        ((month_day >= 1010) & (month_day <= 1130)).astype(float),   # Oct 10 - Nov 30: stubble burning season
        proximity,
    ], axis=1)
    return out.astype(np.float32)
