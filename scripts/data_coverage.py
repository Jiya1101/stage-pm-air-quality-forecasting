"""Report how much real data each train/val/test split actually has (observed PM2.5, real ERA5 weather).

Usage:
    python scripts/data_coverage.py [--path data/real_2022_2026/raw.npz]
"""
import argparse

import _pathfix  # noqa: F401
import numpy as np
import pandas as pd

from aqf.config import DataConfig
from aqf.data.schema import ATMOS_FEATURE_COLS, RawSeries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/real_2022_2026/raw.npz")
    args = p.parse_args()

    raw = RawSeries.load(args.path)
    cfg = DataConfig()
    ts = raw.timestamps
    pm = raw.local[:, :, 0]
    obs = raw.local_observed_mask[:, :, 0] if raw.local_observed_mask is not None else ~np.isnan(pm)
    blh = ATMOS_FEATURE_COLS.index("blh_m")

    # ERA5 hours differ from the placeholder climatology; the placeholder is smooth/periodic so detect real
    # coverage as "atmos row is not identical to the same hour one day earlier".
    era5_real = np.zeros(len(ts), dtype=bool)
    era5_real[24:] = np.abs(raw.atmos[24:, blh] - raw.atmos[:-24, blh]) > 1e-3

    splits = {
        "train": (cfg.train_start, cfg.train_end), "val": (cfg.val_start, cfg.val_end), "test": (cfg.test_start, cfg.test_end),
    }
    print(f"{'split':6} {'range':25} {'hours':>6} {'PM2.5 observed':>15} {'ERA5 real':>10}")
    for name, (a, b) in splits.items():
        m = (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1))
        print(f"{name:6} {a}..{b:11} {m.sum():6d} {obs[m].mean():14.1%} {era5_real[m].mean():10.1%}")

    print("\nObserved PM2.5 per station (train / val / test):")
    for i, sid in enumerate(raw.local_ids):
        row = []
        for a, b in splits.values():
            m = (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1))
            row.append(f"{obs[m, i].mean():6.1%}")
        print(f"  {sid}  " + "  ".join(row))

    for name, (a, b) in splits.items():
        m = (ts >= pd.Timestamp(a)) & (ts < pd.Timestamp(b) + pd.Timedelta(days=1))
        v = pm[m & obs.any(axis=1)][obs[m & obs.any(axis=1)]]
        print(f"\n{name}: observed PM2.5 mean {np.nanmean(v):.0f}, p95 {np.nanpercentile(v, 95):.0f}, max {np.nanmax(v):.0f} ug/m3")
    for c in ["pm10", "no2", "co", "o3"]:
        k = ["pm25", "pm10", "no2", "co", "o3"].index(c)
        o = raw.local_observed_mask[:, :, k] if raw.local_observed_mask is not None else ~np.isnan(raw.local[:, :, k])
        m = (ts >= pd.Timestamp(cfg.test_start)) & (ts < pd.Timestamp(cfg.test_end) + pd.Timedelta(days=1))
        mt = (ts >= pd.Timestamp(cfg.train_start)) & (ts < pd.Timestamp(cfg.train_end) + pd.Timedelta(days=1))
        print(f"{c}: observed in train {o[mt].mean():.1%}, in test {o[m].mean():.1%}")


if __name__ == "__main__":
    main()
