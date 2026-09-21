"""Replace the placeholder atmos / regional-wind columns in an already-fetched RawSeries with real ERA5.

fetch_real_data.py falls back to placeholder atmos data when ERA5 fails, and
re-running it re-pulls FIRMS and opencity.in (slow). This only touches the
ERA5-derived columns of data/real/raw.npz, so the OpenAQ/opencity PM2.5 and
FIRMS fire series already in it are left untouched.

CDS requests queue server-side and can take minutes each; every accepted
chunk is cached in data/real/.era5_cache, so this is safe to interrupt and
re-run.

Usage:
    python scripts/add_era5.py
"""
import argparse

import _pathfix  # noqa: F401

from aqf.data.real_pipeline import apply_era5_to_raw, fetch_era5
from aqf.data.schema import RawSeries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/real/raw.npz")
    args = p.parse_args()

    raw = RawSeries.load(args.path)
    date_from = str(raw.timestamps.min().date())
    date_to = str(raw.timestamps.max().date())
    print(f"Fetching ERA5 for {date_from} .. {date_to} ...", flush=True)

    atmos_df, regional_wind, local_met = fetch_era5(date_from, date_to)
    stats = apply_era5_to_raw(raw, atmos_df, regional_wind, local_met)
    raw.save(args.path)
    print(f"Saved {args.path}. Coverage: {stats}")


if __name__ == "__main__":
    main()
