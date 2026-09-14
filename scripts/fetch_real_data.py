"""Fetch real OpenAQ + FIRMS data and cache it as a RawSeries, same shape as the synthetic dataset.

Requires OPENAQ_API_KEY and FIRMS_MAP_KEY (in .env or the real environment).
See src/aqf/data/real_pipeline.py's module docstring for exactly what's real
data vs. placeholder (BLH/inversion/regional-wind need ERA5, not yet wired).

Usage:
    python scripts/fetch_real_data.py --days 30                      # last 30 days (quick validation)
    python scripts/fetch_real_data.py --start 2019-01-01 --end 2023-12-31   # full historical pull (slow)
"""
import argparse
import datetime
import os

import _pathfix  # noqa: F401

from aqf.data.real_pipeline import assemble_real_raw_series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=None, help="pull the last N days (mutually exclusive with --start/--end)")
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--out", default="data/real")
    args = p.parse_args()

    if args.days:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=args.days)
        date_from, date_to = str(start), str(end)
    elif args.start and args.end:
        date_from, date_to = args.start, args.end
    else:
        raise SystemExit("Pass either --days N, or both --start and --end")

    print(f"Fetching real STAGE-PM data from {date_from} to {date_to} "
          f"(OpenAQ for local stations, FIRMS for regional fires)...")
    raw = assemble_real_raw_series(date_from, date_to)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "raw.npz")
    raw.save(out_path)

    import numpy as np
    local_coverage = 1.0 - np.isnan(raw.local[..., 0]).mean()  # pm25 column
    print(f"Saved {len(raw.timestamps)} hourly timesteps -> {out_path}")
    print(f"Local PM2.5 coverage: {local_coverage:.1%} non-NaN (gaps are real sensor downtime -- see "
          f"data/dataset.py for how the training pipeline should handle NaNs before this is training-ready)")


if __name__ == "__main__":
    main()
