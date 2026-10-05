"""Package a trained checkpoint + a small replay window for deployment (deploy/ is what the Docker image ships).

The full raw.npz is tens of MB and git-ignored; the container only needs the hours the demo replays plus the
48 h lookback before them, so this slices that window out, imputes gaps (keeping the observed mask), and saves it.

Usage:
    python scripts/export_replay.py --checkpoint runs/H_full_model/best.pt --raw data/real_2022_2026/raw.npz \
        --start 2025-10-01 --end 2025-12-15
"""
import argparse
import os
import shutil

import _pathfix  # noqa: F401
import numpy as np
import pandas as pd

from aqf.data.real_pipeline import impute_for_training
from aqf.data.schema import RawSeries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--start", required=True, help="first replay hour (YYYY-MM-DD); 48 h of lookback are kept before it")
    p.add_argument("--end", required=True)
    p.add_argument("--out", default="deploy")
    p.add_argument("--lookback", type=int, default=48)
    args = p.parse_args()

    raw = RawSeries.load(args.raw)
    if np.isnan(raw.local).any():
        raw = impute_for_training(raw)

    ts = raw.timestamps
    lo = int(np.searchsorted(ts.values, pd.Timestamp(args.start).to_datetime64())) - args.lookback
    hi = int(np.searchsorted(ts.values, (pd.Timestamp(args.end) + pd.Timedelta(days=1)).to_datetime64()))
    lo = max(lo, 0)

    def cut(a):
        return None if a is None else a[lo:hi]

    out = RawSeries(
        timestamps=ts[lo:hi], local=raw.local[lo:hi], regional=raw.regional[lo:hi], atmos=raw.atmos[lo:hi],
        local_ids=raw.local_ids, regional_ids=raw.regional_ids,
        source_contrib_gt=cut(raw.source_contrib_gt), regime_gt=cut(raw.regime_gt),
        local_observed_mask=cut(raw.local_observed_mask),
    )
    os.makedirs(os.path.join(args.out, "model"), exist_ok=True)
    out.save(os.path.join(args.out, "replay.npz"))
    shutil.copy(args.checkpoint, os.path.join(args.out, "model", "best.pt"))
    size = os.path.getsize(os.path.join(args.out, "replay.npz")) / 1e6
    print(f"Replay: {ts[lo]} .. {ts[hi-1]} ({hi-lo} h, {size:.1f} MB) + checkpoint -> {args.out}/")


if __name__ == "__main__":
    main()
