"""Generate the synthetic dataset used for pipeline validation before real data access is wired in.

Usage:
    python scripts/generate_synthetic_data.py [--start 2018-01-01] [--end 2023-12-31 23:00] [--out data/synthetic]
"""
import argparse
import os

import _pathfix  # noqa: F401

from aqf.data.synthetic import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2023-12-31 23:00")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Generating synthetic STAGE-PM dataset from {args.start} to {args.end} ...")
    raw = generate(start=args.start, end=args.end, seed=args.seed)
    out_path = os.path.join(args.out, "raw.npz")
    raw.save(out_path)
    print(f"Saved {len(raw.timestamps)} hourly timesteps x {raw.local.shape[1]} local stations "
          f"x {raw.regional.shape[1]} regional sources -> {out_path}")


if __name__ == "__main__":
    main()
