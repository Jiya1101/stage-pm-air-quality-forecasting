"""Train STAGE-PM (single config, or the full A-H ablation suite).

Usage:
    python scripts/train.py --config full
    python scripts/train.py --ablation-suite
"""
import argparse

import _pathfix  # noqa: F401

from aqf.config import Config
from aqf.training.ablation import run_suite
from aqf.training.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="full", help="run name (used for checkpoint/log directory)")
    p.add_argument("--ablation-suite", action="store_true", help="run experiments A-H instead of a single config")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    cfg = Config(name=args.config)
    cfg.train.device = args.device
    if args.epochs:
        cfg.train.epochs = args.epochs

    if args.ablation_suite:
        results = run_suite(cfg)
        print("\n\n=== Ablation suite summary (best val MAE) ===")
        for name, r in results.items():
            print(f"  {name:24s} best_val_mae={r['best_val_mae']:.3f}")
    else:
        result = run(cfg)
        print(f"\nBest val MAE: {result['best_val_mae']:.3f}  (checkpoint: {result['out_dir']}/best.pt)")


if __name__ == "__main__":
    main()
