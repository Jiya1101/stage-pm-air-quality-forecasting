"""Train chosen variants (A-H or the calendar follow-ups) for longer, one after another.

Usage:
    python scripts/train_long.py --variants G_physics_constraint G2_physics_calendar H_full_model H2_full_calendar --epochs 30
Score with: python scripts/eval_suite.py --runs-dir runs/real_long --variants <names>
"""
import argparse
import copy

import _pathfix  # noqa: F401

from aqf.config import Config
from aqf.training.ablation import ALL_EXPERIMENTS
from aqf.training.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/real_long")
    args = p.parse_args()

    for name in args.variants:
        cfg = Config(name=name)
        cfg.data.source = "real"
        cfg.data.stride_hours = args.stride
        cfg.model.hidden_dim = args.hidden_dim
        cfg.train.epochs = args.epochs
        cfg.train.seed = args.seed
        cfg.train.out_dir = args.out_dir
        cfg.ablation = copy.deepcopy(ALL_EXPERIMENTS[name])
        print(f"\n=== {name} ({args.epochs} epochs) ===", flush=True)
        run(cfg)


if __name__ == "__main__":
    main()
