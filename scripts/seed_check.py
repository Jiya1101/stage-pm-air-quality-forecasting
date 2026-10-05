"""Re-train selected ablation variants with other seeds, to see how much of the A..H differences is just seed noise.

Usage:
    python scripts/seed_check.py --variants A_baseline G_physics_constraint --seeds 1 2 --out-dir runs/real_seeds
Then score with: python scripts/eval_suite.py --runs-dir runs/real_seeds/seed1   (one per seed)
"""
import argparse
import copy

import _pathfix  # noqa: F401

from aqf.config import Config
from aqf.training.ablation import EXPERIMENTS
from aqf.training.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["A_baseline", "G_physics_constraint"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out-dir", default="runs/real_seeds")
    args = p.parse_args()

    for seed in args.seeds:
        for name in args.variants:
            cfg = Config(name=name)
            cfg.data.source = "real"
            cfg.data.stride_hours = args.stride
            cfg.model.hidden_dim = args.hidden_dim
            cfg.train.epochs = args.epochs
            cfg.train.seed = seed
            cfg.train.out_dir = f"{args.out_dir}/seed{seed}"
            cfg.ablation = copy.deepcopy(EXPERIMENTS[name])
            print(f"\n=== {name} seed {seed} ===", flush=True)
            run(cfg)


if __name__ == "__main__":
    main()
