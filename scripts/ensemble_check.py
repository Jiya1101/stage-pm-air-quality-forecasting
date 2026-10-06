"""Does averaging several trained models beat the best single one? Scores on the held-out test period.

Usage:
    python scripts/ensemble_check.py --runs-dir runs/real_suite_v2
"""
import argparse
import os

import _pathfix  # noqa: F401
import numpy as np
import torch

from aqf.data.dataset import AQFWindowDataset, make_split
from aqf.evaluation.evaluate import collect_predictions
from aqf.training.ablation import EXPERIMENTS
from aqf.training.train import build_model, load_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs/real_suite_v2")
    args = p.parse_args()

    preds, ref = {}, None
    raw = None
    for name in EXPERIMENTS:
        path = os.path.join(args.runs_dir, name, "best.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        model = build_model(cfg, "cpu")
        model.load_state_dict(ckpt["model_state"])
        if raw is None:
            raw = load_raw(cfg)
            split = make_split(raw, cfg.data, stride_hours=cfg.data.stride_hours, seed=cfg.train.seed)
            ds = AQFWindowDataset(raw, cfg.data, split.test_t0)
            horizons = cfg.data.horizons_hours
        p_, _, targets, observed, _, persist = collect_predictions(model, ds, "cpu")
        preds[name] = p_
        ref = (targets, observed, persist)
        print("scored", name, flush=True)

    targets, observed, persist = ref

    def report(label, pred):
        row = []
        for i in range(len(horizons)):
            o = observed[:, :, i]
            m = np.abs(pred[:, :, i][o] - targets[:, :, i][o]).mean()
            pm = np.abs(persist[:, :, i][o] - targets[:, :, i][o]).mean()
            row.append(f"+{horizons[i]}h MAE {m:6.2f} skill {1 - m / pm:+.3f}")
        overall = np.abs(pred[observed] - targets[observed]).mean()
        print(f"{label:34s} overall {overall:6.2f} | " + " | ".join(row))

    for n, pr in preds.items():
        report(n, pr)
    stack = np.stack(list(preds.values()))
    print()
    report("ENSEMBLE mean of all", stack.mean(axis=0))
    report("ENSEMBLE median of all", np.median(stack, axis=0))
    top = ["G_physics_constraint", "H_full_model", "B_wind_graph", "A_baseline"]
    report("ENSEMBLE mean of G,H,B,A", np.stack([preds[n] for n in top if n in preds]).mean(axis=0))


if __name__ == "__main__":
    main()
