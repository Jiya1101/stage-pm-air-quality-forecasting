"""Evaluate a trained STAGE-PM checkpoint: overall + per-horizon + per-bucket + per-regime + spatial-holdout metrics.

Usage:
    python scripts/evaluate.py --checkpoint runs/full/best.pt
"""
import argparse

import _pathfix  # noqa: F401
import torch

from aqf.data.dataset import AQFWindowDataset, make_split
from aqf.evaluation.evaluate import evaluate, print_report
from aqf.training.train import build_model, load_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg.train.device = args.device

    model = build_model(cfg, args.device)
    model.load_state_dict(ckpt["model_state"])

    raw = load_raw(cfg)
    split = make_split(raw, cfg.data, stride_hours=6, seed=cfg.train.seed)
    test_ds = AQFWindowDataset(raw, cfg.data, split.test_t0)

    report = evaluate(
        model, test_ds, split.held_out_local_mask,
        exceedance_threshold=cfg.data.exceedance_threshold,
        horizons=cfg.data.horizons_hours,
        device=args.device,
    )
    print_report(cfg.name, report)


if __name__ == "__main__":
    main()
