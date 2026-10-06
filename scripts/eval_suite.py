"""Score every trained ablation variant (A..H) on the held-out test period and compare them to persistence.

Usage:
    python scripts/eval_suite.py --runs-dir runs/real_suite [--out docs/ablation_results.md]
"""
import argparse
import json
import os

import _pathfix  # noqa: F401
import torch

from aqf.data.dataset import AQFWindowDataset, make_split
from aqf.evaluation.evaluate import evaluate
from aqf.training.ablation import ALL_EXPERIMENTS
from aqf.training.train import build_model, load_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs/real_suite")
    p.add_argument("--out", default=None, help="write a markdown table here")
    p.add_argument("--device", default="cpu")
    p.add_argument("--variants", nargs="+", default=None, help="only these (default: every known variant found in --runs-dir)")
    args = p.parse_args()

    rows, raw_cache = {}, {}
    for name in (args.variants or ALL_EXPERIMENTS):
        path = os.path.join(args.runs_dir, name, "best.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        cfg = ckpt["cfg"]
        model = build_model(cfg, args.device)
        model.load_state_dict(ckpt["model_state"])
        raw = raw_cache.setdefault(cfg.data.real_dir, load_raw(cfg))
        split = make_split(raw, cfg.data, stride_hours=cfg.data.stride_hours, seed=cfg.train.seed)
        rep = evaluate(model, AQFWindowDataset(raw, cfg.data, split.test_t0), split.held_out_local_mask,
                       cfg.data.exceedance_threshold, cfg.data.horizons_hours, args.device)
        rows[name] = {
            "val_mae": float(ckpt["val_mae"]), "best_epoch": int(ckpt["epoch"]) + 1,
            "test_mae": rep["overall"]["mae"], "coverage95": rep["overall"]["uncertainty_coverage_95pct"],
            "per_horizon_mae": {h: m["mae"] for h, m in rep["per_horizon"].items()},
            "csi": {h: m["csi"] for h, m in rep["per_horizon"].items()},
            "skill": rep["skill_vs_persistence"],
            "persistence_mae": rep["persistence"]["per_horizon_mae"],
            "held_out_stations_mae": (rep["spatial_holdout"]["held_out_stations"] or {}).get("mae"),
            "in_sample_stations_mae": rep["spatial_holdout"]["in_sample_stations"]["mae"],
        }
        print(f"scored {name}", flush=True)

    with open(os.path.join(args.runs_dir, "suite_results.json"), "w") as f:
        json.dump(rows, f, indent=2)

    hs = list(next(iter(rows.values()))["skill"].keys()) if rows else []
    lines = ["| Variant | best ep | val MAE | test MAE | " + " | ".join(f"skill {h}" for h in hs) + " | CSI +1h | 95% cov |",
             "|---|---|---|---|" + "---|" * len(hs) + "---|---|"]
    for n, r in rows.items():
        lines.append(f"| {n} | {r['best_epoch']} | {r['val_mae']:.2f} | {r['test_mae']:.2f} | "
                     + " | ".join(f"{r['skill'][h]:+.3f}" for h in hs)
                     + f" | {list(r['csi'].values())[0]:.3f} | {r['coverage95']:.2f} |")
    if rows:
        pm = next(iter(rows.values()))["persistence_mae"]
        lines.append("\nPersistence MAE per horizon (same observed hours): " + ", ".join(f"{h}: {v:.2f}" for h, v in pm.items()))
    table = "\n".join(lines)
    print("\n" + table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("# Ablation results (real data, test = 2025-09-01 .. 2026-09-20)\n\n" + table + "\n")


if __name__ == "__main__":
    main()
