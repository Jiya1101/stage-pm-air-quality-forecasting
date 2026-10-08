"""PM2.5 target distribution for the train / validation / test splits, how rare the extremes are, and the fitted weights.

Targets are exactly what the loss sees: genuinely observed PM2.5 at every forecast horizon, windows at the training
stride, non-held-out stations. Writes docs/figures/pm25_target_distribution.png and runs/loss_exp/target_distribution.json.

Usage:
    python scripts/target_distribution.py
"""
import json
import os

import _pathfix  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aqf.config import Config
from aqf.data.dataset import make_split
from aqf.losses.frequency import FrequencyWeights, collect_targets
from aqf.training.train import load_raw

EDGES = [0, 25, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1e9]
THRESHOLDS = [200, 400, 600]


def main():
    cfg = Config(name="dist")
    cfg.data.source = "real"
    raw = load_raw(cfg)
    split = make_split(raw, cfg.data, stride_hours=cfg.data.stride_hours, seed=cfg.train.seed)
    horizons, held = cfg.data.horizons_hours, split.held_out_local_mask
    targets = {name: collect_targets(raw, idx, horizons, held) for name, idx in
               [("train", split.train_t0), ("val", split.val_t0), ("test", split.test_t0)]}

    out = {"n": {k: int(len(v)) for k, v in targets.items()}, "rarity": {}, "bins": {}, "quantiles": {}}
    print(f"{'split':6s} {'n targets':>10s} {'mean':>7s} {'median':>7s} {'p95':>6s} {'p99':>6s} {'max':>6s}   "
          + "  ".join(f"{'>' + str(t):>14s}" for t in THRESHOLDS))
    for name, y in targets.items():
        rare = {t: (int((y > t).sum()), float((y > t).mean())) for t in THRESHOLDS}
        out["rarity"][name] = {str(t): {"count": c, "share": s} for t, (c, s) in rare.items()}
        out["quantiles"][name] = {q: float(np.quantile(y, q / 100)) for q in (5, 25, 50, 75, 90, 95, 99, 99.9)}
        print(f"{name:6s} {len(y):10d} {y.mean():7.1f} {np.median(y):7.1f} {np.quantile(y, .95):6.0f} {np.quantile(y, .99):6.0f} {y.max():6.0f}   "
              + "  ".join(f"{c:7d} ({100 * s:4.2f}%)" for c, s in rare.values()))

    print("\nFrequency table (share of targets per PM2.5 bin):")
    labels = [f"{int(a)}-{int(b)}" if b < 1e8 else f">{int(a)}" for a, b in zip(EDGES[:-1], EDGES[1:])]
    print(f"{'bin':>9s} " + " ".join(f"{k + ' n':>9s} {k + ' %':>8s}" for k in targets))
    for i, lab in enumerate(labels):
        row = []
        for name, y in targets.items():
            c = int(((y >= EDGES[i]) & (y < EDGES[i + 1])).sum())
            out["bins"].setdefault(name, {})[lab] = {"count": c, "share": c / len(y)}
            row.append(f"{c:9d} {100 * c / len(y):7.2f}%")
        print(f"{lab:>9s} " + " ".join(row))

    fw = FrequencyWeights.fit(targets["train"], cfg.train.freq_alpha, cfg.train.freq_cap_quantile)
    out["weights"] = fw.info

    os.makedirs("docs/figures", exist_ok=True)
    os.makedirs("runs/loss_exp", exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    bins = np.arange(0, 1050, 25)
    colors = {"train": "#2b59c3", "val": "#e8801f", "test": "#2e9e5b"}
    for name, y in targets.items():
        ax[0].hist(np.clip(y, 0, 1049), bins=bins, density=True, histtype="step", lw=1.8, color=colors[name], label=f"{name} (n={len(y):,})")
        ax[1].hist(np.clip(y, 0, 1049), bins=bins, density=True, histtype="step", lw=1.8, color=colors[name], label=name)
    ax[0].set_title("PM2.5 targets: density"); ax[0].set_xlabel("PM2.5 (ug/m3)"); ax[0].set_ylabel("density"); ax[0].legend()
    ax[1].set_yscale("log"); ax[1].set_title("same, log scale: the long tail"); ax[1].set_xlabel("PM2.5 (ug/m3)")
    for t in THRESHOLDS:
        ax[1].axvline(t, color="grey", ls=":", lw=1)
    grid_y = np.linspace(0, 1000, 500)
    ax[2].plot(grid_y, fw(__import__("torch").tensor(grid_y, dtype=__import__("torch").float32)).numpy(), color="#8a2a6b", lw=2)
    ax[2].axhline(1.0, color="grey", ls=":", lw=1)
    ax[2].set_title(f"fitted weight w(y)  (alpha={fw.info['alpha']}, cap at q={fw.info['cap_quantile']})")
    ax[2].set_xlabel("true PM2.5 (ug/m3)"); ax[2].set_ylabel("weight (mean over training = 1)")
    fig.tight_layout()
    fig.savefig("docs/figures/pm25_target_distribution.png", dpi=130)
    with open("runs/loss_exp/target_distribution.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nweights:", {k: v for k, v in fw.info.items() if k != "weight_by_value"})
    print("weight by value:", {k: round(v, 2) for k, v in fw.info["weight_by_value"].items()})
    print("saved docs/figures/pm25_target_distribution.png and runs/loss_exp/target_distribution.json")


if __name__ == "__main__":
    main()
