"""Compare a baseline model with a frequency-weighted-loss model on the test year, against run-to-run noise.

Metrics: val MAE, test MAE, skill vs persistence per horizon, seasonal MAE / bias / skill, error on actual >200/>400/>600
hours, the actual >400 subset, prediction-vs-actual distribution, a calibration table (mean actual per predicted-value bin,
which does not suffer from the selection effect of conditioning on the actual value), and paired day-block bootstrap
confidence intervals for baseline -> weighted differences. A second baseline seed (and any other pair) gives the
training-noise yardstick: |metric(seed 1) - metric(seed 0)|.

Usage:
    python scripts/loss_experiment_eval.py --baseline runs/cpcb_long/H_full_model/best.pt \
        --weighted runs/loss_exp/H_full_model_freqw/best.pt --noise runs/loss_exp_seed1/H_full_model/best.pt \
        --out runs/loss_exp/comparison.json
"""
import argparse
import json
import os

import _pathfix  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aqf.data.dataset import PM25_IDX, AQFWindowDataset, make_split
from aqf.training.train import build_model, load_raw

SEASONS = {
    "stubble+Diwali (Oct 10-Nov 30)": lambda m, d: ((m == 10) & (d >= 10)) | (m == 11),
    "winter smog (Dec-Feb)": lambda m, d: np.isin(m, [12, 1, 2]),
    "spring/dust (Mar-Jun)": lambda m, d: np.isin(m, [3, 4, 5, 6]),
    "monsoon / early post-monsoon (Jul-Oct 9)": lambda m, d: np.isin(m, [7, 8, 9]) | ((m == 10) & (d < 10)),
}
THRESHOLDS = [200, 400, 600]
VALUE_BINS = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 1e9)]
QUANTILES = [5, 25, 50, 75, 90, 95, 99, 99.9]


def predict(ckpt_path, ds):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"], "cpu")
    model.load_state_dict(ck["model_state"])
    model.eval()
    P = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, shuffle=False):
            P.append(model(b)["pm25_mean"].numpy())
    return np.concatenate(P), float(ck["val_mae"]), int(ck["epoch"]) + 1


def mae(P, T, O, mask=None):
    m = O if mask is None else O & mask
    return float((np.abs(P - T) * m).sum() / max(m.sum(), 1))


def bias(P, T, O, mask=None):
    m = O if mask is None else O & mask
    return float(((P - T) * m).sum() / max(m.sum(), 1))


def metrics(P, T, O, PS, season_of, horizons):
    r = {"test_mae": mae(P, T, O), "bias": bias(P, T, O), "per_horizon": {}, "seasons": {}, "extreme": {}, "value_bins": {}}
    for i, h in enumerate(horizons):
        mh, mp = mae(P[..., i], T[..., i], O[..., i]), mae(PS[..., i], T[..., i], O[..., i])
        r["per_horizon"][f"+{h}h"] = {"mae": mh, "persistence_mae": mp, "skill": 1 - mh / mp, "bias": bias(P[..., i], T[..., i], O[..., i])}
    r["skill_overall"] = 1 - r["test_mae"] / mae(PS, T, O)
    for k, name in enumerate(SEASONS):
        mk = (season_of == k)[:, None, None] & np.ones_like(O)
        mh, mp = mae(P, T, O, mk), mae(PS, T, O, mk)
        r["seasons"][name] = {"n": int((O & mk).sum()), "mae": mh, "bias": bias(P, T, O, mk), "persistence_mae": mp, "skill": 1 - mh / mp}
    for t in THRESHOLDS:
        a = T > t
        pred_hi = P > t
        n_a = int((a & O).sum())
        r["extreme"][f">{t}"] = {
            "n": n_a, "mae": mae(P, T, O, a), "bias": bias(P, T, O, a), "mean_actual": float(T[a & O].mean()), "mean_pred": float(P[a & O].mean()),
            "persistence_mae": mae(PS, T, O, a),
            "recall": float((pred_hi & a & O).sum() / max(n_a, 1)),
            "precision": float((pred_hi & a & O).sum() / max((pred_hi & O).sum(), 1)),
            "n_pred_above": int((pred_hi & O).sum()),
        }
    for lo, hi in VALUE_BINS:
        mk = (T >= lo) & (T < hi)
        lab = f"{int(lo)}-{int(hi)}" if hi < 1e8 else f">{int(lo)}"
        r["value_bins"][lab] = {"n": int((mk & O).sum()), "mae": mae(P, T, O, mk), "bias": bias(P, T, O, mk), "persistence_mae": mae(PS, T, O, mk)}
    big = (T > 400) & O
    r["gt400_by_horizon"] = {f"+{h}h": {"n": int(big[..., i].sum()), "mean_actual": float(T[..., i][big[..., i]].mean()),
                                        "mean_pred": float(P[..., i][big[..., i]].mean()), "mae": mae(P[..., i], T[..., i], O[..., i], T[..., i] > 400),
                                        "persistence_mae": mae(PS[..., i], T[..., i], O[..., i], T[..., i] > 400)} for i, h in enumerate(horizons)}
    pa, ta = P[O], T[O]
    r["distribution"] = {
        "quantiles_pred": {str(q): float(np.percentile(pa, q)) for q in QUANTILES},
        "quantiles_actual": {str(q): float(np.percentile(ta, q)) for q in QUANTILES},
        "mean_pred": float(pa.mean()), "mean_actual": float(ta.mean()),
        "share_pred_above": {str(t): float((pa > t).mean()) for t in THRESHOLDS},
        "share_actual_above": {str(t): float((ta > t).mean()) for t in THRESHOLDS},
    }
    edges = np.percentile(pa, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99, 100])
    cal = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (pa >= lo) & (pa <= hi)
        cal.append({"pred_range": [float(lo), float(hi)], "n": int(mk.sum()), "mean_pred": float(pa[mk].mean()), "mean_actual": float(ta[mk].mean())})
    r["calibration"] = cal
    return r


def day_block_bootstrap(P_b, P_w, T, O, day_id, n_boot=2000, seed=0):
    """Paired bootstrap over days: CI for (weighted - baseline) on test MAE, MAE on actual>400, and bias on actual>400."""
    rng = np.random.default_rng(seed)
    days = np.unique(day_id)
    groups = [np.where(day_id == d)[0] for d in days]

    def per_day(fn):
        return np.array([fn(g) for g in groups])

    def sums(P, mask_fn):
        num_abs = per_day(lambda g: (np.abs(P[g] - T[g]) * (O[g] & mask_fn(g))).sum())
        num_sig = per_day(lambda g: ((P[g] - T[g]) * (O[g] & mask_fn(g))).sum())
        den = per_day(lambda g: (O[g] & mask_fn(g)).sum())
        return num_abs, num_sig, den

    out = {}
    for name, mask_fn in {"overall": lambda g: np.ones_like(O[g]), "actual>400": lambda g: T[g] > 400}.items():
        ab, sb, db = sums(P_b, mask_fn)
        aw, sw, dw = sums(P_w, mask_fn)
        for metric, nb, nw in [("mae", ab, aw), ("bias", sb, sw)]:
            diffs = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(days), len(days))
                d = max(db[idx].sum(), 1)
                diffs.append(nw[idx].sum() / d - nb[idx].sum() / d)
            out[f"{name}:{metric}"] = {"diff_mean": float(np.mean(diffs)), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--weighted", required=True)
    p.add_argument("--noise", default=None, help="a second-seed baseline checkpoint (training-noise yardstick)")
    p.add_argument("--out", default="runs/loss_exp/comparison.json")
    p.add_argument("--fig-dir", default="docs/figures")
    p.add_argument("--label-weighted", default="frequency-weighted")
    args = p.parse_args()

    ck = torch.load(args.baseline, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    raw = load_raw(cfg)
    split = make_split(raw, cfg.data, stride_hours=cfg.data.stride_hours, seed=cfg.train.seed)
    ds = AQFWindowDataset(raw, cfg.data, split.test_t0)
    horizons = cfg.data.horizons_hours

    T, O, PS, t0 = [], [], [], []
    for b in DataLoader(ds, batch_size=64, shuffle=False):
        T.append(b["y_pm25"].numpy()); O.append(b["y_observed"].numpy().astype(bool))
        PS.append(np.repeat(b["local_seq"][:, -1, :, PM25_IDX].numpy()[:, :, None], len(horizons), -1)); t0.append(b["t0"].numpy())
    T, O, PS, t0 = [np.concatenate(x) for x in (T, O, PS, t0)]
    ist = raw.timestamps[t0] + pd.Timedelta(hours=5, minutes=30)
    m, d = ist.month.to_numpy(), ist.day.to_numpy()
    season_of = np.full(len(t0), -1)
    for k, f in enumerate(SEASONS.values()):
        season_of[f(m, d)] = k
    day_id = (ist.year.to_numpy() * 10000 + m * 100 + d)

    models = {"baseline (seed 0)": args.baseline, args.label_weighted: args.weighted}
    if args.noise:
        models["baseline (seed 1)"] = args.noise
    preds, res, meta = {}, {}, {}
    for name, path in models.items():
        preds[name], val_mae, best_ep = predict(path, ds)
        res[name] = metrics(preds[name], T, O, PS, season_of, horizons)
        res[name]["val_mae"], res[name]["best_epoch"] = val_mae, best_ep
        print(f"scored {name}: test MAE {res[name]['test_mae']:.2f}, val MAE {val_mae:.2f}, best epoch {best_ep}", flush=True)

    boot = day_block_bootstrap(preds["baseline (seed 0)"], preds[args.label_weighted], T, O, day_id)
    res["_bootstrap_weighted_minus_baseline"] = boot
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    names = list(models)
    print("\n=== headline ===")
    print(f"{'':28s}" + "".join(f"{n:>24s}" for n in names))
    def row(label, fn, fmt="{:.2f}"):
        print(f"{label:28s}" + "".join(f"{fmt.format(fn(res[n])):>24s}" for n in names))
    row("val MAE (checkpoint)", lambda r: r["val_mae"]); row("test MAE", lambda r: r["test_mae"]); row("test bias (pred-actual)", lambda r: r["bias"], "{:+.2f}")
    for h in res[names[0]]["per_horizon"]:
        row(f"skill {h}", lambda r, h=h: r["per_horizon"][h]["skill"], "{:+.3f}")
    print("\n=== seasons: MAE | bias | skill ===")
    for s in SEASONS:
        print(f"{s:42s}" + "".join(f"  {res[n]['seasons'][s]['mae']:6.2f} | {res[n]['seasons'][s]['bias']:+6.2f} | {res[n]['seasons'][s]['skill']:+.3f}" for n in names))
    print("\n=== actual > threshold: n | MAE | bias | recall | precision ===")
    for t in THRESHOLDS:
        k = f">{t}"
        print(f"{k:6s} n={res[names[0]]['extreme'][k]['n']:5d} " + "".join(
            f"  {res[n]['extreme'][k]['mae']:6.1f} | {res[n]['extreme'][k]['bias']:+7.1f} | {res[n]['extreme'][k]['recall']:.2f} | {res[n]['extreme'][k]['precision']:.2f}" for n in names))
    print("\n=== by actual value bin: MAE | bias ===")
    for b in res[names[0]]["value_bins"]:
        print(f"{b:9s} n={res[names[0]]['value_bins'][b]['n']:6d} " + "".join(f"  {res[n]['value_bins'][b]['mae']:6.1f} | {res[n]['value_bins'][b]['bias']:+7.1f}" for n in names))
    print("\n=== paired day-block bootstrap, weighted minus baseline(seed 0) (95% CI) ===")
    for k, v in boot.items():
        print(f"{k:22s} {v['diff_mean']:+7.2f}   [{v['ci95'][0]:+7.2f}, {v['ci95'][1]:+7.2f}]")

    if args.noise:
        # Training-noise yardstick: how far does the *same* recipe move when only the seed changes?
        # With a single extra seed this is one noise sample, so "exceeds noise" is a rough screen, not a test.
        paths = [("test MAE", ("test_mae",)), ("overall bias", ("bias",))]
        paths += [(f"skill {h}", ("per_horizon", h, "skill")) for h in res[names[0]]["per_horizon"]]
        for s in SEASONS:
            paths += [(f"MAE  {s[:22]}", ("seasons", s, "mae")), (f"bias {s[:22]}", ("seasons", s, "bias"))]
        for t in THRESHOLDS:
            paths += [(f"MAE  actual>{t}", ("extreme", f">{t}", "mae")), (f"bias actual>{t}", ("extreme", f">{t}", "bias")), (f"recall actual>{t}", ("extreme", f">{t}", "recall"))]
        paths += [("MAE  actual<100", ("value_bins", "0-100", "mae")), ("bias actual<100", ("value_bins", "0-100", "bias")),
                  ("MAE  actual 100-200", ("value_bins", "100-200", "mae")), ("bias actual 100-200", ("value_bins", "100-200", "bias"))]

        def get(r, path):
            for k in path:
                r = r[k]
            return r

        print("\n=== change from the loss vs seed-to-seed noise ===")
        print(f"{'metric':36s} {'baseline s0':>12s} {'weighted':>10s} {'d_loss':>9s} {'d_seed (s1-s0)':>15s} {'|d_loss|/|d_seed|':>18s}")
        noise_table = {}
        for label, path in paths:
            b0, w, b1 = (get(res[n], path) for n in (names[0], names[1], names[2]))
            d_loss, d_seed = w - b0, b1 - b0
            ratio = abs(d_loss) / max(abs(d_seed), 1e-9)
            noise_table[label] = {"baseline_s0": b0, "weighted": w, "baseline_s1": b1, "d_loss": d_loss, "d_seed": d_seed, "ratio": ratio}
            print(f"{label:36s} {b0:12.3f} {w:10.3f} {d_loss:+9.3f} {d_seed:+15.3f} {ratio:18.2f}")
        res["_noise_comparison"] = noise_table
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)

    # figures
    os.makedirs(args.fig_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    bins = np.arange(0, 1050, 25)
    ax[0].hist(np.clip(T[O], 0, 1049), bins=bins, histtype="step", lw=2, color="black", label="actual", density=True)
    for n, c in zip(names, ["#2b59c3", "#d9453b", "#8fb83a"]):
        ax[0].hist(np.clip(preds[n][O], 0, 1049), bins=bins, histtype="step", lw=1.5, color=c, label=n, density=True)
    ax[0].set_yscale("log"); ax[0].set_xlabel("PM2.5 (ug/m3)"); ax[0].set_title("test year: predictions vs actual (log density)"); ax[0].legend(fontsize=8)
    for n, c in zip(names, ["#2b59c3", "#d9453b", "#8fb83a"]):
        cal = res[n]["calibration"]
        ax[1].plot([x["mean_pred"] for x in cal], [x["mean_actual"] for x in cal], "o-", color=c, label=n, ms=4)
    lim = max(x["mean_actual"] for x in res[names[0]]["calibration"]) * 1.1
    ax[1].plot([0, lim], [0, lim], ":", color="grey"); ax[1].set_xlabel("mean predicted (per prediction-decile bin)"); ax[1].set_ylabel("mean actual")
    ax[1].set_title("calibration: above the line = under-prediction"); ax[1].legend(fontsize=8)
    labs = list(res[names[0]]["value_bins"])
    w = 0.8 / len(names)
    for j, (n, c) in enumerate(zip(names, ["#2b59c3", "#d9453b", "#8fb83a"])):
        ax[2].bar(np.arange(len(labs)) + j * w, [res[n]["value_bins"][b]["bias"] for b in labs], w, color=c, label=n)
    ax[2].axhline(0, color="black", lw=0.8); ax[2].set_xticks(np.arange(len(labs)) + 0.4 - w / 2); ax[2].set_xticklabels(labs)
    ax[2].set_xlabel("actual PM2.5 bin"); ax[2].set_ylabel("mean (pred - actual)"); ax[2].set_title("bias by actual value"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.fig_dir, "frequency_weighted_loss_comparison.png"), dpi=130)
    print("saved", os.path.join(args.fig_dir, "frequency_weighted_loss_comparison.png"), "and", args.out)


if __name__ == "__main__":
    main()
