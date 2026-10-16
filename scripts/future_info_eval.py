"""Does knowing the weather (and calendar) of the forecast period make +6 h / +24 h forecasts better, especially for sudden spikes?

Compares a baseline model with variants that are also given the future weather and/or the calendar at each forecast time.
Subsets scored (test year, observed hours only, +6 h and +24 h):
  * sudden spikes ("onsets"): last reading below 250 but the actual value above 400 - the spikes that cannot be seen coming
  * severe hours: actual above 400
  * sharp rises: actual at least 150 above the last reading
  * calendar days: the Diwali window (Oct 17-23 2025), weekends vs weekdays, and four seasons
Differences are given with a paired day-block bootstrap interval (test-day sampling only) and set against measured training noise.

Usage:
    python scripts/future_info_eval.py --baseline runs/cpcb_long/H_full_model/best.pt \
        --models HF=runs/future_test/HF_future_weather/best.pt HFC=runs/future_test/HFC_future_weather_calendar/best.pt \
                 HC=runs/future_test/HC_future_calendar/best.pt --out runs/future_test/comparison.json
"""
import argparse
import json
import os

import _pathfix  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aqf.data.dataset import PM25_IDX, AQFWindowDataset, make_split
from aqf.training.train import build_model, load_raw

HORIZON_NAMES = {0: "+1h", 1: "+6h", 2: "+24h"}
SEASONS = {
    "crop-burning + Diwali (Oct 10-Nov 30)": lambda m, d: ((m == 10) & (d >= 10)) | (m == 11),
    "winter smog (Dec-Feb)": lambda m, d: np.isin(m, [12, 1, 2]),
    "spring/dust (Mar-Jun)": lambda m, d: np.isin(m, [3, 4, 5, 6]),
    "monsoon / early post-monsoon (Jul-Oct 9)": lambda m, d: np.isin(m, [7, 8, 9]) | ((m == 10) & (d < 10)),
}


def predict(path, ds):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"], "cpu")
    model.load_state_dict(ck["model_state"])
    model.eval()
    P = []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=64, shuffle=False):
            P.append(model(b)["pm25_mean"].numpy())
    return np.concatenate(P), float(ck["val_mae"]), int(ck["epoch"]) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--models", nargs="+", required=True, help="name=checkpoint ...")
    p.add_argument("--out", default="runs/future_test/comparison.json")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    ck = torch.load(args.baseline, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    raw = load_raw(cfg)
    split = make_split(raw, cfg.data, stride_hours=cfg.data.stride_hours, seed=cfg.train.seed)
    ds = AQFWindowDataset(raw, cfg.data, split.test_t0)

    T, O, L, t0 = [], [], [], []
    for b in DataLoader(ds, batch_size=64, shuffle=False):
        T.append(b["y_pm25"].numpy()); O.append(b["y_observed"].numpy().astype(bool))
        L.append(b["local_seq"][:, -1, :, PM25_IDX].numpy()); t0.append(b["t0"].numpy())
    T, O, L, t0 = [np.concatenate(x) for x in (T, O, L, t0)]
    ist = raw.timestamps[t0] + pd.Timedelta(hours=5, minutes=30)
    m, d = ist.month.to_numpy(), ist.day.to_numpy()
    date = pd.DatetimeIndex(ist.date)
    day_id = (ist.year.to_numpy() * 10000 + m * 100 + d)
    n_w = len(t0)
    ones = np.ones((n_w, T.shape[1]), dtype=bool)
    win = lambda v: np.broadcast_to(np.asarray(v)[:, None], (n_w, T.shape[1]))

    models = {"baseline": args.baseline}
    models.update(dict(kv.split("=", 1) for kv in args.models))
    preds, meta = {}, {}
    for name, path in models.items():
        preds[name], val, ep = predict(path, ds)
        meta[name] = {"val_mae": val, "best_epoch": ep}
        print(f"scored {name}: val MAE {val:.2f} (best epoch {ep})", flush=True)

    def subsets(i):
        t, last = T[..., i], L
        s = {
            "all hours": ones,
            "sudden spikes (last < 250, actual > 400)": (last < 250) & (t > 400),
            "all severe hours (actual > 400)": t > 400,
            "sharp rises (actual - last >= 150)": (t - last) >= 150,
            "Diwali window (Oct 17-23 2025)": win((date >= "2025-10-17") & (date <= "2025-10-23")),
            "weekends": win(ist.dayofweek.to_numpy() >= 5),
            "weekdays": win(ist.dayofweek.to_numpy() < 5),
        }
        for k, f in SEASONS.items():
            s[k] = win(f(m, d))
        return s

    res = {n: {} for n in preds}
    for hi in (0, 1, 2):
        hn = HORIZON_NAMES[hi]
        pers = np.abs(L - T[..., hi])
        for sname, mask in subsets(hi).items():
            mk = mask & O[..., hi]
            n_obs = int(mk.sum())
            if n_obs == 0:
                continue
            for name, P in preds.items():
                e = np.abs(P[..., hi] - T[..., hi])
                row = {
                    "n": n_obs, "mae": float(e[mk].mean()), "bias": float((P[..., hi] - T[..., hi])[mk].mean()),
                    "persistence_mae": float(pers[mk].mean()), "mean_actual": float(T[..., hi][mk].mean()),
                    "mean_pred": float(P[..., hi][mk].mean()), "mean_last": float(L[mk].mean()),
                }
                row["skill"] = 1 - row["mae"] / row["persistence_mae"]
                jump = row["mean_actual"] - row["mean_last"]
                row["share_of_change_captured"] = (row["mean_pred"] - row["mean_last"]) / jump if abs(jump) > 1 else None
                row["recall_gt400"] = float(((P[..., hi] > 400) & (T[..., hi] > 400) & mk).sum() / max(((T[..., hi] > 400) & mk).sum(), 1))
                res[name].setdefault(hn, {})[sname] = row

    # paired day-block bootstrap: model minus baseline, ratio of sums over resampled days
    days = np.unique(day_id)
    groups = [np.where(day_id == dd)[0] for dd in days]
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(days), len(days)) for _ in range(args.n_boot)]
    boot = {n: {} for n in preds if n != "baseline"}
    for hi in (1, 2):
        hn = HORIZON_NAMES[hi]
        for sname, mask in subsets(hi).items():
            mk = mask & O[..., hi]
            if mk.sum() < 20:
                continue
            den = np.array([mk[g].sum() for g in groups], dtype=float)
            nb = np.array([(np.abs(preds["baseline"][g][..., hi] - T[g][..., hi]) * mk[g]).sum() for g in groups])
            for name in boot:
                nm = np.array([(np.abs(preds[name][g][..., hi] - T[g][..., hi]) * mk[g]).sum() for g in groups])
                diffs = np.array([(nm[ix].sum() - nb[ix].sum()) / max(den[ix].sum(), 1) for ix in boot_idx])
                boot[name].setdefault(hn, {})[sname] = {"diff": float((nm.sum() - nb.sum()) / den.sum()), "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "results": res, "bootstrap_vs_baseline": boot}, f, indent=2)

    names = list(preds)
    print("\n=== headline: test MAE and skill vs persistence (all hours) ===")
    for hi in (0, 1, 2):
        hn = HORIZON_NAMES[hi]
        print(f"{hn:5s} " + "  ".join(f"{n}: {res[n][hn]['all hours']['mae']:6.2f} (skill {res[n][hn]['all hours']['skill']:+.3f})" for n in names))
    for hn in ("+6h", "+24h"):
        print(f"\n=== {hn}: MAE | bias by subset (n) ===")
        for sname in res["baseline"][hn]:
            print(f"{sname:44s} n={res['baseline'][hn][sname]['n']:6d} " + "  ".join(
                f"{n}: {res[n][hn][sname]['mae']:6.1f} | {res[n][hn][sname]['bias']:+7.1f}" for n in names))
        print(f"--- share of the actual rise captured (sudden spikes), {hn}:", {n: (None if res[n][hn].get('sudden spikes (last < 250, actual > 400)', {}).get('share_of_change_captured') is None else round(res[n][hn]['sudden spikes (last < 250, actual > 400)']['share_of_change_captured'], 2)) for n in names})
        print(f"--- recall of >400 among severe hours, {hn}:", {n: round(res[n][hn]["all severe hours (actual > 400)"]["recall_gt400"], 2) for n in names})
    print("\n=== paired day-block bootstrap, model minus baseline MAE (95% CI; test-day sampling only) ===")
    for name in boot:
        for hn in boot[name]:
            for sname, v in boot[name][hn].items():
                if sname in ("all hours", "sudden spikes (last < 250, actual > 400)", "all severe hours (actual > 400)", "Diwali window (Oct 17-23 2025)", "weekends"):
                    print(f"{name:6s} {hn:5s} {sname:44s} {v['diff']:+8.2f}  [{v['ci95'][0]:+8.2f}, {v['ci95'][1]:+8.2f}]")


if __name__ == "__main__":
    main()
