"""Would a regime-aware mixture of experts help? Measure, don't guess.

1. Error, bias and skill of one model by season (a proxy for regime: stubble burning / winter smog / dust / monsoon).
2. "Oracle selection": if we could pick the best of our already-trained models *per season* (what a perfect gate
   over experts would do at best), how much would test error drop vs the single best model?
3. The noise floor for that number: the same oracle, but over three seeds of the *same* architecture.
   If architectures give no more oracle gain than seeds do, the differences are noise and experts won't help.

Usage: python scripts/regime_diagnostic.py
"""
import os

import _pathfix  # noqa: F401
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aqf.data.dataset import PM25_IDX, AQFWindowDataset, make_split
from aqf.training.train import build_model, load_raw

MODELS = {  # name -> checkpoint
    "A": "runs/real_suite_v2/A_baseline", "B": "runs/real_suite_v2/B_wind_graph", "C": "runs/real_suite_v2/C_blh",
    "E": "runs/real_suite_v2/E_external_sources", "G10": "runs/real_suite_v2/G_physics_constraint",
    "H10": "runs/real_suite_v2/H_full_model", "G30": "runs/real_long/G_physics_constraint",
    "G30cal": "runs/real_long/G2_physics_calendar", "H30": "runs/real_long/H_full_model",
    "H30cal": "runs/real_long/H2_full_calendar",
    "A_s1": "runs/real_seeds/seed1/A_baseline", "A_s2": "runs/real_seeds/seed2/A_baseline",
    "G_s1": "runs/real_seeds/seed1/G_physics_constraint", "G_s2": "runs/real_seeds/seed2/G_physics_constraint",
}
SEASONS = {"stubble+Diwali (Oct10-Nov30)": lambda m, d: (m == 10 and d >= 10) or m == 11,
           "winter smog (Dec-Feb)": lambda m, d: m in (12, 1, 2),
           "spring/dust (Mar-Jun)": lambda m, d: m in (3, 4, 5, 6),
           "monsoon (Jul-Sep)": lambda m, d: m in (7, 8, 9) or (m == 10 and d < 10)}


def main():
    first = torch.load(os.path.join(MODELS["A"], "best.pt"), map_location="cpu", weights_only=False)["cfg"]
    raw = load_raw(first)
    split = make_split(raw, first.data, stride_hours=first.data.stride_hours, seed=first.train.seed)
    ds = AQFWindowDataset(raw, first.data, split.test_t0)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    preds, S, targets, obs, persist, t0s = {}, [], [], [], [], []
    for name, d in MODELS.items():
        ck = torch.load(os.path.join(d, "best.pt"), map_location="cpu", weights_only=False)
        m = build_model(ck["cfg"], "cpu"); m.load_state_dict(ck["model_state"]); m.eval()
        out_p = []
        with torch.no_grad():
            for b in loader:
                o = m(b)
                out_p.append(o["pm25_mean"].numpy())
                if name == "H30":
                    S.append(o["_S_t_final"].numpy())
                    if not t0s:
                        pass
                    targets.append(b["y_pm25"].numpy()); obs.append(b["y_observed"].numpy().astype(bool))
                    persist.append(np.repeat(b["local_seq"][:, -1, :, PM25_IDX].numpy()[:, :, None], b["y_pm25"].shape[-1], -1))
                    t0s.append(b["t0"].numpy())
        preds[name] = np.concatenate(out_p)
        print("scored", name, flush=True)
    S = np.concatenate(S); targets = np.concatenate(targets); obs = np.concatenate(obs)
    persist = np.concatenate(persist); t0 = np.concatenate(t0s)
    ts = raw.timestamps[t0] + pd.Timedelta(hours=5, minutes=30)
    month, day = ts.month.to_numpy(), ts.day.to_numpy()
    season_of = np.full(len(t0), -1)
    for k, f in enumerate(SEASONS.values()):
        season_of[np.array([f(a, b) for a, b in zip(month, day)])] = k

    def mae(p, mask):
        e = np.abs(p - targets) * obs
        return e[mask].sum() / max(obs[mask].sum(), 1)

    print("\n=== H (30 epochs) by season: error, bias, skill vs persistence (all horizons pooled) ===")
    for k, sname in enumerate(SEASONS):
        mk = season_of == k
        if not mk.any():
            continue
        o = obs[mk]; bias = ((preds["H30"][mk] - targets[mk]) * o).sum() / o.sum()
        mh, mp = mae(preds["H30"], mk), mae(persist, mk)
        print(f"{sname:30s} windows {mk.sum():5d}  PM mean {np.sum(targets[mk]*o)/o.sum():6.1f}  MAE {mh:6.2f}  bias {bias:+6.2f}  persistence {mp:6.2f}  skill {1-mh/mp:+.3f}")

    print("\n=== H (30 epochs) by stability index S_t at issue time (terciles) ===")
    q = np.quantile(S, [1/3, 2/3])
    for lab, mk in [("low S_t", S <= q[0]), ("mid S_t", (S > q[0]) & (S <= q[1])), ("high S_t (stable)", S > q[1])]:
        mh, mp = mae(preds["H30"], mk), mae(persist, mk)
        print(f"{lab:20s} windows {mk.sum():5d}  MAE {mh:6.2f}  persistence {mp:6.2f}  skill {1-mh/mp:+.3f}")

    def oracle(names):
        """Overall MAE if a perfect gate picked the best of `names` per season (and per horizon), vs best single of `names`."""
        tot_e, tot_n = 0.0, 0.0
        for k in range(len(SEASONS)):
            for h in range(targets.shape[-1]):
                mk = (season_of == k)
                o = obs[mk][..., h]
                if o.sum() == 0:
                    continue
                errs = [(np.abs(preds[n][mk][..., h] - targets[mk][..., h]) * o).sum() for n in names]
                tot_e += min(errs); tot_n += o.sum()
        best_single = min(mae(preds[n], np.ones(len(t0), bool)) for n in names)
        return tot_e / tot_n, best_single

    print("\n=== Oracle gate over trained models (a perfect season-wise gate, selected on the test set itself, so optimistic) ===")
    sets = {
        "different architectures (A,B,C,E,G10,H10,G30,G30cal,H30,H30cal)": ["A", "B", "C", "E", "G10", "H10", "G30", "G30cal", "H30", "H30cal"],
        "NOISE FLOOR: 3 seeds of A": ["A", "A_s1", "A_s2"],
        "NOISE FLOOR: 3 seeds of G (10 epochs)": ["G10", "G_s1", "G_s2"],
    }
    for lab, names in sets.items():
        o, b = oracle(names)
        print(f"{lab:68s} best single {b:6.2f}  oracle gate {o:6.2f}  gain {100*(b-o)/b:4.2f}%")


if __name__ == "__main__":
    main()
