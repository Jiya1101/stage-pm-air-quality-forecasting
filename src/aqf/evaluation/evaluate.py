"""Full evaluation: overall metrics, per-pollution-bucket, per-regime, and spatial-holdout generalization.

This is what Experiments A-H (training/ablation.py) are each run through, and
what H1/H2/H3 are ultimately judged against.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from aqf.data.dataset import AQFWindowDataset
from aqf.data.schema import REGIME_CATEGORIES
from aqf.evaluation.metrics import coverage, critical_success_index, mae, metrics_by_bucket, pollution_bucket, rmse
from aqf.losses.physics import PM_SCALE
from aqf.models.stage_pm import StagePM


@torch.no_grad()
def collect_predictions(model: StagePM, dataset: AQFWindowDataset, device: str, batch_size: int = 32):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, stds, targets, observed_all, regime_all = [], [], [], [], []
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        out = model(batch_dev)
        preds.append(out["pm25_mean"].cpu().numpy())
        # logvar is defined in PM_SCALE-normalized units (see losses/physics.py) -> convert back to real PM2.5 std.
        stds.append((torch.exp(0.5 * out["pm25_logvar"]) * PM_SCALE).cpu().numpy())
        targets.append(batch["y_pm25"].numpy())
        observed_all.append(batch["y_observed"].numpy() if "y_observed" in batch else np.ones_like(batch["y_pm25"].numpy()))
        if "y_regime" in batch:
            regime_all.append(batch["y_regime"].numpy())
    preds = np.concatenate(preds, axis=0)     # (Ntest, N_local, H)
    stds = np.concatenate(stds, axis=0)
    targets = np.concatenate(targets, axis=0)
    observed = np.concatenate(observed_all, axis=0).astype(bool)  # False where the real target was imputed, not observed
    regime = np.concatenate(regime_all, axis=0) if regime_all else None
    return preds, stds, targets, observed, regime


def evaluate(
    model: StagePM,
    test_ds: AQFWindowDataset,
    held_out_local_mask: np.ndarray,
    exceedance_threshold: float,
    horizons: tuple,
    device: str = "cpu",
) -> dict:
    preds, stds, targets, observed, regime = collect_predictions(model, test_ds, device)
    in_sample = ~held_out_local_mask
    report: dict = {"overall": {}, "per_horizon": {}, "per_bucket": {}, "per_regime": {}, "spatial_holdout": {}}

    # Every metric below is computed only over genuinely-observed target hours (see
    # data/real_pipeline.py::impute_for_training) -- on real data, ~65% of a naive dense
    # array would otherwise be scoring interpolation smoothness, not forecasting skill.
    report["overall"] = {
        "mae": mae(preds[observed], targets[observed]), "rmse": rmse(preds[observed], targets[observed]),
        "uncertainty_coverage_95pct": coverage(preds[observed], stds[observed], targets[observed]),
        "observed_fraction": float(observed.mean()),
    }

    for h_i, h in enumerate(horizons):
        p, t, o = preds[:, :, h_i], targets[:, :, h_i], observed[:, :, h_i]
        report["per_horizon"][f"+{h}h"] = {
            "mae": mae(p[o], t[o]), "rmse": rmse(p[o], t[o]),
            "csi": critical_success_index(p[o], t[o], exceedance_threshold),
        }

    bucket = pollution_bucket(targets)
    report["per_bucket"] = metrics_by_bucket(preds[observed], targets[observed], bucket[observed])

    if regime is not None:
        regime_expanded = np.repeat(regime[:, :, None], preds.shape[-1], axis=-1)
        for r_id, r_name in enumerate(REGIME_CATEGORIES):
            m = (regime_expanded == r_id) & observed
            if m.sum() == 0:
                continue
            report["per_regime"][r_name] = {"n": int(m.sum()), "mae": mae(preds[m], targets[m]), "rmse": rmse(preds[m], targets[m])}

    in_sample_m = np.zeros_like(observed)
    in_sample_m[:, in_sample, :] = True
    in_sample_m &= observed
    held_out_m = np.zeros_like(observed)
    held_out_m[:, held_out_local_mask, :] = True
    held_out_m &= observed
    report["spatial_holdout"] = {
        "in_sample_stations": {"mae": mae(preds[in_sample_m], targets[in_sample_m])},
        "held_out_stations": {"mae": mae(preds[held_out_m], targets[held_out_m])}
        if held_out_m.any() else None,
    }

    return report


def print_report(name: str, report: dict) -> None:
    print(f"\n===== {name} =====")
    print(f"Overall: MAE={report['overall']['mae']:.2f}  RMSE={report['overall']['rmse']:.2f}  "
          f"95% interval coverage={report['overall']['uncertainty_coverage_95pct']:.3f} (target ~0.95)  "
          f"[scored on {report['overall']['observed_fraction']:.1%} of hours that were genuinely observed]")
    print("Per horizon:")
    for h, m in report["per_horizon"].items():
        print(f"  {h}: MAE={m['mae']:.2f} RMSE={m['rmse']:.2f} CSI={m['csi']:.3f}")
    print("Per pollution bucket:")
    for b, m in report["per_bucket"].items():
        print(f"  {b}: n={m['n']} MAE={m['mae']:.2f} RMSE={m['rmse']:.2f}")
    if report["per_regime"]:
        print("Per regime:")
        for r, m in report["per_regime"].items():
            print(f"  {r}: n={m['n']} MAE={m['mae']:.2f} RMSE={m['rmse']:.2f}")
    sh = report["spatial_holdout"]
    print(f"Spatial holdout: in-sample MAE={sh['in_sample_stations']['mae']:.2f}"
          + (f"  held-out MAE={sh['held_out_stations']['mae']:.2f}" if sh["held_out_stations"] else "  (no held-out stations)"))
