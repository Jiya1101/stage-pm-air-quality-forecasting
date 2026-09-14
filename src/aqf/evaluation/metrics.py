"""Forecast quality metrics, including the per-regime and extreme-episode breakdowns Dr. Priya required:

"Your average MAE may look excellent while the model completely fails during
severe pollution... The proposed model gains the largest advantage precisely
when conventional models fail."
"""
from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def critical_success_index(pred: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """CSI = hits / (hits + misses + false_alarms) for an exceedance event."""
    pred_event = pred > threshold
    target_event = target > threshold
    hits = np.sum(pred_event & target_event)
    misses = np.sum(~pred_event & target_event)
    false_alarms = np.sum(pred_event & ~target_event)
    denom = hits + misses + false_alarms
    return float(hits / denom) if denom > 0 else float("nan")


def coverage(pred_mean: np.ndarray, pred_std: np.ndarray, target: np.ndarray, z: float = 1.96) -> float:
    """Fraction of targets falling inside the predicted (mean +/- z*std) interval -- should be ~0.95 for z=1.96 if well calibrated."""
    lo, hi = pred_mean - z * pred_std, pred_mean + z * pred_std
    inside = (target >= lo) & (target <= hi)
    return float(np.mean(inside))


def pollution_bucket(target: np.ndarray) -> np.ndarray:
    """Normal / Moderate / Severe / Extreme buckets per Dr. Priya's extreme-episode testing requirement."""
    bins = np.array([0, 60, 120, 250, np.inf])
    labels = np.array(["normal", "moderate", "severe", "extreme"])
    idx = np.digitize(target, bins) - 1
    idx = np.clip(idx, 0, len(labels) - 1)
    return labels[idx]


def metrics_by_bucket(pred: np.ndarray, target: np.ndarray, bucket: np.ndarray) -> dict[str, dict]:
    out = {}
    for b in np.unique(bucket):
        m = bucket == b
        if m.sum() == 0:
            continue
        out[str(b)] = {"n": int(m.sum()), "mae": mae(pred[m], target[m]), "rmse": rmse(pred[m], target[m])}
    return out
