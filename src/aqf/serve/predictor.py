"""Inference wrapper used by the web API: load a trained checkpoint + a replay dataset, forecast from any hour in it.

"Replay" means the demo walks through stored hourly data (observed PM2.5 and weather) instead of a live feed:
ERA5 weather lags real time by ~5 days, so a live forecast would need a different weather source. A replay
over held-out hours also lets the dashboard show forecast vs what actually happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from aqf.data.dataset import PM25_IDX, AQFWindowDataset
from aqf.data.schema import REGIME_CATEGORIES, SOURCE_CATEGORIES, RawSeries
from aqf.graph.stations import LOCAL_STATIONS
from aqf.losses.physics import PM_SCALE
from aqf.training.train import build_model

# CPCB AQI categories for PM2.5 (24h, ug/m3)
_BANDS = [(30, "Good"), (60, "Satisfactory"), (90, "Moderate"), (120, "Poor"), (250, "Very poor"), (1e9, "Severe")]


def band(pm25: float) -> str:
    for upper, name in _BANDS:
        if pm25 <= upper:
            return name
    return "Severe"


class Predictor:
    def __init__(self, checkpoint_path: str, replay_path: str, device: str = "cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.cfg = ckpt["cfg"]
        self.cfg.train.device = device
        self.device = device
        self.model = build_model(self.cfg, device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.run_name = self.cfg.name
        self.val_mae = float(ckpt.get("val_mae", float("nan")))

        raw = RawSeries.load(replay_path)
        if np.isnan(raw.local).any():
            from aqf.data.real_pipeline import impute_for_training

            raw = impute_for_training(raw)
        self.raw = raw
        self.horizons = tuple(self.cfg.data.horizons_hours)

        lo = self.cfg.data.lookback_hours - 1
        hi = len(raw.timestamps) - max(self.horizons) - 1  # keep the +24h actual inside the data
        self.first_idx, self.last_idx = lo, hi
        self.ds = AQFWindowDataset(raw, self.cfg.data, np.arange(lo, hi + 1))
        self.stations = [{"id": s.id, "name": s.name, "lat": s.lat, "lon": s.lon} for s in LOCAL_STATIONS]

    # ---- metadata ----
    def meta(self) -> dict:
        ts = self.raw.timestamps
        return {
            "model": self.run_name,
            "horizons_hours": list(self.horizons),
            "first": str(ts[self.first_idx]),
            "last": str(ts[self.last_idx]),
            "n_steps": int(self.last_idx - self.first_idx + 1),
            "stations": self.stations,
            "has_source_head": hasattr(self.model, "source_head") and self.model.source_head is not None,
            "has_regime_head": hasattr(self.model, "regime_head") and self.model.regime_head is not None,
            "source_categories": SOURCE_CATEGORIES,
            "regime_categories": REGIME_CATEGORIES,
        }

    def index_for(self, when: str | None) -> int:
        if when is None:
            return self.last_idx
        t = pd.Timestamp(when)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        idx = int(self.raw.timestamps.get_indexer([t.floor("h")], method="nearest")[0])
        return int(np.clip(idx, self.first_idx, self.last_idx))

    # ---- forecast ----
    @torch.no_grad()
    def forecast(self, when: str | None = None) -> dict:
        idx = self.index_for(when)
        sample = self.ds[idx - self.first_idx]
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in sample.items()}
        out = self.model(batch)

        mean = out["pm25_mean"][0].cpu().numpy()                                  # (N, H)
        std = (torch.exp(0.5 * out["pm25_logvar"][0]) * PM_SCALE).cpu().numpy()
        exceed = out["exceed_prob"][0].cpu().numpy()
        src = out["source_contrib"][0].cpu().numpy() if "source_contrib" in out else None
        reg = torch.softmax(out["regime_logits"][0], dim=-1).cpu().numpy() if "regime_logits" in out else None

        last = sample["local_seq"][-1, :, PM25_IDX].numpy()
        actual = sample["y_pm25"].numpy()
        observed = sample["y_observed"].numpy().astype(bool)

        stations = []
        for i, st in enumerate(self.stations):
            fc = []
            for j, h in enumerate(self.horizons):
                m = float(max(mean[i, j], 0.0))
                s = float(std[i, j])
                fc.append({
                    "horizon_h": h,
                    "pm25": round(m, 1),
                    "lo95": round(max(m - 1.96 * s, 0.0), 1),
                    "hi95": round(m + 1.96 * s, 1),
                    "exceed_prob": round(float(exceed[i, j]), 3),
                    "band": band(m),
                    "actual": round(float(actual[i, j]), 1) if observed[i, j] else None,
                })
            entry = {**st, "current_pm25": round(float(last[i]), 1), "current_band": band(float(last[i])), "forecast": fc}
            if src is not None:
                entry["sources"] = {c: round(float(v), 3) for c, v in zip(SOURCE_CATEGORIES, src[i])}
            if reg is not None:
                k = int(reg[i].argmax())
                entry["regime"] = {"label": REGIME_CATEGORIES[k], "probs": {c: round(float(v), 3) for c, v in zip(REGIME_CATEGORIES, reg[i])}}
            stations.append(entry)

        return {
            "issued_at": str(self.raw.timestamps[idx]),
            "stability_index": round(float(out["_S_t_final"][0]), 3),
            "stations": stations,
        }

    def series(self, station_id: str, hours: int = 72, end: str | None = None) -> dict:
        """Observed PM2.5 for one station over the `hours` before `end` (for the dashboard's trend chart)."""
        i = [s["id"] for s in self.stations].index(station_id)
        end_idx = self.index_for(end)
        lo = max(end_idx - hours + 1, 0)
        vals = self.raw.local[lo:end_idx + 1, i, PM25_IDX]
        obs = (
            self.raw.local_observed_mask[lo:end_idx + 1, i, PM25_IDX]
            if self.raw.local_observed_mask is not None
            else np.ones_like(vals, dtype=bool)
        )
        return {
            "station": station_id,
            "timestamps": [str(t) for t in self.raw.timestamps[lo:end_idx + 1]],
            "pm25": [round(float(v), 1) for v in vals],
            "observed": [bool(o) for o in obs],
        }
