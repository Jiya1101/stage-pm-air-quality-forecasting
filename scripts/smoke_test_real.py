"""Quick smoke test: does STAGE-PM train and produce sane, honest numbers on real OpenAQ+FIRMS(+ERA5) data?

Uses data/real_pipeline.py::impute_for_training, which fills gaps for the
model's inputs but records which target hours were genuinely observed vs.
imputed (raw.local_observed_mask). losses/physics.py::forecast_loss and
evaluation/evaluate.py both mask on this, so -- unlike an earlier version of
this script -- the reported val_mae here is scored only against real CPCB
sensor readings, not interpolated filler. Still a *smoke* test though: the
fetched window may be short and the train/val split is time-ordered rather
than by calendar year (too short a pull for that), so treat this as "does
the pipeline work correctly," not a final accuracy benchmark.

Usage:
    python scripts/smoke_test_real.py
"""
import _pathfix  # noqa: F401
import numpy as np
import torch

from aqf.config import Config
from aqf.data.dataset import AQFWindowDataset, PM25_IDX
from aqf.data.real_pipeline import impute_for_training
from aqf.data.schema import RawSeries
from aqf.losses.physics import total_loss
from aqf.training.train import build_model, _val_mae


def main():
    raw = RawSeries.load("data/real/raw.npz")
    pm25_observed_frac = 1.0 - np.isnan(raw.local[..., PM25_IDX]).mean()
    print(f"Loaded real data: {len(raw.timestamps)} hours, {raw.local.shape[1]} stations, "
          f"PM2.5 observed: {pm25_observed_frac:.1%}")

    raw = impute_for_training(raw)
    print("Gaps filled for model input; genuinely-observed hours tracked separately for scoring.")

    cfg = Config(name="real_smoke")
    cfg.data.source = "real"
    cfg.data.lookback_hours = 24
    cfg.data.horizons_hours = (1, 6, 24)
    cfg.model.hidden_dim = 32
    cfg.model.n_operator_layers = 2
    cfg.model.n_transformer_layers = 1
    cfg.train.epochs = 8
    cfg.train.batch_size = 8
    cfg.train.lr = 1e-3
    # real data has no synthetic ground-truth source-attribution/regime labels -- those heads
    # still run (full ablation config) but their loss terms are skipped automatically since
    # AQFWindowDataset won't attach y_source_contrib/y_regime when raw.source_contrib_gt is None.

    T = len(raw.timestamps)
    max_h = max(cfg.data.horizons_hours)
    lo, hi = cfg.data.lookback_hours - 1, T - max_h - 1
    all_t0 = np.arange(lo, hi, 2)  # stride 2h -- short window, so don't skip too much
    split_idx = int(len(all_t0) * 0.8)
    train_t0, val_t0 = all_t0[:split_idx], all_t0[split_idx:]
    print(f"Windows: {len(train_t0)} train, {len(val_t0)} val (time-ordered split, not by calendar year -- "
          f"too short a pull for that; see data/dataset.py::make_split for the real multi-year version)")

    train_ds = AQFWindowDataset(raw, cfg.data, train_t0)
    val_ds = AQFWindowDataset(raw, cfg.data, val_t0)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = build_model(cfg, "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    station_mask = torch.ones(raw.local.shape[1], dtype=torch.bool)

    for epoch in range(cfg.train.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            out = model(batch)
            loss, log = total_loss(out, batch, cfg.train, station_mask, use_physics=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            losses.append(log["total"])

        val_mae = _val_mae(model, val_loader, station_mask, "cpu")
        print(f"epoch {epoch+1}/{cfg.train.epochs}  train_loss={np.mean(losses):.3f}  val_mae={val_mae:.3f} (observed-only)")

    print("\nSmoke test complete: pipeline trains end-to-end on real data, scored only on genuinely-observed hours.")


if __name__ == "__main__":
    main()
