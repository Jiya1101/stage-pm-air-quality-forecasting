"""Quick smoke test: does STAGE-PM train and produce sane numbers on the real 21-day OpenAQ+FIRMS pull?

Not a real evaluation -- the fetched window is short (505 hours) and only
~30-36% of local PM2.5 readings are actually observed (real CPCB sensor
downtime, see data/real_pipeline.py). This script forward/backward-fills
gaps so the pipeline has a complete array to train on, which means the
reported MAE is optimistic (partly measuring interpolation smoothness, not
genuine forecasting skill on unseen data) -- it is meant to catch pipeline
bugs and sanity-check the loss/gradients behave on a real feature
distribution, not to report a trustworthy accuracy number. A proper
held-out evaluation needs the full historical pull (more data, a real
temporal split, and per-sample loss masking on originally-observed hours).

Usage:
    python scripts/smoke_test_real.py
"""
import _pathfix  # noqa: F401
import numpy as np
import pandas as pd
import torch

from aqf.config import Config
from aqf.data.dataset import AQFWindowDataset, PM25_IDX
from aqf.data.schema import RawSeries, LOCAL_FEATURE_COLS
from aqf.losses.physics import total_loss
from aqf.training.train import build_model


def impute_local_gaps(local: np.ndarray) -> np.ndarray:
    """Per-station, per-feature forward-fill -> backward-fill -> global-mean fallback."""
    T, N, F = local.shape
    out = local.copy()
    for n in range(N):
        df = pd.DataFrame(out[:, n, :], columns=LOCAL_FEATURE_COLS)
        df = df.ffill().bfill()
        out[:, n, :] = df.to_numpy()
    # any feature that was NaN for an entire station's whole window (ffill/bfill can't help) -> global column mean
    for f in range(F):
        col = out[:, :, f]
        if np.isnan(col).any():
            fallback = np.nanmean(col) if not np.isnan(col).all() else 0.0
            col[np.isnan(col)] = fallback
            out[:, :, f] = col
    return out


def main():
    raw = RawSeries.load("data/real/raw.npz")
    pm25_observed_frac = 1.0 - np.isnan(raw.local[..., PM25_IDX]).mean()
    print(f"Loaded real data: {len(raw.timestamps)} hours, {raw.local.shape[1]} stations, "
          f"PM2.5 observed (pre-imputation): {pm25_observed_frac:.1%}")

    raw.local = impute_local_gaps(raw.local)
    print("Gaps filled via ffill/bfill/global-mean (see module docstring caveat).")

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

        model.eval()
        val_errs = []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch)
                err = (out["pm25_mean"] - batch["y_pm25"]).abs().mean().item()
                val_errs.append(err)

        print(f"epoch {epoch+1}/{cfg.train.epochs}  train_loss={np.mean(losses):.3f}  val_mae={np.mean(val_errs):.3f}")

    print("\nSmoke test complete: pipeline trains end-to-end on real OpenAQ+FIRMS data without errors.")
    print("Remember the caveat above -- this MAE is not a trustworthy accuracy number, just a sanity check.")


if __name__ == "__main__":
    main()
