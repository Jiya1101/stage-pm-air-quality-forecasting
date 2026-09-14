"""Single-config training loop for STAGE-PM."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from aqf.config import Config
from aqf.data.dataset import AQFWindowDataset, make_split
from aqf.data.schema import ATMOS_FEATURE_COLS, LOCAL_FEATURE_COLS, REGIONAL_FEATURE_COLS, RawSeries
from aqf.losses.physics import total_loss
from aqf.models.stage_pm import StagePM


def load_raw(cfg: Config) -> RawSeries:
    if cfg.data.source == "synthetic":
        path = os.path.join(cfg.data.synthetic_dir, "raw.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No synthetic dataset at {path}. Run `python scripts/generate_synthetic_data.py` first."
            )
        return RawSeries.load(path)
    raise NotImplementedError(
        "config.data.source == 'real' requires assembling CPCB/ERA5/FIRMS pulls into the RawSeries "
        "schema (see src/aqf/data/schema.py) -- the individual clients are implemented in "
        "src/aqf/data/{cpcb,era5,firms}.py but the assembly/alignment step is deployment-specific "
        "(depends on which stations + date range you pull)."
    )


def build_model(cfg: Config, device: str) -> StagePM:
    return StagePM(
        n_local_features=len(LOCAL_FEATURE_COLS),
        n_regional_features=len(REGIONAL_FEATURE_COLS),
        n_atmos_features=len(ATMOS_FEATURE_COLS),
        n_horizons=len(cfg.data.horizons_hours),
        model_cfg=cfg.model,
        graph_cfg=cfg.graph,
        ablation=cfg.ablation,
        device=device,
    ).to(device)


def run(cfg: Config, verbose: bool = True) -> dict:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = cfg.train.device

    raw = load_raw(cfg)
    split = make_split(raw, cfg.data, stride_hours=6, seed=cfg.train.seed)

    train_ds = AQFWindowDataset(raw, cfg.data, split.train_t0)
    val_ds = AQFWindowDataset(raw, cfg.data, split.val_t0)
    test_ds = AQFWindowDataset(raw, cfg.data, split.test_t0)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    station_mask = torch.tensor(~split.held_out_local_mask)  # True = used in training loss

    out_dir = os.path.join(cfg.train.out_dir, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    history = {"train_loss": [], "val_mae": []}
    best_val = float("inf")

    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.time()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            loss, log = total_loss(out, batch, cfg.train, station_mask, use_physics=cfg.ablation.use_physics_loss)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            epoch_losses.append(log["total"])

        val_mae = _val_mae(model, val_loader, station_mask, device)
        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_mae"].append(val_mae)

        if verbose:
            print(f"[{cfg.name}] epoch {epoch+1}/{cfg.train.epochs} "
                  f"train_loss={np.mean(epoch_losses):.4f} val_mae={val_mae:.4f} "
                  f"({time.time()-t0:.1f}s)")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_mae": val_mae},
                os.path.join(out_dir, "best.pt"),
            )

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {"history": history, "best_val_mae": best_val, "out_dir": out_dir,
            "split": split, "test_ds": test_ds, "raw": raw}


@torch.no_grad()
def _val_mae(model: StagePM, loader: DataLoader, station_mask: torch.Tensor, device: str) -> float:
    model.eval()
    total_err, total_n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        m = station_mask.view(1, -1, 1).to(device).float()
        err = ((out["pm25_mean"] - batch["y_pm25"]).abs() * m).sum().item()
        n = m.sum().item() * batch["y_pm25"].shape[-1]
        total_err += err
        total_n += n
    return total_err / max(total_n, 1)
