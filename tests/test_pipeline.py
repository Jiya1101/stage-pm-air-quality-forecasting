"""End-to-end smoke test: synthetic data -> graph -> model forward -> one training step -> ablation flags change output.

Runs as a plain script (no pytest dependency required):
    python tests/test_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from aqf.config import Config
from aqf.data.dataset import AQFWindowDataset, make_split
from aqf.data.synthetic import generate
from aqf.losses.physics import total_loss
from aqf.training.ablation import EXPERIMENTS
from aqf.training.train import build_model


def test_synthetic_generation_shapes():
    raw = generate(start="2022-10-01", end="2022-12-31 23:00", seed=0)
    T, N_local, F_local = raw.local.shape
    assert N_local == 15
    assert raw.regional.shape[1] == 4
    assert raw.atmos.shape[0] == T
    assert raw.source_contrib_gt.shape == (T, N_local, 5)
    assert (raw.local[..., 0] >= 0).all(), "PM2.5 must be non-negative"
    print(f"OK: synthetic generation shapes ({T} hours, {N_local} local, 4 regional)")
    return raw


def test_forward_and_one_train_step(raw):
    cfg = Config(name="smoke_test")
    cfg.data.lookback_hours = 24
    cfg.data.horizons_hours = (1, 6, 24)
    cfg.data.train_end_year = 2022
    cfg.data.val_year = 2022
    cfg.data.test_year = 2022
    cfg.model.hidden_dim = 16
    cfg.model.n_operator_layers = 2
    cfg.model.n_transformer_layers = 1

    split = make_split(raw, cfg.data, stride_hours=12, seed=0)
    assert len(split.train_t0) > 0, "need at least one training window"

    ds = AQFWindowDataset(raw, cfg.data, split.train_t0[:8])
    batch = torch.utils.data.default_collate([ds[i] for i in range(min(4, len(ds)))])

    model = build_model(cfg, "cpu")
    out = model(batch)

    n_horizons = len(cfg.data.horizons_hours)
    assert out["pm25_mean"].shape == (batch["local_seq"].shape[0], 15, n_horizons)
    assert out["source_contrib"].shape == (batch["local_seq"].shape[0], 15, 5)
    assert torch.allclose(out["source_contrib"].sum(-1), torch.ones_like(out["source_contrib"].sum(-1)), atol=1e-4)
    assert out["regime_logits"].shape == (batch["local_seq"].shape[0], 15, 4)

    station_mask = torch.tensor(~split.held_out_local_mask)
    loss, log = total_loss(out, batch, cfg.train, station_mask, use_physics=True)
    assert torch.isfinite(loss), f"loss is not finite: {log}"

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    total_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert total_grad_norm > 0, "no gradient flowed -- graph is disconnected from the loss"
    opt.step()

    print(f"OK: forward pass shapes correct, loss={loss.item():.3f}, grad_norm={total_grad_norm:.3f}")
    return batch, model


def test_ablation_flags_change_predictions(raw):
    """Sanity check that the ablation ladder is actually wired -- e.g. removing external sources changes output."""
    cfg_full = Config(name="H_full_model")
    cfg_full.data.lookback_hours = 24
    cfg_full.data.train_end_year = 2022
    cfg_full.data.val_year = 2022
    cfg_full.data.test_year = 2022
    cfg_full.model.hidden_dim = 16
    cfg_full.model.n_operator_layers = 2
    cfg_full.model.n_transformer_layers = 1
    cfg_full.ablation = EXPERIMENTS["H_full_model"]

    cfg_baseline = Config(name="A_baseline")
    cfg_baseline.data.lookback_hours = 24
    cfg_baseline.data.train_end_year = 2022
    cfg_baseline.data.val_year = 2022
    cfg_baseline.data.test_year = 2022
    cfg_baseline.model.hidden_dim = 16
    cfg_baseline.model.n_operator_layers = 2
    cfg_baseline.model.n_transformer_layers = 1
    cfg_baseline.ablation = EXPERIMENTS["A_baseline"]

    split = make_split(raw, cfg_full.data, stride_hours=12, seed=0)
    ds = AQFWindowDataset(raw, cfg_full.data, split.train_t0[:4])
    batch = torch.utils.data.default_collate([ds[i] for i in range(min(2, len(ds)))])

    torch.manual_seed(0)
    model_full = build_model(cfg_full, "cpu")
    torch.manual_seed(0)
    model_base = build_model(cfg_baseline, "cpu")

    out_full = model_full(batch)
    out_base = model_base(batch)

    assert "source_contrib" in out_full and "source_contrib" not in out_base
    assert "regime_logits" in out_full and "regime_logits" not in out_base
    print("OK: ablation flags correctly toggle model capacity (source/regime heads present only in full model)")


if __name__ == "__main__":
    raw = test_synthetic_generation_shapes()
    test_forward_and_one_train_step(raw)
    test_ablation_flags_change_predictions(raw)
    print("\nAll smoke tests passed.")
