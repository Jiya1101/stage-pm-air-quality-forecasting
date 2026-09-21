"""Print the per-term loss breakdown for one real-data training batch of a freshly initialised model.

Diagnostic for checking that no single term (e.g. the physics regularisers,
which are computed on PM2.5 and so scale with its units) dominates the
forecast loss.

Usage:
    python scripts/loss_breakdown.py [--source real|synthetic]
"""
import argparse

import _pathfix  # noqa: F401
import torch

from aqf.config import Config
from aqf.data.dataset import AQFWindowDataset, make_split
from aqf.losses.physics import PM_SCALE, total_loss
from aqf.training.train import build_model, load_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "real"], default="real")
    args = p.parse_args()

    cfg = Config(name="loss_breakdown")
    cfg.data.source = args.source
    cfg.model.hidden_dim = 32
    cfg.data.stride_hours = 24

    raw = load_raw(cfg)
    split = make_split(raw, cfg.data, stride_hours=cfg.data.stride_hours, seed=0)
    ds = AQFWindowDataset(raw, cfg.data, split.train_t0[:64])
    batch = torch.utils.data.default_collate([ds[i] for i in range(16)])

    torch.manual_seed(0)
    model = build_model(cfg, "cpu")
    out = model(batch)
    mask = torch.tensor(~split.held_out_local_mask)
    _, log = total_loss(out, batch, cfg.train, mask, use_physics=True)

    tc = cfg.train
    print(f"source={args.source}")
    for k, v in log.items():
        print(f"  {k:20s} {v:14.4f}")
    print("weighted contributions to total:")
    print(f"  forecast (mae+nll+bce)  {log['mae'] / PM_SCALE + tc.lambda_nll * log['nll'] + 0.5 * log['exceed_bce']:14.4f}")
    print(f"  lambda_transport*L_t    {tc.lambda_transport * log['transport_residual']:14.4f}")
    print(f"  lambda_mass*L_m         {tc.lambda_mass * log['mass_residual']:14.4f}")
    print(f"  lambda_stability*L_s    {tc.lambda_stability * log['stability_penalty']:14.4f}")


if __name__ == "__main__":
    main()
