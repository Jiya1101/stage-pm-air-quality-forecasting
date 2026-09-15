"""Total STAGE-PM loss.

    L_total = L_forecast
            + lambda1 * L_transport   (ADR residual: dC/dt vs -u.gradC + div(K gradC) - R + E)
            + lambda2 * L_mass        (whole-domain mass conservation)
            + lambda3 * L_stability   (suppress cross-station diffusion under high S_t)
            + lambda_source * L_source (source-attribution supervision, when ground truth available)
            + lambda_regime * L_regime (regime classification, when ground truth available)

L_forecast itself combines a Gaussian NLL (for calibrated uncertainty, "Add
uncertainty" section) with plain MAE (stabilizes early training before the
learned variance is well calibrated), plus an exceedance BCE term.

Every term is masked to exclude spatially-held-out stations from contributing
to training gradients (see data/dataset.py::SplitIndices.held_out_local_mask),
AND (for real data) to exclude target hours that were imputed rather than
genuinely observed (see data/real_pipeline.py::impute_for_training and
batch["y_observed"] from data/dataset.py) -- without the latter, a model
trained on real data with ~30-36% station uptime would be scored partly on
how well it reproduces forward/backward-fill interpolation, not real
forecasting skill.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from aqf.config import TrainConfig

# PM2.5 values run into the hundreds (severe-episode range), which makes the
# raw-unit Gaussian NLL numerically top-heavy: driving the predicted variance
# up is a cheaper way to shrink (target-mean)^2/var than actually improving
# the mean prediction, especially early in training. Normalizing by a fixed
# scale before computing NLL keeps its gradient magnitude comparable to the
# MAE term's, so training can't "cheat" toward wide, uninformative intervals.
PM_SCALE = 100.0


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NLL computed in PM_SCALE-normalized units -- see PM_SCALE docstring above."""
    mean_n, target_n = mean / PM_SCALE, target / PM_SCALE
    var_n = torch.exp(logvar)
    return 0.5 * (logvar + (target_n - mean_n) ** 2 / var_n)


def forecast_loss(out: dict, batch: dict, station_mask: torch.Tensor, train_cfg: TrainConfig | None = None) -> tuple[torch.Tensor, dict]:
    """station_mask: (N_local,) bool, True = included in loss (i.e. NOT held out).

    Combined with batch["y_observed"] (B, N_local, n_horizons) -- 1 where
    the target was genuinely observed, 0 where it was imputed (real data
    only; defaults to all-ones for synthetic data, see data/dataset.py). A
    sample masked out here contributes nothing to MAE/NLL/BCE or their
    reported values, not just zero gradient with a nonzero denominator.
    """
    mean, logvar, exceed_prob = out["pm25_mean"], out["pm25_logvar"], out["exceed_prob"]
    y, y_exceed = batch["y_pm25"], batch["y_exceed"]
    lambda_nll = train_cfg.lambda_nll if train_cfg is not None else 0.05

    station_m = station_mask.view(1, -1, 1).to(mean.device).float()
    observed_m = batch.get("y_observed")
    m = station_m * observed_m.to(mean.device) if observed_m is not None else station_m.expand_as(mean)
    n_active = m.sum().clamp(min=1)

    nll = (gaussian_nll(mean, logvar, y) * m).sum() / n_active
    mae = (F.l1_loss(mean, y, reduction="none") * m).sum() / n_active
    bce = (F.binary_cross_entropy(exceed_prob.clamp(1e-6, 1 - 1e-6), y_exceed, reduction="none") * m).sum() / n_active
    return mae + lambda_nll * nll + 0.5 * bce, {"mae": mae.item(), "nll": nll.item(), "exceed_bce": bce.item()}


def physics_loss(physics: dict, station_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = station_mask.view(1, 1, -1).to(physics["dCdt_pred"].device).float()

    residual = physics["dCdt_pred"] - physics["dCdt_actual"]
    l_transport = ((residual ** 2) * m).sum() / m.sum().clamp(min=1) / residual.shape[0] / residual.shape[1]

    total_dcdt = (physics["dCdt_actual"] * m).sum(dim=-1)          # (B, L)
    total_ext = (physics["external_term"] * m).sum(dim=-1)
    total_reaction = (physics["reaction_term"] * m).sum(dim=-1)
    mass_residual = total_dcdt - (total_ext - total_reaction)
    l_mass = (mass_residual ** 2).mean()

    # discourage horizontal diffusion from dominating under high stability (S_t -> 1)
    S_t = physics["S_t"]  # (B, L)
    l_stability = ((S_t.unsqueeze(-1) * (physics["diffusion_term"] ** 2)) * m).sum() / m.sum().clamp(min=1) / S_t.shape[0] / S_t.shape[1]

    return l_transport, l_mass, l_stability


def source_contrib_loss(out: dict, batch: dict, station_mask: torch.Tensor) -> torch.Tensor | None:
    if "source_contrib" not in out or "y_source_contrib" not in batch:
        return None
    pred = out["source_contrib"].clamp(min=1e-6)
    target = batch["y_source_contrib"].clamp(min=1e-6)
    m = station_mask.view(1, -1).to(pred.device)
    kl = (target * (target.log() - pred.log())).sum(dim=-1)  # (B, N_local)
    return (kl * m).sum() / m.sum().clamp(min=1) / kl.shape[0]


def regime_loss(out: dict, batch: dict, station_mask: torch.Tensor) -> torch.Tensor | None:
    if "regime_logits" not in out or "y_regime" not in batch:
        return None
    logits = out["regime_logits"]  # (B, N_local, n_regimes)
    target = batch["y_regime"]      # (B, N_local)
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    ce = ce.reshape(target.shape)
    m = station_mask.view(1, -1).to(logits.device)
    return (ce * m).sum() / m.sum().clamp(min=1) / ce.shape[0]


def total_loss(out: dict, batch: dict, train_cfg: TrainConfig, station_mask: torch.Tensor, use_physics: bool) -> tuple[torch.Tensor, dict]:
    f_loss, f_log = forecast_loss(out, batch, station_mask, train_cfg)
    total = f_loss
    log = dict(f_log)

    if use_physics:
        l_t, l_m, l_s = physics_loss(out["_physics"], station_mask)
        total = total + train_cfg.lambda_transport * l_t + train_cfg.lambda_mass * l_m + train_cfg.lambda_stability * l_s
        log.update({"transport_residual": l_t.item(), "mass_residual": l_m.item(), "stability_penalty": l_s.item()})

    src_loss = source_contrib_loss(out, batch, station_mask)
    if src_loss is not None:
        total = total + train_cfg.lambda_source * src_loss
        log["source_kl"] = src_loss.item()

    reg_loss = regime_loss(out, batch, station_mask)
    if reg_loss is not None:
        total = total + train_cfg.lambda_regime * reg_loss
        log["regime_ce"] = reg_loss.item()

    log["total"] = total.item()
    return total, log
