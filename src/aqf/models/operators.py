"""Physics-constrained graph neural operator.

Two things live here:

1. `PhysicsOperatorLayer` -- the learned-embedding-space message-passing layer
   used in the model's forward pass. It decouples message passing into an
   asymmetric **advection** operator (directed, wind-driven) and a symmetric
   **diffusion** operator (distance-based), plus a local **reaction** term and
   an **external** (regional, transport-lagged) injection term -- mirroring
   TransNet/AirPhyNet's operator decoupling rather than doing generic
   attention over a single blended adjacency.

2. `lagged_regional_message` -- gathers, for every (batch, time, local
   station) triple, the *time-lagged* regional source embedding
   `E_s(t - tau_{s,i}(t))` and aggregates it over sources weighted by the
   probabilistic transport corridor `P(i, s, t)`. This is the concrete
   implementation of "fire activity increased in Punjab 8 hours ago -> Delhi
   station rises now".

3. `ScalarADRPhysics` -- a lightweight *scalar*-field companion used only for
   the physics-regularization loss (losses/physics.py). It evaluates the
   literal advection-diffusion-reaction balance directly on observed PM2.5
   values (not learned embeddings) with a handful of learnable physical
   coefficients (advection scale, eddy diffusivity K, reaction/removal rate),
   so the regularizer is checking the model's own learned graph (A_adv,
   A_diff, S_t) against the real historical concentration field, not an
   abstract representation.
"""
from __future__ import annotations

import torch
from torch import nn

from aqf.features.transport import torch_transport_lag


def lagged_regional_message(
    h_R: torch.Tensor,                 # (B, L, N_reg, D) source embeddings over the window
    transport_p: torch.Tensor,         # (B, L, N_reg, N_local) probabilistic transport corridor
    regional_distance_km: torch.Tensor,  # (N_reg, N_local) static
    regional_wind_speed_seq: torch.Tensor,  # (B, L, N_reg)
    min_speed_ms: float,
    max_speed_ms: float,
    max_lag_hours: float,
) -> torch.Tensor:
    """Returns m_R: (B, L, N_local, D), the stability- and lag-aware regional source message."""
    B, L, N_reg, D = h_R.shape
    N_local = transport_p.shape[-1]
    device = h_R.device

    tau = torch_transport_lag(
        regional_distance_km.view(1, 1, N_reg, N_local).expand(B, L, N_reg, N_local),
        regional_wind_speed_seq.unsqueeze(-1).expand(B, L, N_reg, N_local),
        min_speed_ms=min_speed_ms,
        max_speed_ms=max_speed_ms,
        max_lag_hours=max_lag_hours,
    )  # (B, L, N_reg, N_local)
    lag_steps = tau.round().long()

    time_idx = torch.arange(L, device=device).view(1, L, 1, 1).expand(B, L, N_reg, N_local)
    origin_idx = (time_idx - lag_steps).clamp(min=0, max=L - 1)  # (B, L, N_reg, N_local)

    messages = []
    for s in range(N_reg):
        h_R_s = h_R[:, :, s, :]  # (B, L, D)
        h_R_s_exp = h_R_s.unsqueeze(2).expand(B, L, N_local, D)         # broadcast over receptor dim
        idx_s = origin_idx[:, :, s, :].unsqueeze(-1).expand(B, L, N_local, D)
        gathered = torch.gather(h_R_s_exp, dim=1, index=idx_s)          # (B, L, N_local, D)
        messages.append(gathered)
    h_R_lagged = torch.stack(messages, dim=2)  # (B, L, N_reg, N_local, D)

    weights = transport_p.unsqueeze(-1)  # (B, L, N_reg, N_local, 1)
    m_R = (h_R_lagged * weights).sum(dim=2)  # (B, L, N_local, D)
    return m_R


class PhysicsOperatorLayer(nn.Module):
    """One decoupled advection / diffusion / reaction message-passing step, in embedding space."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.adv_proj = nn.Linear(hidden_dim, hidden_dim)
        self.diff_proj = nn.Linear(hidden_dim, hidden_dim)
        self.reaction_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,           # (B, L, N_local, D)
        A_adv: torch.Tensor,       # (B, L, N_local, N_local) directed, wind-driven
        A_diff: torch.Tensor,      # (B, L, N_local, N_local) symmetric, distance-driven
        m_external: torch.Tensor,  # (B, L, N_local, D) regional/source injection
        g_A: torch.Tensor,         # (B, L, D) global atmospheric context
    ) -> torch.Tensor:
        h_adv = self.adv_proj(h)
        h_diff = self.diff_proj(h)

        # message_i = sum_j A[i,j] * (h_j - h_i)  -- flux relative to the receiving node, not raw mixing
        agg_adv = torch.einsum("blij,bljd->blid", A_adv, h_adv)
        deg_adv = A_adv.sum(dim=-1, keepdim=True)
        m_adv = agg_adv - deg_adv * h_adv

        agg_diff = torch.einsum("blij,bljd->blid", A_diff, h_diff)
        deg_diff = A_diff.sum(dim=-1, keepdim=True)
        m_diff = agg_diff - deg_diff * h_diff

        r = self.reaction_mlp(h)

        g_A_expand = g_A.unsqueeze(2).expand(-1, -1, h.shape[2], -1)
        update = self.update_mlp(torch.cat([m_adv, m_diff, r + m_external, g_A_expand], dim=-1))
        return self.norm(h + update)


class ScalarADRPhysics(nn.Module):
    """Evaluates the literal ADR balance on raw PM2.5 for the physics-regularization loss.

    dC/dt =~ -u.grad(C) + div(K grad(C)) + E - R, discretized on the graph as:
        advection_i = sum_j A_adv[i,j] * (C_j - C_i)
        diffusion_i = sum_j A_diff[i,j] * (C_j - C_i)
        reaction_i  = -lambda_R * C_i          (linear removal / deposition)
        external_i  = beta * (regional message reaching i, scalar-collapsed)
    with a small number of *learnable scalar* physical coefficients so the
    regularizer can calibrate advection strength, effective diffusivity, and
    removal rate jointly with the rest of the network, rather than assuming
    unit coefficients.
    """

    def __init__(self):
        super().__init__()
        self.log_advection_scale = nn.Parameter(torch.tensor(0.0))
        self.log_diffusion_scale = nn.Parameter(torch.tensor(0.0))
        self.log_reaction_rate = nn.Parameter(torch.tensor(-2.0))  # small initial removal rate
        self.log_external_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        C: torch.Tensor,          # (B, L, N_local) raw PM2.5
        A_adv: torch.Tensor,      # (B, L, N_local, N_local)
        A_diff: torch.Tensor,     # (B, L, N_local, N_local)
        external_raw: torch.Tensor,  # (B, L, N_local) scalar regional injection (e.g. transport-weighted FEP)
    ) -> dict[str, torch.Tensor]:
        adv_scale = torch.exp(self.log_advection_scale)
        diff_scale = torch.exp(self.log_diffusion_scale)
        reaction_rate = torch.exp(self.log_reaction_rate)
        ext_scale = torch.exp(self.log_external_scale)

        # Cj_minus_Ci[b,l,i,j] = C_j - C_i
        Cj_minus_Ci = C.unsqueeze(2).expand(-1, -1, C.shape[-1], -1) - C.unsqueeze(3).expand(-1, -1, -1, C.shape[-1])

        advection_term = adv_scale * (A_adv * Cj_minus_Ci).sum(dim=-1)   # (B, L, N)
        diffusion_term = diff_scale * (A_diff * Cj_minus_Ci).sum(dim=-1)  # (B, L, N)
        reaction_term = reaction_rate * C
        external_term = ext_scale * external_raw

        dCdt_pred = advection_term + diffusion_term - reaction_term + external_term
        dCdt_actual = torch.zeros_like(C)
        dCdt_actual[:, 1:, :] = C[:, 1:, :] - C[:, :-1, :]

        return {
            "dCdt_pred": dCdt_pred,
            "dCdt_actual": dCdt_actual,
            "advection_term": advection_term,
            "diffusion_term": diffusion_term,
            "reaction_term": reaction_term,
            "external_term": external_term,
        }
