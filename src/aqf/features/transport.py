"""Dynamic transport lag and probabilistic transport corridors.

Implements, in vectorized numpy/torch:

  - great-circle distance d(s,i) between every source and receptor node
  - dynamic transport lag  tau_{s,i}(t) = d_{s,i} / U_eff(t)
  - wind-alignment based directed edge weight (source pushes toward receptor
    only when wind blows roughly from s to i)
  - a probabilistic transport corridor P(i, s, t): softmax over receptors of
    (wind alignment) * (distance decay) * (stability gate), so a Punjab fire
    spreads probability mass over several plausible Delhi stations rather
    than a single deterministic edge.

All of this operates on plain arrays/tensors so it's shared between the
synthetic data generator (numpy, for building ground truth) and the model's
transport encoder (torch, differentiable, for training).
"""
from __future__ import annotations

import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Broadcasts over numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_unit_vector(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Unit direction vector (east, north) approx, pointing from (lat1,lon1) to (lat2,lon2).

    Uses an equirectangular local approximation (fine at the ~500km regional
    scale relevant here); returns shape (..., 2) as (dx_east, dy_north) in an
    arbitrary consistent unit (not true meters, but consistent for computing
    an alignment cosine with a wind vector expressed the same way).
    """
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    mean_lat = np.radians((np.asarray(lat1) + np.asarray(lat2)) / 2.0)
    dx = np.radians(np.asarray(lon2) - np.asarray(lon1)) * np.cos(mean_lat)
    dy = np.radians(np.asarray(lat2) - np.asarray(lat1))
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-9
    return np.stack([dx / norm, dy / norm], axis=-1)


def static_distance_matrix(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Pairwise distance matrix (N_src, N_dst) in km."""
    src_lat = np.asarray(src_lat)[:, None]
    src_lon = np.asarray(src_lon)[:, None]
    dst_lat = np.asarray(dst_lat)[None, :]
    dst_lon = np.asarray(dst_lon)[None, :]
    return haversine_km(src_lat, src_lon, dst_lat, dst_lon)


def static_direction_matrix(src_lat, src_lon, dst_lat, dst_lon) -> np.ndarray:
    """Pairwise unit direction vectors (N_src, N_dst, 2) from src to dst."""
    n_src, n_dst = len(src_lat), len(dst_lat)
    out = np.zeros((n_src, n_dst, 2))
    for i in range(n_src):
        out[i] = bearing_unit_vector(src_lat[i], src_lon[i], np.asarray(dst_lat), np.asarray(dst_lon))
    return out


def effective_wind_speed(wind_speed_ms: np.ndarray, min_speed_ms: float = 0.3, max_speed_ms: float = 20.0) -> np.ndarray:
    """Clip wind speed to a physically sane range to avoid tau -> inf (calm) or tau -> 0 (unphysical)."""
    return np.clip(wind_speed_ms, min_speed_ms, max_speed_ms)


def transport_lag_hours(
    distance_km: np.ndarray,
    wind_speed_ms: np.ndarray,
    min_speed_ms: float = 0.3,
    max_speed_ms: float = 20.0,
    max_lag_hours: float = 48.0,
) -> np.ndarray:
    """tau_{s,i}(t) = d_{s,i} / U_eff(t), in hours. distance_km and wind_speed_ms broadcast together."""
    u_eff_km_per_h = effective_wind_speed(wind_speed_ms, min_speed_ms, max_speed_ms) * 3.6
    tau = distance_km / u_eff_km_per_h
    return np.clip(tau, 0.0, max_lag_hours)


def wind_alignment_weight(
    wind_unit_vec: np.ndarray,      # (..., 2) unit wind direction at source (blowing TOWARD)
    direction_unit_vec: np.ndarray,  # (..., 2) unit vector source -> receptor
) -> np.ndarray:
    """cosine alignment, clipped at 0: only downwind receptors get positive weight."""
    cos_sim = np.sum(wind_unit_vec * direction_unit_vec, axis=-1)
    return np.clip(cos_sim, 0.0, None)


def probabilistic_transport_weights(
    wind_unit_vec: np.ndarray,       # (N_src, 2)
    direction_unit_vec: np.ndarray,  # (N_src, N_dst, 2)
    distance_km: np.ndarray,         # (N_src, N_dst)
    stability_gate: float,           # scalar or (N_src,1)/(N_src,N_dst) broadcastable, in [0,1]; regional edges expand as gate->1
    distance_sigma_km: float = 220.0,
) -> np.ndarray:
    """P(i | s, t): softmax over receptors i of alignment * distance-decay * stability gate.

    This is the "Probabilistic Transport Corridor" -- a fire in Punjab spreads
    influence-probability over several plausible Delhi stations rather than a
    single deterministic wind-blows-there edge.
    """
    align = wind_alignment_weight(wind_unit_vec[:, None, :], direction_unit_vec)  # (N_src, N_dst)
    decay = np.exp(-(distance_km ** 2) / (2.0 * distance_sigma_km ** 2))
    score = align * decay * stability_gate
    # softmax over receptor axis (dst), per source
    score = score - score.max(axis=1, keepdims=True)
    exp_score = np.exp(score) * (align > 0)  # zero out upwind (non-contributing) receptors
    denom = exp_score.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return exp_score / denom


def torch_transport_lag(
    distance_km: torch.Tensor,
    wind_speed_ms: torch.Tensor,
    min_speed_ms: float = 0.3,
    max_speed_ms: float = 20.0,
    max_lag_hours: float = 48.0,
) -> torch.Tensor:
    """Differentiable counterpart of transport_lag_hours for use inside the model."""
    u_eff = wind_speed_ms.clamp(min=min_speed_ms, max=max_speed_ms) * 3.6
    tau = distance_km / u_eff
    return tau.clamp(min=0.0, max=max_lag_hours)


def torch_probabilistic_transport_weights(
    wind_unit_vec: torch.Tensor,       # (B, N_src, 2)
    direction_unit_vec: torch.Tensor,  # (N_src, N_dst, 2), static
    distance_km: torch.Tensor,         # (N_src, N_dst), static
    stability_gate: torch.Tensor,      # (B, 1, 1) or broadcastable, in [0,1]
    distance_sigma_km: float = 220.0,
) -> torch.Tensor:
    """Differentiable batched version -> (B, N_src, N_dst) transport probability."""
    align = (wind_unit_vec.unsqueeze(2) * direction_unit_vec.unsqueeze(0)).sum(-1)  # (B, N_src, N_dst)
    align = align.clamp(min=0.0)
    decay = torch.exp(-(distance_km ** 2) / (2.0 * distance_sigma_km ** 2)).unsqueeze(0)
    score = align * decay * stability_gate
    mask = align > 0
    score = score.masked_fill(~mask, float("-inf"))
    # rows that are entirely masked (no downwind receptor) -> uniform-zero, handled after softmax
    all_masked = (~mask).all(dim=-1, keepdim=True)
    score = torch.where(all_masked, torch.zeros_like(score), score)
    weights = torch.softmax(score, dim=-1)
    weights = weights * (~all_masked)
    return weights
