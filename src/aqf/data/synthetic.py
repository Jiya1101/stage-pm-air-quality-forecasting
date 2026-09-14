"""Physically-plausible synthetic data generator.

Purpose: let the whole STAGE-PM pipeline (graph construction, physics losses,
ablation study) be built and validated *before* real CPCB/ERA5/FIRMS
credentials are wired in (see README "Connecting real data"). The generator
deliberately bakes in the three mechanisms the architecture is supposed to
discover:

  1. wind- and stability-gated transport from regional source nodes with a
     genuine travel-time lag (tau = distance / effective wind speed),
  2. AR(1) local accumulation whose persistence depends on a stability index
     (slow turnover / trapping under inversion, fast venting when well-mixed),
  3. a stubble-burning season (Oct-Nov) at the Punjab/Haryana source nodes,
     a pre-monsoon dust season at Rajasthan, and a Diwali fireworks pulse.

Ground-truth source-contribution fractions and regime labels are also
generated, purely so the model's source-attribution and regime-classifier
heads can be sanity-checked against a known generative process (real data
will not have this luxury).

This is a data-generating process, not a claim about real Delhi
concentrations -- do not use its numbers for anything beyond pipeline
validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqf.data.schema import (
    ATMOS_FEATURE_COLS,
    LOCAL_FEATURE_COLS,
    REGIME_CATEGORIES,
    REGIONAL_FEATURE_COLS,
    SOURCE_CATEGORIES,
    RawSeries,
)
from aqf.features.stability import heuristic_stability_index
from aqf.features.transport import (
    static_direction_matrix,
    static_distance_matrix,
    probabilistic_transport_weights,
    transport_lag_hours,
)
from aqf.graph.stations import LOCAL_STATIONS, REGIONAL_SOURCES


def _wind_field(t_hours: np.ndarray, day_of_year: np.ndarray, rng: np.random.Generator, n: int, base_angle_deg_fn):
    """Diurnal-and-seasonal wind speed/direction with autocorrelated noise, for n co-located sources sharing a regional flow."""
    hour_of_day = t_hours % 24
    speed_base = 2.0 + 2.5 * np.sin((hour_of_day - 6) / 24 * 2 * np.pi).clip(min=0) + 1.0
    # random-walk perturbation, smoothed
    noise = rng.normal(0, 0.4, size=len(t_hours))
    speed_noise = pd.Series(noise).rolling(6, min_periods=1).mean().to_numpy()
    speed = np.clip(speed_base + speed_noise, 0.3, 18.0)

    angle_deg = base_angle_deg_fn(day_of_year) + rng.normal(0, 15, size=len(t_hours))
    angle_noise_smooth = pd.Series(angle_deg).rolling(12, min_periods=1).mean().to_numpy()
    angle_rad = np.radians(angle_noise_smooth)
    # meteorological "blowing FROM" -> convert to "blowing TOWARD" unit vector (east, north)
    u = -speed * np.sin(angle_rad)
    v = -speed * np.cos(angle_rad)
    return speed, u, v


def _seasonal_wind_angle(day_of_year: np.ndarray) -> np.ndarray:
    """Approximate Delhi seasonal wind rose: winter NW (stubble/dust transport), monsoon SE (clean)."""
    # day_of_year in [1,366]; encode a smooth annual cycle peaking at NW (~315deg) in Nov-Jan,
    # shifting to SE (~135deg) around Jul-Aug (monsoon), via a cosine blend.
    phase = 2 * np.pi * (day_of_year - 15) / 365.25
    # angle oscillates between 300 (winter) and 140 (monsoon)
    return 220 + 80 * np.cos(phase)


def _fire_activity(day_of_year: np.ndarray, peak_start=258, peak_end=335, base=2.0, peak=140.0) -> np.ndarray:
    """Stubble-burning seasonal curve (day-of-year ~258=Sep15 .. ~335=Dec1), bell-shaped, plus daily noise floor."""
    center = (peak_start + peak_end) / 2.0
    width = (peak_end - peak_start) / 2.2
    bell = np.exp(-0.5 * ((day_of_year - center) / width) ** 2)
    return base + peak * bell


def _dust_activity(day_of_year: np.ndarray, base=3.0, peak=45.0) -> np.ndarray:
    """Pre-monsoon dust season (Apr-Jun, day ~91-172), bell-shaped."""
    center, width = 130, 30
    bell = np.exp(-0.5 * ((day_of_year - center) / width) ** 2)
    return base + peak * bell


def generate(
    start: str = "2018-01-01",
    end: str = "2023-12-31 23:00",
    seed: int = 0,
    local_sigma_km: float = 12.0,
    regional_sigma_km: float = 220.0,
) -> RawSeries:
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start, end, freq="h")
    T = len(timestamps)
    t_hours = np.arange(T, dtype=float)
    day_of_year = timestamps.dayofyear.to_numpy().astype(float)
    hour_of_day = timestamps.hour.to_numpy().astype(float)
    is_weekend = np.asarray(timestamps.dayofweek >= 5)

    local_lat = np.array([s.lat for s in LOCAL_STATIONS])
    local_lon = np.array([s.lon for s in LOCAL_STATIONS])
    regional_lat = np.array([s.lat for s in REGIONAL_SOURCES])
    regional_lon = np.array([s.lon for s in REGIONAL_SOURCES])
    N_local, N_reg = len(LOCAL_STATIONS), len(REGIONAL_SOURCES)

    regional_distance = static_distance_matrix(regional_lat, regional_lon, local_lat, local_lon)  # (N_reg, N_local)
    regional_direction = static_direction_matrix(regional_lat, regional_lon, local_lat, local_lon)

    # ---- Regional meteorology (shared broad regional flow + per-source jitter) ----
    speed_reg, u_reg, v_reg = _wind_field(t_hours, day_of_year, rng, N_reg, _seasonal_wind_angle)
    # small per-source direction jitter so the 4 regional nodes aren't perfectly identical
    reg_wind_u = np.tile(u_reg[:, None], (1, N_reg)) + rng.normal(0, 0.3, size=(T, N_reg))
    reg_wind_v = np.tile(v_reg[:, None], (1, N_reg)) + rng.normal(0, 0.3, size=(T, N_reg))
    reg_speed = np.sqrt(reg_wind_u ** 2 + reg_wind_v ** 2)

    # ---- Regional source activity ----
    # Source order: SRC_PB, SRC_HR, SRC_RJ, SRC_UP (must match graph/stations.py)
    frp = np.zeros((T, N_reg))
    frp[:, 0] = _fire_activity(day_of_year, peak=160) + rng.gamma(1.0, 3.0, size=T)   # Punjab
    frp[:, 1] = _fire_activity(day_of_year, peak=110) + rng.gamma(1.0, 2.5, size=T)   # Haryana
    frp[:, 2] = rng.gamma(1.0, 1.0, size=T)                                            # Rajasthan (fires: minimal)
    frp[:, 3] = rng.gamma(1.0, 1.0, size=T)                                            # W-UP (fires: minimal)
    frp = np.clip(frp, 0, None)
    fep = frp  # already FRP-like units for the synthetic process; real pipeline uses features/fire_proxy.py

    dust_index = np.zeros((T, N_reg))
    dust_index[:, 2] = _dust_activity(day_of_year) + rng.gamma(1.0, 2.0, size=T)  # Rajasthan dust
    dust_index[:, [0, 1, 3]] = rng.gamma(1.0, 0.5, size=(T, 3))
    dust_index = np.clip(dust_index, 0, None)

    industry_index = np.zeros((T, N_reg))
    weekday_factor = np.where(is_weekend, 0.6, 1.0)
    industry_index[:, 3] = (30 + 10 * np.sin(2 * np.pi * hour_of_day / 24)) * weekday_factor  # W-UP industrial
    industry_index[:, :3] = (10 + 3 * np.sin(2 * np.pi * hour_of_day / 24))[:, None] * weekday_factor[:, None]

    # ---- Local meteorology ----
    speed_loc, u_loc, v_loc = _wind_field(t_hours, day_of_year, rng, N_local, _seasonal_wind_angle)
    loc_wind_u = np.tile(u_loc[:, None], (1, N_local)) + rng.normal(0, 0.3, size=(T, N_local))
    loc_wind_v = np.tile(v_loc[:, None], (1, N_local)) + rng.normal(0, 0.3, size=(T, N_local))
    loc_speed = np.sqrt(loc_wind_u ** 2 + loc_wind_v ** 2)

    temp_c = 25 + 10 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25) + 6 * np.sin(2 * np.pi * (hour_of_day - 15) / 24)
    rh_pct = np.clip(55 - 15 * np.sin(2 * np.pi * (hour_of_day - 15) / 24) + rng.normal(0, 5, T), 10, 100)

    # BLH: diurnal cycle, seasonally suppressed ceiling in winter (Nov-Jan -> low inversion-prone BLH)
    winter_suppression = 0.35 + 0.65 * (1 - np.exp(-0.5 * ((np.minimum(np.abs(day_of_year - 0), np.abs(day_of_year - 365)) - 0) / 45) ** 2))
    diurnal = np.clip(np.sin(2 * np.pi * (hour_of_day - 6) / 24), -0.2, 1.0)
    blh_m = np.clip(150 + 1800 * diurnal.clip(min=0) * winter_suppression + rng.normal(0, 60, T), 30, 3000)

    inversion_strength_k = np.clip(4.0 * (1 - blh_m / 1800.0) + rng.normal(0, 0.4, T), -1.0, 9.0)
    solar_rad_wm2 = np.clip(700 * diurnal.clip(min=0) * (0.5 + 0.5 * (1 - winter_suppression)) + rng.normal(0, 20, T), 0, 950)
    pressure_hpa = 1008 + 6 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25) + rng.normal(0, 1.5, T)
    precip_mm = np.clip(rng.gamma(0.3, 1.5, T) * ((day_of_year > 152) & (day_of_year < 258)), 0, None)

    atmos_wind_speed = (reg_speed.mean(axis=1) + loc_speed.mean(axis=1)) / 2.0
    S_t = heuristic_stability_index(blh_m, inversion_strength_k, rh_pct, atmos_wind_speed, solar_rad_wm2)

    # ---- Dynamic transport: lag + probabilistic corridor, per hour ----
    tau = transport_lag_hours(regional_distance[None, :, :], reg_speed[:, :, None], max_lag_hours=48.0)  # (T, N_reg, N_local)
    lag_steps = np.clip(np.round(tau).astype(int), 0, T - 1)

    reg_wind_unit = np.stack([reg_wind_u, reg_wind_v], axis=-1)
    reg_wind_unit = reg_wind_unit / (np.linalg.norm(reg_wind_unit, axis=-1, keepdims=True) + 1e-9)

    # ---- Local emission processes ----
    morning = np.exp(-0.5 * ((hour_of_day - 8.5) / 1.3) ** 2)
    evening = np.exp(-0.5 * ((hour_of_day - 19.5) / 1.6) ** 2)
    traffic_index = (100 * (morning + 0.85 * evening) + 25) * np.where(is_weekend, 0.55, 1.0)
    traffic_index = np.tile(traffic_index[:, None], (1, N_local)) * (0.85 + 0.3 * rng.random((T, N_local)))

    local_industry = 20 + 6 * np.sin(2 * np.pi * hour_of_day / 24)
    local_industry = np.tile(local_industry[:, None], (1, N_local)) * (0.8 + 0.4 * rng.random((T, N_local))) * np.where(is_weekend, 0.7, 1.0)[:, None]

    local_dust_baseline = np.full((T, N_local), 5.0) + rng.gamma(1.0, 1.5, size=(T, N_local))
    background = np.full((T, N_local), 18.0)

    # Diwali-like fireworks pulse: a single night each generated year around day-of-year 300
    years = timestamps.year.to_numpy()
    diwali_pulse = np.zeros(T)
    for yr in np.unique(years):
        mask = (years == yr) & (day_of_year > 298) & (day_of_year < 300) & ((hour_of_day >= 18) | (hour_of_day <= 1))
        diwali_pulse[mask] = 220.0
    diwali_pulse = np.tile(diwali_pulse[:, None], (1, N_local)) * (0.7 + 0.6 * rng.random((T, N_local)))

    # ---- AR(1) accumulation loop (sequential in time; vectorized across stations) ----
    pm25 = np.zeros((T, N_local))
    biomass_contrib_hist = np.zeros((T, N_local))
    dust_contrib_hist = np.zeros((T, N_local))
    alpha = 0.45 + 0.5 * S_t  # persistence: 0.45 (well-mixed) .. 0.95 (trapped)

    pm25[0] = 60.0
    for t in range(T):
        # lagged regional biomass FEP reaching each local station at time t
        s_idx = np.arange(N_reg)
        lagged_idx = lag_steps[t]  # (N_reg, N_local): the hour each source's signal originated to arrive "now"
        origin_t = (t - lagged_idx).clip(min=0)
        # gather FEP at origin_t per (source, local) pair
        fep_origin = fep[origin_t, s_idx[:, None]]  # (N_reg, N_local)
        dust_origin = dust_index[origin_t, s_idx[:, None]]

        transport_p = probabilistic_transport_weights(
            wind_unit_vec=reg_wind_unit[t],
            direction_unit_vec=regional_direction,
            distance_km=regional_distance,
            stability_gate=(1.0 - S_t[t]),
            distance_sigma_km=regional_sigma_km,
        )  # (N_reg, N_local)

        biomass_in = (fep_origin * transport_p)[:2].sum(axis=0)   # PB + HR -> biomass channel
        dust_in = (dust_origin * transport_p)[2].reshape(-1) + (fep_origin * transport_p)[3]  # RJ dust + UP industry-transport as minor dust proxy
        biomass_contrib_hist[t] = biomass_in
        dust_contrib_hist[t] = dust_in

        trapping_boost = (1.0 + 1.2 * S_t[t])
        equilibrium = (
            0.28 * traffic_index[t]
            + 0.18 * local_industry[t]
            + 1.1 * biomass_in
            + 0.9 * dust_in
            + local_dust_baseline[t]
            + background[t]
            + diwali_pulse[t]
        ) * trapping_boost

        prev = pm25[t - 1] if t > 0 else equilibrium
        noise = rng.normal(0, 3.0, size=N_local)
        pm25[t] = np.clip(alpha[t] * prev + (1 - alpha[t]) * equilibrium + noise, 2.0, 999.0)

    # ---- Derived pollutants (rough proportional relationships for realism) ----
    pm10 = pm25 * (1.55 + 0.25 * rng.random((T, N_local)))
    no2 = 15 + 0.12 * traffic_index + rng.normal(0, 4, (T, N_local))
    co = 0.4 + 0.01 * traffic_index / 10 + rng.normal(0, 0.05, (T, N_local))
    o3 = np.clip(20 + 0.5 * solar_rad_wm2[:, None] / 20 - 0.05 * no2 + rng.normal(0, 5, (T, N_local)), 0, None)

    # ---- Ground-truth source contribution fractions (for sanity-checking the attribution head) ----
    traffic_c = 0.28 * traffic_index
    industry_c = 0.18 * local_industry + np.tile((0.15 * industry_index[:, 3])[:, None], (1, N_local)) + diwali_pulse
    biomass_c = 1.1 * biomass_contrib_hist
    dust_c = 0.9 * dust_contrib_hist + local_dust_baseline
    background_c = np.tile(background, 1)
    stack = np.stack([traffic_c, industry_c, biomass_c, dust_c, background_c], axis=-1)
    stack = np.clip(stack, 1e-6, None)
    source_contrib_gt = stack / stack.sum(axis=-1, keepdims=True)

    # ---- Ground-truth regime label ----
    regime_gt = np.zeros((T, N_local), dtype=np.int64)  # default 0 = normal
    dominant = np.argmax(stack[..., 1:], axis=-1)  # among industry(0)/biomass(1)/dust(2)/background(3)
    is_inversion = (S_t[:, None] > 0.72) & (biomass_c < np.percentile(biomass_c, 60)) & (dust_c < np.percentile(dust_c, 60))
    is_biomass = biomass_c >= np.maximum(dust_c, np.percentile(biomass_c, 70))
    is_dust = (dust_c >= biomass_c) & (dust_c > np.percentile(dust_c, 70))
    regime_gt = np.where(is_biomass, 2, regime_gt)
    regime_gt = np.where((~is_biomass) & is_dust, 3, regime_gt)
    regime_gt = np.where((~is_biomass) & (~is_dust) & np.broadcast_to(is_inversion, regime_gt.shape), 1, regime_gt)

    # ---- Assemble arrays in schema column order ----
    local = np.stack([
        pm25, pm10, no2, co, o3, np.tile(temp_c[:, None], (1, N_local)), rh_pct[:, None] * np.ones((1, N_local)),
        loc_wind_u, loc_wind_v, traffic_index,
    ], axis=-1)
    assert local.shape == (T, N_local, len(LOCAL_FEATURE_COLS))

    regional = np.stack([fep, reg_wind_u, reg_wind_v, industry_index, dust_index], axis=-1)
    assert regional.shape == (T, N_reg, len(REGIONAL_FEATURE_COLS))

    atmos = np.stack([
        blh_m, inversion_strength_k, rh_pct, atmos_wind_speed, solar_rad_wm2, pressure_hpa, precip_mm,
    ], axis=-1)
    assert atmos.shape == (T, len(ATMOS_FEATURE_COLS))

    return RawSeries(
        timestamps=timestamps,
        local=local.astype(np.float32),
        regional=regional.astype(np.float32),
        atmos=atmos.astype(np.float32),
        local_ids=[s.id for s in LOCAL_STATIONS],
        regional_ids=[s.id for s in REGIONAL_SOURCES],
        source_contrib_gt=source_contrib_gt.astype(np.float32),
        regime_gt=regime_gt.astype(np.int64),
    )
