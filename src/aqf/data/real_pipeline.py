"""Assembles a real-data RawSeries from OpenAQ (local pollutants + met) and FIRMS (regional fires).

This is the bridge between the individual API clients (openaq.py, firms.py,
opencity.py, era5.py) and the model, which only ever sees the RawSeries
schema (data/schema.py) -- identical to what synthetic.py produces, so
nothing downstream (graph construction, the model, training) needs to know
or care whether the numbers came from the simulator or from here.

What's real right now vs. placeholder, and why:

  REAL (from live, verified APIs):
    - local pm25, pm10, no2, co, o3, temp, RH, wind      <- OpenAQ (data/openaq.py)
    - regional fire activity (Fire Emission Proxy)        <- FIRMS (data/firms.py, features/fire_proxy.py)
    - domain atmos: BLH, inversion strength, RH, wind
      speed, solar radiation, pressure, precip            <- ERA5 (data/era5.py), when `use_era5=True`
      (default) and ~/.cdsapirc is configured -- this is
      the core physics input (continuous stability index),
      verified live: BLH ~600-735m over Delhi at noon in
      August, physically correct for that season.
    - regional (Punjab/Haryana/Rajasthan/W-UP) wind        <- ERA5, nearest-neighbor per source centroid

  PLACEHOLDER (no free real-data source identified yet):
    - regional industry_index, dust_index -- constants.
      A real industrial-activity or satellite-AOD proxy
      would replace these; out of scope for now.
    - local traffic_index -- diurnal heuristic; no free
      real-time traffic-count API wired in.

If ERA5 isn't configured (cdsapi/xarray missing, ~/.cdsapirc absent, or a
CDS request fails for any reason -- license not accepted, etc.),
assemble_real_raw_series() catches it, warns, and falls back to the
documented diurnal-climatology placeholder so the pipeline still runs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from aqf.data.firms import FIRMSClient, assign_region
from aqf.data.openaq import OpenAQClient, match_local_stations_to_openaq
from aqf.data.schema import ATMOS_FEATURE_COLS, LOCAL_FEATURE_COLS, REGIONAL_FEATURE_COLS, RawSeries
from aqf.features.fire_proxy import fire_emission_proxy
from aqf.graph.stations import LOCAL_STATIONS, REGIONAL_SOURCES


def _hourly_index(date_from: str, date_to: str) -> pd.DatetimeIndex:
    return pd.date_range(date_from, date_to, freq="h", tz="UTC")


def fetch_local_stations(
    date_from: str, date_to: str, api_key: str | None = None, cache_dir: str | None = "data/real/.openaq_cache"
) -> dict[str, pd.DataFrame]:
    """Real OpenAQ pull for every graph/stations.py::LOCAL_STATIONS station.

    A multi-year pull for 15 stations can take hours (OpenAQ's free-tier cap
    is 2000 requests/hour, and a 5-year x ~8-sensor pull needs thousands of
    paginated requests). `cache_dir` (on by default) saves each station's
    result as soon as it's fetched, so an interrupted run resumes instead of
    restarting from zero -- set to None to disable.
    """
    import os

    client = OpenAQClient(api_key=api_key)
    matches = match_local_stations_to_openaq(client)
    missing = [s.name for s in LOCAL_STATIONS if s.id not in matches]
    if missing:
        warnings.warn(f"No OpenAQ location match for: {missing} -- these stations will be all-NaN.")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    out = {}
    for station in LOCAL_STATIONS:
        cache_path = os.path.join(cache_dir, f"{station.id}_{date_from}_{date_to}.parquet") if cache_dir else None
        if cache_path and os.path.exists(cache_path):
            out[station.id] = pd.read_parquet(cache_path)
            continue

        loc = matches.get(station.id)
        if loc is None:
            out[station.id] = pd.DataFrame()
            continue
        df = client.fetch_station_history(loc["id"], date_from, date_to)
        out[station.id] = df
        if cache_path:
            df.to_parquet(cache_path)
    return out


def fetch_regional_fires(date_from: str, date_to: str, map_key: str | None = None) -> pd.DataFrame:
    """Real FIRMS pull, bucketed to REGIONAL_SOURCES and aggregated into hourly Fire Emission Proxy.

    Loops the Area API in <=5-day windows -- verified live: the documented
    "1-10" day_range in some FIRMS docs is stale, the API actually rejects
    anything outside [1, 5] ("Invalid day range. Expects [1..5.]").

    Source selection (verified live via the data_availability endpoint):
    VIIRS_SNPP_NRT only retains a rolling ~2.5-month window, so any request
    older than that returns silently empty, not an error -- caught exactly
    this while testing a 2019 pull. Anything older than ~75 days uses
    VIIRS_SNPP_SP instead (verified coverage: 2012-01-20 to 2026-06-30).
    """
    from aqf.data.firms import DEFAULT_SOURCE, HISTORICAL_SOURCE

    client = FIRMSClient(map_key=map_key)
    bbox = (73.0, 24.0, 79.5, 32.5)  # NCR_TRANSPORT_DOMAIN_BBOX, see firms.py
    DAY_RANGE = 5

    is_historical = pd.Timestamp(date_to) < (pd.Timestamp.now() - pd.Timedelta(days=75))
    source = HISTORICAL_SOURCE if is_historical else DEFAULT_SOURCE

    windows = pd.date_range(date_from, date_to, freq=f"{DAY_RANGE}D")
    if windows.empty or windows[-1] < pd.Timestamp(date_to):
        windows = windows.append(pd.DatetimeIndex([pd.Timestamp(date_to)]))

    frames = []
    for end_date in windows:
        df = client.fetch_area(bbox, day_range=DAY_RANGE, date=end_date.strftime("%Y-%m-%d"), source=source)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    raw["acq_datetime"] = pd.to_datetime(raw["acq_date"]) + pd.to_timedelta(
        raw["acq_time"].astype(str).str.zfill(4).str[:2].astype(int), unit="h"
    )
    frp_col = "frp" if "frp" in raw.columns else "bright_ti4"
    raw = raw.rename(columns={frp_col: "frp_mw"})
    raw = assign_region(raw)
    return fire_emission_proxy(raw, region_col="region_id", time_col="acq_datetime", frp_col="frp_mw")


ERA5_DOMAIN_AREA = (32.5, 73.0, 24.0, 79.5)  # (north, west, south, east) -- same NCR transport domain as FIRMS

_era5_scratch_dir = "data/real/.era5_cache"


def fetch_era5(date_from: str, date_to: str, cache_dir: str = _era5_scratch_dir) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Real ERA5 pull: domain-averaged atmos series + per-REGIONAL_SOURCES wind.

    Chunked by calendar year: a single request spanning multiple years risks
    hitting CDS's per-request size limits (8 variables x hourly x multi-year
    is a lot of data) and, if it fails, loses everything already queued.
    Per-year chunking keeps each CDS request reasonably sized and means a
    failure in one year doesn't lose progress on the others -- each year's
    two downloads (single-levels, pressure-levels) are cached to `cache_dir`
    independently, so a re-run only re-fetches what's missing.
    """
    import os

    from aqf.data.era5 import (
        build_domain_atmos_series,
        download_era5,
        download_era5_pressure_level,
        extract_point_wind_series,
    )

    os.makedirs(cache_dir, exist_ok=True)

    years = range(pd.Timestamp(date_from).year, pd.Timestamp(date_to).year + 1)
    atmos_frames, wind_frames = [], {s.id: [] for s in REGIONAL_SOURCES}

    for year in years:
        year_start = max(pd.Timestamp(date_from), pd.Timestamp(f"{year}-01-01"))
        year_end = min(pd.Timestamp(date_to), pd.Timestamp(f"{year}-12-31"))
        y_from, y_to = year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")

        single_path = os.path.join(cache_dir, f"single_{year}.nc")
        pressure_path = os.path.join(cache_dir, f"pressure925_{year}.nc")

        if not os.path.exists(single_path):
            download_era5(single_path, ERA5_DOMAIN_AREA, y_from, y_to)
        if not os.path.exists(pressure_path):
            download_era5_pressure_level(pressure_path, ERA5_DOMAIN_AREA, y_from, y_to)

        atmos_frames.append(build_domain_atmos_series(single_path, pressure_path))
        for source in REGIONAL_SOURCES:
            wind_frames[source.id].append(extract_point_wind_series(single_path, source.lat, source.lon))

    atmos_df = pd.concat(atmos_frames, ignore_index=True).drop_duplicates(subset="datetime").sort_values("datetime")
    regional_wind = {
        sid: pd.concat(frames, ignore_index=True).drop_duplicates(subset="datetime").sort_values("datetime")
        for sid, frames in wind_frames.items()
    }
    return atmos_df, regional_wind


def _placeholder_atmos(index: pd.DatetimeIndex) -> np.ndarray:
    """Diurnal-climatology placeholder for BLH/inversion/solar/pressure/precip, pending ERA5.

    NOT real data -- see module docstring. Uses the same functional shape as
    the synthetic generator's climatology so the stability index at least
    responds to time-of-day, but carries no actual meteorological
    information for the date in question.
    """
    hour = index.hour.to_numpy()
    diurnal = np.clip(np.sin(2 * np.pi * (hour - 6) / 24), -0.2, 1.0).clip(min=0)
    blh_m = 150 + 1200 * diurnal
    inversion_strength_k = np.clip(3.0 * (1 - blh_m / 1400.0), -1.0, 6.0)
    rh_pct = np.full(len(index), 55.0)
    wind_speed_ms = np.full(len(index), 2.5)
    solar_rad_wm2 = 500 * diurnal
    pressure_hpa = np.full(len(index), 1008.0)
    precip_mm = np.zeros(len(index))
    return np.stack([blh_m, inversion_strength_k, rh_pct, wind_speed_ms, solar_rad_wm2, pressure_hpa, precip_mm], axis=-1)


def assemble_real_raw_series(
    date_from: str,
    date_to: str,
    openaq_api_key: str | None = None,
    firms_map_key: str | None = None,
    use_era5: bool = True,
    use_opencity_fallback: bool = True,
) -> RawSeries:
    """Build a RawSeries from live OpenAQ + FIRMS (+ ERA5, + opencity.in) data.

    See module docstring for what's real vs placeholder. `use_era5=True`
    (default) tries a real ERA5 pull for atmos + regional wind and falls
    back to the documented placeholder climatology with a warning if
    cdsapi/xarray aren't installed or ~/.cdsapirc isn't configured -- so
    this function still works end-to-end without ERA5 access.

    OpenAQ Delhi coverage has a real gap: verified live, every station's
    original sensor generation stopped ~Jan-Feb 2018 and a fresh one only
    resumed 2025-02-18 -- a ~7-year hole for most stations (a couple,
    Wazirpur/Najafgarh, partially bridge it through ~Oct 2022). For any
    hour where OpenAQ has no PM2.5 reading, `use_opencity_fallback=True`
    fills it from data.opencity.in's AQI-derived approximate PM2.5 (real
    2017-2023 hourly coverage for all 15 stations, no gaps, see
    data/opencity.py -- an *estimate* via CPCB's AQI breakpoint table, not a
    direct concentration reading, but genuine per-hour signal rather than
    interpolated filler). Sets `local_observed_mask` to distinguish real
    OpenAQ readings, real-but-estimated opencity readings, and (after a
    later `impute_for_training` call) pure interpolation -- see that
    function and losses/physics.py::forecast_loss for how the distinction
    is used.
    """
    index = _hourly_index(date_from, date_to)
    T, N_local, N_reg = len(index), len(LOCAL_STATIONS), len(REGIONAL_SOURCES)

    local_raw = fetch_local_stations(date_from, date_to, api_key=openaq_api_key)
    local = np.full((T, N_local, len(LOCAL_FEATURE_COLS)), np.nan, dtype=np.float32)
    real_observed = np.zeros((T, N_local, len(LOCAL_FEATURE_COLS)), dtype=bool)
    col_map = {"pm25": 0, "pm10": 1, "no2": 2, "co": 3, "o3": 4, "temperature": 5, "relativehumidity": 6}
    for i, station in enumerate(LOCAL_STATIONS):
        df = local_raw.get(station.id, pd.DataFrame())
        if df.empty:
            continue
        # OpenAQ's hourly-aggregate buckets are timestamped on the half-hour (e.g. ...T05:30:00Z),
        # not on the hour -- verified live. Round to the nearest hour before aligning to `index`,
        # or every value silently reindexes to NaN (caught exactly this: 0% coverage without it).
        ts = pd.to_datetime(df["datetime"], utc=True).dt.round("h")
        df = df.set_index(ts).groupby(level=0).mean(numeric_only=True).reindex(index)
        for src_col, dst_idx in col_map.items():
            if src_col in df.columns:
                local[:, i, dst_idx] = df[src_col].to_numpy()
                real_observed[:, i, dst_idx] = ~np.isnan(local[:, i, dst_idx])
        if "wind_speed" in df.columns and "wind_direction" in df.columns:
            speed = df["wind_speed"].to_numpy()
            angle = np.radians(df["wind_direction"].to_numpy())
            local[:, i, 7] = -speed * np.sin(angle)  # wind_u_ms (blowing toward, meteorological convention)
            local[:, i, 8] = -speed * np.cos(angle)  # wind_v_ms
            real_observed[:, i, 7] = ~np.isnan(local[:, i, 7])
            real_observed[:, i, 8] = ~np.isnan(local[:, i, 8])
        # traffic_index (col 9): no free real-time traffic-count source wired in yet -- diurnal proxy.
        hour = index.hour.to_numpy()
        local[:, i, 9] = 50 + 40 * np.clip(np.sin(2 * np.pi * (hour - 9) / 24), 0, None)

    if use_opencity_fallback:
        _fill_pm25_from_opencity(local, real_observed, index, date_from, date_to)

    regional_fep = fetch_regional_fires(date_from, date_to, map_key=firms_map_key)
    regional = np.zeros((T, N_reg, len(REGIONAL_FEATURE_COLS)), dtype=np.float32)
    if not regional_fep.empty:
        regional_fep = regional_fep.reindex(index.tz_localize(None) if regional_fep.index.tz is None else index)
        for j, source in enumerate(REGIONAL_SOURCES):
            if source.id in regional_fep.columns:
                regional[:, j, 0] = np.nan_to_num(regional_fep[source.id].to_numpy())
    # industry/dust index: no free real-data source wired in yet -- placeholder either way.
    regional[:, :, 3] = 15.0  # industry_index placeholder
    regional[:, :, 4] = 5.0   # dust_index placeholder

    atmos_df, regional_wind = None, None
    if use_era5:
        try:
            atmos_df, regional_wind = fetch_era5(date_from, date_to)
        except Exception as e:  # cdsapi/xarray missing, ~/.cdsapirc not set up, license not accepted, etc.
            warnings.warn(f"ERA5 fetch failed ({e!r}) -- falling back to placeholder atmos/regional-wind data.")

    if atmos_df is not None:
        atmos_df = atmos_df.set_index(pd.to_datetime(atmos_df["datetime"]).dt.tz_localize(None)).reindex(index.tz_localize(None))
        atmos = atmos_df[["blh_m", "inversion_strength_k", "rh_pct", "wind_speed_ms", "solar_rad_wm2", "pressure_hpa", "precip_mm"]].to_numpy(dtype=np.float32)
        atmos = np.where(np.isnan(atmos), _placeholder_atmos(index).astype(np.float32), atmos)  # fill any ERA5 gaps
    else:
        atmos = _placeholder_atmos(index).astype(np.float32)

    if regional_wind is not None:
        for j, source in enumerate(REGIONAL_SOURCES):
            wdf = regional_wind.get(source.id)
            if wdf is None or wdf.empty:
                regional[:, j, 1] = 0.0
                regional[:, j, 2] = -2.0
                continue
            wdf = wdf.set_index(pd.to_datetime(wdf["datetime"]).dt.tz_localize(None)).reindex(index.tz_localize(None))
            regional[:, j, 1] = np.nan_to_num(wdf["wind_u_ms"].to_numpy(), nan=0.0)
            regional[:, j, 2] = np.nan_to_num(wdf["wind_v_ms"].to_numpy(), nan=-2.0)
    else:
        regional[:, :, 1] = 0.0   # wind_u_ms placeholder
        regional[:, :, 2] = -2.0  # wind_v_ms placeholder (slight northerly, i.e. toward Delhi -- a guess, not data)

    return RawSeries(
        timestamps=pd.DatetimeIndex(index.tz_localize(None)),
        local=local,
        regional=regional,
        atmos=atmos,
        local_ids=[s.id for s in LOCAL_STATIONS],
        regional_ids=[s.id for s in REGIONAL_SOURCES],
        source_contrib_gt=None,  # no ground truth for real data -- source_contrib_loss is skipped automatically
        regime_gt=None,
        local_observed_mask=real_observed,  # True where OpenAQ or opencity gave a genuine reading; see docstring above
    )


def _fill_pm25_from_opencity(local: np.ndarray, real_observed: np.ndarray, index: pd.DatetimeIndex, date_from: str, date_to: str) -> None:
    """Fill NaN pm25 gaps (in place) using data.opencity.in's AQI-derived estimate. See assemble_real_raw_series docstring."""
    from aqf.data import opencity

    pm25_col = 0  # LOCAL_FEATURE_COLS.index("pm25")
    still_missing = np.isnan(local[:, :, pm25_col])
    if not still_missing.any():
        return

    try:
        resources = opencity.list_resources()
    except Exception as e:
        warnings.warn(f"opencity.in fallback unavailable ({e!r}) -- PM2.5 gaps outside OpenAQ coverage stay unfilled here.")
        return

    naive_index = index.tz_localize(None) if index.tz is not None else index
    for i, station in enumerate(LOCAL_STATIONS):
        if not still_missing[:, i].any():
            continue
        frag = opencity.STATION_NAME_OVERRIDES.get(station.name, station.name)
        try:
            df = opencity.fetch_station_aqi(frag, resources=resources)
        except Exception as e:
            warnings.warn(f"opencity.in fetch failed for {station.name} ({e!r}) -- skipping fallback for this station.")
            continue
        df = df.set_index(pd.to_datetime(df["datetime"])).reindex(naive_index)
        fill_values = df["pm25_approx"].to_numpy()
        mask = still_missing[:, i] & ~np.isnan(fill_values)
        local[mask, i, pm25_col] = fill_values[mask]
        real_observed[mask, i, pm25_col] = True


def impute_for_training(raw: RawSeries) -> RawSeries:
    """Fill gaps in `raw.local` for model input, while recording which values were genuinely observed.

    Real CPCB/DPCC station uptime is intermittent (observed ~30-36% non-NaN
    in initial testing) -- the model's encoders need a complete array (NaN
    would propagate through every linear layer), but training should only be
    *scored* against genuinely-observed target hours, not values this
    function invented. Sets `raw.local_observed_mask` so
    losses/physics.py::forecast_loss can mask the loss accordingly -- without
    this, a naive "impute everything, train against the imputed values"
    approach is optimistic: it partly measures interpolation smoothness
    rather than real forecasting skill (this was an explicit caveat on the
    first real-data smoke test, scripts/smoke_test_real.py).

    Per station, per feature: forward-fill, then backward-fill (for leading
    gaps with no prior observation yet), then any feature that's NaN for a
    station's *entire* window falls back to the global mean across all
    stations at that timestep, or 0.0 if that's also all-NaN.
    """
    from aqf.data.schema import LOCAL_FEATURE_COLS

    T, N, F = raw.local.shape
    observed_mask = ~np.isnan(raw.local)
    imputed = raw.local.copy()

    for n in range(N):
        df = pd.DataFrame(imputed[:, n, :], columns=LOCAL_FEATURE_COLS)
        imputed[:, n, :] = df.ffill().bfill().to_numpy()

    for f in range(F):
        col = imputed[:, :, f]
        if np.isnan(col).any():
            fallback = np.nanmean(col) if not np.isnan(col).all() else 0.0
            col[np.isnan(col)] = fallback
            imputed[:, :, f] = col

    raw.local = imputed
    raw.local_observed_mask = observed_mask
    return raw
