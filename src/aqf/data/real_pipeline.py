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
    windows = _half_year_windows(date_from, date_to)
    for station in LOCAL_STATIONS:
        loc = matches.get(station.id)
        frames = []
        for w_from, w_to in windows:
            cache_path = os.path.join(cache_dir, f"{station.id}_{w_from}_{w_to}.parquet") if cache_dir else None
            if cache_path and os.path.exists(cache_path):
                frames.append(pd.read_parquet(cache_path))
                continue
            legacy = _slice_legacy_cache(cache_dir, station.id, w_from, w_to) if cache_dir else None
            if legacy is not None:
                frames.append(legacy)
                continue
            if loc is None:
                continue
            df = client.fetch_station_history(loc["id"], w_from, w_to)
            frames.append(df)
            if cache_path:
                df.to_parquet(cache_path)
        frames = [f for f in frames if not f.empty]
        out[station.id] = pd.concat(frames, ignore_index=True).sort_values("datetime") if frames else pd.DataFrame()
    return out


def _half_year_windows(date_from: str, date_to: str) -> list[tuple[str, str]]:
    """Calendar half-year windows clipped to [date_from, date_to].

    OpenAQ's deep-pagination wall (~page 12 = ~500 days of hourly rows) would
    otherwise silently drop the *end* of a long pull, which for a range ending
    today is the most recent data. Half-years stay well under it, and
    calendar-aligned windows line up with earlier cached pulls.
    """
    start, end = pd.Timestamp(date_from), pd.Timestamp(date_to)
    out = []
    cur = start
    while cur <= end:
        boundary = pd.Timestamp(cur.year, 6, 30) if cur.month <= 6 else pd.Timestamp(cur.year, 12, 31)
        w_end = min(boundary, end)
        out.append((cur.strftime("%Y-%m-%d"), w_end.strftime("%Y-%m-%d")))
        cur = w_end + pd.Timedelta(days=1)
    return out


def _slice_legacy_cache(cache_dir: str, station_id: str, w_from: str, w_to: str) -> pd.DataFrame | None:
    """Reuse an earlier whole-range cache file (e.g. DL001_2018-01-01_2023-12-31.parquet) that fully covers this window."""
    import glob
    import os

    for path in glob.glob(os.path.join(cache_dir, f"{station_id}_*_*.parquet")):
        stem = os.path.basename(path)[:-len(".parquet")]
        try:
            l_from, l_to = stem.split("_")[1:3]
        except ValueError:
            continue
        if l_from <= w_from and l_to >= w_to and (l_from, l_to) != (w_from, w_to):
            df = pd.read_parquet(path)
            if df.empty:
                return df
            ts = pd.to_datetime(df["datetime"], utc=True)
            keep = (ts >= pd.Timestamp(w_from, tz="UTC")) & (ts < pd.Timestamp(w_to, tz="UTC") + pd.Timedelta(days=1))
            return df[keep.to_numpy()]
    return None


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

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=75)  # chosen per window: a range ending today spans both products

    windows = pd.date_range(date_from, date_to, freq=f"{DAY_RANGE}D")
    if windows.empty or windows[-1] < pd.Timestamp(date_to):
        windows = windows.append(pd.DatetimeIndex([pd.Timestamp(date_to)]))

    frames = []
    for end_date in windows:
        source = HISTORICAL_SOURCE if end_date < cutoff else DEFAULT_SOURCE
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


def _split_range(start: pd.Timestamp, end: pd.Timestamp):
    """Split [start, end] into two contiguous halves, on a month boundary when the range spans several months."""
    if (start.year, start.month) != (end.year, end.month):
        months = pd.period_range(start, end, freq="M")
        boundary = months[len(months) // 2].start_time
        return (start, boundary - pd.Timedelta(days=1)), (boundary, end)
    if start == end:
        return None
    mid = start + (end - start) // 2
    return (start, mid), (mid + pd.Timedelta(days=1), end)


def _era5_chunks(kind: str, downloader, start: pd.Timestamp, end: pd.Timestamp, cache_dir: str) -> list[str]:
    """Download [start, end] via `downloader`, recursively halving on CDS "cost limits exceeded" errors.

    Verified live: a full-year 8-variable request over the NCR domain is
    rejected with `403 cost limits exceeded -- Your request is too large`,
    while a 21-day one was fine, and the ceiling isn't documented anywhere
    convenient. Rather than guess a chunk size, start big and let the API
    tell us -- each accepted chunk is cached under its own date-range name,
    so re-runs (and a crash halfway through hours of queued requests) only
    re-fetch what's missing. Downloads land in a `.part` file first so an
    interrupted download can't be mistaken for a finished one.
    """
    import os

    path = os.path.join(cache_dir, f"{kind}_{start:%Y%m%d}_{end:%Y%m%d}.nc")
    if os.path.exists(path):
        return [path]
    tmp = path + ".part"
    try:
        downloader(tmp, ERA5_DOMAIN_AREA, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        os.replace(tmp, path)
        print(f"[era5] {kind}: got {start:%Y-%m-%d}..{end:%Y-%m-%d}", flush=True)
        return [path]
    except Exception as e:
        msg = str(e).lower()
        if "cost limits" not in msg and "too large" not in msg:
            raise
        halves = _split_range(start, end)
        if halves is None:
            raise
        print(f"[era5] {kind}: {start:%Y-%m-%d}..{end:%Y-%m-%d} too large for CDS, splitting", flush=True)
        out: list[str] = []
        for a, b in halves:
            out += _era5_chunks(kind, downloader, a, b, cache_dir)
        return out


def fetch_era5(date_from: str, date_to: str, cache_dir: str = _era5_scratch_dir, max_workers: int = 6) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Real ERA5 pull: domain-averaged atmos series, per-REGIONAL_SOURCES wind, and per-station local met.

    Starts from one request per calendar year and adaptively halves any
    request CDS rejects as too large (see `_era5_chunks`). The single-level
    (8 variables) and pressure-level (925hPa temperature only) downloads are
    chunked independently -- the second is ~8x cheaper per request so it may
    tolerate much bigger chunks -- and combined afterwards on `datetime`.
    """
    import os

    from aqf.data.era5 import (
        build_domain_atmos_series,
        domain_t925_series,
        download_era5,
        download_era5_pressure_level,
        extract_points_met,
        inversion_strength_from_profile,
    )

    os.makedirs(cache_dir, exist_ok=True)

    # CDS queues each request server-side and a quarter-year single-level chunk took >13 minutes in
    # testing, so ~24 chunks + the pressure-level ones run one after another would be most of a day.
    # The per-year streams are independent; submit them concurrently (each cdsapi call builds its own
    # Client, so this is thread-safe) and let CDS interleave them.
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for year in range(pd.Timestamp(date_from).year, pd.Timestamp(date_to).year + 1):
        y_start = max(pd.Timestamp(date_from), pd.Timestamp(f"{year}-01-01"))
        y_end = min(pd.Timestamp(date_to), pd.Timestamp(f"{year}-12-31"))
        jobs.append(("single", download_era5, y_start, y_end))
        jobs.append(("pressure925", download_era5_pressure_level, y_start, y_end))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda j: (j[0], _era5_chunks(j[0], j[1], j[2], j[3], cache_dir)), jobs))
    single_paths = [p for kind, paths in results if kind == "single" for p in paths]
    pressure_paths = [p for kind, paths in results if kind == "pressure925" for p in paths]

    single_frames = [build_domain_atmos_series(p, None, keep_t2m=True) for p in single_paths]
    atmos_df = pd.concat(single_frames, ignore_index=True).drop_duplicates(subset="datetime").sort_values("datetime")

    t925 = pd.concat([domain_t925_series(p) for p in pressure_paths], ignore_index=True).drop_duplicates(subset="datetime")
    atmos_df = atmos_df.merge(t925, on="datetime", how="left")
    atmos_df["inversion_strength_k"] = inversion_strength_from_profile(atmos_df["_t2m_k"].to_numpy(), atmos_df["_t925_k"].to_numpy())
    atmos_df = atmos_df.drop(columns=["_t2m_k", "_t925_k"])

    points = {s.id: (s.lat, s.lon) for s in REGIONAL_SOURCES}
    points.update({s.id: (s.lat, s.lon) for s in LOCAL_STATIONS})
    per_chunk = [extract_points_met(p, points) for p in single_paths]
    merged = {
        pid: pd.concat([c[pid] for c in per_chunk], ignore_index=True).drop_duplicates(subset="datetime").sort_values("datetime")
        for pid in points
    }
    regional_wind = {s.id: merged[s.id][["datetime", "wind_u_ms", "wind_v_ms"]] for s in REGIONAL_SOURCES}
    local_met = {s.id: merged[s.id] for s in LOCAL_STATIONS}
    return atmos_df, regional_wind, local_met


def apply_era5_to_raw(
    raw: RawSeries,
    atmos_df: pd.DataFrame,
    regional_wind: dict[str, pd.DataFrame],
    local_met: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Overwrite `raw.atmos` and the regional wind columns with real ERA5 values, in place.

    Any hour ERA5 doesn't cover (e.g. the ~5-day reanalysis latency at the
    recent end) keeps the placeholder climatology rather than becoming NaN.
    Returns coverage stats so callers can see how much of the series is
    real vs placeholder.
    """
    index = pd.DatetimeIndex(raw.timestamps)
    cols = ["blh_m", "inversion_strength_k", "rh_pct", "wind_speed_ms", "solar_rad_wm2", "pressure_hpa", "precip_mm"]

    df = atmos_df.set_index(pd.to_datetime(atmos_df["datetime"]).dt.tz_localize(None)).reindex(index)
    era5 = df[cols].to_numpy(dtype=np.float32)
    placeholder = _placeholder_atmos(index).astype(np.float32)
    raw.atmos = np.where(np.isnan(era5), placeholder, era5).astype(np.float32)

    wind_cov = {}
    for j, sid in enumerate(raw.regional_ids):
        wdf = regional_wind.get(sid)
        if wdf is None or wdf.empty:
            continue
        wdf = wdf.set_index(pd.to_datetime(wdf["datetime"]).dt.tz_localize(None)).reindex(index)
        u, v = wdf["wind_u_ms"].to_numpy(), wdf["wind_v_ms"].to_numpy()
        raw.regional[:, j, 1] = np.where(np.isnan(u), 0.0, u)
        raw.regional[:, j, 2] = np.where(np.isnan(v), -2.0, v)
        wind_cov[sid] = float(1.0 - np.isnan(u).mean())

    local_filled = {}
    if local_met:
        # Fill temp / RH / wind at each station from the ERA5 cell nearest it, but ONLY where the
        # station has no reading of its own (OpenAQ has nothing for most stations across 2018-2023).
        # Without this those columns fall through to impute_for_training's global-mean fallback --
        # i.e. a constant, which for local wind would make the wind-aligned local graph meaningless.
        col_map = {"temp_c": 5, "rh_pct": 6, "wind_u_ms": 7, "wind_v_ms": 8}
        for i, sid in enumerate(raw.local_ids):
            mdf = local_met.get(sid)
            if mdf is None or mdf.empty:
                continue
            mdf = mdf.set_index(pd.to_datetime(mdf["datetime"]).dt.tz_localize(None)).reindex(index)
            n = 0
            for name, dst in col_map.items():
                if name not in mdf.columns:
                    continue
                vals = mdf[name].to_numpy(dtype=np.float32)
                gap = np.isnan(raw.local[:, i, dst]) & ~np.isnan(vals)
                raw.local[gap, i, dst] = vals[gap]
                n += int(gap.sum())
            local_filled[sid] = n

    return {
        "atmos_era5_coverage": float(1.0 - np.isnan(era5).any(axis=1).mean()),
        "regional_wind_coverage": wind_cov,
        "local_met_cells_filled": int(sum(local_filled.values())),
    }


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

    atmos = _placeholder_atmos(index).astype(np.float32)
    regional[:, :, 1] = 0.0   # wind_u_ms placeholder, replaced below if ERA5 is available
    regional[:, :, 2] = -2.0  # wind_v_ms placeholder (slight northerly, i.e. toward Delhi -- a guess, not data)

    raw = RawSeries(
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

    if use_era5:
        try:
            atmos_df, regional_wind, local_met = fetch_era5(date_from, date_to)
            stats = apply_era5_to_raw(raw, atmos_df, regional_wind, local_met)
            print(f"ERA5 applied: {stats}", flush=True)
        except Exception as e:  # cdsapi/xarray missing, ~/.cdsapirc not set up, license not accepted, etc.
            warnings.warn(f"ERA5 fetch failed ({e!r}) -- falling back to placeholder atmos/regional-wind data.")

    return raw


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
    # Keep a mask assemble_real_raw_series already set (it knows which non-NaN cells are opencity
    # estimates vs. proxies vs. real readings); only derive one from NaN-ness if there isn't one.
    observed_mask = raw.local_observed_mask if raw.local_observed_mask is not None else ~np.isnan(raw.local)
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
