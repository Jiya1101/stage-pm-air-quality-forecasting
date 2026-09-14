"""Assembles a real-data RawSeries from OpenAQ (local pollutants + met) and FIRMS (regional fires).

This is the bridge between the individual API clients (openaq.py, firms.py,
opencity.py, era5.py) and the model, which only ever sees the RawSeries
schema (data/schema.py) -- identical to what synthetic.py produces, so
nothing downstream (graph construction, the model, training) needs to know
or care whether the numbers came from the simulator or from here.

What's real right now vs. placeholder, and why:

  REAL (from live, verified APIs):
    - local pm25, pm10, no2, co, o3, temp, RH, wind  <- OpenAQ (data/openaq.py)
    - regional fire activity (Fire Emission Proxy)    <- FIRMS (data/firms.py, features/fire_proxy.py)

  PLACEHOLDER (until ERA5 is wired in -- see data/era5.py):
    - atmospheric BLH, inversion strength, solar radiation, pressure, precip
      -- these fundamentally require ERA5 reanalysis; ground stations don't
      measure boundary-layer height. Filled with a documented diurnal
      climatology, NOT real data -- the stability index computed from these
      will be a rough proxy, not the real physics, until ERA5 is added.
    - regional (Punjab/Haryana/Rajasthan/W-UP) wind vectors and
      industry/dust indices -- OpenAQ station density is sparse outside
      Delhi, so these fall back to the same climatology. FIRMS gives fire
      location/intensity, not wind.

Once ERA5 access is set up, replace `_placeholder_atmos()` and
`_placeholder_regional_met()` with real pulls (era5.py already has the
retrieval + NetCDF-extraction code) -- the RawSeries shape does not change.
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


def fetch_local_stations(date_from: str, date_to: str, api_key: str | None = None) -> dict[str, pd.DataFrame]:
    """Real OpenAQ pull for every graph/stations.py::LOCAL_STATIONS station."""
    client = OpenAQClient(api_key=api_key)
    matches = match_local_stations_to_openaq(client)
    missing = [s.name for s in LOCAL_STATIONS if s.id not in matches]
    if missing:
        warnings.warn(f"No OpenAQ location match for: {missing} -- these stations will be all-NaN.")

    out = {}
    for station in LOCAL_STATIONS:
        loc = matches.get(station.id)
        if loc is None:
            out[station.id] = pd.DataFrame()
            continue
        out[station.id] = client.fetch_station_history(loc["id"], date_from, date_to)
    return out


def fetch_regional_fires(date_from: str, date_to: str, map_key: str | None = None) -> pd.DataFrame:
    """Real FIRMS pull, bucketed to REGIONAL_SOURCES and aggregated into hourly Fire Emission Proxy.

    Loops the Area API in <=5-day windows -- verified live: the documented
    "1-10" day_range in some FIRMS docs is stale, the API actually rejects
    anything outside [1, 5] ("Invalid day range. Expects [1..5].").
    """
    client = FIRMSClient(map_key=map_key)
    bbox = (73.0, 24.0, 79.5, 32.5)  # NCR_TRANSPORT_DOMAIN_BBOX, see firms.py
    DAY_RANGE = 5

    windows = pd.date_range(date_from, date_to, freq=f"{DAY_RANGE}D")
    if windows.empty or windows[-1] < pd.Timestamp(date_to):
        windows = windows.append(pd.DatetimeIndex([pd.Timestamp(date_to)]))

    frames = []
    for end_date in windows:
        df = client.fetch_area(bbox, day_range=DAY_RANGE, date=end_date.strftime("%Y-%m-%d"))
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
) -> RawSeries:
    """Build a RawSeries from live OpenAQ + FIRMS data. See module docstring for what's real vs placeholder."""
    index = _hourly_index(date_from, date_to)
    T, N_local, N_reg = len(index), len(LOCAL_STATIONS), len(REGIONAL_SOURCES)

    local_raw = fetch_local_stations(date_from, date_to, api_key=openaq_api_key)
    local = np.full((T, N_local, len(LOCAL_FEATURE_COLS)), np.nan, dtype=np.float32)
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
        if "wind_speed" in df.columns and "wind_direction" in df.columns:
            speed = df["wind_speed"].to_numpy()
            angle = np.radians(df["wind_direction"].to_numpy())
            local[:, i, 7] = -speed * np.sin(angle)  # wind_u_ms (blowing toward, meteorological convention)
            local[:, i, 8] = -speed * np.cos(angle)  # wind_v_ms
        # traffic_index (col 9): no free real-time traffic-count source wired in yet -- diurnal proxy.
        hour = index.hour.to_numpy()
        local[:, i, 9] = 50 + 40 * np.clip(np.sin(2 * np.pi * (hour - 9) / 24), 0, None)

    regional_fep = fetch_regional_fires(date_from, date_to, map_key=firms_map_key)
    regional = np.zeros((T, N_reg, len(REGIONAL_FEATURE_COLS)), dtype=np.float32)
    if not regional_fep.empty:
        regional_fep = regional_fep.reindex(index.tz_localize(None) if regional_fep.index.tz is None else index)
        for j, source in enumerate(REGIONAL_SOURCES):
            if source.id in regional_fep.columns:
                regional[:, j, 0] = np.nan_to_num(regional_fep[source.id].to_numpy())
    # regional wind/industry/dust: placeholder until ERA5 / a proper industrial-activity proxy is wired in.
    regional[:, :, 1] = 0.0   # wind_u_ms placeholder
    regional[:, :, 2] = -2.0  # wind_v_ms placeholder (slight northerly, i.e. toward Delhi -- a guess, not data)
    regional[:, :, 3] = 15.0  # industry_index placeholder
    regional[:, :, 4] = 5.0   # dust_index placeholder

    atmos = _placeholder_atmos(index).astype(np.float32)

    return RawSeries(
        timestamps=pd.DatetimeIndex(index.tz_localize(None)),
        local=local,
        regional=regional,
        atmos=atmos,
        local_ids=[s.id for s in LOCAL_STATIONS],
        regional_ids=[s.id for s in REGIONAL_SOURCES],
        source_contrib_gt=None,  # no ground truth for real data -- source_contrib_loss is skipped automatically
        regime_gt=None,
    )
