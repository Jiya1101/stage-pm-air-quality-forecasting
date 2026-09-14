"""Delhi Hourly Air Quality Reports (data.opencity.in / CPCB), 2017-2023.

No API key needed. Verified end-to-end while building this: the dataset
covers exactly 39 Delhi CAAQMS/DPCC/IMD/IITM stations at hourly resolution
from 2017 through 2023 -- matching this project's default temporal split
(train 2018-2021 / val 2022 / test 2023) almost exactly, and every station
name in graph/stations.py::LOCAL_STATIONS has a direct match here.

IMPORTANT CAVEAT: this dataset gives the **composite Air Quality Index**
(capped at 500, India's National AQI scale), not raw per-pollutant
concentrations in ug/m3. That's a real limitation for a physics-constrained
model whose mass-conservation loss needs actual PM2.5 concentration, not an
index. Two ways to use it:

  1. `approximate_pm25_from_aqi()` inverts CPCB's official AQI sub-index
     breakpoint table to recover an *approximate* PM2.5 concentration. This
     is only exact when PM2.5 was the "prominent pollutant" driving the AQI
     that hour (usually true for Delhi, especially in winter) -- treat it as
     a reasonable estimate, not ground truth.
  2. Use OpenAQ (data/openaq.py) instead for authoritative multi-pollutant
     concentrations, and use this module's AQI series as a free secondary
     feature / cross-check / gap-filler.

Discovery is done live against opencity.in's CKAN API (package_show) rather
than hardcoding per-station resource UUIDs, so it keeps working if they
re-upload the dataset under new resource IDs.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests

CKAN_BASE = "https://data.opencity.in"
PACKAGE_ID = "delhi-hourly-air-quality-reports"

_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

# CPCB National AQI sub-index breakpoints for PM2.5 (24-hr avg, ug/m3) -> AQI.
# (aqi_lo, aqi_hi, conc_lo, conc_hi); used in reverse to approximate concentration from AQI.
_PM25_BREAKPOINTS = [
    (0, 50, 0.0, 30.0),
    (51, 100, 30.0, 60.0),
    (101, 200, 60.0, 90.0),
    (201, 300, 90.0, 120.0),
    (301, 400, 120.0, 250.0),
    (401, 500, 250.0, 380.0),
]


def approximate_pm25_from_aqi(aqi: np.ndarray) -> np.ndarray:
    """Invert the CPCB PM2.5 AQI sub-index table to an approximate concentration in ug/m3.

    See module docstring caveat -- only reliable when PM2.5 is the prominent
    pollutant. Values above 500 (already capped by the source data) or below
    0 are clamped to the table's outer breakpoints.
    """
    aqi = np.asarray(aqi, dtype=float)
    conc = np.full_like(aqi, np.nan)
    for aqi_lo, aqi_hi, c_lo, c_hi in _PM25_BREAKPOINTS:
        mask = (aqi >= aqi_lo) & (aqi <= aqi_hi)
        frac = (aqi[mask] - aqi_lo) / (aqi_hi - aqi_lo)
        conc[mask] = c_lo + frac * (c_hi - c_lo)
    conc = np.where(aqi > 500, _PM25_BREAKPOINTS[-1][3], conc)
    conc = np.where(aqi < 0, np.nan, conc)
    return conc


def list_resources(package_id: str = PACKAGE_ID, timeout: int = 30) -> list[dict]:
    """Live CKAN resource listing for the dataset (name, download url, etc.)."""
    resp = requests.get(f"{CKAN_BASE}/api/3/action/package_show", params={"id": package_id}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN package_show failed: {payload}")
    return payload["result"]["resources"]


def find_resource_url(station_name_fragment: str, resources: list[dict] | None = None, suffix: str = "2017-2023") -> str:
    """Find the hourly (not 15-minute-2024-25) CSV resource whose name contains `station_name_fragment`."""
    resources = resources if resources is not None else list_resources()
    frag = station_name_fragment.lower()
    candidates = [r for r in resources if frag in r["name"].lower() and suffix in r["name"]]
    if not candidates:
        available = [r["name"] for r in resources if suffix in r["name"]]
        raise ValueError(f"No resource matching '{station_name_fragment}' (suffix={suffix!r}). Available: {available}")
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous match for '{station_name_fragment}': {[c['name'] for c in candidates]}")
    return candidates[0]["url"]


def parse_wide_aqi_csv(text: str) -> pd.DataFrame:
    """Parse opencity.in's wide Year/Month-block CSV format into a long hourly (datetime, aqi) frame.

    Format (verified against the live DTU CPCB file):
        Year,2017
        January-2017,00:00:00,01:00:00,...,23:00:00
        1,434.0,417.0,...,407          <- day-of-month, 24 hourly values
        2,421.0,434.0,...,416
        ...
        February-2017,00:00:00,...
        ...
        Year,2018
        ...
    """
    year = None
    month = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if parts[0] == "Year" and len(parts) == 2:
            year = int(parts[1])
            continue
        first = parts[0]
        month_name = first.split("-")[0] if "-" in first else None
        if month_name in _MONTHS and len(parts) >= 2 and parts[1].strip().count(":") == 2:
            month = _MONTHS[month_name]
            continue
        if year is not None and month is not None:
            try:
                day = int(first)
            except ValueError:
                continue
            for hour, val in enumerate(parts[1:25]):
                val = val.strip()
                if val == "" or val.lower() in ("na", "none", "null"):
                    continue
                try:
                    aqi = float(val)
                except ValueError:
                    continue
                try:
                    ts = pd.Timestamp(year=year, month=month, day=day, hour=hour)
                except ValueError:
                    continue  # e.g. Feb 30 padding rows some exports include
                rows.append((ts, aqi))

    df = pd.DataFrame(rows, columns=["datetime", "aqi"]).sort_values("datetime").reset_index(drop=True)
    df["pm25_approx"] = approximate_pm25_from_aqi(df["aqi"].to_numpy())
    return df


def to_hourly_series(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Reindex a parsed (datetime, aqi, pm25_approx) frame to a *continuous* hourly index.

    parse_wide_aqi_csv only emits rows for cells that had a value, so missing
    hours (sensor downtime) are simply absent rather than present-as-NaN.
    Downstream windowing (data/dataset.py) assumes a contiguous hourly array,
    so reindex here and leave real gaps as NaN for the caller to impute or
    drop as appropriate.
    """
    start = start or df["datetime"].min()
    end = end or df["datetime"].max()
    full_index = pd.date_range(start, end, freq="h")
    return df.set_index("datetime").reindex(full_index).rename_axis("datetime").reset_index()


def fetch_station_aqi(station_name_fragment: str, resources: list[dict] | None = None, timeout: int = 120) -> pd.DataFrame:
    """End-to-end: discover the station's resource URL, download, and parse. No API key required."""
    url = find_resource_url(station_name_fragment, resources=resources)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return parse_wide_aqi_csv(resp.content.decode("utf-8", errors="replace"))


# Maps graph/stations.py::LOCAL_STATIONS names to their opencity.in resource-name fragment,
# for the handful that don't match verbatim (e.g. a hyphen difference).
STATION_NAME_OVERRIDES = {
    "Okhla Phase 2": "Okhla Phase-2",
}


def fetch_all_local_stations(timeout: int = 120) -> dict[str, pd.DataFrame]:
    """Fetch hourly AQI (+ approximate PM2.5) for every station in graph/stations.py::LOCAL_STATIONS."""
    from aqf.graph.stations import LOCAL_STATIONS

    resources = list_resources()
    out = {}
    for station in LOCAL_STATIONS:
        frag = STATION_NAME_OVERRIDES.get(station.name, station.name)
        out[station.id] = fetch_station_aqi(frag, resources=resources, timeout=timeout)
    return out
