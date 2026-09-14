"""CPCB / CAAQMS ground station data via the data.gov.in open API.

Requires a free API key from https://data.gov.in (Sign Up -> My Account ->
API Keys), passed as CPCB_API_KEY or `api_key=`.

The live real-time AQI resource id below (`RESOURCE_ID`) is data.gov.in's
public "Real Time Air Quality Index" dataset, which covers CPCB CAAQMS
stations nationwide including Delhi-NCR. If data.gov.in rotates the resource
id, update the constant here -- nothing else in the pipeline needs to change,
since downstream code only depends on `fetch_station_hourly` returning a
DataFrame with the columns declared in `CPCB_RETURN_COLS`.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.data.gov.in/resource"
# data.gov.in "Real Time Air Quality Index from Various Locations" dataset.
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"

CPCB_RETURN_COLS = ["station", "city", "datetime", "pollutant_id", "pollutant_avg", "latitude", "longitude"]


class CPCBClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("CPCB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No CPCB API key found. Register at https://data.gov.in and set "
                "CPCB_API_KEY, or pass api_key=... explicitly."
            )
        self.timeout = timeout

    def fetch_raw(self, city: str = "Delhi", limit: int = 1000, offset: int = 0) -> pd.DataFrame:
        """One page of the live real-time AQI feed, filtered by city."""
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": offset,
            "filters[city]": city,
        }
        resp = requests.get(f"{BASE_URL}/{RESOURCE_ID}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        return pd.DataFrame.from_records(records)

    def fetch_all(self, city: str = "Delhi", page_size: int = 1000, max_pages: int = 50, sleep_s: float = 0.2) -> pd.DataFrame:
        """Paginate through all currently-available records for a city."""
        frames = []
        for page in range(max_pages):
            df = self.fetch_raw(city=city, limit=page_size, offset=page * page_size)
            if df.empty:
                break
            frames.append(df)
            time.sleep(sleep_s)
        if not frames:
            return pd.DataFrame(columns=CPCB_RETURN_COLS)
        return pd.concat(frames, ignore_index=True)


def fetch_station_hourly(city: str = "Delhi", api_key: str | None = None) -> pd.DataFrame:
    """Convenience entry point: raw pollutant readings for a city, long format.

    NOTE: the live data.gov.in feed is a real-time snapshot API (current
    readings per station), not a historical hourly archive. For the multi-year
    temporal-holdout training design in config.py (2018-2021 train / 2022 val
    / 2023 test), point-in-time snapshots must be collected into your own
    archive over time (e.g. via a scheduled job calling this function hourly),
    or sourced from CPCB's historical data portal (https://cpcb.nic.in /
    https://airquality.cpcb.gov.in) which requires a manual bulk-download
    request. This function is the live-ingestion half of that pipeline.
    """
    client = CPCBClient(api_key=api_key)
    raw = client.fetch_all(city=city)
    if raw.empty:
        return raw
    raw = raw.rename(columns={"last_update": "datetime"})
    keep = [c for c in CPCB_RETURN_COLS if c in raw.columns]
    return raw[keep]


def pivot_pollutants_wide(raw: pd.DataFrame) -> pd.DataFrame:
    """Long (station, pollutant_id, pollutant_avg) -> wide (one column per pollutant)."""
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["pollutant_avg"] = pd.to_numeric(raw["pollutant_avg"], errors="coerce")
    wide = raw.pivot_table(
        index=["station", "city", "datetime", "latitude", "longitude"],
        columns="pollutant_id",
        values="pollutant_avg",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide
