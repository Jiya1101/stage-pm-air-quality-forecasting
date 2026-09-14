"""NASA FIRMS active-fire detections (VIIRS/MODIS) via the FIRMS area API.

Requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/
(Area API section). Set FIRMS_MAP_KEY or pass map_key=.

Output feeds directly into features/fire_proxy.py::fire_emission_proxy --
this module only handles retrieval, never converts raw detections into an
"emission" value itself (see fire_proxy.py docstring for why that
distinction matters to reviewers).
"""
from __future__ import annotations

import io
import os

import pandas as pd
import requests

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# VIIRS S-NPP/NOAA-20/NOAA-21 near-real-time (NRT) products, ~375m nominal resolution.
DEFAULT_SOURCE = "VIIRS_SNPP_NRT"


class FIRMSClient:
    def __init__(self, map_key: str | None = None, timeout: int = 30):
        self.map_key = map_key or os.environ.get("FIRMS_MAP_KEY")
        if not self.map_key:
            raise ValueError(
                "No FIRMS MAP_KEY found. Register at "
                "https://firms.modaps.eosdis.nasa.gov/api/ and set FIRMS_MAP_KEY, "
                "or pass map_key=... explicitly."
            )
        self.timeout = timeout

    def fetch_area(
        self,
        bbox: tuple[float, float, float, float],  # (west, south, east, north)
        day_range: int = 1,
        date: str | None = None,
        source: str = DEFAULT_SOURCE,
    ) -> pd.DataFrame:
        """Fetch fire detections within `bbox` for `day_range` days ending at `date` (YYYY-MM-DD, default today).

        Returns raw FIRMS columns: latitude, longitude, brightness/bright_ti4,
        scan, track, acq_date, acq_time, satellite, confidence, version,
        frp, daynight.
        """
        west, south, east, north = bbox
        parts = [FIRMS_BASE_URL, self.map_key, source, f"{west},{south},{east},{north}", str(day_range)]
        if date:
            parts.append(date)
        url = "/".join(parts)
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        if not resp.text.strip() or resp.text.lower().startswith("invalid"):
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(resp.text))


# Bounding box loosely covering Punjab, Haryana, Rajasthan (NW), western UP,
# and Delhi-NCR -- the domain the STAGE-PM regional source nodes represent.
NCR_TRANSPORT_DOMAIN_BBOX = (73.0, 24.0, 79.5, 32.5)  # (west, south, east, north)


def fetch_regional_fires(day_range: int = 10, date: str | None = None, map_key: str | None = None) -> pd.DataFrame:
    client = FIRMSClient(map_key=map_key)
    df = client.fetch_area(NCR_TRANSPORT_DOMAIN_BBOX, day_range=day_range, date=date)
    if df.empty:
        return df
    df["acq_datetime"] = pd.to_datetime(df["acq_date"]) + pd.to_timedelta(
        df["acq_time"].astype(str).str.zfill(4).str[:2].astype(int), unit="h"
    )
    frp_col = "frp" if "frp" in df.columns else "bright_ti4"
    df = df.rename(columns={frp_col: "frp_mw"})
    return df


def assign_region(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket raw detections into the 4 STAGE-PM regional source nodes by nearest centroid.

    Import is local to avoid a hard dependency from this module on the graph
    package during simple retrieval-only usage.
    """
    from aqf.features.transport import haversine_km
    from aqf.graph.stations import REGIONAL_SOURCES

    df = df.copy()
    dists = {
        src.id: haversine_km(df["latitude"].to_numpy(), df["longitude"].to_numpy(), src.lat, src.lon)
        for src in REGIONAL_SOURCES
    }
    dist_df = pd.DataFrame(dists)
    df["region_id"] = dist_df.idxmin(axis=1).to_numpy()
    return df
