"""OpenAQ v3 API -- authoritative multi-pollutant concentrations for Delhi CAAQMS stations.

This is the primary real-data source for LOCAL_FEATURE_COLS (pm25, pm10, no2,
co, o3 in real ug/m3 units, plus RH/wind/temp for many stations), unlike
data/opencity.py which only has composite AQI. Verified while building this:
CPCB's DTU New Delhi station reports PM2.5, PM10, NOx, O3, CO, SO2, RH, temp
and wind speed/direction on OpenAQ, with data "reporting since 09/03/2018" --
directly covering this project's default 2018-2023 temporal split.

Requires a free API key: register at https://explore.openaq.org/register,
then set OPENAQ_API_KEY (or pass api_key=).

Rate limits apply (see https://docs.openaq.org/using-the-api/rate-limits) --
fetch_station_history paginates responsibly with a small delay between pages.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

BASE_URL = "https://api.openaq.org/v3"

# NCR_BBOX = (west, south, east, north), a superset of graph/stations.py::LOCAL_STATIONS
# and generous enough to also catch nearby DPCC/IMD/IITM-operated stations OpenAQ ingests.
NCR_BBOX = (76.75, 28.35, 77.55, 28.95)

TARGET_PARAMETERS = {"pm25", "pm10", "no2", "co", "o3", "relativehumidity", "temperature", "wind_speed", "wind_direction"}


class OpenAQClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAQ API key found. Register free at https://explore.openaq.org/register "
                "and set OPENAQ_API_KEY, or pass api_key=... explicitly."
            )
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self.api_key})

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self._session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def find_locations(self, bbox: tuple[float, float, float, float] = NCR_BBOX, limit: int = 100) -> list[dict]:
        """List monitoring locations within `bbox` (west, south, east, north)."""
        params = {"bbox": ",".join(str(v) for v in bbox), "limit": limit}
        return self._get("/locations", params).get("results", [])

    def get_sensors(self, location_id: int) -> list[dict]:
        """List sensors (one per pollutant/met-variable) at a location, with their units and coverage dates."""
        return self._get(f"/locations/{location_id}/sensors").get("results", [])

    def get_hourly_measurements(
        self, sensor_id: int, date_from: str, date_to: str, limit: int = 1000, sleep_s: float = 0.25
    ) -> pd.DataFrame:
        """Paginate through hourly-aggregated measurements for one sensor over [date_from, date_to] (YYYY-MM-DD)."""
        rows, page = [], 1
        while True:
            payload = self._get(
                f"/sensors/{sensor_id}/measurements_hourly",
                {"date_from": date_from, "date_to": date_to, "limit": limit, "page": page},
            )
            results = payload.get("results", [])
            if not results:
                break
            for r in results:
                rows.append({
                    "datetime": r["period"]["datetimeFrom"]["utc"],
                    "value": r["value"],
                    "parameter": r.get("parameter", {}).get("name"),
                    "unit": r.get("parameter", {}).get("units"),
                })
            found = payload.get("meta", {}).get("found", 0)
            if isinstance(found, int) and page * limit >= found:
                break
            if len(results) < limit:
                break
            page += 1
            time.sleep(sleep_s)
        df = pd.DataFrame(rows)
        if not df.empty:
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df

    def fetch_station_history(self, location_id: int, date_from: str, date_to: str) -> pd.DataFrame:
        """All target parameters for one location over a date range, wide format (one column per parameter)."""
        sensors = self.get_sensors(location_id)
        frames = []
        for sensor in sensors:
            param_name = sensor.get("parameter", {}).get("name")
            if param_name not in TARGET_PARAMETERS:
                continue
            df = self.get_hourly_measurements(sensor["id"], date_from, date_to)
            if df.empty:
                continue
            df = df.rename(columns={"value": param_name})[["datetime", param_name]]
            frames.append(df.set_index("datetime"))
        if not frames:
            return pd.DataFrame()
        wide = pd.concat(frames, axis=1)
        return wide.sort_index().reset_index()


def match_local_stations_to_openaq(client: OpenAQClient) -> dict[str, dict]:
    """Best-effort name match between graph/stations.py::LOCAL_STATIONS and OpenAQ location IDs.

    Returns {station_id: openaq_location_dict}. Matches are by substring on
    station name -- verify manually before relying on this for a full
    pipeline run, since OpenAQ station naming doesn't always exactly mirror
    CPCB's station names.
    """
    from aqf.graph.stations import LOCAL_STATIONS

    locations = client.find_locations()
    matches = {}
    for station in LOCAL_STATIONS:
        name_l = station.name.lower()
        candidates = [loc for loc in locations if name_l in loc.get("name", "").lower() or loc.get("name", "").lower() in name_l]
        if candidates:
            matches[station.id] = candidates[0]
    return matches
