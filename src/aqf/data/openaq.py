"""OpenAQ v3 API -- authoritative multi-pollutant concentrations for Delhi CAAQMS stations.

This is the primary real-data source for LOCAL_FEATURE_COLS (pm25, pm10, no2,
co, o3 in real ug/m3 units, plus RH/wind/temp for many stations), unlike
data/opencity.py which only has composite AQI. Verified while building this:
CPCB's DTU New Delhi station reports PM2.5, PM10, NOx, O3, CO, SO2, RH, temp
and wind speed/direction on OpenAQ, with data "reporting since 09/03/2018" --
directly covering this project's default 2018-2023 temporal split.

Requires a free API key: register at https://explore.openaq.org/register,
then set OPENAQ_API_KEY (or pass api_key=).

Rate limits: free tier is 60 requests/minute, 2000/hour (verified against
docs.openaq.org/using-the-api/rate-limits). Fetching all 15 LOCAL_STATIONS
(~10 sensors each) is ~150 requests -- comfortably over the per-minute limit
in a tight loop, so `_get` throttles proactively using the
`x-ratelimit-remaining`/`x-ratelimit-reset` response headers and retries
with backoff on a 429 (confirmed hit during testing without this).
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
    def __init__(self, api_key: str | None = None, timeout: int = 30, min_interval_s: float = 1.1):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAQ API key found. Register free at https://explore.openaq.org/register "
                "and set OPENAQ_API_KEY, or pass api_key=... explicitly."
            )
        self.timeout = timeout
        self.min_interval_s = min_interval_s  # >1s keeps us under the 60/min free-tier ceiling
        self._session = requests.Session()
        self._session.headers.update({"X-API-Key": self.api_key})
        self._last_request_t = 0.0

    def _get(self, path: str, params: dict | None = None, max_retries: int = 5) -> dict:
        for attempt in range(max_retries):
            elapsed = time.time() - self._last_request_t
            if elapsed < self.min_interval_s:
                time.sleep(self.min_interval_s - elapsed)

            try:
                resp = self._session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
            except requests.exceptions.RequestException as e:
                # Transient network blip -- observed live, twice, during a multi-minute, many-request
                # fetch: once as ConnectionError on the initial connect, once as ChunkedEncodingError
                # while requests was reading the response body (both wrap the same underlying
                # ConnectionResetError, but requests surfaces them as different exception subclasses,
                # so catch RequestException broadly rather than trying to enumerate every variant).
                self._last_request_t = time.time()
                if attempt == max_retries - 1:
                    raise
                time.sleep(min((2 ** attempt) * 2, 30.0))
                continue

            self._last_request_t = time.time()

            if resp.status_code == 429:
                reset_s = resp.headers.get("x-ratelimit-reset")
                wait = float(reset_s) if reset_s and reset_s.replace(".", "", 1).isdigit() else (2 ** attempt) * 2
                wait = min(wait, 60.0) + 0.5
                time.sleep(wait)
                continue

            remaining = resp.headers.get("x-ratelimit-remaining")
            if remaining is not None and remaining.isdigit() and int(remaining) <= 1:
                reset_s = resp.headers.get("x-ratelimit-reset")
                if reset_s and reset_s.replace(".", "", 1).isdigit():
                    time.sleep(float(reset_s) + 0.5)

            resp.raise_for_status()
            return resp.json()

        raise RuntimeError(f"OpenAQ API: gave up after {max_retries} retries (429 / connection errors) for {path}")

    def find_locations(self, bbox: tuple[float, float, float, float] = NCR_BBOX, limit: int = 100) -> list[dict]:
        """List ALL monitoring locations within `bbox` (west, south, east, north), paginated.

        Verified live: the NCR bbox alone returns 100+ locations, so a
        single unpaginated page (the previous behavior here) silently missed
        some -- including, in one observed case, the currently-active
        duplicate of a station whose other entry had gone stale in 2018.
        """
        results, page = [], 1
        while True:
            batch = self._get("/locations", {"bbox": ",".join(str(v) for v in bbox), "limit": limit, "page": page}).get("results", [])
            if not batch:
                break
            results.extend(batch)
            if len(batch) < limit:
                break
            page += 1
        return results

    def get_sensors(self, location_id: int) -> list[dict]:
        """List sensors (one per pollutant/met-variable) at a location, with their units and coverage dates."""
        return self._get(f"/locations/{location_id}/sensors").get("results", [])

    def get_hourly_measurements(
        self, sensor_id: int, date_from: str, date_to: str, limit: int = 1000, sleep_s: float = 0.25
    ) -> pd.DataFrame:
        """Paginate through hourly-aggregated measurements for one sensor over [date_from, date_to] (YYYY-MM-DD).

        NOTE: verified live against the OpenAPI spec -- this endpoint's date
        filter params are `datetime_from`/`datetime_to`, NOT `date_from`/
        `date_to` (which are silently ignored, so an unfiltered call pulls
        the sensor's *entire* history -- caught this during testing when a
        10-day request took minutes instead of seconds).
        """
        rows, page = [], 1
        while True:
            payload = self._get(
                f"/sensors/{sensor_id}/measurements/hourly",
                {"datetime_from": date_from, "datetime_to": date_to, "limit": limit, "page": page},
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
        """All target parameters for one location over a date range, wide format (one column per parameter).

        Some locations expose more than one sensor for the same parameter
        (verified live: duplicate/re-registered instrument entries, sometimes
        in different units -- e.g. co/no2 in ppb on one sensor, ug/m3 on
        another). This takes the first sensor encountered per parameter and
        skips the rest, to avoid duplicate-named columns; it does not attempt
        unit harmonization across sensors -- check `units_used` if precise
        units matter downstream.
        """
        sensors = self.get_sensors(location_id)
        frames = []
        seen_params: set[str] = set()
        units_used: dict[str, str] = {}
        for sensor in sensors:
            param_name = sensor.get("parameter", {}).get("name")
            if param_name not in TARGET_PARAMETERS or param_name in seen_params:
                continue
            df = self.get_hourly_measurements(sensor["id"], date_from, date_to)
            if df.empty:
                continue
            seen_params.add(param_name)
            units_used[param_name] = sensor.get("parameter", {}).get("units", "")
            df = df.rename(columns={"value": param_name})[["datetime", param_name]]
            frames.append(df.set_index("datetime"))
        if not frames:
            result = pd.DataFrame()
        else:
            wide = pd.concat(frames, axis=1)
            result = wide.sort_index().reset_index()
        result.attrs["units_used"] = units_used
        return result


# Verified live against the OpenAQ v3 API: a handful of graph/stations.py names don't match
# verbatim (a hyphen difference, same issue as data/opencity.py's STATION_NAME_OVERRIDES).
STATION_NAME_OVERRIDES = {
    "Dwarka Sector 8": "Dwarka-Sector 8",
    "Okhla Phase 2": "Okhla Phase-2",
}


def _datetime_last(loc: dict) -> str:
    return (loc.get("datetimeLast") or {}).get("utc") or ""


def _has_pm25(loc: dict) -> bool:
    return any((s.get("parameter") or {}).get("name") == "pm25" for s in (loc.get("sensors") or []))


def match_local_stations_to_openaq(client: OpenAQClient) -> dict[str, dict]:
    """Best-effort name match between graph/stations.py::LOCAL_STATIONS and OpenAQ location IDs.

    Returns {station_id: openaq_location_dict}. Matches are by substring on
    station name. IMPORTANT (found the hard way): several stations have more
    than one OpenAQ location entry for the same physical site -- an old one
    that stopped reporting in 2018, and a currently-active re-registration.
    Picking "the first candidate" silently grabbed the dead one for ~9 of 15
    stations in testing (0% coverage in any recent window, despite matching
    successfully). This instead prefers, among name matches, whichever
    candidate has a pm25 sensor and the most recent `datetimeLast` -- i.e.
    the one that's actually still reporting.
    """
    from aqf.graph.stations import LOCAL_STATIONS

    locations = client.find_locations()
    matches = {}
    for station in LOCAL_STATIONS:
        name_l = STATION_NAME_OVERRIDES.get(station.name, station.name).lower()
        candidates = [loc for loc in locations if name_l in loc.get("name", "").lower() or loc.get("name", "").lower() in name_l]
        if candidates:
            candidates.sort(key=lambda loc: (_has_pm25(loc), _datetime_last(loc)), reverse=True)
            matches[station.id] = candidates[0]
    return matches
