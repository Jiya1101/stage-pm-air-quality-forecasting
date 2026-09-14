"""ERA5 reanalysis via the Copernicus Climate Data Store (CDS) API.

Requires the `cdsapi` package (`pip install cdsapi`) and a CDS API key. Set up
per https://cds.climate.copernicus.eu/how-to-api: create `~/.cdsapirc`
containing:

    url: https://cds.climate.copernicus.eu/api
    key: <your-uid>:<your-api-key>

Variables pulled here map directly onto ATMOS_FEATURE_COLS /
StabilityIndex's five inputs (src/aqf/features/stability.py):
  boundary_layer_height        -> blh_m
  (derived from temp profile)  -> inversion_strength_k
  2m_dewpoint + 2m_temperature -> rh_pct (via Magnus formula)
  10m u/v wind components      -> wind_speed_ms, and per-node wind vectors
                                   used directly by the transport graph
  surface_solar_radiation_downwards -> solar_rad_wm2
  surface_pressure, total_precipitation -> pressure_hpa, precip_mm
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ERA5_VARIABLES = [
    "boundary_layer_height",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]


def download_era5(
    output_nc_path: str,
    area_box: tuple[float, float, float, float],  # (north, west, south, east)
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> str:
    """Submit a CDS retrieval request for hourly ERA5 single-level fields over `area_box`.

    Blocks until the CDS queue processes the request and writes a NetCDF file
    to `output_nc_path`. Requires `cdsapi` to be installed and configured.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise ImportError(
            "cdsapi is not installed. Run `pip install cdsapi` and configure "
            "~/.cdsapirc per https://cds.climate.copernicus.eu/how-to-api"
        ) from e

    dates = pd.date_range(start_date, end_date, freq="D")
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": variables or ERA5_VARIABLES,
            "year": sorted(set(dates.strftime("%Y"))),
            "month": sorted(set(dates.strftime("%m"))),
            "day": sorted(set(dates.strftime("%d"))),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": list(area_box),
        },
        output_nc_path,
    )
    return output_nc_path


def relative_humidity_from_dewpoint(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    """Magnus-formula RH% from 2m temperature and 2m dewpoint (both in Kelvin)."""
    t_c = temp_k - 273.15
    td_c = dewpoint_k - 273.15
    a, b = 17.625, 243.04
    e_t = np.exp((a * t_c) / (b + t_c))
    e_td = np.exp((a * td_c) / (b + td_c))
    return np.clip(100.0 * e_td / e_t, 0.0, 100.0)


def load_era5_nc(path: str, station_lat: float, station_lon: float) -> pd.DataFrame:
    """Nearest-neighbor extract a single station's hourly time series from a downloaded ERA5 NetCDF file.

    Requires `xarray` (`pip install xarray netCDF4`) -- imported lazily so the
    rest of the package has no hard dependency on it.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray + netCDF4 are required to read ERA5 output: pip install xarray netCDF4") from e

    ds = xr.open_dataset(path)
    point = ds.sel(latitude=station_lat, longitude=station_lon, method="nearest")
    df = point.to_dataframe().reset_index()

    out = pd.DataFrame({"datetime": df["time"] if "time" in df else df.get("valid_time")})
    if "blh" in df:
        out["blh_m"] = df["blh"]
    if "u10" in df and "v10" in df:
        out["wind_u_ms"] = df["u10"]
        out["wind_v_ms"] = df["v10"]
        out["wind_speed_ms"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    if "t2m" in df and "d2m" in df:
        out["rh_pct"] = relative_humidity_from_dewpoint(df["t2m"].to_numpy(), df["d2m"].to_numpy())
        out["temp_c"] = df["t2m"] - 273.15
    if "sp" in df:
        out["pressure_hpa"] = df["sp"] / 100.0
    if "tp" in df:
        out["precip_mm"] = df["tp"] * 1000.0
    if "ssrd" in df:
        out["solar_rad_wm2"] = df["ssrd"] / 3600.0  # J/m^2 accumulated hourly -> W/m^2
    return out


def inversion_strength_from_profile(temp_2m_k: np.ndarray, temp_925hpa_k: np.ndarray) -> np.ndarray:
    """Vertical temperature inversion strength: T(925hPa) - T(2m), positive = inversion (warmer aloft).

    Requires pulling `temperature` on pressure level 925hPa from
    `reanalysis-era5-pressure-levels` in addition to the single-levels
    request above (add via a second `download_era5`-style call with dataset
    name changed and `"pressure_level": ["925"]`, `"variable": ["temperature"]`).
    """
    return temp_925hpa_k - temp_2m_k
