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


def open_era5_dataset(path: str):
    """Open a downloaded ERA5 file, transparently handling both plain NetCDF and CDS's zip-wrapped output.

    Verified live: a multi-variable reanalysis-era5-single-levels request
    (e.g. BLH + wind + temp + dewpoint + pressure + precip + solar) comes
    back as a ZIP archive -- even when saved with a ".nc" extension and
    "format": "netcdf" requested -- containing two files, one for
    instantaneous fields (data_stream-oper_stepType-instant.nc) and one for
    accumulated fields like precip/solar radiation
    (data_stream-oper_stepType-accum.nc). A single-variable request (e.g.
    just 925hPa temperature) came back as plain NetCDF, no zip. This merges
    however many .nc members are present into one dataset either way.
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError("xarray + netCDF4 are required to read ERA5 output: pip install xarray netCDF4") from e

    import zipfile

    if zipfile.is_zipfile(path):
        import tempfile

        extract_dir = tempfile.mkdtemp(prefix="era5_")
        with zipfile.ZipFile(path) as z:
            nc_members = [n for n in z.namelist() if n.endswith(".nc")]
            z.extractall(extract_dir, members=nc_members)
        import os

        datasets = [xr.open_dataset(os.path.join(extract_dir, m)) for m in nc_members]
        return xr.merge(datasets, compat="override", join="outer")

    return xr.open_dataset(path)


def load_era5_nc(path: str, station_lat: float, station_lon: float) -> pd.DataFrame:
    """Nearest-neighbor extract a single station's hourly time series from a downloaded ERA5 NetCDF file.

    Requires `xarray` (`pip install xarray netCDF4`) -- imported lazily so the
    rest of the package has no hard dependency on it.
    """
    ds = open_era5_dataset(path)
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
    request above -- see `download_era5_pressure_level` below.
    """
    return temp_925hpa_k - temp_2m_k


def download_era5_pressure_level(
    output_nc_path: str,
    area_box: tuple[float, float, float, float],  # (north, west, south, east)
    start_date: str,
    end_date: str,
    pressure_level: str = "925",
) -> str:
    """Pull 925hPa temperature -- the second dataset needed for inversion_strength_from_profile.

    Same request shape as download_era5, verified live against the same CDS
    account/license (reanalysis-era5-pressure-levels requires accepting its
    own license separately at the CDS dataset page, same one-time step as
    reanalysis-era5-single-levels).
    """
    try:
        import cdsapi
    except ImportError as e:
        raise ImportError("cdsapi is not installed. Run `pip install cdsapi`.") from e

    dates = pd.date_range(start_date, end_date, freq="D")
    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": ["temperature"],
            "pressure_level": [pressure_level],
            "year": sorted(set(dates.strftime("%Y"))),
            "month": sorted(set(dates.strftime("%m"))),
            "day": sorted(set(dates.strftime("%d"))),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": list(area_box),
        },
        output_nc_path,
    )
    return output_nc_path


def build_domain_atmos_series(single_level_nc: str, pressure_level_nc: str | None = None) -> pd.DataFrame:
    """Spatially-averaged hourly atmospheric series over the whole downloaded domain.

    This is the real replacement for real_pipeline.py's `_placeholder_atmos`
    -- one global vector per hour (BLH, inversion, RH, wind speed, solar,
    pressure, precip) matching ATMOS_FEATURE_COLS, averaged across the NCR
    bounding box rather than tied to a single station (matches how the
    synthetic generator and the model's G_A atmospheric-state layer treat
    it: a domain-wide conditioning signal, not a per-station one).
    """
    ds = open_era5_dataset(single_level_nc)
    spatial_dims = [d for d in ("latitude", "longitude") if d in ds.dims]
    mean_ds = ds.mean(dim=spatial_dims)
    df = mean_ds.to_dataframe().reset_index()
    time_col = "valid_time" if "valid_time" in df.columns else "time"

    out = pd.DataFrame({"datetime": pd.to_datetime(df[time_col])})
    out["blh_m"] = df.get("blh", np.nan)
    if "u10" in df.columns and "v10" in df.columns:
        out["wind_speed_ms"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    if "t2m" in df.columns and "d2m" in df.columns:
        out["rh_pct"] = relative_humidity_from_dewpoint(df["t2m"].to_numpy(), df["d2m"].to_numpy())
        out["_t2m_k"] = df["t2m"]
    out["pressure_hpa"] = df.get("sp", np.nan) / 100.0 if "sp" in df.columns else np.nan
    out["precip_mm"] = df.get("tp", np.nan) * 1000.0 if "tp" in df.columns else np.nan
    out["solar_rad_wm2"] = df.get("ssrd", np.nan) / 3600.0 if "ssrd" in df.columns else np.nan

    if pressure_level_nc is not None and "_t2m_k" in out.columns:
        pds = open_era5_dataset(pressure_level_nc)
        p_spatial_dims = [d for d in ("latitude", "longitude") if d in pds.dims]
        pmean = pds.mean(dim=p_spatial_dims).to_dataframe().reset_index()
        p_time_col = "valid_time" if "valid_time" in pmean.columns else "time"
        pmean = pmean.rename(columns={p_time_col: "datetime", "t": "_t925_k"})
        pmean["datetime"] = pd.to_datetime(pmean["datetime"])
        out = out.merge(pmean[["datetime", "_t925_k"]], on="datetime", how="left")
        out["inversion_strength_k"] = inversion_strength_from_profile(out["_t2m_k"].to_numpy(), out["_t925_k"].to_numpy())
        out = out.drop(columns=["_t2m_k", "_t925_k"])
    else:
        out["inversion_strength_k"] = 0.0
        out = out.drop(columns=[c for c in ("_t2m_k",) if c in out.columns])

    return out


def extract_point_wind_series(nc_path: str, lat: float, lon: float) -> pd.DataFrame:
    """Nearest-neighbor 10m wind time series at one point (used for REGIONAL_SOURCES wind, real replacement
    for real_pipeline.py's constant wind placeholder)."""
    ds = open_era5_dataset(nc_path)
    point = ds.sel(latitude=lat, longitude=lon, method="nearest")
    df = point.to_dataframe().reset_index()
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    out = pd.DataFrame({"datetime": pd.to_datetime(df[time_col])})
    out["wind_u_ms"] = df.get("u10", 0.0)
    out["wind_v_ms"] = df.get("v10", 0.0)
    return out
