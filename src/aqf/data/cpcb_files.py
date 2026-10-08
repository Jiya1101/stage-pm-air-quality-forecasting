"""Load hourly station files downloaded by hand from the CPCB Central Control Room portal.

Portal: https://airquality.cpcb.gov.in/ccr/#/repository/data  (frequency 1H, state Delhi). The download sits behind a
CAPTCHA, so the files are fetched manually and this module only reads them.

File name:  Raw_Data_<year>_site_<site id>_<name>_delhi_<dpcc|cpcb>_1Hr.csv  -- one file per station per year.

Time: the `Timestamp` column is Indian Standard Time and labels the *start* of the hour. The hour [t, t+1h) IST has its
centre at t + 30 min IST = t - 5 h UTC exactly, so UTC hour = label - 5 h. Verified against OpenAQ's copy of the same
stations (correlation 0.996-0.997 at zero offset over ~6,500 overlapping hours at three stations).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from aqf.graph.stations import LOCAL_STATIONS

# graph/stations.py station id -> CPCB site id (the number in the file name)
SITE_IDS = {
    "DL001": "301",    # Anand Vihar
    "DL002": "124",    # R K Puram
    "DL003": "125",    # Punjabi Bagh
    "DL004": "1422",   # Dwarka Sector 8
    "DL005": "117",    # ITO
    "DL006": "122",    # Mandir Marg
    "DL007": "1428",   # Okhla Phase 2
    "DL008": "1430",   # Rohini
    "DL009": "113",    # Shadipur
    "DL010": "1434",   # Wazirpur
    "DL011": "1427",   # Najafgarh
    "DL012": "1426",   # Narela
    "DL013": "1432",   # Sonia Vihar
    "DL014": "1435",   # Vivek Vihar
    "DL015": "1420",   # Ashok Vihar
}

# LOCAL_FEATURE_COLS index -> (CSV column, plausible range). Values outside the range are sensor glitches -> NaN.
COLUMNS = {
    0: ("PM2.5 (µg/m³)", (0.0, 1000.0)),
    1: ("PM10 (µg/m³)", (0.0, 1000.0)),
    2: ("NO2 (µg/m³)", (0.0, 500.0)),
    3: ("CO (mg/m³)", (0.0, 50.0)),
    4: ("Ozone (µg/m³)", (0.0, 500.0)),
    5: ("AT (°C)", (-5.0, 50.0)),
    6: ("RH (%)", (0.0, 100.0)),
}
WS_COL, WD_COL = "WS (m/s)", "WD (deg)"


def load_station(cpcb_dir: str, site_id: str, first_year: int, last_year: int) -> pd.DataFrame:
    """Hourly frame indexed by UTC hour (naive), with the raw CSV columns."""
    frames = []
    for year in range(first_year, last_year + 1):
        for path in glob.glob(os.path.join(cpcb_dir, f"Raw_Data_{year}_site_{site_id}_*_1Hr.csv")):
            df = pd.read_csv(path)
            df.index = pd.to_datetime(df["Timestamp"]) - pd.Timedelta(hours=5)
            frames.append(df.drop(columns=["Timestamp"]))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out.groupby(level=0).mean(numeric_only=True).sort_index()


def build_local_array(cpcb_dir: str, index: pd.DatetimeIndex, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """(T, N_local, n_features) values and a same-shape bool mask of genuinely measured entries.

    Wind (u, v) from speed and direction, same convention as the OpenAQ path. traffic_index (col 9) has no sensor
    source; it stays a diurnal placeholder and is never marked observed.
    """
    naive = index.tz_localize(None) if index.tz is not None else index
    T = len(naive)
    local = np.full((T, len(LOCAL_STATIONS), n_features), np.nan, dtype=np.float32)
    observed = np.zeros_like(local, dtype=bool)
    hour = naive.hour.to_numpy()

    for i, station in enumerate(LOCAL_STATIONS):
        site = SITE_IDS[station.id]
        df = load_station(cpcb_dir, site, naive[0].year - 1, naive[-1].year + 1)
        if df.empty:
            continue
        df = df.reindex(naive)
        for col_idx, (name, (lo, hi)) in COLUMNS.items():
            v = df[name].to_numpy(dtype=float) if name in df.columns else np.full(T, np.nan)
            v = np.where((v >= lo) & (v <= hi), v, np.nan)
            local[:, i, col_idx] = v
            observed[:, i, col_idx] = ~np.isnan(v)
        if WS_COL in df.columns and WD_COL in df.columns:
            speed = df[WS_COL].to_numpy(dtype=float)
            ang = np.radians(df[WD_COL].to_numpy(dtype=float))
            speed = np.where((speed >= 0) & (speed <= 30), speed, np.nan)
            local[:, i, 7] = -speed * np.sin(ang)
            local[:, i, 8] = -speed * np.cos(ang)
            observed[:, i, 7] = observed[:, i, 8] = ~np.isnan(local[:, i, 7]) & ~np.isnan(local[:, i, 8])
        local[:, i, 9] = 50 + 40 * np.clip(np.sin(2 * np.pi * (hour - 9) / 24), 0, None)  # traffic placeholder
    return local, observed
