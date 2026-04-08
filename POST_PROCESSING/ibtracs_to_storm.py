"""
ibtracs_to_storm.py  —  Convert IBTrACS to STORM 13-column .txt format
========================================================================

Reads IBTrACS (NetCDF or CSV) and writes STORM-format files that can
be fed directly into evaluation.py.

Outputs per basin:
    STORM_DATA_IBTRACS_{basin}_1000_YEARS_0.txt          (all phases)
    STORM_DATA_IBTRACS_{basin}_EN_1000_YEARS_0.txt       (El Niño)
    STORM_DATA_IBTRACS_{basin}_NEU_1000_YEARS_0.txt      (Neutral)
    STORM_DATA_IBTRACS_{basin}_LN_1000_YEARS_0.txt       (La Niña)

Usage
-----
    # From NetCDF (preferred):
    python ibtracs_to_storm.py \\
        --input IBTrACS.ALL.v04r01.nc \\
        --format netcdf \\
        --land_mask_dir /path/to/repo/ \\
        --outdir ibtracs_storm_format/ \\
        --basins NA WP

    # From CSV:
    python ibtracs_to_storm.py \\
        --input ibtracs.ALL.list.v04r01.csv \\
        --format csv \\
        --outdir ibtracs_storm_format/

Download IBTrACS from:
    NetCDF: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/IBTrACS.ALL.v04r01.nc
    CSV:    https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv

Dependencies: numpy, pandas, (xarray + netCDF4 for NetCDF input)
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

BASIN_ID = {"EP": 0, "NA": 1, "NI": 2, "SI": 3, "SP": 4, "WP": 5}

# IBTrACS basin codes that map to STORM basins
IBTRACS_BASIN_MAP = {
    "EP": "EP",
    "NA": "NA",
    "NI": "NI",
    "SI": "SI",
    "SP": "SP",
    "WP": "WP",
    "AS": "NI",  # Arabian Sea → North Indian
    "BB": "NI",  # Bay of Bengal → North Indian
}

# Basin boundaries (lon in 0–360), matching preprocessing.py
BASIN_BOUNDS = {
    "EP": (5, 60, 180, 285),  # (lat0, lat1, lon0, lon1)
    "NA": (5, 60, 255, 360),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}

# Active-season months per basin
ACTIVE_MONTHS = {
    "EP": {6, 7, 8, 9, 10, 11},
    "NA": {6, 7, 8, 9, 10, 11},
    "NI": {10, 11},
    "SI": {1, 2, 3, 4, 11, 12},
    "SP": {1, 2, 3, 4, 11, 12},
    "WP": {5, 6, 7, 8, 9, 10, 11},
}

# Wind conversion: 1-min knots → 10-min m/s
# Factor = kn_to_ms × wmo_1min_to_10min = 0.5144 × 0.88
KN1MIN_TO_MS10MIN = 0.5144 * 0.88  # ≈ 0.4527
KN_TO_MS = 0.5144  # knots → m/s (no averaging period change)
NM_TO_KM = 1.852  # nautical miles → km

# Saffir-Simpson thresholds in 10-min m/s (matching STORM convention)
SS_THRESHOLDS = [18.0, 33.0, 43.0, 50.0, 58.0, 70.0]

# Minimum wind to keep a storm (10-min m/s) — tropical storm threshold
MIN_WIND_MS = 18.0

# Default RMAX when missing (km)
DEFAULT_RMAX_KM = 50.0


# ═══════════════════════════════════════════════════════════════════════
# ONI TABLE (same as evaluation.py)
# ═══════════════════════════════════════════════════════════════════════


def _default_oni_table() -> pd.DataFrame:
    """Hardcoded ONI 3.4 monthly values 1980–2021."""
    _oni = {
        1980: [0.0, -0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 0.0, -0.1, -0.1, -0.1, -0.1],
        1981: [-0.3, -0.5, -0.5, -0.4, -0.3, -0.3, -0.4, -0.3, -0.2, -0.1, -0.2, -0.1],
        1982: [-0.1, 0.0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 1.9, 2.1, 2.1],
        1983: [2.1, 1.8, 1.5, 1.2, 1.0, 0.6, 0.2, -0.2, -0.5, -0.8, -0.9, -0.8],
        1984: [-0.5, -0.3, -0.3, -0.4, -0.5, -0.4, -0.3, -0.2, -0.3, -0.6, -0.9, -1.1],
        1985: [-0.9, -0.7, -0.7, -0.7, -0.7, -0.6, -0.4, -0.4, -0.4, -0.3, -0.2, -0.3],
        1986: [-0.4, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2, 0.4, 0.7, 0.9, 1.0, 1.1],
        1987: [1.1, 1.2, 1.1, 1.0, 0.9, 1.1, 1.4, 1.6, 1.6, 1.4, 1.2, 1.1],
        1988: [0.8, 0.5, 0.1, -0.3, -0.8, -1.2, -1.2, -1.1, -1.2, -1.6, -1.9, -1.8],
        1989: [-1.6, -1.4, -1.1, -0.8, -0.6, -0.3, -0.3, -0.3, -0.3, -0.3, -0.2, -0.1],
        1990: [0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.3, 0.4, 0.4],
        1991: [0.4, 0.3, 0.2, 0.2, 0.4, 0.6, 0.7, 0.7, 0.7, 0.8, 1.2, 1.4],
        1992: [1.6, 1.5, 1.4, 1.2, 1.0, 0.7, 0.3, 0.0, -0.1, -0.1, 0.0, 0.0],
        1993: [0.2, 0.3, 0.5, 0.7, 0.8, 0.6, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1],
        1994: [0.1, 0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.9, 1.0],
        1995: [0.9, 0.7, 0.5, 0.3, 0.2, 0.0, -0.2, -0.5, -0.7, -0.9, -0.9, -0.9],
        1996: [-0.9, -0.7, -0.6, -0.4, -0.2, -0.2, -0.2, -0.3, -0.3, -0.4, -0.4, -0.5],
        1997: [-0.5, -0.4, -0.1, 0.3, 0.8, 1.2, 1.6, 1.9, 2.1, 2.3, 2.4, 2.3],
        1998: [2.1, 1.8, 1.4, 1.0, 0.5, -0.1, -0.7, -1.0, -1.2, -1.3, -1.4, -1.5],
        1999: [-1.5, -1.3, -1.0, -0.9, -0.9, -1.0, -1.0, -1.0, -1.1, -1.2, -1.4, -1.7],
        2000: [-1.7, -1.4, -1.1, -0.8, -0.7, -0.6, -0.6, -0.5, -0.6, -0.7, -0.7, -0.7],
        2001: [-0.7, -0.5, -0.4, -0.3, -0.2, -0.1, -0.1, 0.0, -0.1, -0.2, -0.3, -0.3],
        2002: [-0.1, 0.0, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.1],
        2003: [0.9, 0.6, 0.4, 0.0, -0.3, -0.2, 0.1, 0.2, 0.3, 0.3, 0.4, 0.3],
        2004: [0.3, 0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7],
        2005: [0.6, 0.6, 0.5, 0.4, 0.3, 0.1, 0.0, -0.1, -0.1, -0.3, -0.6, -0.7],
        2006: [-0.7, -0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.9],
        2007: [0.7, 0.3, -0.1, -0.2, -0.3, -0.3, -0.4, -0.6, -0.8, -1.1, -1.2, -1.3],
        2008: [-1.5, -1.4, -1.2, -0.9, -0.7, -0.5, -0.4, 0.0, 0.0, -0.3, -0.6, -0.7],
        2009: [-0.8, -0.7, -0.5, -0.2, 0.1, 0.4, 0.5, 0.5, 0.7, 1.0, 1.3, 1.6],
        2010: [1.5, 1.3, 0.9, 0.4, -0.1, -0.6, -1.0, -1.4, -1.6, -1.7, -1.7, -1.6],
        2011: [-1.4, -1.1, -0.8, -0.6, -0.3, -0.1, -0.3, -0.5, -0.7, -0.9, -1.0, -0.9],
        2012: [-0.8, -0.6, -0.5, -0.4, -0.2, 0.1, 0.3, 0.3, 0.3, 0.2, 0.0, -0.2],
        2013: [-0.4, -0.3, -0.2, -0.2, -0.3, -0.3, -0.4, -0.3, -0.2, -0.2, -0.2, -0.3],
        2014: [-0.4, -0.4, -0.2, 0.1, 0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.6, 0.7],
        2015: [0.6, 0.6, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.1, 2.4, 2.6, 2.6],
        2016: [2.5, 2.1, 1.7, 1.0, 0.5, 0.0, -0.3, -0.6, -0.7, -0.7, -0.7, -0.6],
        2017: [-0.3, -0.1, 0.1, 0.3, 0.4, 0.4, 0.2, 0.0, -0.4, -0.7, -0.9, -1.0],
        2018: [-0.9, -0.8, -0.6, -0.4, -0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 0.8],
        2019: [0.8, 0.8, 0.8, 0.8, 0.6, 0.5, 0.3, 0.1, 0.1, 0.3, 0.5, 0.5],
        2020: [0.5, 0.5, 0.4, 0.2, -0.1, -0.3, -0.5, -0.6, -0.9, -1.2, -1.3, -1.2],
        2021: [-1.0, -0.8, -0.6, -0.4, -0.2, -0.1, -0.4, -0.5, -0.7, -0.8, -0.9, -0.9],
    }
    rows = []
    for yr, vals in _oni.items():
        for m, v in enumerate(vals, 1):
            rows.append({"year": yr, "month": m, "oni": v})
    df = pd.DataFrame(rows)
    df["phase"] = np.where(
        df["oni"] > 0.5, "EN", np.where(df["oni"] < -0.5, "LN", "NEU")
    )
    return df


def _oni_phase_lookup(oni_df: pd.DataFrame) -> Dict[Tuple[int, int], str]:
    """Build (year, month) → phase lookup."""
    return {(int(r["year"]), int(r["month"])): r["phase"] for _, r in oni_df.iterrows()}


# ═══════════════════════════════════════════════════════════════════════
# LAND MASK UTILITIES
# ═══════════════════════════════════════════════════════════════════════


class LandMask:
    """
    Wraps a STORM-format land–ocean mask (0.1° resolution).

    Attributes
    ----------
    mask : 2D array, 1 = land, 0 = ocean
    lat0, lat1, lon0, lon1 : basin bounds
    step : grid cells per degree (10 → 0.1°)
    """

    def __init__(
        self,
        filepath: str,
        lat0: float,
        lat1: float,
        lon0: float,
        lon1: float,
        step: int = 10,
    ):
        self.mask = np.loadtxt(filepath)
        self.lat0 = lat0
        self.lat1 = lat1
        self.lon0 = lon0
        self.lon1 = lon1
        self.step = step

    def is_land(self, lat: float, lon: float) -> bool:
        """Return True if (lat, lon) is over land."""
        row = int(round((self.lat1 - lat) * self.step))
        col = int(round((lon - self.lon0) * self.step))
        if 0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1]:
            return self.mask[row, col] > 0.5
        return False

    def dist_to_land_approx(
        self, lat: float, lon: float, max_search_deg: float = 5.0
    ) -> float:
        """
        Approximate distance to nearest land cell (km).
        Uses expanding box search on the mask grid.
        Returns 0 if on land.
        """
        if self.is_land(lat, lon):
            return 0.0

        row_c = int(round((self.lat1 - lat) * self.step))
        col_c = int(round((lon - self.lon0) * self.step))
        max_cells = int(max_search_deg * self.step)

        best_dist = np.inf
        for radius in range(1, max_cells + 1):
            found_any = False
            for dr in range(-radius, radius + 1):
                for dc in [-radius, radius]:
                    r, c = row_c + dr, col_c + dc
                    if 0 <= r < self.mask.shape[0] and 0 <= c < self.mask.shape[1]:
                        if self.mask[r, c] > 0.5:
                            dlat = dr / self.step
                            dlon = dc / self.step
                            # Rough km conversion
                            km = np.sqrt(
                                (dlat * 111.0) ** 2
                                + (dlon * 111.0 * np.cos(np.radians(lat))) ** 2
                            )
                            best_dist = min(best_dist, km)
                            found_any = True
            for dc in range(-radius + 1, radius):
                for dr in [-radius, radius]:
                    r, c = row_c + dr, col_c + dc
                    if 0 <= r < self.mask.shape[0] and 0 <= c < self.mask.shape[1]:
                        if self.mask[r, c] > 0.5:
                            dlat = dr / self.step
                            dlon = dc / self.step
                            km = np.sqrt(
                                (dlat * 111.0) ** 2
                                + (dlon * 111.0 * np.cos(np.radians(lat))) ** 2
                            )
                            best_dist = min(best_dist, km)
                            found_any = True
            if found_any:
                break

        return best_dist if best_dist < np.inf else 9999.0


def _load_land_masks(mask_dir: str, basins: List[str]) -> Dict[str, LandMask]:
    """Load land masks for requested basins."""
    masks = {}
    for b in basins:
        fpath = os.path.join(mask_dir, f"Land_ocean_mask_{b}.txt")
        if os.path.exists(fpath):
            lat0, lat1, lon0, lon1 = BASIN_BOUNDS[b]
            masks[b] = LandMask(fpath, lat0, lat1, lon0, lon1)
        else:
            warnings.warn(
                f"Land mask not found: {fpath}. Using IBTrACS dist2land instead."
            )
    return masks


# ═══════════════════════════════════════════════════════════════════════
# SAFFIR–SIMPSON CATEGORY (10-min m/s)
# ═══════════════════════════════════════════════════════════════════════


def _ss_category(wind_ms: float) -> int:
    """Saffir-Simpson category from 10-min sustained wind (m/s)."""
    for cat in range(len(SS_THRESHOLDS) - 1, -1, -1):
        if wind_ms >= SS_THRESHOLDS[cat]:
            return cat
    return 0  # below TS but still in track


# ═══════════════════════════════════════════════════════════════════════
# IBTRACS READERS
# ═══════════════════════════════════════════════════════════════════════


def _read_ibtracs_netcdf(
    filepath: str, year_range: Tuple[int, int], basins: List[str]
) -> pd.DataFrame:
    """
    Read IBTrACS v04 NetCDF into a flat DataFrame.
    Uses USA agency data where available, falls back to WMO.
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)

    # Decode times
    times = pd.to_datetime(ds["iso_time"].values.astype(str).flatten())

    # Core variables — shape (storm, time_along_track)
    sid = ds["sid"].values.astype(str)  # storm IDs
    basin_raw = ds["basin"].values.astype(str)  # basin at each time step
    lat = ds["lat"].values  # degrees_north
    lon = ds["lon"].values  # degrees_east

    # Wind: prefer USA, fall back to WMO
    usa_wind = ds["usa_wind"].values if "usa_wind" in ds else None  # 1-min kn
    wmo_wind = (
        ds["wmo_wind"].values if "wmo_wind" in ds else None
    )  # 10-min kn (varies by agency)
    usa_pres = ds["usa_pres"].values if "usa_pres" in ds else None  # hPa
    wmo_pres = ds["wmo_pres"].values if "wmo_pres" in ds else None

    # RMW and dist2land
    usa_rmw = ds["usa_rmw"].values if "usa_rmw" in ds else None  # nm
    dist2land = ds["dist2land"].values if "dist2land" in ds else None  # km
    landfall_raw = ds["landfall"].values if "landfall" in ds else None

    n_storms, n_times = lat.shape
    iso_time_2d = ds["iso_time"].values.astype(str)

    rows = []
    for i in range(n_storms):
        storm_sid = str(sid[i]).strip()
        for j in range(n_times):
            la = float(lat[i, j])
            lo = float(lon[i, j])
            if np.isnan(la) or np.isnan(lo):
                continue

            # Parse time
            try:
                t = pd.Timestamp(str(iso_time_2d[i, j]).strip())
            except Exception:
                continue
            if t.year < year_range[0] or t.year > year_range[1]:
                continue

            # Basin
            b = str(basin_raw[i, j]).strip()[:2]
            storm_basin = IBTRACS_BASIN_MAP.get(b)
            if storm_basin is None or storm_basin not in basins:
                continue

            # Wind (prefer USA → WMO)
            w = np.nan
            wind_is_1min = False
            if usa_wind is not None:
                w_usa = float(usa_wind[i, j])
                if not np.isnan(w_usa) and w_usa > 0:
                    w = w_usa
                    wind_is_1min = True
            if np.isnan(w) and wmo_wind is not None:
                w_wmo = float(wmo_wind[i, j])
                if not np.isnan(w_wmo) and w_wmo > 0:
                    w = w_wmo
                    wind_is_1min = False  # WMO is nominally 10-min

            # Pressure (prefer USA → WMO)
            p = np.nan
            if usa_pres is not None:
                p_usa = float(usa_pres[i, j])
                if not np.isnan(p_usa) and p_usa > 0:
                    p = p_usa
            if np.isnan(p) and wmo_pres is not None:
                p_wmo = float(wmo_pres[i, j])
                if not np.isnan(p_wmo) and p_wmo > 0:
                    p = p_wmo

            # Convert wind to 10-min m/s
            if not np.isnan(w):
                if wind_is_1min:
                    w_ms = w * KN1MIN_TO_MS10MIN
                else:
                    w_ms = w * KN_TO_MS  # already 10-min, just convert units
            else:
                w_ms = np.nan

            # RMW
            rmw_km = DEFAULT_RMAX_KM
            if usa_rmw is not None:
                rmw_val = float(usa_rmw[i, j])
                if not np.isnan(rmw_val) and rmw_val > 0:
                    rmw_km = rmw_val * NM_TO_KM

            # Distance to land
            d2l = np.nan
            if dist2land is not None:
                d2l = float(dist2land[i, j])

            # Longitude to 0–360
            if lo < 0:
                lo += 360.0

            rows.append(
                {
                    "sid": storm_sid,
                    "time": t,
                    "year": t.year,
                    "month": t.month,
                    "basin": storm_basin,
                    "lat": la,
                    "lon": lo,
                    "wind_ms": w_ms,
                    "pressure": p,
                    "rmax_km": rmw_km,
                    "dist2land_km": d2l,
                }
            )

    ds.close()
    return pd.DataFrame(rows)


def _read_ibtracs_csv(
    filepath: str, year_range: Tuple[int, int], basins: List[str]
) -> pd.DataFrame:
    """
    Read IBTrACS v04 CSV.  Skips the units row (row 1).
    Uses USA agency columns where available.
    """
    # Read with the first header row, skip the units row
    df = pd.read_csv(
        filepath, skiprows=[1], low_memory=False, na_values=[" ", "", "  "]
    )

    # Standardise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse time
    df["time"] = pd.to_datetime(df["iso_time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # Filter years
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Map basin
    df["basin_raw"] = df["basin"].astype(str).str.strip().str[:2]
    df["basin_mapped"] = df["basin_raw"].map(IBTRACS_BASIN_MAP)
    df = df[df["basin_mapped"].isin(basins)]

    # Lat / lon
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    # Wind
    usa_wind_col = "usa_wind" if "usa_wind" in df.columns else None
    wmo_wind_col = "wmo_wind" if "wmo_wind" in df.columns else None

    wind_ms = pd.Series(np.nan, index=df.index)
    if usa_wind_col:
        usa_w = pd.to_numeric(df[usa_wind_col], errors="coerce")
        valid = usa_w.notna() & (usa_w > 0)
        wind_ms[valid] = usa_w[valid] * KN1MIN_TO_MS10MIN
    if wmo_wind_col:
        still_nan = wind_ms.isna()
        wmo_w = pd.to_numeric(df[wmo_wind_col], errors="coerce")
        valid = still_nan & wmo_w.notna() & (wmo_w > 0)
        wind_ms[valid] = wmo_w[valid] * KN_TO_MS
    df["wind_ms"] = wind_ms

    # Pressure
    usa_pres_col = "usa_pres" if "usa_pres" in df.columns else None
    wmo_pres_col = "wmo_pres" if "wmo_pres" in df.columns else None
    pres = pd.Series(np.nan, index=df.index)
    if usa_pres_col:
        p = pd.to_numeric(df[usa_pres_col], errors="coerce")
        valid = p.notna() & (p > 0)
        pres[valid] = p[valid]
    if wmo_pres_col:
        still_nan = pres.isna()
        p = pd.to_numeric(df[wmo_pres_col], errors="coerce")
        valid = still_nan & p.notna() & (p > 0)
        pres[valid] = p[valid]
    df["pressure"] = pres

    # RMW
    rmw = pd.Series(DEFAULT_RMAX_KM, index=df.index)
    if "usa_rmw" in df.columns:
        r = pd.to_numeric(df["usa_rmw"], errors="coerce")
        valid = r.notna() & (r > 0)
        rmw[valid] = r[valid] * NM_TO_KM
    df["rmax_km"] = rmw

    # Dist to land
    d2l = pd.Series(np.nan, index=df.index)
    if "dist2land" in df.columns:
        d2l = pd.to_numeric(df["dist2land"], errors="coerce")
    df["dist2land_km"] = d2l

    # Lon to 0–360
    df.loc[df["lon"] < 0, "lon"] += 360.0

    out = df[
        [
            "sid",
            "time",
            "year",
            "month",
            "basin_mapped",
            "lat",
            "lon",
            "wind_ms",
            "pressure",
            "rmax_km",
            "dist2land_km",
        ]
    ].copy()
    out.rename(columns={"basin_mapped": "basin"}, inplace=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
# INTERPOLATION TO 3-HOURLY
# ═══════════════════════════════════════════════════════════════════════


def _interpolate_storm_3h(storm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate a single storm's track to 3-hourly resolution.
    Input must be sorted by time and have a 'time' column.
    """
    storm_df = storm_df.sort_values("time").reset_index(drop=True)
    if len(storm_df) < 2:
        return storm_df

    t_start = storm_df["time"].iloc[0]
    t_end = storm_df["time"].iloc[-1]
    new_times = pd.date_range(t_start, t_end, freq="3h")

    if len(new_times) < 2:
        return storm_df.iloc[:1]

    # Build time in hours for interpolation
    t_orig_h = (storm_df["time"] - t_start).dt.total_seconds() / 3600.0
    t_new_h = (new_times - t_start).total_seconds() / 3600.0

    result = pd.DataFrame({"time": new_times})
    result["year"] = new_times.year
    result["month"] = new_times.month

    for col in ["lat", "lon", "wind_ms", "pressure", "rmax_km", "dist2land_km"]:
        vals = storm_df[col].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() >= 2:
            result[col] = np.interp(t_new_h, t_orig_h[valid], vals[valid])
        elif valid.sum() == 1:
            result[col] = vals[valid][0]
        else:
            result[col] = np.nan

    # Carry over constant fields
    result["sid"] = storm_df["sid"].iloc[0]
    result["basin"] = storm_df["basin"].iloc[0]

    return result


# ═══════════════════════════════════════════════════════════════════════
# MAIN CONVERSION PIPELINE
# ═══════════════════════════════════════════════════════════════════════


def convert_ibtracs(
    input_path: str,
    fmt: str = "netcdf",
    outdir: str = "ibtracs_storm_format",
    basins: Optional[List[str]] = None,
    year_range: Tuple[int, int] = (1980, 2021),
    land_mask_dir: Optional[str] = None,
    split_phases: bool = True,
    interpolate_3h: bool = True,
):
    """
    Full conversion pipeline.

    Parameters
    ----------
    input_path : path to IBTrACS file (NetCDF or CSV)
    fmt : "netcdf" or "csv"
    outdir : output directory
    basins : list of basin codes (default: all 6)
    year_range : (start_year, end_year) inclusive
    land_mask_dir : path to directory with Land_ocean_mask_{basin}.txt files.
                    If None, uses IBTrACS native dist2land.
    split_phases : if True, also write phase-split files
    interpolate_3h : if True, interpolate to 3-hourly (recommended)
    """
    if basins is None:
        basins = list(BASIN_ID.keys())

    os.makedirs(outdir, exist_ok=True)

    # ── 1. Read IBTrACS ──
    print(f"Reading IBTrACS ({fmt}) from {input_path} ...")
    if fmt == "netcdf":
        raw = _read_ibtracs_netcdf(input_path, year_range, basins)
    elif fmt == "csv":
        raw = _read_ibtracs_csv(input_path, year_range, basins)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    print(f"  {len(raw)} track points, {raw['sid'].nunique()} storms")

    # ── 2. Load land masks (optional) ──
    masks = {}
    if land_mask_dir:
        masks = _load_land_masks(land_mask_dir, basins)

    # ── 3. ONI lookup ──
    oni_df = _default_oni_table()
    oni_lookup = _oni_phase_lookup(oni_df)

    # ── 4. Process per basin ──
    for basin in basins:
        print(f"\n{'─' * 50}")
        print(f"  Basin: {basin}")
        print(f"{'─' * 50}")

        basin_data = raw[raw["basin"] == basin].copy()
        if basin_data.empty:
            print("  No storms found, skipping.")
            continue

        storm_ids = basin_data["sid"].unique()
        print(f"  {len(storm_ids)} storms")

        all_storm_rows = []
        year_base = year_range[0]

        for sid in storm_ids:
            sdf = basin_data[basin_data["sid"] == sid].copy()

            # ── 4a. Interpolate to 3-hourly ──
            if interpolate_3h:
                sdf = _interpolate_storm_3h(sdf)

            if sdf.empty:
                continue

            # ── 4b. Filter: keep only TS+ track points ──
            # STORM preprocessing drops every point with wmo_wind < 18 m/s:
            #   df1 = df1[df1["wmo_wind"] >= 18]
            # and synthetic output only contains TS+ points by construction.
            # Apply the same filter here for a fair comparison.
            sdf = sdf[sdf["wind_ms"] >= MIN_WIND_MS].copy()
            if len(sdf) < 2:
                continue
            sdf = sdf.reset_index(drop=True)

            # ── 4c. Compute landfall & dist_land ──
            if basin in masks:
                lm = masks[basin]
                sdf["landfall"] = sdf.apply(
                    lambda r: 1 if lm.is_land(r["lat"], r["lon"]) else 0, axis=1
                )
                sdf["dist_land"] = sdf.apply(
                    lambda r: lm.dist_to_land_approx(r["lat"], r["lon"]), axis=1
                )
            else:
                # Use IBTrACS native
                d2l = sdf["dist2land_km"].values
                sdf["landfall"] = np.where(np.isnan(d2l), 0, np.where(d2l <= 0, 1, 0))
                sdf["dist_land"] = np.where(np.isnan(d2l), 9999.0, d2l)

            # ── 4d. SS category (10-min m/s) ──
            sdf["ss_cat"] = sdf["wind_ms"].apply(
                lambda w: _ss_category(w) if not np.isnan(w) else 0
            )

            # ── 4e. Fill missing pressure with wind-pressure relationship ──
            # Atkinson-Holliday: P ≈ 1010 - 0.6 * (V_kn)^0.644 (rough)
            missing_p = sdf["pressure"].isna()
            if missing_p.any():
                wind_kn = sdf.loc[missing_p, "wind_ms"] / KN_TO_MS
                sdf.loc[missing_p, "pressure"] = 1013.0 - 3.0 * wind_kn**0.76

            # Fill remaining with 1013
            sdf["pressure"] = sdf["pressure"].fillna(1013.0)

            # ── 4f. Fill missing RMW ──
            sdf["rmax_km"] = sdf["rmax_km"].fillna(DEFAULT_RMAX_KM)

            # ── 4g. Assign ENSO phase (genesis month) ──
            genesis_year = int(sdf["year"].iloc[0])
            genesis_month = int(sdf["month"].iloc[0])
            phase = oni_lookup.get((genesis_year, genesis_month), "NEU")

            # ── 4h. Build output rows ──
            storm_year = genesis_year - year_base  # 0-indexed
            storm_id_in_year = 0  # will be renumbered below

            for idx_t, (_, row) in enumerate(sdf.iterrows()):
                all_storm_rows.append(
                    {
                        "orig_year": genesis_year,
                        "storm_year": storm_year,
                        "month": int(row["month"]),
                        "storm_id": 0,  # placeholder
                        "timestep": idx_t,
                        "basin_id": BASIN_ID[basin],
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "pressure": row["pressure"],
                        "wind": row["wind_ms"],
                        "rmax": row["rmax_km"],
                        "ss_cat": int(row.get("ss_cat", 0)),
                        "landfall": int(row.get("landfall", 0)),
                        "dist_land": row.get("dist_land", 9999.0),
                        "phase": phase,
                        "sid": sid,
                    }
                )

        if not all_storm_rows:
            print("  No TS+ storms after filtering, skipping.")
            continue

        df_out = pd.DataFrame(all_storm_rows)

        # ── 5. Renumber storm_id per year (0, 1, 2, ...) ──
        for yr in df_out["storm_year"].unique():
            yr_mask = df_out["storm_year"] == yr
            sids_in_year = df_out.loc[yr_mask, "sid"].unique()
            sid_to_id = {s: i for i, s in enumerate(sids_in_year)}
            df_out.loc[yr_mask, "storm_id"] = df_out.loc[yr_mask, "sid"].map(sid_to_id)

        # ── 6. Write output files ──
        cols_out = [
            "storm_year",
            "month",
            "storm_id",
            "timestep",
            "basin_id",
            "lat",
            "lon",
            "pressure",
            "wind",
            "rmax",
            "ss_cat",
            "landfall",
            "dist_land",
        ]

        def _write_storm_file(subset: pd.DataFrame, filepath: str):
            arr = subset[cols_out].values
            np.savetxt(
                filepath,
                arr,
                delimiter=",",
                fmt=[
                    "%d",
                    "%d",
                    "%d",
                    "%d",
                    "%d",
                    "%.4f",
                    "%.4f",
                    "%.2f",
                    "%.2f",
                    "%.2f",
                    "%d",
                    "%d",
                    "%.2f",
                ],
            )
            print(
                f"  Wrote {filepath}  ({len(subset)} rows, "
                f"{subset['sid'].nunique()} storms)"
            )

        # All-phase file
        fname_all = f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_0.txt"
        _write_storm_file(df_out, os.path.join(outdir, fname_all))

        # Phase-split files
        if split_phases:
            for ph in ["EN", "NEU", "LN"]:
                sub = df_out[df_out["phase"] == ph]
                if sub.empty:
                    print(f"  Phase {ph}: no storms")
                    continue
                fname = f"STORM_DATA_IBTRACS_{basin}_{ph}_1000_YEARS_0.txt"
                _write_storm_file(sub, os.path.join(outdir, fname))

    # ── 7. Write metadata ──
    meta = {
        "source": input_path,
        "year_range": f"{year_range[0]}-{year_range[1]}",
        "basins": ",".join(basins),
        "n_years": year_range[1] - year_range[0] + 1,
        "wind_convention": "10-min sustained m/s",
        "land_mask": land_mask_dir or "IBTrACS native dist2land",
        "interpolated_3h": str(interpolate_3h),
    }
    meta_path = os.path.join(outdir, "metadata.txt")
    with open(meta_path, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    print(f"\n  Metadata written to {meta_path}")
    print(f"  n_years = {meta['n_years']}  (use this value in evaluation.py)")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Convert IBTrACS to STORM 13-column format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # NetCDF input (recommended):
  python ibtracs_to_storm.py \\
      --input IBTrACS.ALL.v04r01.nc \\
      --format netcdf \\
      --land_mask_dir ./SIENA-IH-STORM-VWS-Track/ \\
      --outdir ibtracs_ref/ \\
      --basins NA WP

  # CSV input:
  python ibtracs_to_storm.py \\
      --input ibtracs.ALL.list.v04r01.csv \\
      --format csv \\
      --outdir ibtracs_ref/

  # Then use in evaluation:
  python evaluation.py \\
      --folder ibtracs_ref/ \\
      --basin NA \\
      --n_years 42
        """,
    )
    parser.add_argument("--input", required=True, help="Path to IBTrACS file")
    parser.add_argument("--format", choices=["netcdf", "csv"], default="netcdf")
    parser.add_argument("--outdir", default="ibtracs_storm_format")
    parser.add_argument(
        "--basins", nargs="+", default=None, choices=list(BASIN_ID.keys())
    )
    parser.add_argument("--year_start", type=int, default=1980)
    parser.add_argument("--year_end", type=int, default=2021)
    parser.add_argument(
        "--land_mask_dir",
        default=None,
        help="Directory with STORM Land_ocean_mask_{basin}.txt files",
    )
    parser.add_argument(
        "--no_phase_split",
        action="store_true",
        help="Skip ENSO phase-split output files",
    )
    parser.add_argument(
        "--no_interpolate", action="store_true", help="Skip 3-hourly interpolation"
    )

    args = parser.parse_args()

    convert_ibtracs(
        input_path=args.input,
        fmt=args.format,
        outdir=args.outdir,
        basins=args.basins,
        year_range=(args.year_start, args.year_end),
        land_mask_dir=args.land_mask_dir,
        split_phases=not args.no_phase_split,
        interpolate_3h=not args.no_interpolate,
    )


if __name__ == "__main__":
    main()
