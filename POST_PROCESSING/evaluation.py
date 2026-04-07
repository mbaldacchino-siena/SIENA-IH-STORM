"""
SIENA-IH-STORM  –  Evaluation & Metrics Module
================================================

Computes every metric described in Section 5.4 of the whitepaper:

  Climatological fidelity (§5.4.1)
  ─────────────────────────────────
  1. Annual genesis count  (mean, std, Poisson λ̂)
  2. Genesis density        (1°×1° grid, events per 10 000 yr)
  3. Track density          (1°×1° grid, 3-hourly fixes per yr)
  4. Intensity distributions (CDF of Pmin & Vmax + two-sample KS)
  5. Average TC lifetime    (mean 3-hourly steps per storm)

  Hazard-oriented metrics (§5.4.2)
  ─────────────────────────────────
  6. Landfall counts        (annual, per basin/phase)
  7. Landfall intensity     (CDF of Pmin & Vmax at first landfall)
  8. Return periods at coastal cities  (proximity-based Weibull RP)

  Composite catalog builder
  ─────────────────────────
  9. assemble_all_catalog() — build the ALL catalog from phase-specific
     folders weighted by historical ONI active-month frequencies.

Usage
-----
    from evaluation import (
        load_catalog,
        compute_all_metrics,
        assemble_all_catalog,
        return_periods_at_cities,
        compare_candidates,
    )

    cat = load_catalog("/path/to/S3_NA_LN/", basin="NA")
    metrics = compute_all_metrics(cat, n_years=10_000)

File format
-----------
STORM_DATA_IBTRACS_{basin}_{phase}_1000_YEARS_{id}.txt   (phase variant)
STORM_DATA_IBTRACS_{basin}_1000_YEARS_{id}.txt           (legacy B0)
  13 comma-separated columns, 0-indexed year counter per 1000-yr chunk.

Dependencies: numpy, scipy, pandas  (all in the standard scientific stack)
Optional:     matplotlib (for plotting helpers only)

Author : Mathys Baldacchino / evaluation framework
License: same as SIENA-IH-STORM repository
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass, field
from math import radians, cos, sin, asin, sqrt
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    from climada.hazard import Centroids, TCTracks
    from climada.hazard.trop_cyclone import TropCyclone

    HAS_CLIMADA = True
except ImportError:
    HAS_CLIMADA = False

import numpy as np
import pandas as pd
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

BASIN_ID_MAP = {"EP": 0, "NA": 1, "NI": 2, "SI": 3, "SP": 4, "WP": 5}
ID_BASIN_MAP = {v: k for k, v in BASIN_ID_MAP.items()}

# Active-season months per basin  (from input.dat / whitepaper)
ACTIVE_MONTHS = {
    "EP": [6, 7, 8, 9, 10, 11],
    "NA": [6, 7, 8, 9, 10, 11],
    "NI": [10, 11],
    "SI": [1, 2, 3, 4, 11, 12],
    "SP": [1, 2, 3, 4, 11, 12],
    "WP": [5, 6, 7, 8, 9, 10, 11],
}

# Basin geographic bounds (lon in 0–360 convention)
BASIN_BOUNDS = {
    "EP": {"lon": (180, 285), "lat": (0, 60)},
    "NA": {"lon": (255, 359), "lat": (0, 60)},
    "NI": {"lon": (30, 100), "lat": (0, 35)},
    "SI": {"lon": (30, 135), "lat": (-60, 0)},
    "SP": {"lon": (135, 240), "lat": (-60, 0)},
    "WP": {"lon": (100, 180), "lat": (0, 60)},
}

# Saffir-Simpson thresholds (10-min sustained, m/s)
SS_THRESHOLDS_MS = {
    "TS": 18.0,
    "Cat1": 33.0,  # ~64 kn  (10-min ≈ 29 kn, but STORM uses 10-min conversion)
    "Cat2": 43.0,
    "Cat3": 50.0,
    "Cat4": 58.0,
    "Cat5": 70.0,
}

# Default coastal cities (Table 3 of whitepaper)
DEFAULT_CITIES = [
    {"city": "Miami", "lat": 25.8, "lon": -80.2, "basin": "NA"},
    {"city": "Houston", "lat": 29.8, "lon": -95.4, "basin": "NA"},
    {"city": "New Orleans", "lat": 30.0, "lon": -90.1, "basin": "NA"},
    {"city": "Tampa", "lat": 28.0, "lon": -82.5, "basin": "NA"},
    {"city": "New York", "lat": 40.7, "lon": -74.0, "basin": "NA"},
    {"city": "Charleston", "lat": 32.8, "lon": -80.0, "basin": "NA"},
    {"city": "Tokyo", "lat": 35.7, "lon": 139.7, "basin": "WP"},
    {"city": "Manila", "lat": 14.6, "lon": 121.0, "basin": "WP"},
    {"city": "Hong Kong", "lat": 22.3, "lon": 114.2, "basin": "WP"},
    {"city": "Mumbai", "lat": 19.1, "lon": 72.9, "basin": "NI"},
    {"city": "Saint-Denis", "lat": -20.9, "lon": 55.5, "basin": "SI"},
]

# Column names for the STORM .txt format
STORM_COLUMNS = [
    "year",
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


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════


def load_storm_file(filepath: str) -> pd.DataFrame:
    """Load a single STORM .txt file into a DataFrame."""
    
    return df

def load_storm_file(filepath: str) -> pd.DataFrame:
    """Load a single STORM .txt file into a DataFrame."""
    df = pd.read_csv(
        filepath,
        header=None,
        names=STORM_COLUMNS,
        dtype={
            "year": np.int32,
            "month": np.int32,
            "storm_id": np.int32,
            "timestep": np.int32,
            "basin_id": np.int32,
            "lat": np.float64,
            "lon": np.float64,
            "pressure": np.float64,
            "wind": np.float64,
            "rmax": np.float64,
            "ss_cat": np.int32,
            "landfall": np.int32,
            "dist_land": np.float64,
        },
    )
    # Drop rows where wind is NaN (can occur from numerical issues in simulation)
    n_before = len(df)
    df = df.dropna(subset=["wind", "pressure"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Warning: dropped {n_dropped} rows with NaN wind in {os.path.basename(filepath)}")
    return df


def _find_catalog_files(
    folder: str,
    basin: str,
    phase: Optional[str] = None,
    file_pattern: Optional[str] = None,
) -> List[str]:
    """
    Auto-discover STORM .txt files in *folder* for a given basin (and
    optional ENSO phase).  Returns a sorted list of file paths.

    Supports both naming conventions:
      STORM_DATA_IBTRACS_{basin}_{phase}_1000_YEARS_{id}.txt   (SIENA / IH-STORM)
      STORM_DATA_IBTRACS_{basin}_1000_YEARS_{id}.txt           (original STORM / B0)
    """
    if file_pattern is not None:
        pattern = file_pattern.format(basin=basin, phase=phase or "")
        hits = sorted(glob.glob(os.path.join(folder, pattern)))
        if hits:
            return hits

    # Try phase-aware naming first
    if phase is not None:
        pattern = f"STORM_DATA_IBTRACS_{basin}_{phase}_1000_YEARS_*.txt"
        hits = sorted(glob.glob(os.path.join(folder, pattern)))
        if hits:
            return hits

    # Fallback to legacy naming  (no phase in filename)
    pattern = f"STORM_DATA_IBTRACS_{basin}_1000_YEARS_*.txt"
    hits = sorted(glob.glob(os.path.join(folder, pattern)))
    return hits


def load_catalog(
    folder: str,
    basin: str,
    phase: Optional[str] = None,
    max_files: Optional[int] = None,
    file_pattern: Optional[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all STORM .txt chunks from *folder* for a given basin/phase
    into a single DataFrame.  Year counters are offset so that
    chunk 0 → years 0–999, chunk 1 → years 1000–1999, etc.

    Parameters
    ----------
    folder : str
        Directory containing the STORM .txt files.
    basin : str
        Basin code: EP, NA, NI, SI, SP, WP.
    phase : str or None
        ENSO phase label (EN, NEU, LN) or None for legacy/B0 files.
    max_files : int or None
        If set, only load the first N files (useful for quick checks).
    file_pattern : str or None
        Optional glob pattern (may contain {basin} and {phase} placeholders).

    Returns
    -------
    pd.DataFrame with columns from STORM_COLUMNS plus 'global_year' and
    'global_storm_uid'.
    """
    files = _find_catalog_files(folder, basin, phase, file_pattern)
    if not files:
        raise FileNotFoundError(
            f"No STORM files found in {folder} for basin={basin}, phase={phase}"
        )

    if max_files is not None:
        files = files[:max_files]

    chunks = []
    year_offset = 0
    storm_uid_offset = 0

    for fpath in files:
        df = load_storm_file(fpath)
        # Infer number of years in this chunk from max year + 1
        n_years_chunk = int(df["year"].max()) + 1

        df["global_year"] = df["year"] + year_offset

        # Build a globally unique storm identifier
        # Within each file, (year, storm_id) is unique
        df["global_storm_uid"] = df["global_year"] * 10_000 + df["storm_id"]

        chunks.append(df)
        year_offset += n_years_chunk

    catalog = pd.concat(chunks, ignore_index=True)
    return catalog, files


def pool_tctracks(track_input, deduplicate=True):
    """
    Accept either:
      - one TCTracks object
      - a list/tuple of TCTracks objects
    and return a single pooled TCTracks object.
    """
    if not HAS_CLIMADA:
        raise ImportError("CLIMADA is required for pool_tctracks. pip install climada")
    if isinstance(track_input, TCTracks):
        pooled = TCTracks(data=list(track_input.data))
    elif isinstance(track_input, (list, tuple)):
        pooled = TCTracks()
        for obj in track_input:
            if not isinstance(obj, TCTracks):
                raise TypeError("Each element must be a TCTracks object.")
            pooled.append(list(obj.data))
    else:
        raise TypeError("track_input must be a TCTracks or a list/tuple of TCTracks.")

    if deduplicate:
        unique_data = []
        seen = set()
        for ds in pooled.data:
            sid = ds.attrs.get("sid", ds.attrs.get("name"))
            if sid not in seen:
                unique_data.append(ds)
                seen.add(sid)
        pooled = TCTracks(data=unique_data)

    return pooled


# ═══════════════════════════════════════════════════════════════════════
# §5.3  — COMPOSITE ALL CATALOG
# ═══════════════════════════════════════════════════════════════════════


def _default_oni_table() -> pd.DataFrame:
    """
    Hardcoded ONI 3.4 monthly values 1980–2021.
    Source: NOAA CPC  (https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)
    """
    # year, month, oni_value  — complete table 1980-2021
    # fmt: off
    _oni_raw = {
        1980: [0.0,-0.1,0.1,0.2,0.3,0.3,0.2,0.0,-0.1,-0.1,-0.1,-0.1],
        1981: [-0.3,-0.5,-0.5,-0.4,-0.3,-0.3,-0.4,-0.3,-0.2,-0.1,-0.2,-0.1],
        1982: [-0.1,0.0,0.1,0.3,0.5,0.6,0.8,1.0,1.5,1.9,2.1,2.1],
        1983: [2.1,1.8,1.5,1.2,1.0,0.6,0.2,-0.2,-0.5,-0.8,-0.9,-0.8],
        1984: [-0.5,-0.3,-0.3,-0.4,-0.5,-0.4,-0.3,-0.2,-0.3,-0.6,-0.9,-1.1],
        1985: [-0.9,-0.7,-0.7,-0.7,-0.7,-0.6,-0.4,-0.4,-0.4,-0.3,-0.2,-0.3],
        1986: [-0.4,-0.4,-0.3,-0.2,-0.1,0.0,0.2,0.4,0.7,0.9,1.0,1.1],
        1987: [1.1,1.2,1.1,1.0,0.9,1.1,1.4,1.6,1.6,1.4,1.2,1.1],
        1988: [0.8,0.5,0.1,-0.3,-0.8,-1.2,-1.2,-1.1,-1.2,-1.6,-1.9,-1.8],
        1989: [-1.6,-1.4,-1.1,-0.8,-0.6,-0.3,-0.3,-0.3,-0.3,-0.3,-0.2,-0.1],
        1990: [0.1,0.2,0.2,0.2,0.2,0.3,0.3,0.3,0.4,0.3,0.4,0.4],
        1991: [0.4,0.3,0.2,0.2,0.4,0.6,0.7,0.7,0.7,0.8,1.2,1.4],
        1992: [1.6,1.5,1.4,1.2,1.0,0.7,0.3,0.0,-0.1,-0.1,0.0,0.0],
        1993: [0.2,0.3,0.5,0.7,0.8,0.6,0.3,0.2,0.2,0.2,0.1,0.1],
        1994: [0.1,0.1,0.2,0.3,0.4,0.4,0.4,0.4,0.4,0.6,0.9,1.0],
        1995: [0.9,0.7,0.5,0.3,0.2,0.0,-0.2,-0.5,-0.7,-0.9,-0.9,-0.9],
        1996: [-0.9,-0.7,-0.6,-0.4,-0.2,-0.2,-0.2,-0.3,-0.3,-0.4,-0.4,-0.5],
        1997: [-0.5,-0.4,-0.1,0.3,0.8,1.2,1.6,1.9,2.1,2.3,2.4,2.3],
        1998: [2.1,1.8,1.4,1.0,0.5,-0.1,-0.7,-1.0,-1.2,-1.3,-1.4,-1.5],
        1999: [-1.5,-1.3,-1.0,-0.9,-0.9,-1.0,-1.0,-1.0,-1.1,-1.2,-1.4,-1.7],
        2000: [-1.7,-1.4,-1.1,-0.8,-0.7,-0.6,-0.6,-0.5,-0.6,-0.7,-0.7,-0.7],
        2001: [-0.7,-0.5,-0.4,-0.3,-0.2,-0.1,-0.1,0.0,-0.1,-0.2,-0.3,-0.3],
        2002: [-0.1,0.0,0.1,0.2,0.5,0.7,0.8,0.9,1.0,1.2,1.3,1.1],
        2003: [0.9,0.6,0.4,0.0,-0.3,-0.2,0.1,0.2,0.3,0.3,0.4,0.3],
        2004: [0.3,0.3,0.2,0.2,0.2,0.3,0.5,0.7,0.7,0.7,0.7,0.7],
        2005: [0.6,0.6,0.5,0.4,0.3,0.1,0.0,-0.1,-0.1,-0.3,-0.6,-0.7],
        2006: [-0.7,-0.6,-0.4,-0.2,0.0,0.1,0.2,0.3,0.5,0.7,0.9,0.9],
        2007: [0.7,0.3,-0.1,-0.2,-0.3,-0.3,-0.4,-0.6,-0.8,-1.1,-1.2,-1.3],
        2008: [-1.5,-1.4,-1.2,-0.9,-0.7,-0.5,-0.4,0.0,0.0,-0.3,-0.6,-0.7],
        2009: [-0.8,-0.7,-0.5,-0.2,0.1,0.4,0.5,0.5,0.7,1.0,1.3,1.6],
        2010: [1.5,1.3,0.9,0.4,-0.1,-0.6,-1.0,-1.4,-1.6,-1.7,-1.7,-1.6],
        2011: [-1.4,-1.1,-0.8,-0.6,-0.3,-0.1,-0.3,-0.5,-0.7,-0.9,-1.0,-0.9],
        2012: [-0.8,-0.6,-0.5,-0.4,-0.2,0.1,0.3,0.3,0.3,0.2,0.0,-0.2],
        2013: [-0.4,-0.3,-0.2,-0.2,-0.3,-0.3,-0.4,-0.3,-0.2,-0.2,-0.2,-0.3],
        2014: [-0.4,-0.4,-0.2,0.1,0.3,0.2,0.1,0.0,0.2,0.4,0.6,0.7],
        2015: [0.6,0.6,0.6,0.8,1.0,1.2,1.5,1.8,2.1,2.4,2.6,2.6],
        2016: [2.5,2.1,1.7,1.0,0.5,0.0,-0.3,-0.6,-0.7,-0.7,-0.7,-0.6],
        2017: [-0.3,-0.1,0.1,0.3,0.4,0.4,0.2,0.0,-0.4,-0.7,-0.9,-1.0],
        2018: [-0.9,-0.8,-0.6,-0.4,-0.1,0.1,0.1,0.2,0.4,0.7,0.9,0.8],
        2019: [0.8,0.8,0.8,0.8,0.6,0.5,0.3,0.1,0.1,0.3,0.5,0.5],
        2020: [0.5,0.5,0.4,0.2,-0.1,-0.3,-0.5,-0.6,-0.9,-1.2,-1.3,-1.2],
        2021: [-1.0,-0.8,-0.6,-0.4,-0.2,-0.1,-0.4,-0.5,-0.7,-0.8,-0.9,-0.9],
    }
    # fmt: on
    rows = []
    for yr, vals in _oni_raw.items():
        for m, v in enumerate(vals, 1):
            rows.append({"year": yr, "month": m, "oni": v})
    df = pd.DataFrame(rows)
    df["phase"] = np.where(
        df["oni"] > 0.5, "EN", np.where(df["oni"] < -0.5, "LN", "NEU")
    )
    return df


def compute_phase_fractions(
    basin: str,
    oni_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Empirical ENSO phase frequencies for *basin* active-season months
    over 1980–2021.  Returns dict {"EN": f_en, "NEU": f_neu, "LN": f_ln}.
    """
    if oni_df is None:
        oni_df = _default_oni_table()
    active = set(ACTIVE_MONTHS[basin])
    sub = oni_df[oni_df["month"].isin(active)]
    counts = sub["phase"].value_counts()
    total = counts.sum()
    return {ph: counts.get(ph, 0) / total for ph in ("EN", "NEU", "LN")}


# NEW (entire function added)
def compute_effective_years(
    basin: str,
    total_years: float = 42,
    oni_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Effective observation years per ENSO phase for a given basin.

    For IBTrACS over 1980–2021 (42 years), the EN subset doesn't cover
    42 years — it covers only the active-season months classified as EN.
    The effective years are:

        Y_eff(phase) = (n_active_months_in_phase / season_length)

    This is the correct denominator for annual rates (genesis/yr, etc.)
    when evaluating phase-specific IBTrACS subsets.

    Parameters
    ----------
    basin : basin code
    total_years : total observation period (default 42 for 1980–2021)
    oni_df : ONI table (uses built-in if None)

    Returns
    -------
    dict : {"EN": eff_years, "NEU": eff_years, "LN": eff_years, "ALL": total_years}
    """
    if oni_df is None:
        oni_df = _default_oni_table()
    active = set(ACTIVE_MONTHS[basin])
    season_length = len(active)
    sub = oni_df[oni_df["month"].isin(active)]
    counts = sub["phase"].value_counts()

    result = {"ALL": float(total_years)}
    for ph in ("EN", "NEU", "LN"):
        n_months = counts.get(ph, 0)
        result[ph] = float(n_months) / season_length
    return result


def assemble_all_catalog(
    phase_folders: Dict[str, str],
    basin: str,
    total_years: int = 10_000,
    oni_df: Optional[pd.DataFrame] = None,
    file_pattern: Optional[str] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the composite ALL catalog (§5.3) by drawing simulated years
    from phase-specific catalogs in proportion to historical ENSO
    active-month frequencies.

    Parameters
    ----------
    phase_folders : dict
        {"EN": "/path/to/EN/", "NEU": "/path/to/NEU/", "LN": "/path/to/LN/"}
    basin : str
    total_years : int
    oni_df : pd.DataFrame or None  (uses built-in ONI if None)
    file_pattern : str or None
    seed : int
        Random seed for reproducible year sampling.

    Returns
    -------
    pd.DataFrame  — concatenated catalog with reassigned global_year.
    """
    fracs = compute_phase_fractions(basin, oni_df)
    rng = np.random.default_rng(seed)

    # Compute how many years to draw from each phase
    raw_years = {ph: fracs[ph] * total_years for ph in ("EN", "NEU", "LN")}
    phase_years = {ph: int(np.round(v)) for ph, v in raw_years.items()}

    # Adjust rounding residual: assign remainder to largest-remainder phase
    diff = total_years - sum(phase_years.values())
    if diff != 0:
        remainders = {ph: raw_years[ph] - int(raw_years[ph]) for ph in phase_years}
        sorted_ph = sorted(remainders, key=lambda p: remainders[p], reverse=(diff > 0))
        for i in range(abs(diff)):
            phase_years[sorted_ph[i % 3]] += int(np.sign(diff))

    parts = []
    year_offset = 0

    for ph in ("EN", "NEU", "LN"):
        n_draw = phase_years[ph]
        if n_draw == 0:
            continue

        cat, files = load_catalog(
            phase_folders[ph], basin, phase=ph, file_pattern=file_pattern
        )
        available_years = cat["global_year"].unique()
        if len(available_years) < n_draw:
            warnings.warn(
                f"Phase {ph}: requested {n_draw} years but only "
                f"{len(available_years)} available; sampling with replacement."
            )
            chosen = rng.choice(available_years, size=n_draw, replace=True)
        else:
            chosen = rng.choice(available_years, size=n_draw, replace=False)

        for new_yr, old_yr in enumerate(chosen, start=year_offset):
            sub = cat[cat["global_year"] == old_yr].copy()
            sub["global_year"] = new_yr
            sub["enso_phase"] = ph
            parts.append(sub)

        year_offset += n_draw

    all_cat = pd.concat(parts, ignore_index=True)
    # Rebuild UIDs
    all_cat["global_storm_uid"] = all_cat["global_year"] * 10_000 + all_cat["storm_id"]
    return all_cat, files


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in km between two points (decimal degrees)."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def _per_storm_agg(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    One-row-per-storm summary.  Computes genesis point, lifetime-min
    pressure, lifetime-max wind, number of timesteps, first-landfall info.
    """
    g = catalog.groupby("global_storm_uid", sort=False)

    agg = g.agg(
        year=("global_year", "first"),
        month=("month", "first"),
        n_steps=("timestep", "count"),
        lat_genesis=("lat", "first"),
        lon_genesis=("lon", "first"),
        pmin=("pressure", "min"),
        vmax=("wind", "max"),
        max_ss=("ss_cat", "max"),
        has_landfall=("landfall", "max"),
    )

    # First landfall point
    lf = catalog[catalog["landfall"] == 1].groupby("global_storm_uid", sort=False)
    first_lf = lf.first()[["pressure", "wind", "lat", "lon"]].rename(
        columns={
            "pressure": "lf_pressure",
            "wind": "lf_wind",
            "lat": "lf_lat",
            "lon": "lf_lon",
        }
    )
    agg = agg.join(first_lf, how="left")
    return agg


# ═══════════════════════════════════════════════════════════════════════
# §5.4.1  —  CLIMATOLOGICAL FIDELITY
# ═══════════════════════════════════════════════════════════════════════

# ---------- 1. Annual genesis count ----------


def annual_genesis_count(catalog: pd.DataFrame, n_years: float) -> dict:
    """..."""
    storms = _per_storm_agg(catalog)
    total = len(storms)
    mean_rate = total / n_years

    if "year" in storms.columns:
        actual_years = catalog["global_year"].nunique()
        annual = (
            storms.groupby("year").size().reindex(range(actual_years), fill_value=0)
        )
        std_val = float(annual.std())
    else:
        std_val = np.sqrt(mean_rate)

    return {
        "mean": float(mean_rate),
        "std": std_val,
        "lambda_hat": float(mean_rate),
        "total_storms": int(total),
        "n_years": float(n_years),
    }


# ---------- 2. Genesis density ----------


def genesis_density(
    catalog: pd.DataFrame,
    n_years: float,
    resolution: float = 1.0,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2-D genesis frequency on a regular grid.

    Returns
    -------
    density : 2D array, events per 10 000 years per cell
    lon_edges, lat_edges : 1D edge arrays
    """
    storms = _per_storm_agg(catalog)
    lons = storms["lon_genesis"].values
    lats = storms["lat_genesis"].values

    if lon_range is None:
        lon_range = (np.floor(lons.min()), np.ceil(lons.max()))
    if lat_range is None:
        lat_range = (np.floor(lats.min()), np.ceil(lats.max()))

    lon_edges = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    lat_edges = np.arange(lat_range[0], lat_range[1] + resolution, resolution)

    H, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])
    # Normalize: events per 10 000 years per cell
    density = H.T * (10_000 / n_years)
    return density, lon_edges, lat_edges


# ---------- 3. Track density ----------


def track_density(
    catalog: pd.DataFrame,
    n_years: float,
    resolution: float = 1.0,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Number of 3-hourly TC fixes per 1°×1° cell, normalized to annual rate.

    Returns
    -------
    density : 2D array (fixes per year per cell)
    lon_edges, lat_edges
    """
    lons = catalog["lon"].values
    lats = catalog["lat"].values

    if lon_range is None:
        lon_range = (np.floor(lons.min()), np.ceil(lons.max()))
    if lat_range is None:
        lat_range = (np.floor(lats.min()), np.ceil(lats.max()))

    lon_edges = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    lat_edges = np.arange(lat_range[0], lat_range[1] + resolution, resolution)

    H, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])
    density = H.T / n_years
    return density, lon_edges, lat_edges


# ---------- 4. Intensity distributions ----------


def intensity_distributions(
    catalog: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
    ref_n_years: Optional[int] = None,
) -> dict:
    """
    Empirical CDFs of lifetime-minimum pressure (Pmin) and lifetime-maximum
    wind (Vmax).  If *reference* is provided, computes two-sample KS
    statistics.

    Parameters
    ----------
    catalog : synthetic catalog DataFrame
    reference : optional IBTrACS-like DataFrame (same column schema or
                with columns 'pmin', 'vmax' per storm)
    ref_n_years : n_years for the reference (for info only)

    Returns
    -------
    dict with:
      pmin_values, pmin_cdf       — synthetic Pmin ECDF
      vmax_values, vmax_cdf       — synthetic Vmax ECDF
      ks_pmin, ks_vmax            — KS statistics (if reference given)
      ks_pmin_pvalue, ks_vmax_pvalue
      ref_pmin_values, ref_pmin_cdf  (if reference given)
      ref_vmax_values, ref_vmax_cdf
    """
    storms_syn = _per_storm_agg(catalog)
    pmin_syn = np.sort(storms_syn["pmin"].dropna().values)
    vmax_syn = np.sort(storms_syn["vmax"].dropna().values)

    def ecdf(x):
        return np.arange(1, len(x) + 1) / len(x)

    result = {
        "pmin_values": pmin_syn,
        "pmin_cdf": ecdf(pmin_syn),
        "vmax_values": vmax_syn,
        "vmax_cdf": ecdf(vmax_syn),
    }

    if reference is not None:
        if "pmin" in reference.columns:
            ref_storms = reference
        else:
            ref_storms = _per_storm_agg(reference)
        pmin_ref = np.sort(ref_storms["pmin"].dropna().values)
        vmax_ref = np.sort(ref_storms["vmax"].dropna().values)

        ks_p = stats.ks_2samp(pmin_syn, pmin_ref)
        ks_v = stats.ks_2samp(vmax_syn, vmax_ref)

        result.update(
            {
                "ref_pmin_values": pmin_ref,
                "ref_pmin_cdf": ecdf(pmin_ref),
                "ref_vmax_values": vmax_ref,
                "ref_vmax_cdf": ecdf(vmax_ref),
                "ks_pmin": float(ks_p.statistic),
                "ks_pmin_pvalue": float(ks_p.pvalue),
                "ks_vmax": float(ks_v.statistic),
                "ks_vmax_pvalue": float(ks_v.pvalue),
            }
        )

    return result


# ---------- 5. Average TC lifetime ----------


def average_lifetime(catalog: pd.DataFrame) -> dict:
    """
    Summary statistics of TC lifetime.  See also lifetime_distribution()
    for the full per-storm DataFrame.

    Returns
    -------
    dict: mean_steps, std_steps, mean_hours, median_steps
    """
    storms = _per_storm_agg(catalog)
    steps = storms["n_steps"]
    return {
        "mean_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "median_steps": float(steps.median()),
        "mean_hours": float(steps.mean() * 3),
    }


def lifetime_distribution(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Per-storm lifetime table — one row per TC.

    Returns
    -------
    DataFrame with columns:
        storm_uid, year, month, n_steps, duration_hours, pmin, vmax,
        max_ss_cat, lat_genesis, lon_genesis, has_landfall
    Export this to CSV to overlay distributions across candidates.
    """
    storms = _per_storm_agg(catalog)
    out = storms[
        [
            "year",
            "month",
            "n_steps",
            "lat_genesis",
            "lon_genesis",
            "pmin",
            "vmax",
            "max_ss",
            "has_landfall",
        ]
    ].copy()
    out.index.name = "storm_uid"
    out["duration_hours"] = out["n_steps"] * 3
    out = out.reset_index()
    return out


# ═══════════════════════════════════════════════════════════════════════
# §5.4.2  —  HAZARD-ORIENTED METRICS
# ═══════════════════════════════════════════════════════════════════════

# ---------- 6. Landfall counts ----------


def landfall_counts(catalog: pd.DataFrame, n_years: float) -> dict:
    """..."""
    storms = _per_storm_agg(catalog)
    lf_storms = storms[storms["has_landfall"] == 1]
    n_lf = len(lf_storms)

    mean_rate = n_lf / n_years
    if "year" in lf_storms.columns:
        actual_years = catalog["global_year"].nunique()
        annual = (
            lf_storms.groupby("year").size().reindex(range(actual_years), fill_value=0)
        )
        std_val = float(annual.std())
    else:
        std_val = np.sqrt(mean_rate)

    return {
        "total_landfalls": int(n_lf),
        "annual_mean": float(mean_rate),
        "annual_std": std_val,
    }


# ---------- 7. Landfall intensity ----------


def landfall_intensity(
    catalog: pd.DataFrame,
    reference: Optional[pd.DataFrame] = None,
) -> dict:
    """
    CDF of pressure and wind at first landfall.

    Returns
    -------
    dict with lf_pmin, lf_vmax arrays and CDFs; KS stats if ref given.
    """
    storms = _per_storm_agg(catalog)
    lf = storms.dropna(subset=["lf_pressure"])

    pmin_lf = np.sort(lf["lf_pressure"].values)
    vmax_lf = np.sort(lf["lf_wind"].values)

    def ecdf(x):
        return np.arange(1, len(x) + 1) / len(x)

    result = {
        "lf_pmin_values": pmin_lf,
        "lf_pmin_cdf": ecdf(pmin_lf),
        "lf_vmax_values": vmax_lf,
        "lf_vmax_cdf": ecdf(vmax_lf),
        "n_landfall_storms": len(lf),
    }

    if reference is not None:
        ref_storms = (
            _per_storm_agg(reference) if "pmin" not in reference.columns else reference
        )
        ref_lf = ref_storms.dropna(subset=["lf_pressure"])
        if len(ref_lf) > 0:
            ref_pmin = np.sort(ref_lf["lf_pressure"].values)
            ref_vmax = np.sort(ref_lf["lf_wind"].values)
            ks_p = stats.ks_2samp(pmin_lf, ref_pmin) if len(pmin_lf) > 0 else None
            ks_v = stats.ks_2samp(vmax_lf, ref_vmax) if len(vmax_lf) > 0 else None
            result.update(
                {
                    "ref_lf_pmin_values": ref_pmin,
                    "ref_lf_vmax_values": ref_vmax,
                    "ks_lf_pmin": float(ks_p.statistic) if ks_p else np.nan,
                    "ks_lf_pmin_pvalue": float(ks_p.pvalue) if ks_p else np.nan,
                    "ks_lf_vmax": float(ks_v.statistic) if ks_v else np.nan,
                    "ks_lf_vmax_pvalue": float(ks_v.pvalue) if ks_v else np.nan,
                }
            )

    return result


# ---------- 8. Return periods at coastal cities ----------


def _extract_city_max_winds(
    catalog: pd.DataFrame,
    city_lat: float,
    city_lon: float,
    radius_km: float = 111.0,
    min_wind: float = 18.0,
) -> np.ndarray:
    """
    For each TC in the catalog, find the maximum wind speed within
    *radius_km* of the city.  Returns array of per-storm max winds
    (only storms that passed within radius and exceeded min_wind).
    """
    # Convert city lon to 0–360 if negative
    clon = city_lon if city_lon >= 0 else city_lon + 360.0

    # Vectorised pre-filter: rough lat/lon box (~ 1° ≈ 111 km)
    dlat_deg = radius_km / 111.0
    dlon_deg = radius_km / (111.0 * max(cos(radians(city_lat)), 0.1))

    mask = (
        (catalog["lat"] >= city_lat - dlat_deg)
        & (catalog["lat"] <= city_lat + dlat_deg)
        & (catalog["lon"] >= clon - dlon_deg)
        & (catalog["lon"] <= clon + dlon_deg)
    )
    nearby = catalog.loc[mask].copy()
    if nearby.empty:
        return np.array([])

    # Exact haversine distance
    nearby["dist_km"] = nearby.apply(
        lambda r: haversine(r["lon"], r["lat"], clon, city_lat), axis=1
    )
    within = nearby[nearby["dist_km"] <= radius_km]
    if within.empty:
        return np.array([])

    # Max wind per storm
    storm_max = within.groupby("global_storm_uid")["wind"].max()
    storm_max = storm_max[storm_max >= min_wind]
    return storm_max.values


def return_periods_at_city(
    catalog: list[str],
    n_years: float,
    city_lat: float,
    city_lon: float,
    radius_km: float = 111.0,
    min_wind: float = 18.0,
) -> pd.DataFrame:
    """
    Compute empirical wind-speed return periods at a single city using
    the Weibull plotting position (consistent with IH-STORM / STORM
    methodology).

    Parameters
    ----------
    catalog : full catalog DataFrame
    n_years : number of simulated years
    city_lat, city_lon : city coordinates (lon can be negative)
    radius_km : search radius
    min_wind : minimum wind threshold (m/s)

    Returns
    -------
    DataFrame with columns [wind_ms, rank, exceedance_prob, return_period_yr]
    sorted by descending wind.
    """
    winds = _extract_city_max_winds(catalog, city_lat, city_lon, radius_km, min_wind)
    if len(winds) == 0:
        return pd.DataFrame(
            columns=["wind_ms", "rank", "exceedance_prob", "return_period_yr"]
        )

    winds_sorted = np.sort(winds)[::-1]  # descending
    n_events = len(winds_sorted)
    ranks = np.arange(1, n_events + 1)

    # Weibull plotting position: P_exc = rank / (N + 1)
    # Then annualise: P_annual = P_exc * (N / n_years)
    weibull_prob = ranks / (n_events + 1.0)
    annual_exc = weibull_prob * (n_events / n_years)
    rp = 1.0 / annual_exc

    return pd.DataFrame(
        {
            "wind_ms": winds_sorted,
            "rank": ranks,
            "exceedance_prob": annual_exc,
            "return_period_yr": rp,
        }
    )


def return_periods_at_cities(
    catalog: list[str],
    n_years: float,
    cities: Optional[List[dict]] = None,
    radius_km: float = 111.0,
    min_wind: float = 18.0,
    target_rp: Optional[np.ndarray] = None,
    model: str | None = None,
) -> pd.DataFrame:
    """
    Compute return-period curves at multiple cities, optionally
    interpolated to specific return-period values.

    Parameters
    ----------
    catalog, n_years, radius_km, min_wind : see return_periods_at_city
    cities : list of dicts with keys "city", "lat", "lon"
             (defaults to DEFAULT_CITIES)
    target_rp : array of return periods (years) to interpolate onto.
                If None, returns the raw empirical RP table.
    model : parametric model for wind field to use (H08, H1980,...)

    Returns
    -------
    If target_rp is None:
        DataFrame with all city RP curves stacked, columns
        [city, wind_ms, rank, exceedance_prob, return_period_yr].
    If target_rp is given:
        DataFrame pivoted: rows = target_rp, columns = city names,
        values = interpolated wind speed (m/s).
    """
    if not HAS_CLIMADA:
        raise ImportError(
            "CLIMADA is required for Holland 2008 return periods. pip install climada"
        )
    if cities is None:
        cities = DEFAULT_CITIES
    if target_rp is not None:
        target_rp = np.asarray(target_rp)
    if model is None:
        model = "H08"
    rows = []
    interp_rows = []
    wide_frames = []

    for basin in np.unique([i["basin"] for i in cities]):
        basin_cities = [i for i in cities if i["basin"] == basin]
        city_df = (
            pd.DataFrame(basin_cities)
            .drop_duplicates(subset=["city", "lat", "lon"])
            .reset_index(drop=True)
        )

        centroids = Centroids(
            lat=city_df["lat"].to_numpy(),
            lon=city_df["lon"].to_numpy(),
            crs="EPSG:4326",
        )

        scenarios = [
            TCTracks.from_simulations_storm(i)
            for i in catalog
            if "_" + basin + "_" in i.split("/")[-1]
        ]
        pooled_tracks = pool_tctracks(scenarios, deduplicate=False)
        haz = TropCyclone.from_tracks(
            pooled_tracks,
            centroids=centroids,
            model=model,
            intensity_thres=min_wind,
        )
        haz.check()

        # Use the caller-provided n_years for frequency, not a hardcoded assumption.
        # For synthetic catalogs: n_years = len(files) * 1000 (set by caller)
        # For IBTrACS: n_years = effective observation years (e.g. 42, or phase-adjusted)
        eff_years = float(n_years)
        haz.frequency = np.full(haz.size, 1.0 / eff_years)
        haz.frequency_unit = "1/year"

        gdf_rp, _, _ = haz.local_exceedance_intensity(
            return_periods=target_rp,
            method="interpolate",
            min_intensity=0,
            log_frequency=True,
            log_intensity=True,
            # bin_decimals=bin_decimals,
        )

        rp_cols = [str(rp) for rp in target_rp]

        phase_wide = pd.concat(
            [
                pd.DataFrame({"scenario": [basin] * len(city_df)}),
                city_df[["city", "lat", "lon"]].reset_index(drop=True),
                gdf_rp[rp_cols].reset_index(drop=True),
            ],
            axis=1,
        )
        phase_wide.columns = ["basin", "city", "lat", "lon"] + target_rp.tolist()

        wide_frames.append(phase_wide)

    curves_wide = pd.concat(wide_frames, ignore_index=True)
    print(curves_wide)
    curves_long = curves_wide.melt(
        id_vars=["basin", "city", "lat", "lon"],
        var_name="return_period",
        value_name="wind_ms",
    )
    curves_long["return_period"] = curves_long["return_period"].astype(float)
    curves_long = curves_long.sort_values(["city", "return_period"]).reset_index(
        drop=True
    )

    return curves_long


def return_periods_all_catalog(
    phase_folders: Dict[str, str],
    basin: str,
    total_years: int = 10_000,
    cities: Optional[List[dict]] = None,
    target_rp: Optional[np.ndarray] = None,
    model: str = "H08",
    min_wind: float = 18.0,
    oni_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
    file_pattern: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute return periods for the composite ALL catalog using
    ENSO-weighted year sampling via CLIMADA Holland 2008.

    Instead of loading all years from all files, this function
    replicates the §5.3 weighting:
      1. Compute Y_ph = f_ph × total_years for each ENSO phase
      2. For each phase, discover STORM files and randomly select
         Y_ph year-indices across those files
      3. Load only the selected years via
         TCTracks.from_simulations_storm(path, years=[...])
      4. Pool tracks, set frequency = 1/total_years per event
      5. Run CLIMADA TropCyclone.from_tracks() with Holland 2008

    Parameters
    ----------
    phase_folders : {"EN": "/path/", "NEU": "/path/", "LN": "/path/"}
    basin : basin code
    total_years : target ALL catalog length
    cities : list of dicts with "city", "lat", "lon", "basin"
    target_rp : return period values (years)
    model : parametric wind model (default "H08")
    seed : random seed for reproducible sampling

    Returns
    -------
    curves_long : DataFrame with columns
        [basin, city, lat, lon, return_period, wind_ms]
    """
    if not HAS_CLIMADA:
        raise ImportError("CLIMADA required for return_periods_all_catalog")

    if cities is None:
        cities = [c for c in DEFAULT_CITIES if c.get("basin") == basin]
    if target_rp is None:
        target_rp = np.array([2, 5, 10, 25, 50, 100, 250, 500, 1000])
    target_rp = np.asarray(target_rp)

    fracs = compute_phase_fractions(basin, oni_df)
    rng = np.random.default_rng(seed)

    # ── Compute years to draw per phase ──
    raw = {ph: fracs[ph] * total_years for ph in ("EN", "NEU", "LN")}
    phase_n = {ph: int(np.round(v)) for ph, v in raw.items()}
    diff = total_years - sum(phase_n.values())
    if diff != 0:
        remainders = {ph: raw[ph] - int(raw[ph]) for ph in phase_n}
        for i, ph in enumerate(
            sorted(remainders, key=remainders.get, reverse=(diff > 0))
        ):
            if i >= abs(diff):
                break
            phase_n[ph] += int(np.sign(diff))

    # ── Discover files and sample years ──
    all_tracks = []
    for ph in ("EN", "NEU", "LN"):
        n_draw = phase_n[ph]
        if n_draw == 0:
            continue

        files = _find_catalog_files(phase_folders[ph], basin, ph, file_pattern)
        if not files:
            warnings.warn(f"No files for phase {ph} in {phase_folders[ph]}")
            continue

        # Each file has 1000 years (indices 0–999).
        # Build pool of (file_path, local_year_idx) pairs.
        year_pool = [(f, y) for f in files for y in range(1000)]
        if len(year_pool) < n_draw:
            warnings.warn(
                f"Phase {ph}: need {n_draw} years but only "
                f"{len(year_pool)} available; sampling with replacement."
            )
            chosen_idx = rng.choice(len(year_pool), size=n_draw, replace=True)
        else:
            chosen_idx = rng.choice(len(year_pool), size=n_draw, replace=False)

        # Group selected years by file
        file_years: Dict[str, list] = {}
        for idx in chosen_idx:
            fpath, yr = year_pool[idx]
            file_years.setdefault(fpath, []).append(yr)

        # Load via CLIMADA with year filtering
        for fpath, years_list in file_years.items():
            tc = TCTracks.from_simulations_storm(fpath, years=sorted(years_list))
            all_tracks.extend(tc.data)

    if not all_tracks:
        warnings.warn("No tracks loaded for ALL catalog RP computation")
        return pd.DataFrame()

    pooled = TCTracks(data=all_tracks)
    print(f"  ALL RP: {len(pooled.data)} tracks from {total_years} effective years")

    # ── Compute hazard per basin ──
    basin_cities = [c for c in cities if c.get("basin") == basin]
    if not basin_cities:
        basin_cities = cities  # use all if no basin tag

    city_df = (
        pd.DataFrame(basin_cities)
        .drop_duplicates(subset=["city", "lat", "lon"])
        .reset_index(drop=True)
    )
    centroids = Centroids(
        lat=city_df["lat"].to_numpy(),
        lon=city_df["lon"].to_numpy(),
        crs="EPSG:4326",
    )

    haz = TropCyclone.from_tracks(
        pooled,
        centroids=centroids,
        model=model,
        intensity_thres=min_wind,
    )
    haz.check()
    haz.frequency = np.full(haz.size, 1.0 / float(total_years))
    haz.frequency_unit = "1/year"

    gdf_rp, _, _ = haz.local_exceedance_intensity(
        return_periods=target_rp,
        method="interpolate",
        min_intensity=0,
        log_frequency=True,
        log_intensity=True,
    )

    rp_cols = [str(rp) for rp in target_rp]
    wide = pd.concat(
        [
            pd.DataFrame({"basin": [basin] * len(city_df)}),
            city_df[["city", "lat", "lon"]].reset_index(drop=True),
            gdf_rp[rp_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    wide.columns = ["basin", "city", "lat", "lon"] + target_rp.tolist()

    curves_long = wide.melt(
        id_vars=["basin", "city", "lat", "lon"],
        var_name="return_period",
        value_name="wind_ms",
    )
    curves_long["return_period"] = curves_long["return_period"].astype(float)
    curves_long = curves_long.sort_values(["city", "return_period"]).reset_index(
        drop=True
    )

    return curves_long


# ═══════════════════════════════════════════════════════════════════════
# ACE  (bonus metric, used in your notebook already)
# ═══════════════════════════════════════════════════════════════════════


def ace_density(
    catalog: pd.DataFrame,
    n_years: float,
    resolution: float = 2.0,
    lon_range: Optional[Tuple[float, float]] = None,
    lat_range: Optional[Tuple[float, float]] = None,
    dt_hours: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulated Cyclone Energy density (10^{-4} kn² per year per cell).
    ACE at each fix = V_max² × dt/6  (with V in knots, dt in hours).

    10-min m/s → 1-min knots: multiply by 1.9438  (≈ kn per m/s),
    then Saffir–Simpson convention factor ~1.0 (STORM already stores 10-min).
    We follow the standard: keep 10-min m/s, square, scale by 1e-4.
    """
    lons = catalog["lon"].values
    lats = catalog["lat"].values
    # ACE contribution per fix: V² * (dt / 6) in (m/s)²·h
    # Convert 10-min m/s to knots: ×1.9438
    wind_kn = catalog["wind"].values * 1.9438
    ace_fix = wind_kn**2 * (dt_hours / 6.0)  # ×10^{-4} below

    if lon_range is None:
        lon_range = (np.floor(lons.min()), np.ceil(lons.max()))
    if lat_range is None:
        lat_range = (np.floor(lats.min()), np.ceil(lats.max()))

    lon_edges = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    lat_edges = np.arange(lat_range[0], lat_range[1] + resolution, resolution)

    H, _, _ = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges], weights=ace_fix)
    density = H.T * 1e-4 / n_years  # 10^{-4} kn² per year per cell
    return density, lon_edges, lat_edges


# ═══════════════════════════════════════════════════════════════════════
# DENSITY GRID & LIFETIME EXPORT HELPERS
# ═══════════════════════════════════════════════════════════════════════


def export_density_csv(
    density: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    filepath: str,
    value_label: str = "value",
):
    """
    Export a 2-D density grid to a long-format CSV with columns
    [lat_center, lon_center, <value_label>].

    Suitable for reloading in Python/R for overlaying difference maps.
    """
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    rows = []
    for i, la in enumerate(lat_c):
        for j, lo in enumerate(lon_c):
            rows.append(
                {"lat_center": la, "lon_center": lo, value_label: density[i, j]}
            )
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, float_format="%.6f")
    return df


def export_all_densities(
    catalog: pd.DataFrame,
    n_years: float,
    outdir: str,
    label: str = "",
    basin: Optional[str] = None,
    grid_resolution: float = 1.0,
    ace_resolution: float = 2.0,
):
    """
    Compute and export genesis, track, and ACE density grids as CSVs.

    Files written:
        {outdir}/genesis_density_{label}.csv
        {outdir}/track_density_{label}.csv
        {outdir}/ace_density_{label}.csv

    Parameters
    ----------
    catalog : loaded STORM-format catalog
    n_years : total simulated years
    outdir : output directory
    label : suffix for filenames (e.g. "S3_NA_LN")
    basin : basin code for setting grid bounds
    grid_resolution : degrees for genesis & track grids
    ace_resolution : degrees for ACE grid
    """
    os.makedirs(outdir, exist_ok=True)

    lon_range = lat_range = None
    if basin and basin in BASIN_BOUNDS:
        b = BASIN_BOUNDS[basin]
        lon_range = b["lon"]
        lat_range = b["lat"]

    # Genesis density
    gd, glon, glat = genesis_density(
        catalog, n_years, grid_resolution, lon_range, lat_range
    )
    export_density_csv(
        gd,
        glon,
        glat,
        os.path.join(outdir, f"genesis_density_{label}.csv"),
        "genesis_per_10kyr",
    )

    # Track density
    td, tlon, tlat = track_density(
        catalog, n_years, grid_resolution, lon_range, lat_range
    )
    export_density_csv(
        td,
        tlon,
        tlat,
        os.path.join(outdir, f"track_density_{label}.csv"),
        "fixes_per_yr",
    )

    # ACE density
    ad, alon, alat = ace_density(catalog, n_years, ace_resolution, lon_range, lat_range)
    export_density_csv(
        ad,
        alon,
        alat,
        os.path.join(outdir, f"ace_density_{label}.csv"),
        "ace_1e4_kn2_per_yr",
    )

    print(f"  Exported density grids to {outdir}/ (*_{label}.csv)")
    return {
        "genesis": (gd, glon, glat),
        "track": (td, tlon, tlat),
        "ace": (ad, alon, alat),
    }


# ═══════════════════════════════════════════════════════════════════════
# COMBINED METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════


def compute_all_metrics(
    catalog: pd.DataFrame,
    n_years: float,
    file_paths: list[str],
    basin: Optional[str] = None,
    reference: Optional[pd.DataFrame] = None,
    ref_n_years: Optional[int] = None,
    cities: Optional[List[dict]] = None,
    target_rp: Optional[np.ndarray] = None,
    grid_resolution: float = 1.0,
) -> dict:
    """
    Compute all Section 5.4 metrics in one call.

    Parameters
    ----------
    catalog : loaded catalog DataFrame (output of load_catalog)
    n_years : total simulated years
    file_paths : path to STORM files, for Return Periods
    basin : basin code (used to set geographic bounds for grids)
    reference : optional IBTrACS catalog for KS comparisons
    ref_n_years : years spanned by reference
    cities : list of city dicts for return periods
    target_rp : return period values to interpolate (years)
    grid_resolution : degrees for density grids

    Returns
    -------
    dict of dicts, keyed by metric name.
    """
    # Set grid bounds from basin if available
    lon_range = lat_range = None
    if basin and basin in BASIN_BOUNDS:
        b = BASIN_BOUNDS[basin]
        lon_range = b["lon"]
        lat_range = b["lat"]

    # Filter cities to basin if specified
    if cities is None and basin:
        cities = [c for c in DEFAULT_CITIES if c.get("basin") == basin]
    elif cities is None:
        cities = DEFAULT_CITIES

    results = {}

    # §5.4.1
    results["genesis_count"] = annual_genesis_count(catalog, n_years)

    gd, glon, glat = genesis_density(
        catalog, n_years, grid_resolution, lon_range, lat_range
    )
    results["genesis_density"] = {"density": gd, "lon_edges": glon, "lat_edges": glat}

    td, tlon, tlat = track_density(
        catalog, n_years, grid_resolution, lon_range, lat_range
    )
    results["track_density"] = {"density": td, "lon_edges": tlon, "lat_edges": tlat}

    results["intensity"] = intensity_distributions(catalog, reference)
    results["lifetime"] = average_lifetime(catalog)

    # §5.4.2
    results["landfall_counts"] = landfall_counts(catalog, n_years)
    results["landfall_intensity"] = landfall_intensity(catalog, reference)

    if cities:
        if target_rp is None:
            target_rp = np.array([2, 5, 10, 25, 50, 100, 250, 500, 1000])
        results["return_periods"] = return_periods_at_cities(
            file_paths,
            n_years,
            cities,
            target_rp=target_rp,
        )

    # Bonus
    ace_res = 2.0
    ad, alon, alat = ace_density(catalog, n_years, ace_res, lon_range, lat_range)
    results["ace_density"] = {"density": ad, "lon_edges": alon, "lat_edges": alat}

    return results


# ═══════════════════════════════════════════════════════════════════════
# MULTI-CANDIDATE COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def compare_candidates(
    candidates: Dict[str, dict],
    basin: str,
    reference: Optional[pd.DataFrame] = None,
    ref_n_years: Optional[int] = None,
    cities: Optional[List[dict]] = None,
    target_rp: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build a summary comparison table across multiple candidates.

    Parameters
    ----------
    candidates : dict of {name: {"folder": str, "phase": str or None, "n_years": int}}
        Example:
        {
            "B0_ALL": {"folder": "/data/B0/", "phase": None, "n_years": 10000},
            "S3_LN":  {"folder": "/data/S3/LN/", "phase": "LN", "n_years": 10000},
        }
    basin, reference, ref_n_years, cities, target_rp : forwarded to compute_all_metrics

    Returns
    -------
    pd.DataFrame with one row per candidate, columns = scalar metrics.
    """
    summary_rows = []

    for name, spec in candidates.items():
        cat, files = load_catalog(
            spec["folder"],
            basin,
            phase=spec.get("phase"),
            file_pattern=spec.get("file_pattern"),
        )
        n_years = spec["n_years"]

        m = compute_all_metrics(
            cat,
            n_years,
            basin=basin,
            reference=reference,
            ref_n_years=ref_n_years,
            cities=cities,
            target_rp=target_rp,
            file_paths=files,
        )

        row = {"candidate": name}
        # Genesis
        row["genesis_mean"] = m["genesis_count"]["mean"]
        row["genesis_std"] = m["genesis_count"]["std"]
        row["total_storms"] = m["genesis_count"]["total_storms"]
        # Lifetime
        row["lifetime_mean_steps"] = m["lifetime"]["mean_steps"]
        row["lifetime_mean_hours"] = m["lifetime"]["mean_hours"]
        # Landfall
        row["landfall_annual_mean"] = m["landfall_counts"]["annual_mean"]
        row["landfall_annual_std"] = m["landfall_counts"]["annual_std"]
        # KS tests (if reference was provided)
        if "ks_pmin" in m["intensity"]:
            row["ks_pmin"] = m["intensity"]["ks_pmin"]
            row["ks_pmin_pval"] = m["intensity"]["ks_pmin_pvalue"]
            row["ks_vmax"] = m["intensity"]["ks_vmax"]
            row["ks_vmax_pval"] = m["intensity"]["ks_vmax_pvalue"]
        if "ks_lf_pmin" in m["landfall_intensity"]:
            row["ks_lf_pmin"] = m["landfall_intensity"]["ks_lf_pmin"]
            row["ks_lf_vmax"] = m["landfall_intensity"]["ks_lf_vmax"]

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


# ═══════════════════════════════════════════════════════════════════════
# PLOTTING  (requires matplotlib; cartopy optional but recommended)
# ═══════════════════════════════════════════════════════════════════════

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Expanded plot extents for cartopy (slightly wider than BASIN_BOUNDS)
BASIN_PLOT_EXTENT = {
    "EP": [175, 290, 0, 65],
    "NA": [250, 365, 0, 65],
    "NI": [25, 105, 0, 40],
    "SI": [5, 140, -65, 0],
    "SP": [130, 245, -65, 0],
    "WP": [95, 185, 0, 65],
}

PHASE_COLORS = {
    "EN": "#E53935",
    "NEU": "#212121",
    "LN": "#1E88E5",
    "ALL": "#4CAF50",
    "IBTrACS": "#F9A825",
}
PHASE_LABELS = {
    "EN": "El Niño",
    "NEU": "Neutral",
    "LN": "La Niña",
    "ALL": "All phases",
    "IBTrACS": "IBTrACS",
}

# Saffir-Simpson thresholds in 10-min m/s (for annotations)
SS_THRESH_10MIN = {"Cat 1": 33, "Cat 2": 43, "Cat 3": 50, "Cat 4": 58, "Cat 5": 70}


# ── Map axis helpers ──


def _make_map_axes(n_panels, basin, figwidth=16):
    """Create a row of map subplots with optional cartopy."""
    import matplotlib.pyplot as plt

    ncols = min(4, n_panels) if n_panels > 2 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    height = figwidth / ncols * 0.6 * nrows
    kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figwidth, height), subplot_kw=kw, squeeze=False
    )
    axes_flat = axes.flatten()
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)
    return fig, axes_flat[:n_panels]


def _dress_map_axis(ax, basin):
    """Add coastlines, land shading and set extent."""
    extent = BASIN_PLOT_EXTENT.get(basin)
    if HAS_CARTOPY:
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    elif extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])


def _pcolormesh(ax, grid, lat_edges, lon_edges, **kwargs):
    """pcolormesh wrapper handling cartopy transform."""
    import matplotlib.colors as mcolors

    grid_plot = np.ma.masked_where(np.abs(grid) < 1e-12, grid)
    transform_kw = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    return ax.pcolormesh(
        lon_edges, lat_edges, grid_plot, shading="flat", **transform_kw, **kwargs
    )


def _compute_map_grid(df, basin, mode, resolution, n_years):
    """
    Compute a 2-D grid from catalog data for map plotting.

    Parameters
    ----------
    df : catalog DataFrame
    mode : "genesis" | "track" | "ace"
    resolution, n_years : grid cell size, normalization
    Returns
    -------
    grid, lat_edges, lon_edges
    """
    lat0, lat1, lon0, lon1 = (
        BASIN_BOUNDS[basin]["lat"][0],
        BASIN_BOUNDS[basin]["lat"][1],
        BASIN_BOUNDS[basin]["lon"][0],
        BASIN_BOUNDS[basin]["lon"][1],
    )
    lat_edges = np.arange(lat0, lat1 + resolution, resolution)
    lon_edges = np.arange(lon0, lon1 + resolution, resolution)
    wind_col = "wind" if "wind" in df.columns else "wind_ms"

    if mode == "genesis":
        if "timestep" in df.columns:
            sub = df[df["timestep"] == 0]
        elif "global_storm_uid" in df.columns:
            sub = df.drop_duplicates("global_storm_uid", keep="first")
        else:
            sub = df.drop_duplicates("storm_id", keep="first")
        grid, _, _ = np.histogram2d(
            sub["lat"].values, sub["lon"].values, bins=[lat_edges, lon_edges]
        )
    elif mode == "ace":
        wind_kt = df[wind_col].values * 1.94384
        weights = wind_kt**2 * 1e-4
        grid, _, _ = np.histogram2d(
            df["lat"].values,
            df["lon"].values,
            bins=[lat_edges, lon_edges],
            weights=weights,
        )
    else:  # track
        grid, _, _ = np.histogram2d(
            df["lat"].values, df["lon"].values, bins=[lat_edges, lon_edges]
        )
    return grid / max(n_years, 1.0), lat_edges, lon_edges


# ── Absolute density panel maps ──


def plot_density_panels(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    n_years_dict: Dict[str, float],
    mode: str = "track",
    resolution: float = 1.0,
    outdir: str = ".",
    cmap: Optional[str] = None,
    figwidth: float = 16,
):
    """
    Side-by-side density maps for multiple datasets/phases.

    Parameters
    ----------
    datasets : {"EN": df, "NEU": df, "LN": df, "IBTrACS": df, ...}
    n_years_dict : {"EN": 10000, "IBTrACS": 42, ...}
    mode : "genesis", "track", or "ace"
    """
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = {"genesis": "YlOrRd", "track": "YlGnBu", "ace": "hot_r"}.get(
            mode, "YlOrRd"
        )
    unit = {
        "genesis": "storms/yr/cell",
        "track": "fixes/yr/cell",
        "ace": "ACE/yr/cell",
    }.get(mode, "")
    mode_label = {
        "genesis": "Genesis Density",
        "track": "Track Density",
        "ace": "ACE Density",
    }.get(mode, mode)

    n = len(datasets)
    fig, axes = _make_map_axes(n, basin, figwidth)

    grids = {}
    for label, df in datasets.items():
        g, lat_e, lon_e = _compute_map_grid(
            df, basin, mode, resolution, n_years_dict.get(label, 1.0)
        )
        grids[label] = g

    all_vals = np.concatenate([g[g > 0] for g in grids.values() if np.any(g > 0)])
    vmax = np.percentile(all_vals, 95) if len(all_vals) > 0 else 1.0

    for ax, label in zip(axes, datasets.keys()):
        _dress_map_axis(ax, basin)
        im = _pcolormesh(ax, grids[label], lat_e, lon_e, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(PHASE_LABELS.get(label, label), fontsize=11)

    plt.colorbar(im, ax=list(axes), shrink=0.6, label=unit, pad=0.02)
    fig.suptitle(f"{mode_label} — {basin} ({resolution}° grid)", fontsize=14, y=1.02)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"validation_{mode}_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ── Difference maps (3-row triplet) ──


def plot_difference_triplet(
    datasets_new: Dict[str, pd.DataFrame],
    dataset_ref: pd.DataFrame,
    basin: str,
    n_years_new: float,
    n_years_ref: float,
    mode: str = "track",
    resolution: float = 2.0,
    datasets_old: Optional[Dict[str, pd.DataFrame]] = None,
    n_years_old: Optional[float] = None,
    outdir: str = ".",
    diff_cmap: str = "RdBu_r",
):
    """
    Three-row publication figure:
      Row 0: Absolute panels  (IBTrACS + per-phase new [+ per-phase old])
      Row 1: New − IBTrACS    (per phase)
      Row 2: New − Old        (per phase, if old provided)
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    abs_cmap = {"genesis": "YlOrRd", "track": "YlGnBu", "ace": "hot_r"}.get(
        mode, "YlOrRd"
    )
    unit = {
        "genesis": "storms/yr/cell",
        "track": "fixes/yr/cell",
        "ace": "ACE/yr/cell",
    }.get(mode, "")
    mode_label = {
        "genesis": "Genesis Density",
        "track": "Track Density",
        "ace": "ACE Density",
    }.get(mode, mode)

    phases = [ph for ph in ["EN", "NEU", "LN"] if ph in datasets_new]
    has_old = datasets_old is not None and len(datasets_old) > 0

    # Compute grids
    grids = {}
    g_ref, lat_e, lon_e = _compute_map_grid(
        dataset_ref, basin, mode, resolution, n_years_ref
    )
    grids["ibtracs"] = g_ref
    for ph in phases:
        g, _, _ = _compute_map_grid(
            datasets_new[ph], basin, mode, resolution, n_years_new
        )
        grids[f"new_{ph}"] = g
    if has_old:
        for ph in phases:
            if ph in datasets_old:
                g, _, _ = _compute_map_grid(
                    datasets_old[ph], basin, mode, resolution, n_years_old
                )
                grids[f"old_{ph}"] = g

    # Layout
    abs_panels = [("ibtracs", "IBTrACS")]
    for ph in phases:
        abs_panels.append((f"new_{ph}", f"SIENA {PHASE_LABELS.get(ph, ph)}"))
    if has_old:
        for ph in phases:
            if f"old_{ph}" in grids:
                abs_panels.append((f"old_{ph}", f"Old {PHASE_LABELS.get(ph, ph)}"))

    ncols = max(len(abs_panels), len(phases))
    nrows = 2 + (1 if has_old else 0)

    kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.2 * nrows + 0.8),
        subplot_kw=kw,
        squeeze=False,
    )

    # Row 0: Absolute
    abs_grids = [grids[k] for k, _ in abs_panels if k in grids]
    all_pos = (
        np.concatenate([g[g > 0] for g in abs_grids]) if abs_grids else np.array([1])
    )
    vmax_abs = np.percentile(all_pos, 99) if len(all_pos) > 0 else 1.0
    for i, (key, label) in enumerate(abs_panels):
        if i >= ncols:
            break
        _dress_map_axis(axes[0, i], basin)
        if key in grids:
            im0 = _pcolormesh(
                axes[0, i],
                grids[key],
                lat_e,
                lon_e,
                cmap=abs_cmap,
                vmin=0,
                vmax=vmax_abs,
            )
        axes[0, i].set_title(label, fontsize=9)
    plt.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.6, label=unit, pad=0.02)
    for i in range(len(abs_panels), ncols):
        axes[0, i].set_visible(False)

    # Row 1: New − IBTrACS
    diff1 = []
    for ph in phases:
        key = f"new_{ph}"
        if key in grids:
            shape = (
                min(grids[key].shape[0], g_ref.shape[0]),
                min(grids[key].shape[1], g_ref.shape[1]),
            )
            d = grids[key][: shape[0], : shape[1]] - g_ref[: shape[0], : shape[1]]
            diff1.append((d, f"SIENA−IBTrACS ({PHASE_LABELS.get(ph, ph)})"))

    vlim1 = (
        max(
            (np.percentile(np.abs(d[d != 0]), 99) if np.any(d != 0) else 0.01)
            for d, _ in diff1
        )
        if diff1
        else 0.01
    )
    vlim1 = max(vlim1, 0.01)
    for i, (d, label) in enumerate(diff1):
        _dress_map_axis(axes[1, i], basin)
        im1 = _pcolormesh(
            axes[1, i],
            d,
            lat_e,
            lon_e,
            cmap=diff_cmap,
            norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim1, vmax=vlim1),
        )
        axes[1, i].set_title(label, fontsize=9)
    plt.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.6, label=f"Δ {unit}", pad=0.02)
    for i in range(len(diff1), ncols):
        axes[1, i].set_visible(False)

    # Row 2: New − Old
    if has_old:
        diff2 = []
        for ph in phases:
            k_new, k_old = f"new_{ph}", f"old_{ph}"
            if k_new in grids and k_old in grids:
                shape = (
                    min(grids[k_new].shape[0], grids[k_old].shape[0]),
                    min(grids[k_new].shape[1], grids[k_old].shape[1]),
                )
                d = (
                    grids[k_new][: shape[0], : shape[1]]
                    - grids[k_old][: shape[0], : shape[1]]
                )
                diff2.append((d, f"SIENA−Old ({PHASE_LABELS.get(ph, ph)})"))
        if diff2:
            vlim2 = max(
                (np.percentile(np.abs(d[d != 0]), 99) if np.any(d != 0) else 0.01)
                for d, _ in diff2
            )
            vlim2 = max(vlim2, 0.01)
            for i, (d, label) in enumerate(diff2):
                _dress_map_axis(axes[2, i], basin)
                im2 = _pcolormesh(
                    axes[2, i],
                    d,
                    lat_e,
                    lon_e,
                    cmap=diff_cmap,
                    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim2, vmax=vlim2),
                )
                axes[2, i].set_title(label, fontsize=9)
            plt.colorbar(
                im2, ax=axes[2, :].tolist(), shrink=0.6, label=f"Δ {unit}", pad=0.02
            )
        for i in range(len(diff2) if diff2 else 0, ncols):
            axes[2, i].set_visible(False)

    fig.suptitle(f"{mode_label} — {basin} ({resolution}° grid)", fontsize=14, y=1.01)
    os.makedirs(outdir, exist_ok=True)
    safe = mode_label.lower().replace(" ", "_")
    outpath = os.path.join(outdir, f"diff_{safe}_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ── LMI distribution ──


def plot_lmi_distribution(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    outdir: str = ".",
):
    """
    Per-storm lifetime maximum intensity histogram, overlaid.
    datasets : {"EN": catalog_df, "IBTrACS": catalog_df, ...}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    wind_col = "wind" if "wind" in list(datasets.values())[0].columns else "wind_ms"
    bins = np.arange(15, 95, 2.5)

    for label, df in datasets.items():
        uid_col = (
            "global_storm_uid" if "global_storm_uid" in df.columns else "storm_uid"
        )
        if uid_col in df.columns:
            lmi = df.groupby(uid_col)[wind_col].max().dropna()
        else:
            lmi = _per_storm_agg(df)["vmax"].dropna()
        color = PHASE_COLORS.get(label, "gray")
        display = PHASE_LABELS.get(label, label)
        ax.hist(
            lmi.values,
            bins=bins,
            density=True,
            alpha=0.35,
            label=display,
            color=color,
            edgecolor=color,
            linewidth=0.8,
        )
        counts, edges = np.histogram(lmi.values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, color=color, linewidth=2)

    for cat_label, v in SS_THRESH_10MIN.items():
        ax.axvline(v, color="gray", ls=":", alpha=0.5, lw=0.8)
        ax.text(
            v + 0.5,
            ax.get_ylim()[1] * 0.95,
            cat_label,
            fontsize=7,
            color="gray",
            va="top",
            rotation=90,
        )

    ax.set_xlabel("Lifetime Maximum Intensity [m/s, 10-min sustained]")
    ax.set_ylabel("Probability density")
    ax.set_title(f"LMI Distribution — {basin}")
    ax.legend(frameon=False)
    ax.set_xlim(15, 85)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"validation_lmi_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ── Wind-pressure scatter ──


def plot_wind_pressure_scatter(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    outdir: str = ".",
    max_points: int = 50_000,
):
    """Wind vs pressure scatter with WPR reference curves overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    wind_col = "wind" if "wind" in list(datasets.values())[0].columns else "wind_ms"

    for label, df in datasets.items():
        mask = (df["pressure"] > 850) & (df["pressure"] < 1020) & (df[wind_col] > 10)
        sub = df[mask]
        if len(sub) > max_points:
            sub = sub.sample(max_points, random_state=42)
        color = PHASE_COLORS.get(label, "gray")
        display = PHASE_LABELS.get(label, label)
        ax.scatter(
            sub["pressure"],
            sub[wind_col],
            s=1,
            alpha=0.08,
            color=color,
            rasterized=True,
        )
        ax.scatter([], [], s=30, color=color, label=display, alpha=0.8)

    dp = np.linspace(1, 120, 200)
    for a, b, ls, wpr_lbl in [
        (0.7, 0.62, "-", r"WPR: V=0.70·ΔP$^{0.62}$"),
        (0.6, 0.65, "--", r"WPR: V=0.60·ΔP$^{0.65}$"),
    ]:
        ax.plot(
            1013 - dp, a * dp**b, ls, color="black", lw=1.5, alpha=0.6, label=wpr_lbl
        )

    ax.set_xlabel("Central pressure [hPa]")
    ax.set_ylabel("Maximum wind speed [m/s, 10-min]")
    ax.set_title(f"Wind-Pressure Relationship — {basin}")
    ax.set_xlim(880, 1020)
    ax.set_ylim(10, 85)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"validation_wpr_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ── Intensity CDFs per phase ──


def plot_intensity_cdfs(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    outdir: str = ".",
):
    """ECDF of lifetime Pmin and Vmax, one curve per dataset/phase."""
    import matplotlib.pyplot as plt

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    wind_col = "wind" if "wind" in list(datasets.values())[0].columns else "wind_ms"
    ecdf = lambda x: np.arange(1, len(x) + 1) / len(x)

    for label, df in datasets.items():
        storms = _per_storm_agg(df)
        color = PHASE_COLORS.get(label, "gray")
        display = PHASE_LABELS.get(label, label)

        pmin = np.sort(storms["pmin"].dropna().values)
        vmax_vals = np.sort(storms["vmax"].dropna().values)
        ax_p.plot(pmin, ecdf(pmin), color=color, label=display, lw=1.5)
        ax_v.plot(vmax_vals, ecdf(vmax_vals), color=color, label=display, lw=1.5)

    ax_p.set_xlabel("Minimum central pressure (hPa)")
    ax_p.set_ylabel("CDF")
    ax_p.set_title(f"Lifetime Pmin — {basin}")
    ax_p.legend(frameon=False, fontsize=8)
    ax_p.grid(True, alpha=0.2)

    ax_v.set_xlabel("Maximum wind speed (m/s, 10-min)")
    ax_v.set_ylabel("CDF")
    ax_v.set_title(f"Lifetime Vmax — {basin}")
    ax_v.legend(frameon=False, fontsize=8)
    ax_v.grid(True, alpha=0.2)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"validation_cdfs_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


def plot_return_period_curves(
    rp_table: pd.DataFrame,
    city_col: str = "city",
    ncols: int = 3,
    figsize_per_ax: Tuple[float, float] = (4.5, 3.5),
    **scatter_kw,
):
    """
    Plot RP curves from the raw output of return_periods_at_cities().

    Parameters
    ----------
    rp_table : DataFrame with columns [city, wind_ms, return_period_yr]
    """
    import matplotlib.pyplot as plt

    cities = rp_table[city_col].unique()
    n = len(cities)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )

    for i, city in enumerate(cities):
        ax = axes[i // ncols, i % ncols]
        sub = rp_table[rp_table[city_col] == city]
        rp_col = (
            "return_period" if "return_period" in sub.columns else "return_period_yr"
        )
        ax.scatter(sub[rp_col], sub["wind_ms"], s=8, **scatter_kw)
        ax.set_xscale("log")
        ax.set_xlabel("Return period (yr)")
        ax.set_ylabel("Wind speed (m/s)")
        ax.set_title(city)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_lifetime_distribution(
    lifetime_dfs: Dict[str, pd.DataFrame],
    kind: str = "duration_hours",
    bins: int = 50,
    figsize: Tuple[float, float] = (8, 5),
    density: bool = True,
):
    """
    Overlay lifetime histograms for multiple candidates.

    Parameters
    ----------
    lifetime_dfs : {"S3_LN": df, "B1_LN": df, "IBTrACS": df, ...}
        Each df is the output of lifetime_distribution().
    kind : column to plot — "duration_hours" or "n_steps"
    bins : number of histogram bins
    density : if True, plot as probability density (comparable across catalogs)

    Returns
    -------
    fig, ax
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    xlabel = "Duration (hours)" if kind == "duration_hours" else "Time steps (3h)"

    for label, df in lifetime_dfs.items():
        ax.hist(
            df[kind].dropna(),
            bins=bins,
            density=density,
            alpha=0.5,
            label=label,
            histtype="stepfilled",
            linewidth=1.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("TC Lifetime Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ── Generate full validation suite ──


def run_all_plots(
    datasets: Dict[str, pd.DataFrame],
    n_years_dict: Dict[str, float],
    basin: str,
    outdir: str = "validation_plots",
    resolution_density: float = 2.0,
    resolution_track: float = 1.0,
    dataset_ref: Optional[pd.DataFrame] = None,
    n_years_ref: Optional[float] = None,
    datasets_old: Optional[Dict[str, pd.DataFrame]] = None,
    n_years_old: Optional[float] = None,
):
    """
    Generate the full validation plot suite for one basin.

    Parameters
    ----------
    datasets : {"EN": df, "NEU": df, "LN": df}
    n_years_dict : {"EN": 10000, ...}
    dataset_ref : IBTrACS reference catalog (optional)
    datasets_old : old model catalogs for difference maps (optional)
    """
    os.makedirs(outdir, exist_ok=True)
    all_ds = dict(datasets)
    all_ny = dict(n_years_dict)
    if dataset_ref is not None:
        all_ds["IBTrACS"] = dataset_ref
        all_ny["IBTrACS"] = n_years_ref or 42

    for mode, res in [
        ("genesis", resolution_density),
        ("track", resolution_track),
        ("ace", resolution_density),
    ]:
        print(f"\n  Plotting {mode} density panels...")
        plot_density_panels(
            all_ds, basin, all_ny, mode=mode, resolution=res, outdir=outdir
        )

    if dataset_ref is not None:
        n_new = list(n_years_dict.values())[0]
        for mode, res in [
            ("genesis", resolution_density),
            ("track", resolution_track),
            ("ace", resolution_density),
        ]:
            print(f"  Plotting {mode} difference maps...")
            plot_difference_triplet(
                datasets,
                dataset_ref,
                basin,
                n_years_new=n_new,
                n_years_ref=n_years_ref or 42,
                mode=mode,
                resolution=res,
                datasets_old=datasets_old,
                n_years_old=n_years_old,
                outdir=outdir,
            )

    print("\n  Plotting LMI distribution...")
    plot_lmi_distribution(all_ds, basin, outdir=outdir)
    print("  Plotting wind-pressure scatter...")
    plot_wind_pressure_scatter(all_ds, basin, outdir=outdir)
    print("  Plotting intensity CDFs...")
    plot_intensity_cdfs(all_ds, basin, outdir=outdir)
    print(f"\n  All plots saved to: {outdir}/")


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: print a text summary
# ═══════════════════════════════════════════════════════════════════════


def print_summary(metrics: dict, label: str = "Catalog"):
    """Pretty-print the scalar metrics from compute_all_metrics()."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")

    g = metrics["genesis_count"]
    print(
        f"  Genesis count  : λ̂ = {g['mean']:.2f} ± {g['std']:.2f}  "
        f"(total {g['total_storms']} storms / {g['n_years']} yr)"
    )

    lt = metrics["lifetime"]
    print(
        f"  TC lifetime    : {lt['mean_steps']:.1f} steps "
        f"({lt['mean_hours']:.0f} h)  median {lt['median_steps']:.0f}"
    )

    lf = metrics["landfall_counts"]
    print(
        f"  Landfall rate  : {lf['annual_mean']:.2f} ± {lf['annual_std']:.2f} /yr  "
        f"(total {lf['total_landfalls']})"
    )

    inten = metrics["intensity"]
    if "ks_pmin" in inten:
        print(
            f"  KS (Pmin)      : D = {inten['ks_pmin']:.4f}  "
            f"p = {inten['ks_pmin_pvalue']:.2e}"
        )
        print(
            f"  KS (Vmax)      : D = {inten['ks_vmax']:.4f}  "
            f"p = {inten['ks_vmax_pvalue']:.2e}"
        )

    lfi = metrics["landfall_intensity"]
    if "ks_lf_pmin" in lfi:
        print(f"  KS landfall P  : D = {lfi['ks_lf_pmin']:.4f}")
        print(f"  KS landfall V  : D = {lfi['ks_lf_vmax']:.4f}")

    if "return_periods" in metrics:
        rp = metrics["return_periods"]
        if isinstance(rp, pd.DataFrame) and not rp.empty:
            print(f"\n  Return-period table (wind m/s at target RPs):")
            with pd.option_context("display.float_format", "{:.1f}".format):
                print(rp.to_string(index=True))
    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT  (optional)
# ═══════════════════════════════════════════════════════════════════════


def main():
    """
    Example usage from command line:

        python evaluation.py --folder /path/to/S3/LN/ --basin NA --phase LN --n_years 10000
    """
    import argparse

    parser = argparse.ArgumentParser(description="SIENA-IH-STORM Evaluation")
    parser.add_argument(
        "--folder", required=True, help="Directory with STORM .txt files"
    )
    parser.add_argument("--basin", required=True, choices=list(BASIN_ID_MAP.keys()))
    parser.add_argument(
        "--phase", default=None, help="ENSO phase (EN, NEU, LN) or omit for B0"
    )
    parser.add_argument("--n_years", type=int, default=10_000)
    parser.add_argument(
        "--ref_folder", default=None, help="Optional reference catalog folder"
    )
    parser.add_argument("--ref_phase", default=None)
    parser.add_argument("--ref_n_years", type=int, default=None)
    parser.add_argument(
        "--resolution", type=float, default=1.0, help="Grid resolution (deg)"
    )
    parser.add_argument("--out_csv", default=None, help="Save scalar summary to CSV")

    args = parser.parse_args()

    print(f"Loading catalog: {args.folder}  basin={args.basin}  phase={args.phase}")
    cat, files = load_catalog(args.folder, args.basin, phase=args.phase)
    print(f"  Loaded {len(cat)} rows, {cat['global_storm_uid'].nunique()} storms")

    ref = None
    if args.ref_folder:
        print(f"Loading reference: {args.ref_folder}")
        ref, _ = load_catalog(args.ref_folder, args.basin, phase=args.ref_phase)

    metrics = compute_all_metrics(
        cat,
        args.n_years,
        basin=args.basin,
        reference=ref,
        ref_n_years=args.ref_n_years,
        grid_resolution=args.resolution,
        file_paths=files,
    )

    print_summary(metrics, label=f"{args.basin} / {args.phase or 'ALL'}")

    if args.out_csv:
        row = {
            "basin": args.basin,
            "phase": args.phase or "ALL",
            "genesis_mean": metrics["genesis_count"]["mean"],
            "genesis_std": metrics["genesis_count"]["std"],
            "lifetime_hours": metrics["lifetime"]["mean_hours"],
            "landfall_mean": metrics["landfall_counts"]["annual_mean"],
        }
        if "ks_pmin" in metrics["intensity"]:
            row["ks_pmin"] = metrics["intensity"]["ks_pmin"]
            row["ks_vmax"] = metrics["intensity"]["ks_vmax"]
        pd.DataFrame([row]).to_csv(args.out_csv, index=False)
        print(f"  Saved to {args.out_csv}")


if __name__ == "__main__":
    main()
