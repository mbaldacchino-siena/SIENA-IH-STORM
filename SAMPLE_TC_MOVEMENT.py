# -*- coding: utf-8 -*-
"""
STORM module for simulation of the TC track.

SIENA VWS update:
  - Track model now uses VWS as a continuous physical covariate instead
    of ENSO phase dummies.
  - At runtime, phase-specific VWS climatology fields are loaded, so
    ENSO modulation flows through the actual shear environment.
  - TC_movement now requires `monthlist` to load the correct monthly
    VWS field for each storm.
"""

import numpy as np
from SELECT_BASIN import Basins_WMO
from siena_utils import (
    normalize_phase,
    load_monthly_field,
    load_field_with_year_fallback,
)
import os
import sys

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


_BASIN_BOUNDS = {
    "EP": (5, 60, 180, 285),
    "NA": (5, 60, 255, 360),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}

# ── VWS field cache (loaded once per (month, phase, year), reused across storms) ──
_VWS_CACHE = {}


def _load_vws_cached(month, phase=None, env_year=None):
    """Load a monthly VWS field, caching the result.
    If env_year is set, load the year-specific field (with phase-mean fallback).
    """
    key = (month, normalize_phase(phase), env_year)
    if key not in _VWS_CACHE:
        try:
            _VWS_CACHE[key] = load_field_with_year_fallback(
                dir_path, "VWS", month, phase=phase, env_year=env_year
            )
        except Exception:
            _VWS_CACHE[key] = None
    return _VWS_CACHE[key]


# ── O(1) grid index lookups (same as SAMPLE_TC_PRESSURE) ──
_LAT_GRID_STEP = 180.0 / 720.0  # 0.25 degrees
_LON_GRID_STEP = 360.0 / 1440.0


def _lat_to_idx(lat):
    """Grid runs from 90 (idx=0) to -90 (idx=720)."""
    return int(round((90.0 - lat) / _LAT_GRID_STEP))


def _lon_to_idx(lon):
    """Grid runs from 0 (idx=0) to 359.75 (idx=1439)."""
    lon = lon % 360.0
    return int(round(lon / _LON_GRID_STEP)) % 1440


def _get_vws_at_point(vws_field, lat, lon):
    """Look up VWS at a point. Returns 0.0 if field is None or index out of range."""
    if vws_field is None:
        return 0.0
    lat_idx = _lat_to_idx(lat)
    lon_idx = _lon_to_idx(lon)
    if 0 <= lat_idx < vws_field.shape[0] and 0 <= lon_idx < vws_field.shape[1]:
        val = float(vws_field[lat_idx, lon_idx])
        return val if np.isfinite(val) else 0.0
    return 0.0


def find_lat_index_bins(basin, lat):
    lat0 = _BASIN_BOUNDS[basin][0]
    base = 5
    return np.floor(float(lat - lat0) / base)


def Check_if_landfall(lat, lon, lat1, lon0, land_mask):
    x_coord = int(10 * (lon - lon0))
    y_coord = int(10 * (lat1 - lat))
    return land_mask[y_coord, x_coord]


def TC_movement(
    lon_genesis_list,
    lat_genesis_list,
    basin,
    monthlist=None,
    phase=None,
    env_years=None,
):
    """
    Simulate TC track movement.

    Parameters
    ----------
    lon_genesis_list : list of float, genesis longitudes
    lat_genesis_list : list of float, genesis latitudes
    basin : str, basin code
    monthlist : list of int, genesis month per storm (required for VWS lookup)
    phase : str or None, ENSO phase for loading phase-specific VWS field
    env_years : dict {month: year} or None. If set, each storm uses the
                historical year assigned to its genesis month.

    Returns
    -------
    latall, lonall, landfallall : lists of lists
    """
    phase = normalize_phase(phase)
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    basin_name = dict(zip(basins, [0, 1, 2, 3, 4, 5]))
    idx = basin_name[basin]
    constants_all = np.load(
        os.path.join(__location__, "TRACK_COEFFICIENTS.npy"),
        allow_pickle=True,
        encoding="latin1",
    ).item()
    land_mask = np.loadtxt(
        os.path.join(__location__, "Land_ocean_mask_" + str(basin) + ".txt")
    )
    constants = constants_all[idx]
    lat0, lat1, lon0, lon1 = _BASIN_BOUNDS[basin]
    latall = []
    lonall = []
    landfallall = []

    for storm_i, (lat_genesis, lon_genesis) in enumerate(
        zip(lat_genesis_list, lon_genesis_list)
    ):
        # ── Load VWS field for this storm's month ──
        storm_month = monthlist[storm_i] if monthlist is not None else None
        vws_field = None
        if storm_month is not None:
            env_year = env_years.get(storm_month) if env_years else None
            vws_field = _load_vws_cached(storm_month, phase=phase, env_year=env_year)

        latlijst = [lat_genesis]
        lonlijst = [lon_genesis]
        landfalllijst = [
            Check_if_landfall(lat_genesis, lon_genesis, lat1, lon0, land_mask)
        ]
        lat = lat_genesis
        lon = lon_genesis
        var = 0
        while var == 0:
            ind = int(find_lat_index_bins(basin, lat))
            ind = max(0, min(ind, len(constants) - 1))
            row = constants[ind]

            if len(row) >= 17:
                [
                    a_lat0,
                    a_lat1,
                    a_lat2,
                    b_vws_lat,  # VWS coeff for latitude (was g_en)
                    _unused1,  # was g_ln
                    b_lon0,
                    b_lon1,
                    b_vws_lon,  # VWS coeff for longitude (was d_en)
                    _unused2,  # was d_ln
                    Elatmu,
                    Elatstd,
                    Elonmu,
                    Elonstd,
                    Dlat0mu,
                    Dlat0std,
                    Dlon0mu,
                    Dlon0std,
                ] = row
            else:
                # Legacy 13-element format (no VWS, no ENSO)
                [
                    b_lon0,
                    b_lon1,
                    a_lat0,
                    a_lat1,
                    a_lat2,
                    Elatmu,
                    Elatstd,
                    Elonmu,
                    Elonstd,
                    Dlat0mu,
                    Dlat0std,
                    Dlon0mu,
                    Dlon0std,
                ] = row
                b_vws_lat = b_vws_lon = 0.0

            if len(latlijst) == 1:
                dlat0 = np.random.normal(Dlat0mu, Dlat0std)
                dlon0 = np.random.normal(Dlon0mu, Dlon0std)

            # ── VWS at current position ──
            vws = _get_vws_at_point(vws_field, lat, lon)

            # ── Latitude update ──
            dlat1 = float(a_lat0 + a_lat1 * dlat0 + a_lat2 * lat + b_vws_lat * vws)

            if basin in ("SP", "SI"):
                dlat1 = float(
                    dlat1
                    + (
                        -abs(np.random.normal(Elatmu, Elatstd))
                        if lat > -10
                        else np.random.normal(Elatmu, Elatstd)
                    )
                )
            else:
                dlat1 = float(
                    dlat1
                    + (
                        abs(np.random.normal(Elatmu, Elatstd))
                        if lat < 10
                        else np.random.normal(Elatmu, Elatstd)
                    )
                )

            # ── Longitude update ──
            dlon1 = float(
                b_lon0
                + b_lon1 * dlon0
                + b_vws_lon * vws
                + np.random.normal(Elonmu, Elonstd)
            )

            if np.abs(lat) >= 45 and dlon1 < 0.0:
                dlon1 = 0

            lat = round(dlat1 + lat, 1)
            lon = round(dlon1 + lon, 1)
            dlat0 = dlat1
            dlon0 = dlon1
            if lat <= lat1 - 0.1 and lat > lat0 and lon <= lon1 - 0.1 and lon > lon0:
                latlijst.append(lat)
                lonlijst.append(lon)
                landfalllijst.append(Check_if_landfall(lat, lon, lat1, lon0, land_mask))
            else:
                var = 1
        latall.append(latlijst)
        lonall.append(lonlijst)
        landfallall.append(landfalllijst)
    return latall, lonall, landfallall
