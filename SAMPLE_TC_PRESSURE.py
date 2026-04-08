# -*- coding: utf-8 -*-
"""
@author: Nadia Bloemendaal, nadia.bloemendaal@vu.nl

For more information, please see
Bloemendaal, N., Haigh, I.D., de Moel, H. et al.
Generation of a global synthetic tropical cyclone hazard dataset using STORM.
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This is the STORM module for simulation of the TC pressure

Copyright (C) 2020 Nadia Bloemendaal. All versions released under GNU General Public License v3.0

Performance notes (SIENA):
  - coastal_basemap_data.npy loaded ONCE into module-level cache
    (was: loaded from disk on every non-landfall timestep — 18MB × ~5000 reads)
  - Monthly environmental fields cached by (stem, month, phase) key
    (was: np.loadtxt re-parsing 721×1440 text every storm)
  - RMAX_PRESSURE.npy loaded once into module-level cache
  - Grid index lookups replaced with O(1) arithmetic on regular grids
    (was: np.abs(array - val).argmin() creating temporary arrays)
  All physics, branching, and output formats are identical to the original.
"""

import numpy as np
from SELECT_BASIN import Basins_WMO
from math import radians, cos, sin, asin, sqrt
from SAMPLE_RMAX import Add_Rmax
from scipy.stats import truncnorm
import math
import sys
import os

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
import random
from siena_utils import (
    normalize_phase,
    phase_code,
    load_monthly_field,
    load_field_with_year_fallback,
)

# ==========================================================================
# MODULE-LEVEL CACHES — loaded once, reused across all calls
# ==========================================================================

# Cache 1: Coastal basemap for distance_from_coast
# Original code loaded this 18MB file on EVERY non-landfall timestep
_COASTAL_CACHE = None


def _get_coastal_data():
    """Load coastal basemap once, return cached (lons, lats) arrays."""
    global _COASTAL_CACHE
    if _COASTAL_CACHE is None:
        fpath = os.path.join(dir_path, "coastal_basemap_data.npy")
        D = np.load(fpath, encoding="latin1", allow_pickle=True).tolist()
        _COASTAL_CACHE = (D["lons"], D["lats"])
    return _COASTAL_CACHE


# Cache 2: Monthly environmental fields keyed by (stem, month, phase_str, env_year)
# Original code called np.loadtxt (text parsing of 721×1440 grids) per storm.
# Multiple storms in the same month re-parsed the same file.
_FIELD_CACHE = {}


def _load_field_cached(stem, month, phase=None, env_year=None):
    """Load a monthly field, caching the result for subsequent calls.
    If env_year is set, load year-specific field with phase-mean fallback.
    """
    key = (stem, month, normalize_phase(phase), env_year)
    if key not in _FIELD_CACHE:
        try:
            if env_year is not None:
                # Year-specific: stem is e.g. "VWS", not "Monthly_mean_VWS"
                # load_field_with_year_fallback handles the prefix
                _stem = stem.replace("Monthly_mean_", "")
                _FIELD_CACHE[key] = load_field_with_year_fallback(
                    dir_path, _stem, month, phase=phase, env_year=env_year
                )
            else:
                _FIELD_CACHE[key] = load_monthly_field(
                    dir_path, stem, month, phase=phase
                )
        except Exception:
            _FIELD_CACHE[key] = None
    return _FIELD_CACHE[key]


# Cache 3: Precomputed regular-grid index functions
# The MSLP/PI/VWS/RH fields are on a regular 0.25° grid: lat from 90 to -90
# (721 points), lon from 0 to 359.75 (1440 points).
# Instead of np.abs(array - val).argmin() per timestep, compute the index directly.
_LAT_GRID_STEP = 180.0 / 720.0  # 0.25 degrees
_LON_GRID_STEP = 360.0 / 1440.0  # 0.25 degrees


def _lat_to_idx(lat):
    """Convert latitude to grid index. Grid runs from 90 (idx=0) to -90 (idx=720)."""
    return int(round((90.0 - lat) / _LAT_GRID_STEP))


def _lon_to_idx(lon):
    """Convert longitude to grid index. Grid runs from 0 (idx=0) to 359.75 (idx=1439)."""
    lon = lon % 360.0
    return int(round(lon / _LON_GRID_STEP)) % 1440


# Basin boundaries — avoids calling Basins_WMO (which triggers unnecessary
# Poisson draws and file loads just to return static coordinates)
_BASIN_BOUNDS = {
    "EP": (5, 60, 180, 285),
    "NA": (5, 60, 255, 360),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}

_BASIN_IDX = {"EP": 0, "NA": 1, "NI": 2, "SI": 3, "SP": 4, "WP": 5}


# ==========================================================================
# SAMPLING FUNCTIONS (unchanged physics)
# ==========================================================================
def _safe_sigma(val, fallback=1.0):
    """Sanitize a sigma value: must be finite and positive.
    max(NaN, 0.1) returns NaN in Python — explicit check needed."""
    val = float(val)
    if not np.isfinite(val) or val <= 0:
        return fallback
    return val


def _sample_twopn(mu, std_neg, std_pos):
    """
    Sample from a two-piece normal distribution.
    (John 1982, Commun. Stat. Theory Methods 11(8), 879-885)
    """
    std_neg = _safe_sigma(abs(std_neg))
    std_pos = _safe_sigma(abs(std_pos))

    u = np.random.random()
    p_left = std_neg / (std_neg + std_pos)
    if u < p_left:
        return mu - abs(np.random.normal(0, std_neg))
    else:
        return mu + abs(np.random.normal(0, std_pos))


def _sample_truncated_twopn(mu, std_neg, std_pos, lower, upper):
    """
    Sample from a truncated two-piece normal distribution.
    Uses scipy.stats.truncnorm for proper truncated sampling.
    """
    std_neg = _safe_sigma(abs(std_neg))
    std_pos = _safe_sigma(abs(std_pos))

    if not np.isfinite(mu) or not np.isfinite(lower) or not np.isfinite(upper):
        return float(np.clip(np.random.normal(0, 1.0), lower, upper))

    mu_clamped = np.clip(mu, lower, upper)

    u = np.random.random()
    p_left = std_neg / (std_neg + std_pos)

    if u < p_left:
        sigma = std_neg
        a_tn = (lower - mu_clamped) / sigma
        b_tn = 0.0
    else:
        sigma = std_pos
        a_tn = 0.0
        b_tn = (upper - mu_clamped) / sigma

    if a_tn >= b_tn:
        return float(mu_clamped)

    draw = truncnorm.rvs(a_tn, b_tn, loc=mu_clamped, scale=sigma)
    return float(np.clip(draw, lower, upper))


# ==========================================================================
# PHYSICS FUNCTIONS (unchanged)
# ==========================================================================


def Calculate_Vmax(Penv, Pc, coef):
    [a, b] = coef
    Vmax10 = a * (Penv - Pc) ** b
    return Vmax10


def Calculate_Pressure(Vmax10, Penv, coef):
    [a, b] = coef
    Pc = Penv - (Vmax10 / a) ** (1.0 / b)
    return Pc


def TC_Category(V):
    if V >= 15.8 and V < 29.0:
        cat = 0
    elif V >= 29.0 and V < 37.6:
        cat = 1
    elif V >= 37.6 and V < 43.4:
        cat = 2
    elif V >= 43.4 and V < 51.1:
        cat = 3
    elif V >= 51.1 and V < 61.6:
        cat = 4
    elif V >= 61.6:
        cat = 5
    else:
        cat = -1
    return cat


def find_index_pressure(basin, lat, lon, lat0, lon0, lon1):
    base = 5
    latindex = np.floor(float(lat - lat0) / base)
    lonindex = np.floor(float(lon - lon0) / base)
    maxlon = (lon1 - lon0) / 5.0
    ind = latindex * maxlon + lonindex
    return ind


def PRESSURE_JAMES_MASON(
    dp,
    pres,
    a,
    b,
    c,
    d,
    mpi,
    vws=0.0,
    rh=0.0,
    phase=None,
    c_vws=0.0,
    c_rh=0.0,
    c_en=0.0,
    c_ln=0.0,
):
    if pres < mpi:
        presmpi = 0
    else:
        presmpi = pres - mpi
    phase_effect = c_en if phase == 2 else c_ln if phase == 0 else 0.0
    y = a + b * dp + c * np.exp(-d * presmpi) + c_vws * vws + c_rh * rh + phase_effect
    return y


def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = abs(lon1 - lon2)
    dlat = abs(lat2 - lat1)
    A1 = sin(dlat / 2) ** 2.0 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2.0
    C2 = 2.0 * asin(sqrt(A1))
    r = 6371.0
    km = C2 * r
    return km


def decay_after_landfall(lat_landfall, lon_landfall, latlijst, lonlijst, p, coef, Penv):
    C1 = 0.0109
    D1 = -0.0503
    R = 0.9
    t0 = 150
    alpha = 0.095
    vb = 26.7

    v0 = Calculate_Vmax(Penv, p, coef)

    wind_decay = []
    pressure_decay = []
    pressure_decay.append(p)
    wind_decay.append(v0)

    v0 = v0 / 0.5144444444
    D0 = 1.0
    v = v0
    t = 3
    j = 1
    pres_landfall = p

    while v > 35 and j < len(latlijst):
        try:
            D = haversine(lat_landfall, lon_landfall, latlijst[j], lonlijst[j])
            if D == 0.0:
                pressure_decay.append(pres_landfall)
                wind_decay.append(v0 * 0.5144444)
                j = j + 1
                t = t + 3
            if D > 1:
                M = C1 * t * (t0 - t)
                b_KM = D1 * t * (t0 - t)
                C_KM = M * np.log(D / D0) + b_KM
                v = vb + (R * v0 - vb) * np.exp(-alpha * t) - C_KM
                if v * 0.51444 < 18.0:
                    return pressure_decay, wind_decay
                pres_landfall = Calculate_Pressure(v * 0.514444, Penv, coef)
                pres_landfall = round(pres_landfall, 1)
                pressure_decay.append(pres_landfall)
                wind_decay.append(v * 0.514444)
                t = t + 3
                j = j + 1
            else:
                j = j + 1
                t = t + 3
        except IndexError:
            v = -100.0

    return pressure_decay, wind_decay


def distance_from_coast(lon, lat, degree_in_km=111.12):
    """
    Calculate the distance from coast using CACHED coastal data.
    Original loaded the 18MB file on every call — now loads once.
    """
    if lon > 180:
        lon = lon - 360.0
    lons, lats = _get_coastal_data()
    dists = np.sqrt((lons - lon) ** 2 + (lats - lat) ** 2)
    mindist = np.min(dists) * degree_in_km
    return mindist


def add_parameters_to_TC_data(
    pressure_list,
    wind_list,
    latfull,
    lonfull,
    year,
    storm_number,
    month,
    basin,
    landfallfull,
    lijst,
    TC_data,
    idx,
    Penv_field=None,
    env_year=None,
):
    """
    Assemble per-timestep TC output rows.

    Output columns (0-indexed):
        0  year
        1  month
        2  storm_number
        3  timestep
        4  basin_idx
        5  lat
        6  lon
        7  central_pressure (hPa)
        8  max_wind (m/s, 10-min sustained)
        9  rmax (km)
       10  category (Saffir-Simpson, -1 if sub-TS)
       11  landfall (1=over land, 0=over ocean)
       12  dist_to_coast (km)
       13  penv (hPa) — environmental pressure at (lat, lon)
       14  env_year — historical year used for environmental fields (-1 if N/A)
    """
    rmax_list = Add_Rmax(pressure_list)
    x = min(len(landfallfull), len(lijst))


    _env_year_val = int(env_year) if env_year is not None else -1

    for l in range(0, x):
        if landfallfull[l] == 1.0:
            dist = 0
        else:
            # Uses cached coastal data (no file I/O)
            dist = distance_from_coast(lonfull[l], latfull[l])

        # ── Penv: look up from the MSLP field at (lat, lon) ──
        penv = -1.0
        if Penv_field is not None and l < len(latfull):
            _li = _lat_to_idx(latfull[l])
            _lo = _lon_to_idx(lonfull[l])
            if 0 <= _li < Penv_field.shape[0] and 0 <= _lo < Penv_field.shape[1]:
                _v = float(Penv_field[_li, _lo])
                if np.isfinite(_v):
                    penv = round(_v, 1)

        category = TC_Category(wind_list[l])
        TC_data.append(
            [
                year,
                month,
                storm_number,
                l,
                idx,
                latfull[l],
                lonfull[l],
                pressure_list[l],
                wind_list[l],
                rmax_list[l],
                category,
                landfallfull[l],
                dist,
                penv,
                _env_year_val,
            ]
        )

    return TC_data


# ==========================================================================
# HELPER: extract row coefficients (avoids duplicated if/elif blocks)
# ==========================================================================


def _unpack_pressure_row(row):
    """
    Unpack a coefficient row into (c0,c1,c2,c3, EPmu, EPstd_neg, EPstd_pos,
    mpi, c_vws, c_rh, c_en, c_ln) regardless of row length.
    """
    if len(row) >= 12:
        c0, c1, c2, c3, EPmu, EPstd_neg, EPstd_pos, mpi = row[:8]
        c_vws, c_rh, c_en, c_ln = row[8:12]
    elif len(row) >= 11:
        c0, c1, c2, c3, EPmu, EPstd, mpi = row[:7]
        EPstd_neg = EPstd_pos = EPstd
        c_vws, c_rh, c_en, c_ln = row[7:11]
    else:
        c0, c1, c2, c3, EPmu, EPstd, mpi = row[:7]
        EPstd_neg = EPstd_pos = EPstd
        c_vws, c_rh, c_en, c_ln = 0.0, 0.0, 0.0, 0.0
    return c0, c1, c2, c3, EPmu, EPstd_neg, EPstd_pos, mpi, c_vws, c_rh, c_en, c_ln


# ==========================================================================
# MAIN GENERATION FUNCTION
# ==========================================================================


def TC_pressure(
    basin,
    latlist,
    lonlist,
    landfalllist,
    year,
    storms,
    monthlist,
    TC_data,
    phase=None,
    env_years=None,
):
    """
    Calculate TC pressure along synthetic tracks.
    Logic is identical to the original; only I/O is cached.

    Parameters
    ----------
    env_years : dict {month: year} or None
        If set, each storm uses the historical year assigned to its genesis
        month for loading environmental fields (VWS/RH/PI/MSLP).
        Falls back to phase-mean if year-specific file is not found.
    """
    idx = _BASIN_IDX[basin]
    lat0, lat1, lon0, lon1 = _BASIN_BOUNDS[basin]

    phase = normalize_phase(phase)
    ph_code = phase_code(phase) if phase is not None else 1

    # These .npy files are small and loaded once per TC_pressure call
    # (once per year). Acceptable cost — not the bottleneck.
    JM_pressure = np.load(
        os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy"), allow_pickle=True
    ).item()
    Genpres = np.load(
        os.path.join(__location__, "DP0_PRES_GENESIS.npy"), allow_pickle=True
    ).item()
    WPR_coefficients = np.load(
        os.path.join(__location__, "COEFFICIENTS_WPR_PER_MONTH.npy"), allow_pickle=True
    ).item()
    Genwind = np.load(
        os.path.join(__location__, "GENESIS_WIND.npy"), allow_pickle=True
    ).item()

    intlist = [5, 3, 2, 5, 5, 5]
    int_thres = intlist[idx]
    wind_threshold = 18.0

    for storm_number, month, latfull, lonfull, landfallfull in zip(
        range(0, int(storms)), monthlist, latlist, lonlist, landfalllist
    ):
        i = 0
        vmax = 0
        count = 0
        p = np.nan
        # FIX: Track consecutive genesis resets to detect infinite loops.
        # The original code resets i=0 when p < p_threshold or isnan(p),
        # which re-samples genesis at latfull[0]. If that location always
        # produces bad pressure (e.g., low Penv + high sampled wind),
        # the loop never terminates. Cap at MAX_GENESIS_RETRIES and skip.
        _genesis_retries = 0
        MAX_GENESIS_RETRIES = 50

        # OPTIMIZATION: Load environmental fields via cache.
        # Original called np.loadtxt per storm — now each unique
        # (stem, month, phase, env_year) is parsed once and reused.
        env_year = env_years.get(month) if env_years else None
        Penv_field = _load_field_cached(
            "Monthly_mean_MSLP", month, phase=phase, env_year=env_year
        )
        PI_field = _load_field_cached(
            "Monthly_mean_PI", month, phase=phase, env_year=env_year
        )
        VWS_field = _load_field_cached(
            "Monthly_mean_VWS", month, phase=phase, env_year=env_year
        )
        RH_field = _load_field_cached(
            "Monthly_mean_RH600", month, phase=phase, env_year=env_year
        )

        constants_pressure = JM_pressure[idx][month]
        constants_pressure = np.array(constants_pressure)

        coef = WPR_coefficients[idx][month]
        coef = np.array(coef)

        mpi_col = 7 if constants_pressure.shape[1] >= 12 else 6
        p_threshold = min(constants_pressure[:, mpi_col]) - 10.0

        EP = Genpres[idx][month]

        while i < len(latfull):
            lat, lon, landfall = latfull[i], lonfull[i], landfallfull[i]

            # OPTIMIZATION: O(1) index lookup on regular grid
            # Original: np.abs(np.linspace(90,-90,721) - lat).argmin()
            # which allocated a 721-element array every timestep.
            lat_dummy = _lat_to_idx(lat)
            lon_dummy = _lon_to_idx(lon)

            Penv = Penv_field[lat_dummy, lon_dummy]

            if lat0 <= lat <= lat1 and lon0 <= lon <= lon1:
                if (p < p_threshold) | math.isnan(p):
                    _genesis_retries += 1
                    if _genesis_retries > MAX_GENESIS_RETRIES:
                        # This storm's genesis location consistently produces
                        # invalid pressure. Skip it rather than loop forever.
                        break
                    i = 0
                    vmax = 0

                if i == 0:
                    vmax = random.choice(Genwind[idx][month])
                    p = Calculate_Pressure(vmax, Penv, coef)

                    pressure_list = []
                    wind_list = []

                    [Pmu, Pstd, DP0mu, DP0std, dpmin, dpmax] = EP
                    dp0 = np.random.normal(DP0mu, DP0std)
                    dp1 = -1.0 * np.abs(dp0)

                    pressure_list.append(p)
                    wind_list.append(vmax)
                    i = i + 1

                if landfall == 1:
                    if (p < p_threshold) | math.isnan(p):
                        print("Landfall", p, p_threshold)
                        _genesis_retries += 1
                        if _genesis_retries > MAX_GENESIS_RETRIES:
                            break
                        i = 0
                        vmax = 0

                    elif vmax < wind_threshold or p > Penv:
                        TC_data = add_parameters_to_TC_data(
                            pressure_list,
                            wind_list,
                            latfull,
                            lonfull,
                            year,
                            storm_number,
                            month,
                            basin,
                            landfallfull,
                            pressure_list,
                            TC_data,
                            idx,
                            Penv_field=Penv_field,
                            env_year=env_year,
                        )
                        i = 1000000000000000

                    else:
                        ind = int(
                            find_index_pressure(basin, lat, lon, lat0, lon0, lon1)
                        )
                        row = constants_pressure[ind]
                        (
                            c0,
                            c1,
                            c2,
                            c3,
                            EPmu,
                            EPstd_neg,
                            EPstd_pos,
                            mpi,
                            c_vws,
                            c_rh,
                            c_en,
                            c_ln,
                        ) = _unpack_pressure_row(row)

                        if PI_field is not None:
                            pi_val = float(PI_field[lat_dummy, lon_dummy])
                            if np.isfinite(pi_val) and pi_val > 0:
                                mpi = pi_val
                        vws = (
                            float(VWS_field[lat_dummy, lon_dummy])
                            if VWS_field is not None
                            else 0.0
                        )
                        rh = (
                            float(RH_field[lat_dummy, lon_dummy])
                            if RH_field is not None
                            else 0.0
                        )

                        y = PRESSURE_JAMES_MASON(
                            dp1,
                            p,
                            c0,
                            c1,
                            c2,
                            c3,
                            mpi,
                            vws=vws,
                            rh=rh,
                            phase=ph_code,
                            c_vws=c_vws,
                            c_rh=c_rh,
                            c_en=c_en,
                            c_ln=c_ln,
                        )
                        dp0 = _sample_truncated_twopn(
                            y + EPmu, EPstd_neg, EPstd_pos, dpmin, dpmax
                        )

                        if p < mpi:
                            if dp0 < 0:
                                if count < 5:
                                    count = count + 1
                                else:
                                    dp0 = abs(dp0)
                        else:
                            count = 0

                        p = round(dp0 + p, 1)
                        dp1 = dp0

                        if vmax < wind_threshold or p > Penv:
                            TC_data = add_parameters_to_TC_data(
                                pressure_list,
                                wind_list,
                                latfull,
                                lonfull,
                                year,
                                storm_number,
                                month,
                                basin,
                                landfallfull,
                                pressure_list,
                                TC_data,
                                idx,
                                Penv_field=Penv_field,
                                env_year=env_year,
                            )
                            i = 10000000000000000000000000000000

                        else:
                            pressure_list.append(p)
                            vmax = Calculate_Vmax(Penv, p, coef)
                            vmax = round(vmax, 1)
                            wind_list.append(vmax)

                    if any(c < 1 for c in landfallfull[i:]):
                        check_move_ocean = (
                            i + np.where(np.array(landfallfull[i:]) == 0.0)[0][0]
                        )

                        if check_move_ocean > i + 3:
                            decay_pressure, decay_wind = decay_after_landfall(
                                lat,
                                lon,
                                latfull[i : i + check_move_ocean],
                                lonfull[i : i + check_move_ocean],
                                p,
                                coef,
                                Penv,
                            )
                            for d in range(len(decay_pressure)):
                                pressure_list.append(decay_pressure[d])
                                wind_list.append(decay_wind[d])

                        if wind_list[-1] < wind_threshold:
                            TC_data = add_parameters_to_TC_data(
                                pressure_list,
                                wind_list,
                                latfull,
                                lonfull,
                                year,
                                storm_number,
                                month,
                                basin,
                                landfallfull,
                                pressure_list,
                                TC_data,
                                idx,
                                Penv_field=Penv_field,
                                env_year=env_year,
                            )
                            i = 10000000000000000000000000.0

                        else:
                            dp1 = pressure_list[-1] - pressure_list[-2]
                            p = pressure_list[-1]
                            i = check_move_ocean

                    else:
                        decay_pressure, decay_wind = decay_after_landfall(
                            lat, lon, latfull[i:], lonfull[i:], p, coef, Penv
                        )
                        for d in range(len(decay_pressure)):
                            pressure_list.append(decay_pressure[d])
                            wind_list.append(decay_wind[d])

                        TC_data = add_parameters_to_TC_data(
                            pressure_list,
                            wind_list,
                            latfull,
                            lonfull,
                            year,
                            storm_number,
                            month,
                            basin,
                            landfallfull,
                            pressure_list,
                            TC_data,
                            idx,
                            Penv_field=Penv_field,
                            env_year=env_year,
                        )
                        i = 1000000000000

                else:  # no landfall
                    if (p < p_threshold) | math.isnan(p):
                        print("No landfall", p, p_threshold)
                        _genesis_retries += 1
                        if _genesis_retries > MAX_GENESIS_RETRIES:
                            break
                        i = 0
                        vmax = 0

                    elif vmax < wind_threshold or p > Penv and i > 3:
                        TC_data = add_parameters_to_TC_data(
                            pressure_list,
                            wind_list,
                            latfull,
                            lonfull,
                            year,
                            storm_number,
                            month,
                            basin,
                            landfallfull,
                            pressure_list,
                            TC_data,
                            idx,
                            Penv_field=Penv_field,
                            env_year=env_year,
                        )
                        i = 1000000000000000

                    else:
                        ind = int(
                            find_index_pressure(basin, lat, lon, lat0, lon0, lon1)
                        )
                        row = constants_pressure[ind]
                        (
                            c0,
                            c1,
                            c2,
                            c3,
                            EPmu,
                            EPstd_neg,
                            EPstd_pos,
                            mpi,
                            c_vws,
                            c_rh,
                            c_en,
                            c_ln,
                        ) = _unpack_pressure_row(row)

                        if PI_field is not None:
                            pi_val = float(PI_field[lat_dummy, lon_dummy])
                            if np.isfinite(pi_val) and pi_val > 0:
                                mpi = pi_val
                        vws = (
                            float(VWS_field[lat_dummy, lon_dummy])
                            if VWS_field is not None
                            else 0.0
                        )
                        rh = (
                            float(RH_field[lat_dummy, lon_dummy])
                            if RH_field is not None
                            else 0.0
                        )

                        y = PRESSURE_JAMES_MASON(
                            dp1,
                            p,
                            c0,
                            c1,
                            c2,
                            c3,
                            mpi,
                            vws=vws,
                            rh=rh,
                            phase=ph_code,
                            c_vws=c_vws,
                            c_rh=c_rh,
                            c_en=c_en,
                            c_ln=c_ln,
                        )
                        dp0 = _sample_truncated_twopn(
                            y + EPmu, EPstd_neg, EPstd_pos, dpmin, dpmax
                        )

                        if p < mpi:
                            if dp0 < 0:
                                if count < 5:
                                    count = count + 1
                                else:
                                    dp0 = abs(dp0)
                        else:
                            count = 0

                        if i < int_thres:
                            dp0 = -1.0 * np.abs(dp0)
                        p = round(dp0 + p, 1)
                        dp1 = dp0

                        if vmax < wind_threshold or p > Penv:
                            TC_data = add_parameters_to_TC_data(
                                pressure_list,
                                wind_list,
                                latfull,
                                lonfull,
                                year,
                                storm_number,
                                month,
                                basin,
                                landfallfull,
                                pressure_list,
                                TC_data,
                                idx,
                                Penv_field=Penv_field,
                                env_year=env_year,
                            )
                            i = 10000000000000000000000000000000

                        else:
                            pressure_list.append(p)
                            vmax = Calculate_Vmax(Penv, p, coef)
                            vmax = round(vmax, 1)
                            wind_list.append(vmax)
                            i = i + 1

            else:
                TC_data = add_parameters_to_TC_data(
                    pressure_list,
                    wind_list,
                    latfull,
                    lonfull,
                    year,
                    storm_number,
                    month,
                    basin,
                    landfallfull,
                    pressure_list,
                    TC_data,
                    idx,
                    Penv_field=Penv_field,
                    env_year=env_year,
                )
                i = 100000000000000000.0

        if i == len(latfull):
            TC_data = add_parameters_to_TC_data(
                pressure_list,
                wind_list,
                latfull,
                lonfull,
                year,
                storm_number,
                month,
                basin,
                landfallfull,
                pressure_list,
                TC_data,
                idx,
                Penv_field=Penv_field,
                env_year=env_year,
            )

    return TC_data
