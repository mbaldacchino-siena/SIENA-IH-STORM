# -*- coding: utf-8 -*-
"""STORM genesis-location sampler with optional ENSO-phase-specific grids."""

import numpy as np
import random
import os
from CODE.SELECT_BASIN import Basins_WMO
from CODE.siena_utils import normalize_phase
from CODE.genesis_matrix import _blend_genesis_with_env, compute_gpi_field

__location__ = os.path.realpath(os.getcwd())  # TEMP FIX?
dir_path = __location__


def Check_NA_formation(lat, lon):
    """
    Reject NA sampling in EP-exclusive waters (Pacific side of Central America).
    Uses a single diagonal separator through the Central American isthmus,
    applied only in the NA–EP overlap longitudes (255–285E = 75–105W).
    Outside that strip, nothing is blocked (NA owns everything east of 285E).
    """
    # Outside the overlap strip, NA has exclusive claim
    if lon >= 285.0:  # east of 75W — pure NA Atlantic/Caribbean
        return False
    # In the overlap strip, use the isthmus separator
    # Isthmus line: from (16N, 92W) = (16, 268E) to (8N, 78W) = (8, 282E)
    # lat_isthmus(lon) = 16 - (16-8)/(282-268) * (lon - 268) = 16 - 0.571*(lon - 268)
    lat_isthmus = 16.0 - 0.571 * (lon - 268.0)
    return lat < lat_isthmus


def Check_EP_formation(lat, lon):
    """
    Reject EP sampling in NA-exclusive waters (Atlantic/Caribbean side).
    Symmetric to above.
    """
    if lon < 255.0:  # west of 105W — pure EP Pacific
        return False
    if lon >= 285.0:  # east of 75W — not in EP range at all, definitely NA
        return True
    lat_isthmus = 16.0 - 0.571 * (lon - 268.0)
    return lat >= lat_isthmus


def Check_if_landfall(lat, lon, basin, land_mask):
    s, monthdummy, lat0_WMO, lat1_WMO, lon0_WMO, lon1_WMO = Basins_WMO(basin)
    x = int(10 * (lon - lon0_WMO))
    y = int(10 * (lat1_WMO - lat))
    return land_mask[y, x]


def _build_weighted_index(grid_copy, round_coeff=4):
    """
    Build the weighted sampling list from a genesis grid.
    Returns an empty list if the grid has no valid positive entries.
    """
    grid_copy = np.array(grid_copy)
    grid_copy = np.round(grid_copy, round_coeff)
    ncols = len(grid_copy[0, :])
    weighted_list_index = []
    for i in range(len(grid_copy[:, 0])):
        for j in range(ncols):
            cell = grid_copy[i, j]
            # ---- FIX: guard against NaN and negative values ----
            if not np.isfinite(cell) or cell <= 0:
                continue
            value = int((10**round_coeff) * cell)
            if value > 0:
                weighted_list_index.extend([i * (ncols) + j] * value)
    return weighted_list_index, grid_copy


_LAND_MASK_CACHE_SP = {}


def _get_land_mask_sp(basin):
    if basin not in _LAND_MASK_CACHE_SP:
        _LAND_MASK_CACHE_SP[basin] = np.loadtxt(
            os.path.join(
                dir_path, "Land_ocean_masks/Land_ocean_mask_" + str(basin) + ".txt"
            )
        )
    return _LAND_MASK_CACHE_SP[basin]


def Startingpoint(
    no_storms,
    monthlist,
    basin,
    phase=None,
    month_phases=None,
    env_years=None,
):
    # Local cache framework for GPI
    gpi_cache = {}

    def _get_runtime_gpi(month, effective_phase, env_year):
        key = (month, effective_phase, env_year)
        if key not in gpi_cache:
            gpi_cache[key] = compute_gpi_field(
                basin,
                month,
                phase=effective_phase,
                env_year=env_year,
            )
        return gpi_cache[key]

    phase = normalize_phase(phase)
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    basin_name = dict(zip(basins, [0, 1, 2, 3, 4, 5]))
    idx = basin_name[basin]
    lon_coordinates = []
    lat_coordinates = []
    s, monthdummy, lat0, lat1, lon0, lon1 = Basins_WMO(basin, phase=phase)
    land_mask = _get_land_mask_sp(basin)

    for month in monthlist:
        effective_phase = phase
        if month_phases is not None and month in month_phases:
            effective_phase = normalize_phase(month_phases[month])
        env_year = None if env_years is None else env_years.get(month)

        # file path to the 1x1 degree matrix for genesis (fallback pooled)
        grid_path_pooled = os.path.join(
            dir_path, f"GRID_GENESIS_MATRIX_{idx}_{month}.txt"
        )
        empirical_path_pooled = os.path.join(
            dir_path, f"GRID_GENESIS_EMPIRICAL_MATRIX_{idx}_{month}.txt"
        )
        grid_path_phase = None
        empirical_path_phase = None

        if effective_phase is not None:  # ← was: if phase is not None
            grid_path_phase = os.path.join(
                dir_path, f"GRID_GENESIS_MATRIX_{idx}_{month}_{effective_phase}.txt"
            )
            empirical_path_phase = os.path.join(
                dir_path,
                f"GRID_GENESIS_EMPIRICAL_MATRIX_{idx}_{month}_{effective_phase}.txt",
            )

        # If a runtime env-year is available, start from the empirical matrix
        # and blend with a runtime year-conditioned GPI using the existing
        # _blend_genesis_with_env() helper.
        weighted_list_index = []
        grid_copy = None

        # Runtime year-conditioned blend: only applicable in FORECAST mode,
        # where month_phases is populated by get_month_phases(forecast_cfg, ...).
        # In historical/classic mode month_phases is None, and the baked
        # GRID_GENESIS_MATRIX_{idx}_{month}[_phase].txt files from preprocessing
        # already encode the chosen genesis_weighting (EMPIRICAL / GPI / GPI-MIX).
        # Using the runtime blend in classic mode would silently bypass that choice.
        if env_year is not None and month_phases is not None:
            empirical_path = empirical_path_pooled
            if empirical_path_phase is not None and os.path.exists(
                empirical_path_phase
            ):
                empirical_path = empirical_path_phase

            if os.path.exists(empirical_path):
                raw_empirical = np.loadtxt(empirical_path)
                env_runtime = _get_runtime_gpi(month, effective_phase, env_year)

                if env_runtime is not None:
                    # Optional dump for blend inspection: set env var
                    # SIENA_BLEND_DUMP_DIR to a directory path and the four
                    # arrays (empirical, gpi, blended) plus metadata.json
                    # will be saved per (basin, month, phase, env_year)
                    # tuple. Off by default — zero overhead.
                    _dump_dir = os.environ.get("SIENA_BLEND_DUMP_DIR")
                    raw = _blend_genesis_with_env(
                        raw_empirical,
                        env_runtime,
                        label=f"runtime_{basin}_{month}_{effective_phase}_{env_year}",
                        dump_dir=_dump_dir,
                    )
                    weighted_list_index, grid_copy = _build_weighted_index(raw)

        if len(weighted_list_index) == 0:
            if grid_path_phase is not None and os.path.exists(grid_path_phase):
                raw = np.loadtxt(grid_path_phase)
                weighted_list_index, grid_copy = _build_weighted_index(raw)

        if len(weighted_list_index) == 0:
            raw = np.loadtxt(grid_path_pooled)
            weighted_list_index, grid_copy = _build_weighted_index(raw)

        if len(weighted_list_index) == 0:
            raise RuntimeError(
                f"No valid genesis sampling weights for basin={basin}, month={month}, "
                f"phase={effective_phase}, env_year={env_year}"
            )

        ncols = len(grid_copy[0, :])
        var = 0
        attempts = 0
        max_attempts = 10000
        while var == 0 and attempts < max_attempts:
            attempts += 1
            idx0 = random.choice(weighted_list_index)
            row = int(np.floor(idx0 / (ncols)))
            col = int(idx0 % (ncols))
            lat_pert = random.uniform(0, 0.99)
            lon_pert = random.uniform(0, 0.99)
            lon_pt = lon0 + round(col + lon_pert, 2)
            lat_pt = lat1 - round(row + lat_pert, 2)
            if lon_pt < lon1 and lat_pt < lat1:
                check = Check_if_landfall(lat_pt, lon_pt, basin, land_mask)
                if basin == "EP":
                    check = check or Check_EP_formation(lat_pt, lon_pt)
                if basin == "NA":
                    check = check or Check_NA_formation(lat_pt, lon_pt)
                if check == 0:
                    var = 1
                    lon_coordinates.append(lon_pt)
                    lat_coordinates.append(lat_pt)
    return lon_coordinates, lat_coordinates
