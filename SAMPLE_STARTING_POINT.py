# -*- coding: utf-8 -*-
"""STORM genesis-location sampler with optional ENSO-phase-specific grids."""

import numpy as np
import random
import os
import sys
from SELECT_BASIN import Basins_WMO
from siena_utils import normalize_phase

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def Check_EP_formation(lat, lon):
    # Block EP genesis in the NA-exclusive zone (lon>276 AND lat>20)
    return lon > 276 and lat > 20


def Check_NA_formation(lat, lon):
    # Block NA genesis in the EP-exclusive zone (lon<276 AND lat<20)
    return lon < 276 and lat < 20


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


def Startingpoint(no_storms, monthlist, basin, phase=None):
    phase = normalize_phase(phase)
    basins = ["EP", "NA", "NI", "SI", "SP", "WP"]
    basin_name = dict(zip(basins, [0, 1, 2, 3, 4, 5]))
    idx = basin_name[basin]
    lon_coordinates = []
    lat_coordinates = []
    s, monthdummy, lat0, lat1, lon0, lon1 = Basins_WMO(basin, phase=phase)
    land_mask = np.loadtxt(
        os.path.join(dir_path, "Land_ocean_mask_" + str(basin) + ".txt")
    )

    for month in monthlist:
        # ---- FIX: Try phase-specific grid first, fall back to pooled ----
        grid_path_phase = None
        grid_path_pooled = os.path.join(
            dir_path, f"GRID_GENESIS_MATRIX_{idx}_{month}.txt"
        )

        if phase is not None:
            candidate = os.path.join(
                dir_path, f"GRID_GENESIS_MATRIX_{idx}_{month}_{phase}.txt"
            )
            if os.path.exists(candidate):
                grid_path_phase = candidate

        # Try phase grid first
        weighted_list_index = []
        grid_copy = None
        if grid_path_phase is not None:
            raw = np.loadtxt(grid_path_phase)
            weighted_list_index, grid_copy = _build_weighted_index(raw)

        # Fallback to pooled if phase grid is empty or missing
        if len(weighted_list_index) == 0:
            raw = np.loadtxt(grid_path_pooled)
            weighted_list_index, grid_copy = _build_weighted_index(raw)

        # If still empty (shouldn't happen for pooled, but be safe)
        if len(weighted_list_index) == 0:
            continue

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
