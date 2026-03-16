# -*- coding: utf-8 -*-
"""STORM module for simulation of the TC track.
SIENA extension: optional phase-aware offsets in pooled coefficients.
"""

import numpy as np
from SELECT_BASIN import Basins_WMO
from siena_utils import phase_code, normalize_phase
import os
import sys

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


_BASIN_BOUNDS = {
    "EP": (5, 60, 180, 285),
    "NA": (5, 60, 255, 359),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}


def find_lat_index_bins(basin, lat):
    lat0 = _BASIN_BOUNDS[basin][0]
    base = 5
    return np.floor(float(lat - lat0) / base)


def Check_if_landfall(lat, lon, lat1, lon0, land_mask):
    x_coord = int(10 * (lon - lon0))
    y_coord = int(10 * (lat1 - lat))
    return land_mask[y_coord, x_coord]


def TC_movement(lon_genesis_list, lat_genesis_list, basin, phase=None):
    phase = normalize_phase(phase)
    ph_code = phase_code(phase) if phase is not None else 1
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
    # ---- FIX: use static bounds ----
    lat0, lat1, lon0, lon1 = _BASIN_BOUNDS[basin]
    latall = []
    lonall = []
    landfallall = []

    for lat_genesis, lon_genesis in zip(lat_genesis_list, lon_genesis_list):
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
            # ---- FIX: clamp index to valid range ----
            ind = max(0, min(ind, len(constants) - 1))
            row = constants[ind]
            if len(row) >= 17:
                [
                    a_lat0,
                    a_lat1,
                    a_lat2,
                    g_en,
                    g_ln,
                    b_lon0,
                    b_lon1,
                    d_en,
                    d_ln,
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
                g_en = g_ln = d_en = d_ln = 0.0
            if len(latlijst) == 1:
                dlat0 = np.random.normal(Dlat0mu, Dlat0std)  # scalar, not array
                dlon0 = np.random.normal(Dlon0mu, Dlon0std)  # scalar, not array
            phase_lat = g_en if ph_code == 2 else g_ln if ph_code == 0 else 0.0
            phase_lon = d_en if ph_code == 2 else d_ln if ph_code == 0 else 0.0
            dlat1 = float(a_lat0 + a_lat1 * dlat0 + a_lat2 * lat + phase_lat)
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
            dlon1 = float(
                b_lon0 + b_lon1 * dlon0 + phase_lon + np.random.normal(Elonmu, Elonstd)
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
