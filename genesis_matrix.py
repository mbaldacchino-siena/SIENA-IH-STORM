# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:50:17 2021
This is the script for generating the genesis matrices in Python 3 (with Cartopy)
"""

import numpy as np
import os
import sys
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import pandas as pd 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

land_shp_fname = shpreader.natural_earth(
    resolution="50m", category="physical", name="land"
)

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)


def is_land(x, y):
    return land.contains(sgeom.Point(x, y))


# print(is_land(150,20))


def create_mask(basin):
    stepsize = 10
    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)
    x = int(abs(lon1 - lon0) * stepsize)
    y = int(abs(lat1 - lat0) * stepsize)
    if lon0 < 180:  # south pacific
        lon_grid, lat_grid = np.mgrid[
            lon0 : lon1 : complex(0, x), lat0 : lat1 : complex(0, y)
        ]
    else:
        lon_grid, lat_grid = np.mgrid[
            lon0 - 360 : lon1 - 360 : complex(0, x), lat0 : lat1 : complex(0, y)
        ]

    mask = np.ones((len(lon_grid[0]), len(lon_grid)))
    for i in range(len(lon_grid)):
        for j in range(len(lon_grid[i])):
            mask[j][i] = is_land(lon_grid[i][j], lat_grid[i][j])

    mask = np.flipud(mask)

    return mask


def BOUNDARIES_BASINS(idx):
    if idx == "EP":  # Eastern Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 180, 285
    if idx == "NA":  # North Atlantic
        lat0, lat1, lon0, lon1 = 5, 60, 255, 359
    if idx == "NI":  # North Indian
        lat0, lat1, lon0, lon1 = 5, 60, 30, 100
    if idx == "SI":  # South Indian
        lat0, lat1, lon0, lon1 = -60, -5, 10, 135
    if idx == "SP":  # South Pacific
        lat0, lat1, lon0, lon1 = -60, -5, 135, 240
    if idx == "WP":  # Western Pacific
        lat0, lat1, lon0, lon1 = 5, 60, 100, 180

    return lat0, lat1, lon0, lon1


def create_5deg_grid(locations, month, basin):
    step = 5.0

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)
    if basin == "NA":
        lonspace = np.linspace(lon0, 360.0, int(abs(lon0 - 360.0) / step) + 1)
    else:
        lonspace = np.linspace(lon0, lon1, int(abs(lon0 - lon1) / step) + 1)

    latspace = np.linspace(lat0, lat1, int(abs(lat0 - lat1) / step) + 1)

    lat_list = [
        locations[month][i][0]
        for i in range(len(locations[month]))
        if (
            lat0 <= locations[month][i][0] <= lat1
            and lon0 <= locations[month][i][1] <= lon1
        )
    ]
    lon_list = [
        locations[month][i][1]
        for i in range(len(locations[month]))
        if (
            lat0 <= locations[month][i][0] <= lat1
            and lon0 <= locations[month][i][1] <= lon1
        )
    ]

    df = pd.DataFrame({"Latitude": lat_list, "Longitude": lon_list})

    to_bin = lambda x: np.floor(x / step) * step
    df["latbin"] = df.Latitude.map(to_bin)
    df["lonbin"] = df.Longitude.map(to_bin)
    groups = df.groupby(["latbin", "lonbin"])
    count_df = pd.DataFrame({"count": groups.size()}).reset_index()
    counts = count_df["count"]
    latbin = groups.count().index.get_level_values("latbin")
    lonbin = groups.count().index.get_level_values("lonbin")
    count_matrix = np.zeros((len(latspace), int(abs(lon0 - lon1) / step) + 1))

    for lat, lon, count in zip(latbin, lonbin, counts):
        i = latspace.tolist().index(lat)
        j = lonspace.tolist().index(lon)
        count_matrix[i, j] = count

    return count_matrix


def create_1deg_grid(delta_count_matrix, basin, month):
    step = 5.0

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)

    latspace = np.linspace(lat0, lat1, int(abs(lat0 - lat1) / step) + 1)
    lonspace = np.linspace(lon0, lon1, int(abs(lon0 - lon1) / step) + 1)

    xg = int(abs(lon1 - lon0))
    yg = int(abs(lat1 - lat0))
    xgrid, ygrid = np.mgrid[lon0 : lon1 : complex(0, xg), lat0 : lat1 : complex(0, yg)]
    points = []
    for i in range(len(lonspace)):
        for j in range(len(latspace)):
            points.append((lonspace[i], latspace[j]))

    values = np.reshape(delta_count_matrix.T, int(len(lonspace)) * int(len(latspace)))
    grid = griddata(points, values, (xgrid, ygrid), method="cubic")
    grid = np.transpose(grid)
    grid = np.flipud(grid)
    grid[grid < 0] = 0

    # overlay data with a land-sea mask
    mdata = create_mask(basin)
    coarseness = 10
    mdata_coarse = mdata.reshape(
        (
            mdata.shape[0] // coarseness,
            coarseness,
            mdata.shape[1] // coarseness,
            coarseness,
        )
    )
    mdata_coarse = np.mean(mdata_coarse, axis=(1, 3))

    (x, y) = mdata_coarse.shape

    for i in range(0, x):
        for j in range(0, y):
            if mdata_coarse[i, j] > 0.50:
                grid[i, j] = 0.0  # ---- FIX: use 0.0 instead of 'nan' string ----

    # ---- FIX: replace any remaining NaN with 0 ----
    grid = np.nan_to_num(grid, nan=0.0)

    plt.imshow(grid)
    plt.savefig("matrix" + str(basin) + str(month) + ".jpg", format="jpg")
    plt.close()

    return grid


def Change_genesis_locations_STORM():
    monthsall = {
        "EP": [6, 7, 8, 9, 10, 11],
        "NA": [6, 7, 8, 9, 10, 11],
        "NI": [4, 5, 6, 9, 10, 11],
        "SI": [1, 2, 3, 4, 11, 12],
        "SP": [1, 2, 3, 4, 11, 12],
        "WP": [5, 6, 7, 8, 9, 10, 11],
    }
    locations = np.load(
        os.path.join(__location__, "GEN_LOC.npy"), allow_pickle=True, encoding="latin1"
    ).item()

    for basin, idx in zip(["EP", "NA", "NI", "SI", "SP", "WP"], range(0, 6)):
        for month in monthsall[basin]:
            print(basin, idx)
            matrix_dict = create_5deg_grid(locations[idx], month, basin)
            genesis_grids = create_1deg_grid(matrix_dict, basin, month)
            np.savetxt(
                os.path.join(
                    __location__, "GRID_GENESIS_MATRIX_{}_{}.txt".format(idx, month)
                ),
                genesis_grids,
            )


# ---- FIX: Minimum genesis count threshold for phase-specific grids ----
MIN_GENESIS_FOR_PHASE_GRID = 5


def Change_genesis_locations(idx_basin, months):

    basin_name = ["EP", "NA", "NI", "SI", "SP", "WP"]
    monthsall = {
        "EP": months[0],
        "NA": months[1],
        "NI": months[2],
        "SI": months[3],
        "SP": months[4],
        "WP": months[5],
    }

    locations = np.load(
        os.path.join(__location__, "GEN_LOC.npy"), allow_pickle=True, encoding="latin1"
    ).item()
    locations_phase = None
    phase_path = os.path.join(__location__, "GEN_LOC_PHASE.npy")
    if os.path.exists(phase_path):
        locations_phase = np.load(
            phase_path, allow_pickle=True, encoding="latin1"
        ).item()

    for ii in range(len(idx_basin)):
        idx = idx_basin[ii]
        basin = basin_name[idx]
        for month in monthsall[basin]:
            print("genesis grid for basin ", basin, "and month ", month)
            matrix_dict = create_5deg_grid(locations[idx], month, basin)
            genesis_grids = create_1deg_grid(matrix_dict, basin, month)
            np.savetxt(
                os.path.join(
                    __location__, "GRID_GENESIS_MATRIX_{}_{}.txt".format(idx, month)
                ),
                genesis_grids,
            )
            if locations_phase is not None:
                for phase in ["LN", "NEU", "EN"]:
                    phase_locs = locations_phase[idx][month].get(phase, [])

                    # ---- FIX: skip phase grid if too few genesis events ----
                    if len(phase_locs) < MIN_GENESIS_FOR_PHASE_GRID:
                        # Don't write a phase-specific file — SAMPLE_STARTING_POINT
                        # will automatically fall back to the pooled grid
                        print(
                            f"  Skipping phase grid {basin}/{month}/{phase}: "
                            f"only {len(phase_locs)} events (min={MIN_GENESIS_FOR_PHASE_GRID})"
                        )
                        # Remove stale file if it exists from a previous run
                        stale = os.path.join(
                            __location__,
                            "GRID_GENESIS_MATRIX_{}_{}_{}.txt".format(
                                idx, month, phase
                            ),
                        )
                        if os.path.exists(stale):
                            os.remove(stale)
                        continue

                    try:
                        phase_loc = {idx: {month: phase_locs}}
                        matrix_dict_phase = create_5deg_grid(
                            phase_loc[idx], month, basin
                        )
                        genesis_phase = create_1deg_grid(
                            matrix_dict_phase, basin, month
                        )
                        np.savetxt(
                            os.path.join(
                                __location__,
                                "GRID_GENESIS_MATRIX_{}_{}_{}.txt".format(
                                    idx, month, phase
                                ),
                            ),
                            genesis_phase,
                        )
                    except Exception as e:
                        print(f"  Phase grid failed for {basin}/{month}/{phase}: {e}")
                        # Remove stale file
                        stale = os.path.join(
                            __location__,
                            "GRID_GENESIS_MATRIX_{}_{}_{}.txt".format(
                                idx, month, phase
                            ),
                        )
                        if os.path.exists(stale):
                            os.remove(stale)
