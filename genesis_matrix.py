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


def build_environmental_genesis_factor(basin, month, phase=None):
    """
    Build an environmental genesis favorability grid based on PI, VWS, and RH.

    Follows the Genesis Potential Index (GPI) logic of Camargo et al. (2007,
    doi:10.1175/JCLI4282.1) and Emanuel & Nolan (2004). The multiplicative
    form captures ENSO-driven spatial shifts in genesis far better than sparse
    observational counts alone.

    The environmental factor is:
        E = PI_term * VWS_term * RH_term

    where:
        PI_term  = clip((PI_ref - PI) / PI_scale, 0, ...)   [lower PI = more intense ceiling = favorable]
        VWS_term = exp(-alpha * VWS)                         [low shear = favorable]
        RH_term  = clip(RH / RH_ref, 0.2, 2.0)              [high RH = favorable]

    Returns a normalized 1-degree grid matching the basin dimensions,
    or None if no environmental fields are available.
    """
    from siena_utils import load_monthly_field

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)
    xg = int(abs(lon1 - lon0))
    yg = int(abs(lat1 - lat0))

    # Try to load environmental fields
    has_any = False
    pi_field = vws_field = rh_field = None

    phase_str = phase if phase is not None else None

    # Load PI (thermodynamic or empirical)
    try:
        pi_global = load_monthly_field(__location__, "Monthly_mean_PI", month, phase=phase_str)
        # Extract basin region — PI is on global SST grid (721 x 1440, 0.25deg)
        lat_grid = np.linspace(90, -90, pi_global.shape[0])
        lon_grid = np.linspace(0, 359.75, pi_global.shape[1])
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        pi_basin = pi_global[lat_0i:lat_1i, lon_0i:lon_1i]
        # Resample to 1-degree grid
        from scipy.ndimage import zoom
        if pi_basin.shape[0] > 0 and pi_basin.shape[1] > 0:
            zy = yg / pi_basin.shape[0]
            zx = xg / pi_basin.shape[1]
            pi_field = zoom(pi_basin, (zy, zx), order=1)
            has_any = True
    except Exception:
        pass

    # Load VWS
    try:
        vws_global = load_monthly_field(__location__, "Monthly_mean_VWS", month, phase=phase_str)
        lat_grid = np.linspace(90, -90, vws_global.shape[0])
        lon_grid = np.linspace(0, 359.75, vws_global.shape[1])
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        vws_basin = vws_global[lat_0i:lat_1i, lon_0i:lon_1i]
        from scipy.ndimage import zoom
        if vws_basin.shape[0] > 0 and vws_basin.shape[1] > 0:
            zy = yg / vws_basin.shape[0]
            zx = xg / vws_basin.shape[1]
            vws_field = zoom(vws_basin, (zy, zx), order=1)
            has_any = True
    except Exception:
        pass

    # Load RH600
    try:
        rh_global = load_monthly_field(__location__, "Monthly_mean_RH600", month, phase=phase_str)
        lat_grid = np.linspace(90, -90, rh_global.shape[0])
        lon_grid = np.linspace(0, 359.75, rh_global.shape[1])
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        rh_basin = rh_global[lat_0i:lat_1i, lon_0i:lon_1i]
        from scipy.ndimage import zoom
        if rh_basin.shape[0] > 0 and rh_basin.shape[1] > 0:
            zy = yg / rh_basin.shape[0]
            zx = xg / rh_basin.shape[1]
            rh_field = zoom(rh_basin, (zy, zx), order=1)
            has_any = True
    except Exception:
        pass

    if not has_any:
        return None

    # Build environmental factor
    env = np.ones((yg, xg))

    if pi_field is not None:
        # PI is minimum central pressure — lower = more intense ceiling = more favorable
        # Normalize: favorable where PI is low (< ~980 hPa)
        pi_field = np.nan_to_num(pi_field, nan=1020.0)
        pi_term = np.clip((1020.0 - pi_field) / 60.0, 0.0, 3.0)
        # Ensure shape match
        pi_term = pi_term[:env.shape[0], :env.shape[1]]
        env[:pi_term.shape[0], :pi_term.shape[1]] *= pi_term

    if vws_field is not None:
        vws_field = np.nan_to_num(vws_field, nan=15.0)
        vws_term = np.exp(-0.10 * np.maximum(vws_field, 0.0))
        vws_term = vws_term[:env.shape[0], :env.shape[1]]
        env[:vws_term.shape[0], :vws_term.shape[1]] *= vws_term

    if rh_field is not None:
        rh_field = np.nan_to_num(rh_field, nan=50.0)
        # RH can be in fraction (0-1) or percent (0-100)
        if np.nanmax(rh_field) < 2.0:
            rh_field = rh_field * 100.0
        rh_term = np.clip(rh_field / 60.0, 0.2, 2.5)
        rh_term = rh_term[:env.shape[0], :env.shape[1]]
        env[:rh_term.shape[0], :rh_term.shape[1]] *= rh_term

    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    env[env < 0] = 0.0

    # Normalize
    s = env.sum()
    if s > 0:
        env = env / s

    return env


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

            # Fix 5: Apply environmental genesis weighting (pooled)
            env_pooled = build_environmental_genesis_factor(basin, month, phase=None)
            if env_pooled is not None:
                # Ensure shape compatibility
                rows = min(genesis_grids.shape[0], env_pooled.shape[0])
                cols = min(genesis_grids.shape[1], env_pooled.shape[1])
                weighted = genesis_grids[:rows, :cols] * env_pooled[:rows, :cols]
                weighted = np.nan_to_num(weighted, nan=0.0)
                weighted[weighted < 0] = 0.0
                if weighted.sum() > 0:
                    # Preserve original total mass so Poisson count is unaffected
                    weighted = weighted * (genesis_grids.sum() / weighted.sum())
                    genesis_grids[:rows, :cols] = weighted[:rows, :cols]
                print(f"  Applied environmental weighting (pooled) for {basin}/{month}")

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
                        print(
                            f"  Skipping phase grid {basin}/{month}/{phase}: "
                            f"only {len(phase_locs)} events (min={MIN_GENESIS_FOR_PHASE_GRID})"
                        )
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

                        # Fix 5: Apply phase-specific environmental weighting
                        env_phase = build_environmental_genesis_factor(basin, month, phase=phase)
                        if env_phase is not None:
                            rows = min(genesis_phase.shape[0], env_phase.shape[0])
                            cols = min(genesis_phase.shape[1], env_phase.shape[1])
                            weighted = genesis_phase[:rows, :cols] * env_phase[:rows, :cols]
                            weighted = np.nan_to_num(weighted, nan=0.0)
                            weighted[weighted < 0] = 0.0
                            if weighted.sum() > 0:
                                weighted = weighted * (genesis_phase.sum() / weighted.sum())
                                genesis_phase[:rows, :cols] = weighted[:rows, :cols]
                            print(f"  Applied environmental weighting ({phase}) for {basin}/{month}")

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
                        stale = os.path.join(
                            __location__,
                            "GRID_GENESIS_MATRIX_{}_{}_{}.txt".format(
                                idx, month, phase
                            ),
                        )
                        if os.path.exists(stale):
                            os.remove(stale)
