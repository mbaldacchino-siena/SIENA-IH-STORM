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
from CODE.siena_utils import load_monthly_field
from scipy.ndimage import zoom


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
        lat0, lat1, lon0, lon1 = 5, 60, 255, 360
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

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)
    xg = int(abs(lon1 - lon0))
    yg = int(abs(lat1 - lat0))

    # Try to load environmental fields
    has_any = False
    pi_field = vws_field = rh_field = None

    phase_str = phase if phase is not None else None

    # Load PI (thermodynamic or empirical)
    try:
        pi_global = load_monthly_field(
            __location__, "Monthly_mean_PI", month, phase=phase_str
        )
        # Extract basin region — PI is on global SST grid (721 x 1440, 0.25deg)
        lat_grid = np.linspace(90, -90, pi_global.shape[0])
        lon_grid = np.linspace(0, 359.75, pi_global.shape[1])
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        pi_basin = pi_global[lat_0i:lat_1i, lon_0i:lon_1i]
        # Resample to 1-degree grid

        if pi_basin.shape[0] > 0 and pi_basin.shape[1] > 0:
            zy = yg / pi_basin.shape[0]
            zx = xg / pi_basin.shape[1]
            pi_field = zoom(pi_basin, (zy, zx), order=1)
            has_any = True
    except Exception:
        pass

    # Load VWS
    try:
        vws_global = load_monthly_field(
            __location__, "Monthly_mean_VWS", month, phase=phase_str
        )
        lat_grid = np.linspace(90, -90, vws_global.shape[0])
        lon_grid = np.linspace(0, 359.75, vws_global.shape[1])
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        vws_basin = vws_global[lat_0i:lat_1i, lon_0i:lon_1i]
        if vws_basin.shape[0] > 0 and vws_basin.shape[1] > 0:
            zy = yg / vws_basin.shape[0]
            zx = xg / vws_basin.shape[1]
            vws_field = zoom(vws_basin, (zy, zx), order=1)
            has_any = True
    except Exception:
        pass

    # Load RH600
    try:
        rh_global = load_monthly_field(
            __location__, "Monthly_mean_RH600", month, phase=phase_str
        )
        lat_grid = np.linspace(90, -90, rh_global.shape[0])
        lon_grid = np.linspace(0, 359.75, rh_global.shape[1])
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        rh_basin = rh_global[lat_0i:lat_1i, lon_0i:lon_1i]
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
        pi_term = pi_term[: env.shape[0], : env.shape[1]]
        env[: pi_term.shape[0], : pi_term.shape[1]] *= pi_term

    if vws_field is not None:
        vws_field = np.nan_to_num(vws_field, nan=15.0)
        vws_term = np.exp(-0.10 * np.maximum(vws_field, 0.0))
        vws_term = vws_term[: env.shape[0], : env.shape[1]]
        env[: vws_term.shape[0], : vws_term.shape[1]] *= vws_term

    if rh_field is not None:
        rh_field = np.nan_to_num(rh_field, nan=50.0)
        # RH can be in fraction (0-1) or percent (0-100)
        if np.nanmax(rh_field) < 2.0:
            rh_field = rh_field * 100.0
        rh_term = np.clip(rh_field / 60.0, 0.2, 2.5)
        rh_term = rh_term[: env.shape[0], : env.shape[1]]
        env[: rh_term.shape[0], : rh_term.shape[1]] *= rh_term

    env = np.nan_to_num(env, nan=0.0, posinf=0.0, neginf=0.0)
    env[env < 0] = 0.0

    # Normalize
    s = env.sum()
    if s > 0:
        env = env / s

    return env


def _blend_genesis_with_env(genesis, env, label=""):
    """
    Additive blend of observed genesis density with environmental (GPI) field.

    Pure multiplication zeros out GPI-favorable cells that lack observed genesis
    in 40 years — exactly the tail events a 10,000-year catalog must capture.
    Pure replacement ignores observational constraints and places genesis wherever
    the thermodynamics are favorable, including regions where synoptic-scale
    organization never occurs.

    The additive blend:
        final = (1 - w) * obs_normalized + w * env_normalized

    preserves the observed core while allowing genesis at unsampled but
    environmentally favorable locations.  The mixing weight w is computed
    adaptively as the fraction of GPI-favorable cells that have no observed
    genesis — larger when observations are sparse (NI, phase-specific grids),
    smaller when dense (NA pooled).

    Total genesis mass is preserved so the Poisson storm count is unaffected.

    Parameters
    ----------
    genesis : 2D array, observed genesis density (1-deg grid, possibly sparse)
    env : 2D array, environmental field (GPI or custom), same grid, normalized
    label : str, for logging

    Returns
    -------
    blended : 2D array, same shape as genesis, with original total mass preserved
    """
    rows = min(genesis.shape[0], env.shape[0])
    cols = min(genesis.shape[1], env.shape[1])
    obs = genesis[:rows, :cols].copy()
    gpi = env[:rows, :cols].copy()

    gpi = np.nan_to_num(gpi, nan=0.0)
    gpi[gpi < 0] = 0.0
    obs = np.nan_to_num(obs, nan=0.0)
    obs[obs < 0] = 0.0

    original_mass = obs.sum()
    if original_mass <= 0 or gpi.sum() <= 0:
        return genesis  # nothing to blend

    # Adaptive mixing weight: fraction of GPI-favorable cells unsampled by obs.
    # "Favorable" = top 75% of nonzero GPI cells (bottom 25% are marginal).
    gpi_nonzero = gpi[gpi > 0]
    if len(gpi_nonzero) < 5:
        return genesis

    threshold = np.percentile(gpi_nonzero, 25)
    favorable_mask = gpi >= threshold
    observed_mask = obs > 0
    n_favorable = favorable_mask.sum()
    n_covered = (favorable_mask & observed_mask).sum()
    coverage = n_covered / n_favorable if n_favorable > 0 else 1.0

    # w = fraction of favorable space not covered by observations
    # Clamp to [0.05, 0.40] — at least 5% GPI contribution (even dense basins
    # have unsampled tail events), at most 40% (observations should dominate)
    w = np.clip(1.0 - coverage, 0.05, 0.50)

    # Normalize each component to unit sum before blending
    obs_norm = obs / obs.sum()
    gpi_norm = gpi / gpi.sum()

    blended = (1.0 - w) * obs_norm + w * gpi_norm
    blended = np.nan_to_num(blended, nan=0.0)
    blended[blended < 0] = 0.0

    # Restore original total mass
    if blended.sum() > 0:
        blended = blended * (original_mass / blended.sum())

    # Write back into full-size grid
    result = genesis.copy()
    result[:rows, :cols] = blended

    print(
        f"  {label}: w={w:.3f} (coverage={coverage:.3f}), "
        f"obs_nonzero={int(observed_mask.sum())}, "
        f"gpi_favorable={int(n_favorable)}"
    )

    return result


def Change_genesis_locations(idx_basin, months, genesis_weighting):

    basin_name = ["EP", "NA", "NI", "SI", "SP", "WP"]
    monthsall = {
        "EP": months[0],
        "NA": months[1],
        "NI": months[2],
        "SI": months[3],
        "SP": months[4],
        "WP": months[5],
    }
    genesis_mode = genesis_weighting

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
            env_weight = None

            matrix_dict = create_5deg_grid(locations[idx], month, basin)
            genesis_grids = create_1deg_grid(matrix_dict, basin, month)

            if genesis_mode == "GPI-MIX":
                env_weight = compute_gpi_field(basin, month, phase=None)
                np.savetxt(
                    os.path.join(
                        __location__, "GRID_GPIMIX_MATRIX_{}_{}.txt".format(idx, month)
                    ),
                    env_weight,
                )
            elif genesis_mode == "GPI":
                genesis_grids = compute_gpi_field(basin, month, phase=None)
                np.savetxt(
                    os.path.join(
                        __location__, "GRID_GPI_MATRIX_{}_{}.txt".format(idx, month)
                    ),
                    genesis_grids,
                )
            elif genesis_mode != "EMPIRICAL":
                env_weight = build_environmental_genesis_factor(
                    basin, month, phase=None
                )
                np.savetxt(
                    os.path.join(
                        __location__, "GRID_ENVPI_MATRIX_{}_{}.txt".format(idx, month)
                    ),
                    env_weight,
                )

            # Additive blend: observed genesis + environmental prior
            if env_weight is not None:
                genesis_grids = _blend_genesis_with_env(
                    genesis_grids, env_weight, label=f"pooled {basin}/{month}"
                )

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

                        # Load phase-specific environmental weighting
                        env_phase = None
                        if genesis_mode == "GPI-MIX":
                            env_phase = compute_gpi_field(basin, month, phase=phase)
                        elif genesis_mode == "GPI":
                            genesis_phase = compute_gpi_field(basin, month, phase=phase)
                        elif genesis_mode != "EMPIRICAL":
                            env_phase = build_environmental_genesis_factor(
                                basin, month, phase=phase
                            )

                        # Additive blend: observed phase genesis + phase-specific GPI
                        if env_phase is not None:
                            genesis_phase = _blend_genesis_with_env(
                                genesis_phase,
                                env_phase,
                                label=f"{phase} {basin}/{month}",
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
                        stale = os.path.join(
                            __location__,
                            "GRID_GENESIS_MATRIX_{}_{}_{}.txt".format(
                                idx, month, phase
                            ),
                        )
                        if os.path.exists(stale):
                            os.remove(stale)


##########################
#    GPI = Genesis Potential Intensity
##########################


def compute_gpi_field(basin, month, phase=None):
    """
        Compute the Genesis Potential Index following Emanuel & Nolan (2004)
        and Camargo et al. (2007, J. Climate, 20, 4819-4834).

        GPI = |10⁵ η|^(3/2) · (H/50)³ · (Vpot/70)³ · (1 + 0.1·VWS)⁻²

        Returns a 1-degree grid matching the basin dimensions, or None
        if required fields are missing.

    *Emanuel, K. A., & Nolan, D. S. (2004). Tropical cyclone activity and the global climate system.
        Preprints, 26th Conf. on Hurricanes and Tropical Meteorology, Miami, FL, Amer. Meteor. Soc., 240–241.
    *Camargo, S. J., Emanuel, K. A., & Sobel, A. H. (2007). Use of a Genesis Potential Index to Diagnose
        ENSO Effects on Tropical Cyclone Genesis. J. Climate, 20, 4819–4834. doi:10.1175/JCLI4282.1
    *Tippett, M. K., Camargo, S. J., & Sobel, A. H. (2011). A Poisson Regression Index for Tropical Cyclone
      Genesis and the Role of Large-Scale Vorticity in Genesis. J. Climate, 24, 2335–2357.

    Reference 3 is important because Tippett et al. (2011) showed that the vorticity term dominates
      GPI variability in the WP during ENSO — this directly supports why your current implementation
        (without vorticity) misses the primary mechanism.
    """

    lat0, lat1, lon0, lon1 = BOUNDARIES_BASINS(basin)
    xg = int(abs(lon1 - lon0))  # 1-degree grid
    yg = int(abs(lat1 - lat0))
    phase_str = phase if phase is not None else None

    # --- Load fields ---
    try:
        vort_global = load_monthly_field(
            __location__, "Monthly_mean_VORT850", month, phase=phase_str
        )
    except Exception:
        print(f"  GPI: Missing VORT850 for month={month}, phase={phase_str}")
        return None

    try:
        rh_global = load_monthly_field(
            __location__, "Monthly_mean_RH600", month, phase=phase_str
        )
    except Exception:
        print(f"  GPI: Missing RH600 for month={month}, phase={phase_str}")
        return None

    try:
        vpot_global = load_monthly_field(
            __location__, "Monthly_mean_VMAX_PI", month, phase=phase_str
        )
    except Exception:
        print(f"  GPI: Missing VMAX_PI for month={month}, phase={phase_str}")
        return None

    try:
        vws_global = load_monthly_field(
            __location__, "Monthly_mean_VWS", month, phase=phase_str
        )
    except Exception:
        print(f"  GPI: Missing VWS for month={month}, phase={phase_str}")
        return None

    # --- Extract basin region and resample to 1-degree ---
    def _extract_and_resample(field_global, target_rows, target_cols):
        """Extract basin sub-region from a global field, resample to target grid."""
        nlat, nlon = field_global.shape
        lat_grid = np.linspace(90, -90, nlat)
        lon_grid = np.linspace(0, 360 * (nlon - 1) / nlon, nlon)
        lat_0i = np.abs(lat_grid - lat1).argmin()
        lat_1i = np.abs(lat_grid - lat0).argmin()
        lon_0i = np.abs(lon_grid - lon0).argmin()
        lon_1i = np.abs(lon_grid - lon1).argmin()
        sub = field_global[lat_0i:lat_1i, lon_0i:lon_1i]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            return None
        zy = target_rows / sub.shape[0]
        zx = target_cols / sub.shape[1]
        return zoom(sub, (zy, zx), order=1)

    vort = _extract_and_resample(vort_global, yg, xg)
    rh = _extract_and_resample(rh_global, yg, xg)
    vpot = _extract_and_resample(vpot_global, yg, xg)
    vws = _extract_and_resample(vws_global, yg, xg)

    if any(f is None for f in [vort, rh, vpot, vws]):
        return None

    # --- Compute GPI ---
    # Absolute vorticity = relative vorticity + Coriolis
    lat_1d = np.linspace(lat1, lat0, yg)  # top to bottom
    f_coriolis = 2.0 * 7.2921e-5 * np.sin(np.deg2rad(lat_1d))
    f_2d = f_coriolis[:, None] * np.ones((yg, xg))
    eta = np.abs(vort + f_2d)
    eta = np.maximum(eta, 2e-6)  # floor near equator

    # RH handling (could be fraction or %)
    rh = np.nan_to_num(rh, nan=50.0)
    if np.nanmax(rh) < 2.0:
        rh = rh * 100.0

    vpot = np.nan_to_num(vpot, nan=0.0)
    vws = np.nan_to_num(vws, nan=15.0)

    # GPI = |10⁵ η|^1.5 · (H/50)³ · (Vpot/70)³ · (1 + 0.1·VWS)⁻²
    vort_term = (1e5 * eta) ** 1.5
    rh_term = (np.clip(rh, 1.0, 100.0) / 50.0) ** 3
    pi_term = (np.clip(vpot, 0.0, 120.0) / 70.0) ** 3
    shear_term = (1.0 + 0.1 * np.clip(vws, 0.0, 50.0)) ** (-2)

    gpi = vort_term * rh_term * pi_term * shear_term
    gpi = np.nan_to_num(gpi, nan=0.0, posinf=0.0, neginf=0.0)
    gpi[gpi < 0] = 0.0

    # Normalize to unit sum
    s = gpi.sum()
    if s > 0:
        gpi = gpi / s

    return gpi
