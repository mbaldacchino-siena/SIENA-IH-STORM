"""
@author: Mathys Baldacchino, mathys.baldacchino@sienacapitalgroup.com

Climatology's cyclone to execute SIENA-IH-STORM
This programs select SIENA-IH-STORM data (cyclones, SST, SLP, VWS, RH) for different climatology periods
It also downloads necessary profile data for Potential Indexing.
It is strongly inspired by I. Oderiz (C) 2025 [Itxaso Oderiz, itxaso.oderiz@unican.es]

Copyright (C) 2026 Mathys Baldacchino
"""

import xarray as xr
import requests
import os
import os.path as op
import cdsapi
import pandas as pd
import numpy as np
import potential_intensity

from pathlib import Path


def _get_IBStrack(url, local_path, file_name):
    # Specify the local file path where you want to save the downloaded file
    local_file_path = local_path + file_name

    if not Path(local_file_path).is_file():
        # Send an HTTP GET request to the URL
        response = requests.get(url + file_name)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the local file in binary write mode and write the content of the response to it
            with open(local_file_path, "wb") as file:
                file.write(response.content)
            print(f"File '{url}' downloaded successfully.")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    else:
        print("Already downloaded")


# ==============================================================================
# Function to get climate index
# ==============================================================================


def _get_climate_index(url, local_path):

    # select the index from https://psl.noaa.gov/data/climateindices/list/
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    data = response.text
    lines = data.split("\n")
    # delete the first row
    lines = lines[1:]
    # create empty arrays and DataFRame
    climate_index0 = pd.DataFrame()
    YEAR = []
    MONTH = []
    CLIMATE_INDEX = []
    # Iterate over the lines and extract the year, month and climate index values
    for line in lines:
        # print(line)
        if line.strip():
            try:
                parts = line.split()
                values = [float(x) for x in parts[1:]]
                year = np.array([int(parts[0])] * 12)
                month = np.array(range(1, 13))
                climate_index = np.array(values)
                YEAR.append(year)
                MONTH.append(month)
                CLIMATE_INDEX.append(climate_index)
            except:
                continue
    climate_index0["year"] = np.concatenate(YEAR)
    climate_index0["month"] = np.concatenate(MONTH)
    climate_index0["climate_index"] = np.concatenate(CLIMATE_INDEX)
    # delete non registered values -999
    climate_index0 = climate_index0[climate_index0["climate_index"] != -999]
    climate_index0 = climate_index0[climate_index0["climate_index"] != -99]
    climate_index0.to_csv(local_path + "/climate_index.csv")
    print(f"File '{url}' downloaded successfully.")


# ==============================================================================
# Function to get SLP from ERA5
# ==============================================================================


def _download_monthly_mean_SLP(dir_data, year_list):

    my_file = op.join(dir_data, "Monthly_mean_MSLP.nc")

    if not Path(my_file).is_file():
        # change years accorign to the slide period
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "format": "netcdf",
                "product_type": "monthly_averaged_reanalysis",
                "variable": "mean_sea_level_pressure",
                "year": year_list,
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "time": "00:00",
            },
            my_file,
        )


# ==============================================================================
# Function to get SST from ERA5
# ==============================================================================


def _download_monthly_mean_SST(dir_data, year_list):
    my_file = op.join(dir_data, "Monthly_mean_SST.nc")

    if not Path(my_file).is_file():
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "format": "netcdf",
                "product_type": "monthly_averaged_reanalysis",
                "variable": "sea_surface_temperature",
                "year": year_list,
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "time": "00:00",
            },
            my_file,
        )


# ==============================================================================
# Function to get VWS from ERA5
# ==============================================================================


def _download_wind_shear_data(dir_data, year_list):
    out_path = op.join(dir_data, "Monthly_mean_VWS_components.nc")
    # Save 850 hPa components for vorticity computation
    u850_path = op.join(dir_data, "Monthly_mean_U850.nc")
    v850_path = op.join(dir_data, "Monthly_mean_V850.nc")

    if not Path(out_path).is_file():
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "format": "netcdf",
                "product_type": "monthly_averaged_reanalysis",
                "variable": ["u_component_of_wind", "v_component_of_wind"],
                "pressure_level": ["200", "850"],
                "year": year_list,
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "time": "00:00",
            },
            out_path,
        )
    my_file = op.join(dir_data, "Monthly_mean_VWS.nc")

    if not Path(my_file).is_file():
        ds = xr.open_dataset(out_path)
        u_name = "u" if "u" in ds.data_vars else list(ds.data_vars)[0]
        v_name = "v" if "v" in ds.data_vars else list(ds.data_vars)[1]
        level_name = "pressure_level" if "pressure_level" in ds.coords else "level"
        u200 = ds[u_name].sel({level_name: 200})
        u850 = ds[u_name].sel({level_name: 850})
        v200 = ds[v_name].sel({level_name: 200})
        v850 = ds[v_name].sel({level_name: 850})
        vws = np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        xr.Dataset({"vws": vws}).to_netcdf(my_file)
        xr.Dataset({"v850": v850}).to_netcdf(v850_path)
        xr.Dataset({"u850": u850}).to_netcdf(u850_path)
        ds.close()


# ==============================================================================
# Function to get Relative Humidity from ERA5
# ==============================================================================


def _download_humidity_data(dir_data, year_list):
    my_file = op.join(dir_data, "Monthly_mean_RH600.nc")
    if not Path(my_file).is_file():
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "format": "netcdf",
                "product_type": "monthly_averaged_reanalysis",
                "variable": "relative_humidity",
                "pressure_level": "600",
                "year": year_list,
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "time": "00:00",
            },
            my_file,
        )


# ==============================================================================
# Function to get T and Q profiles from ERA5 (at 1 degree)
# ==============================================================================


def _download_profile_data(dir_data, year_list):
    """Download ERA5 T and Q on pressure levels for PI computation."""
    for varname, filename in [
        ("temperature", "Monthly_mean_T.nc"),
        ("specific_humidity", "Monthly_mean_Q.nc"),
    ]:
        out = op.join(dir_data, filename)
        if Path(out).is_file():
            continue
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "format": "netcdf",
                "product_type": "monthly_averaged_reanalysis",
                "variable": varname,
                "pressure_level": [
                    "1000",
                    "925",
                    "850",
                    "700",
                    "600",
                    "500",
                    "400",
                    "300",
                    "250",
                    "200",
                    "150",
                    "100",
                ],
                "year": year_list,
                "month": [
                    "01",
                    "02",
                    "03",
                    "04",
                    "05",
                    "06",
                    "07",
                    "08",
                    "09",
                    "10",
                    "11",
                    "12",
                ],
                "time": "00:00",
                "grid": ["1.0", "1.0"],  # coarser grid to save RAM
            },
            out,
        )


def climatology_data(year):
    """Download the necessary data if not yet in local."""

    # identifying local path (where this code is located)
    local_path = os.getcwd()
    year_list = [str(year) for year in range(year[0], year[1] + 1)]

    _download_monthly_mean_SLP(local_path, year_list)
    _download_monthly_mean_SST(local_path, year_list)
    _download_humidity_data(local_path, year_list)
    _download_wind_shear_data(local_path, year_list)
    _download_profile_data(local_path, year_list)

    # download data from IBTrACS
    file_name = "/IBTrACS.ALL.v04r01.nc"

    _get_IBStrack(
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/netcdf/",
        local_path,
        file_name,
    )

    # open file of IBStrack
    cyclones = xr.open_dataset(local_path + file_name)
    filtered_cyclones = cyclones.where(cyclones.season >= year[0], drop=True)
    filtered_cyclones = filtered_cyclones.where(
        filtered_cyclones.season <= year[1], drop=True
    )
    filtered_cyclones.to_netcdf(
        local_path + "/IBTrACS." + str(year[0]) + "_" + str(year[1]) + "v04r01.nc"
    )


###########################################################
#######    COMPUTE    VORTICITY    FOR GENESIS   PI
###########################################################


def _compute_and_save_vorticity(u850_path, v850_path, oni_df, out_dir):
    """
    Compute 850 hPa relative vorticity from u,v components.
    Save pooled and phase-specific monthly climatologies.
    """
    ds_u = xr.open_dataset(u850_path)
    ds_v = xr.open_dataset(v850_path)

    time_dim = "valid_time" if "valid_time" in ds_u.dims else "time"
    lat = ds_u.latitude.values  # degrees, typically 90 to -90
    lon = ds_u.longitude.values

    # Grid spacing in meters
    R = 6.371e6  # Earth radius
    dlat = np.abs(np.mean(np.diff(lat)))  # degrees
    dlon = np.abs(np.mean(np.diff(lon)))  # degrees
    dy = np.deg2rad(dlat) * R  # constant
    # dx varies with latitude
    dx = np.deg2rad(dlon) * R * np.cos(np.deg2rad(lat))  # (nlat,)

    # Phase lookup
    phase_lookup = {}
    for _, row in oni_df.iterrows():
        phase_lookup[(int(row["year"]), int(row["month"]))] = (
            str(row["phase"]).strip().upper()
        )

    times = pd.to_datetime(ds_u[time_dim].values)
    PHASES = ["LN", "NEU", "EN"]
    accum = {m: {ph: [] for ph in PHASES} for m in range(1, 13)}
    accum_pooled = {m: [] for m in range(1, 13)}

    u_var = [v for v in ds_u.data_vars][0]
    v_var = [v for v in ds_v.data_vars][0]

    for t_idx in range(len(times)):
        yr, mo = int(times[t_idx].year), int(times[t_idx].month)
        ph = phase_lookup.get((yr, mo), "NEU")

        u = ds_u[u_var].isel({time_dim: t_idx}).values  # (lat, lon)
        v = ds_v[v_var].isel({time_dim: t_idx}).values

        # Relative vorticity: ζ = ∂v/∂x − ∂u/∂y
        # Central differences
        dvdx = np.gradient(v, axis=1) / dx[:, None]  # (lat, lon)
        dudy = np.gradient(u, axis=0) / dy  # (lat, lon)
        vort = dvdx - dudy

        accum[mo][ph].append(vort)
        accum_pooled[mo].append(vort)

    ds_u.close()
    ds_v.close()

    for m in range(1, 13):
        if accum_pooled[m]:
            pooled = np.nanmean(np.stack(accum_pooled[m]), axis=0)
            np.savetxt(os.path.join(out_dir, f"Monthly_mean_VORT850_{m}.txt"), pooled)
        for ph in PHASES:
            if accum[m][ph]:
                field = np.nanmean(np.stack(accum[m][ph]), axis=0)
            elif m in accum_pooled and accum_pooled[m]:
                field = pooled
            else:
                continue
            np.savetxt(
                os.path.join(out_dir, f"Monthly_mean_VORT850_{m}_{ph}.txt"), field
            )

    print("Vorticity climatologies complete.")


# ==========================
# SIENA additions for pooled + phase-aware climatologies
# ==========================


def _add_phase_labels(climate_df, positive_threshold=0.5, negative_threshold=-0.5):
    climate_df = climate_df.copy()
    climate_df["phase"] = np.where(
        climate_df["climate_index"] >= positive_threshold,
        "EN",
        np.where(climate_df["climate_index"] <= negative_threshold, "LN", "NEU"),
    )
    return climate_df


def _save_phase_table(climate_df, local_path):
    climate_df.to_csv(op.join(local_path, "climate_index.csv"), index=False)


def compute_phase_climatology(
    nc_path: str,
    varname: str | None,
    oni_df: pd.DataFrame,
    out_stem: str,
    out_dir: str | None,
    pressure_level_idx: int | None = None,
    unit_scale: float = 1.0,
) -> tuple[dict, dict]:
    """
    Unified phase-specific climatology builder.

    Parameters
    ----------
    nc_path : str, path to ERA5 .nc file
    varname : str or None. If None, auto-detect first data variable.
    oni_df : DataFrame with [year, month, phase]
    out_stem : str, output file prefix (e.g. 'Monthly_mean_SST')
    out_dir : str | None, output directory, if set to None, no save
    pressure_level_idx : int or None, select a single pressure level if needed
    unit_scale : float, multiply output by this (e.g. 0.01 for Pa→hPa)

    Returns
    -------
    clim : dict {month: {phase: ndarray}}, the phase-specific fields
    pooled : dict {month: ndarray}, the pooled fields
    """
    ds = xr.open_dataset(nc_path)

    # Normalize time dimension name
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"

    # Auto-detect variable
    if varname is None or varname not in ds:
        varname = list(ds.data_vars)[0]

    # Select pressure level if needed
    if pressure_level_idx is not None:
        lvl_dim = "pressure_level" if "pressure_level" in ds.dims else "level"
        ds = ds.isel({lvl_dim: pressure_level_idx})

    # Build (year, month) → phase lookup
    phase_lookup = {}
    for _, row in oni_df.iterrows():
        phase_lookup[(int(row["year"]), int(row["month"]))] = (
            str(row["phase"]).strip().upper()
        )

    times = pd.to_datetime(ds[time_dim].values)
    PHASES = ["LN", "NEU", "EN"]

    # Accumulate per month × phase
    accum = {m: {ph: [] for ph in PHASES} for m in range(1, 13)}
    accum_pooled = {m: [] for m in range(1, 13)}

    for t_idx in range(len(times)):
        yr, mo = int(times[t_idx].year), int(times[t_idx].month)
        ph = phase_lookup.get((yr, mo), "NEU")
        field = ds[varname].isel({time_dim: t_idx}).values
        accum[mo][ph].append(field)
        accum_pooled[mo].append(field)

    ds.close()

    # Average and save
    clim: dict = {m: {} for m in range(1, 13)}
    pooled: dict = {}

    for m in range(1, 13):
        # Pooled
        if accum_pooled[m]:
            p = np.nanmean(np.stack(accum_pooled[m]), axis=0) * unit_scale
            if out_dir is not None:
                np.savetxt(os.path.join(out_dir, f"{out_stem}_{m}.txt"), p)
            pooled[m] = p
        # Phase-specific
        for ph in PHASES:
            if accum[m][ph]:
                f = np.nanmean(np.stack(accum[m][ph]), axis=0) * unit_scale
            elif m in pooled:
                f = pooled[m]  # fallback to pooled
            else:
                continue
            if out_dir is not None:
                np.savetxt(os.path.join(out_dir, f"{out_stem}_{m}_{ph}.txt"), f)
            clim[m][ph] = f

    return clim, pooled


def build_pooled_and_phase_climatologies(
    period,
    climate_index="ONI",
    threshold=0.5,
):
    local_path = os.getcwd()

    # Fetch and label climate index
    _get_climate_index(
        f"https://psl.noaa.gov/data/correlation/{climate_index.lower()}.data",
        local_path,
    )
    climate_df = pd.read_csv(op.join(local_path, "climate_index.csv"))
    climate_df = climate_df[
        (climate_df["year"] >= period[0]) & (climate_df["year"] <= period[1])
    ]
    climate_df = _add_phase_labels(climate_df, abs(threshold), -abs(threshold))
    _save_phase_table(climate_df, local_path)

    # Build all climatologies through the single unified function
    variables = [
        ("Monthly_mean_SST.nc", "sst", "Monthly_mean_SST", None, 1.0),
        ("Monthly_mean_MSLP.nc", "msl", "Monthly_mean_MSLP", None, 0.01),
        ("Monthly_mean_VWS.nc", "vws", "Monthly_mean_VWS", None, 1.0),
        ("Monthly_mean_RH600.nc", "r", "Monthly_mean_RH600", 0, 1.0),
    ]

    for nc_file, varname, stem, plev, scale in variables:
        nc_path = op.join(local_path, nc_file)
        if op.exists(nc_path):
            compute_phase_climatology(
                nc_path,
                varname,
                climate_df,
                stem,
                local_path,
                pressure_level_idx=plev,
                unit_scale=scale,
            )
            print(f"{stem} done")

    # 2. PI — uses T, Q, SST, MSLP internally, saves only PI
    era5_paths = {}
    for key, fname in [
        ("sst", "Monthly_mean_SST.nc"),
        ("mslp", "Monthly_mean_MSLP.nc"),
        ("t", "Monthly_mean_T.nc"),
        ("q", "Monthly_mean_Q.nc"),
    ]:
        p = op.join(local_path, fname)
        if op.exists(p):
            era5_paths[key] = p

    if "sst" in era5_paths and "mslp" in era5_paths:
        potential_intensity.build_phase_specific_pi_climatologies(
            climate_df,
            era5_paths,
            local_path,
        )
        print("PI done")

    # Compute 850 hPa relative vorticity from u,v

    u850_path = op.join(local_path, "Monthly_mean_U850.nc")
    v850_path = op.join(local_path, "Monthly_mean_V850.nc")
    if op.exists(u850_path) and op.exists(v850_path):
        _compute_and_save_vorticity(u850_path, v850_path, climate_df, local_path)
        print("850hPa Vorticity done")

    return climate_df


# ==============================================================================
# Year-level environmental fields for interannual resampling
# ==============================================================================


def save_yearly_env_fields(climate_df, period):
    """
    Extract and save individual year-month environmental fields for runtime
    resampling. Instead of loading phase-mean climatologies (which suppress
    interannual variability), the simulation can draw a random historical
    year and load that year's actual VWS/RH/MSLP/PI.

    Storage: env_yearly/{stem}_{year}_{month}.npy

    Also builds an env_pool.json mapping each (ENSO phase, month) to its
    historical years, used at runtime for sampling.

    For seasonal forecast mode: place forecast fields in the same directory
    with a synthetic year label (e.g. 9999). The runtime code will load them
    when env_year=9999 is passed.

    Parameters
    ----------
    climate_df : DataFrame with [year, month, climate_index, phase]
    period : (start_year, end_year)
    """
    from siena_utils import save_yearly_field, save_env_pool, _env_yearly_dir

    local_path = os.getcwd()
    out_dir = _env_yearly_dir(local_path)
    print(f"Saving yearly fields to {out_dir}")

    # ── 1. Build month-level year pool ──
    # For each (phase, month), store which historical years had that month
    # in that phase. This ensures that when generating an LN catalog, a
    # storm born in September loads environmental fields from a September
    # that was actually classified as LN — not from a year that was LN
    # overall but had a NEU September.
    #
    # Structure: {"LN": {"6": [1988, 1999], "7": [1988, ...]}, ...}
    month_pool = {"LN": {}, "NEU": {}, "EN": {}}
    for _, row in climate_df.iterrows():
        yr = int(row["year"])
        mo = str(int(row["month"]))
        ph = str(row["phase"]).strip().upper()
        if ph not in month_pool:
            continue
        if mo not in month_pool[ph]:
            month_pool[ph][mo] = []
        if yr not in month_pool[ph][mo]:
            month_pool[ph][mo].append(yr)

    # Sort for reproducibility
    for ph in month_pool:
        for mo in month_pool[ph]:
            month_pool[ph][mo].sort()

    save_env_pool(local_path, month_pool)
    for ph in ["LN", "NEU", "EN"]:
        month_counts = {m: len(yrs) for m, yrs in month_pool[ph].items()}
        print(f"  {ph} month pool: {month_counts}")

    # ── 2. Extract per-year-month VWS, RH, MSLP from NetCDF ──
    variables = [
        ("Monthly_mean_VWS.nc", "vws", "VWS", None, 1.0),
        ("Monthly_mean_RH600.nc", "r", "RH600", 0, 1.0),
        ("Monthly_mean_MSLP.nc", "msl", "MSLP", None, 0.01),  # Pa → hPa
        ("Monthly_mean_SST.nc", "sst", "SST", None, 1.0),
    ]

    for nc_file, varname, stem, plev, scale in variables:
        nc_path = op.join(local_path, nc_file)
        if not op.exists(nc_path):
            print(f"  Skipping {stem}: {nc_file} not found")
            continue

        ds = xr.open_dataset(nc_path)
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"

        if varname is None or varname not in ds:
            varname = list(ds.data_vars)[0]

        if plev is not None:
            lvl_dim = "pressure_level" if "pressure_level" in ds.dims else "level"
            ds = ds.isel({lvl_dim: plev})

        times = pd.to_datetime(ds[time_dim].values)
        saved = 0
        for t_idx in range(len(times)):
            yr = int(times[t_idx].year)
            mo = int(times[t_idx].month)
            if yr < period[0] or yr > period[1]:
                continue
            field = ds[varname].isel({time_dim: t_idx}).values * scale
            save_yearly_field(local_path, stem, yr, mo, field)
            saved += 1

        ds.close()
        print(f"  {stem}: saved {saved} year-month fields")

    # ── 3. Compute per-year-month thermodynamic PI ──
    # Uses full Bister & Emanuel (2002) via tcpyPI when T/Q profiles are
    # available at 1°. This preserves the interannual atmospheric profile
    # variation (tropopause temperature, moisture stratification) that the
    # simplified SST-only approximation would miss.
    # Compute at 1° (T/Q native resolution), NaN-fill coastal cells, then
    # upscale to 0.25° for consistency with other environmental fields.
    # One-time cost: ~4-8 hours for 42 years × 12 months = 504 fields.
    sst_nc = op.join(local_path, "Monthly_mean_SST.nc")
    mslp_nc = op.join(local_path, "Monthly_mean_MSLP.nc")
    t_nc = op.join(local_path, "Monthly_mean_T.nc")
    q_nc = op.join(local_path, "Monthly_mean_Q.nc")

    has_profiles = op.exists(t_nc) and op.exists(q_nc)
    use_full_pi = potential_intensity.HAS_TCPYPI and has_profiles

    if not op.exists(sst_nc) or not op.exists(mslp_nc):
        print("  Skipping yearly PI: SST or MSLP .nc not found")
    else:
        if use_full_pi:
            print("  Computing yearly PI with full tcpyPI (Bister & Emanuel 2002)")
        elif has_profiles:
            print("  tcpyPI not installed — falling back to simplified PI")
            print("  Install with: pip install tcpyPI")
        else:
            print("  No T/Q profiles — falling back to simplified PI")

        ds_sst = xr.open_dataset(sst_nc)
        ds_mslp = xr.open_dataset(mslp_nc)
        time_dim_s = "valid_time" if "valid_time" in ds_sst.dims else "time"
        time_dim_m = "valid_time" if "valid_time" in ds_mslp.dims else "time"
        var_sst = "sst" if "sst" in ds_sst else list(ds_sst.data_vars)[0]
        var_mslp = "msl" if "msl" in ds_mslp else list(ds_mslp.data_vars)[0]

        # Determine fine grid shape (0.25°) from SST
        fine_shape = ds_sst[var_sst].isel({time_dim_s: 0}).values.shape

        # Load T/Q if available
        ds_t = ds_q = p_lev_hPa = None
        coarse_shape = None
        if use_full_pi:
            ds_t = xr.open_dataset(t_nc)
            ds_q = xr.open_dataset(q_nc)
            time_dim_t = "valid_time" if "valid_time" in ds_t.dims else "time"
            lvl_dim = "pressure_level" if "pressure_level" in ds_t.dims else "level"
            p_lev_hPa = ds_t[lvl_dim].values.astype(float)
            var_t = "t" if "t" in ds_t else list(ds_t.data_vars)[0]
            var_q = "q" if "q" in ds_q else list(ds_q.data_vars)[0]
            # Coarse shape from T (1° = 181×360)
            sample_t = ds_t[var_t].isel({time_dim_t: 0}).values
            coarse_shape = sample_t.shape[-2:]
            print(
                f"  PI grid: compute at {coarse_shape} (1°), upscale to {fine_shape} (0.25°)"
            )

        # Build time index lookups for MSLP and T/Q
        times_s = pd.to_datetime(ds_sst[time_dim_s].values)
        times_m = pd.to_datetime(ds_mslp[time_dim_m].values)
        mslp_idx = {
            (int(times_m[i].year), int(times_m[i].month)): i
            for i in range(len(times_m))
        }
        t_idx_lookup = {}
        if ds_t is not None:
            times_t = pd.to_datetime(ds_t[time_dim_t].values)
            t_idx_lookup = {
                (int(times_t[i].year), int(times_t[i].month)): i
                for i in range(len(times_t))
            }

        saved_pi = 0
        for s_idx in range(len(times_s)):
            yr = int(times_s[s_idx].year)
            mo = int(times_s[s_idx].month)
            if yr < period[0] or yr > period[1]:
                continue

            # Check if already computed (skip on re-runs)
            _check = os.path.join(out_dir, f"PI_{yr}_{mo}.npy")
            if os.path.exists(_check):
                saved_pi += 1
                continue

            sst_field = ds_sst[var_sst].isel({time_dim_s: s_idx}).values
            m_idx = mslp_idx.get((yr, mo))
            if m_idx is None:
                continue
            mslp_field = ds_mslp[var_mslp].isel({time_dim_m: m_idx}).values * 0.01

            if use_full_pi and (yr, mo) in t_idx_lookup:
                # Full thermodynamic PI
                ti = t_idx_lookup[(yr, mo)]
                t_field = ds_t[var_t].isel({time_dim_t: ti}).values
                q_field = ds_q[var_q].isel({time_dim_t: ti}).values

                # Coarsen SST/MSLP to match T/Q grid (1°)
                sst_c = potential_intensity._coarsen_to_match(sst_field, coarse_shape)
                mslp_c = potential_intensity._coarsen_to_match(mslp_field, coarse_shape)

                pmin, _ = potential_intensity.compute_pi_field_tcpyPI(
                    sst_c, mslp_c, t_field, q_field, p_lev_hPa
                )
                # NaN-fill coastal cells before upscaling
                pmin = potential_intensity._nanfill_nearest(pmin)
                # Upscale to 0.25°
                if pmin.shape != fine_shape:
                    pmin = potential_intensity._upscale_to_target(pmin, fine_shape)
            else:
                # Simplified fallback
                pmin, _ = potential_intensity.compute_pi_field_simplified(
                    sst_field, mslp_field
                )

            save_yearly_field(local_path, "PI", yr, mo, pmin)
            saved_pi += 1
            if saved_pi % 50 == 0:
                print(f"    PI progress: {saved_pi} fields computed...")

        ds_sst.close()
        ds_mslp.close()
        if ds_t is not None:
            ds_t.close()
        if ds_q is not None:
            ds_q.close()
        print(
            f"  PI: saved {saved_pi} year-month fields"
            f" ({'tcpyPI' if use_full_pi else 'simplified'})"
        )

    print("Yearly env fields complete.")
