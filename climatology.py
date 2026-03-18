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
from potential_intensity import build_phase_specific_pi_climatologies

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
    nc_path : str, varname : str | None, oni_df : pd.DataFrame, out_stem : str, out_dir : str | None, pressure_level_idx : int | None =None, unit_scale : float =1.0,
) -> tuple[dict,dict]:
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
    clim : dict = {m: {} for m in range(1, 13)}
    pooled : dict = {}

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
    period, climate_index="ONI", threshold=0.5,
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
        build_phase_specific_pi_climatologies(
            climate_df,
            era5_paths,
            local_path,
        )
        print("PI done")

    return climate_df
