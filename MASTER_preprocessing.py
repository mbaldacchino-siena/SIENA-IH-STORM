# -*- coding: utf-8 -*-
"""
Master program for pooled SIENA-IH-STORM preprocessing.

Fixes applied:
  Fix 1: Phase-specific VWS and RH600 climatologies
  Fix 2: Thermodynamic PI fields (replaces empirical SST-based MPI)
"""

import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import preprocessing
import coefficients
import environmental
import genesis_matrix
import import_data
import potential_intensity
from siena_utils import load_climate_index_table

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

(
    period,
    climate_index,
    threshold,
    idx_basin,
    months,
    mpi_bounds,
    months_for_coef_MPI,
    months_for_coef_PRESS,
    generation_phase,
) = import_data.input_data("input.dat")

TC_file = "IBTrACS." + str(period[0]) + "_" + str(period[1]) + "v04r00.nc"
print("********************************")
data = xr.open_dataset(os.path.join(__location__, TC_file), decode_times=False)

df1 = data.to_dataframe()[
    [
        "lat",
        "lon",
        "sid",
        "season",
        "number",
        "basin",
        "subbasin",
        "name",
        "iso_time",
        "wmo_wind",
        "wmo_pres",
        "track_type",
        "main_track_sid",
        "dist2land",
        "landfall",
        "usa_sshs",
        "usa_rmw",
    ]
]
df1["wmo_wind"] = df1["wmo_wind"] * 0.51444444
df1 = df1[df1["wmo_wind"] >= 18]
nyear = []
for basin in [b"EP", b"NA", b"NI", b"SI", b"SP", b"WP"]:
    nyear.append(len(np.unique(df1[df1["basin"] == basin]["season"].values)))
years = np.unique(data.season.values)
print("number of years: ", int(len(years)), "from", years[0], "to", years[-1])
preprocessing.extract_data(data, int(years[-1]))
data.close()
print("extract_data done")

# ============================================================================
# Load ONI / climate index table
# ============================================================================
print("********************************")
oni_table = load_climate_index_table(os.path.join(__location__, "climate_index.csv"))

# ============================================================================
# Fix 1: Build phase-specific VWS and RH600 climatologies from ERA5 .nc files
# ============================================================================
# If the ERA5 .nc files are available, build phase-conditioned climatologies
# so that VWS and RH fields are ENSO-state-consistent with other fields.
print("********************************")
print("Building phase-specific environmental climatologies...")

vws_nc = os.path.join(__location__, "Monthly_mean_VWS.nc")
rh_nc = os.path.join(__location__, "Monthly_mean_RH600.nc")

if os.path.exists(vws_nc) and len(oni_table) > 0:
    potential_intensity.build_phase_specific_env_climatologies(
        oni_df=oni_table,
        nc_path=vws_nc,
        varname="vws",
        out_stem="Monthly_mean_VWS",
        out_dir=__location__,
    )
    print("Phase-specific VWS climatologies done")
else:
    print("Skipping phase-specific VWS: file or ONI table not available")
    # Fall back to original pooled approach if .nc exists
    if os.path.exists(vws_nc):
        data_vws = xr.open_dataset(vws_nc)
        environmental.monthly_mean_vws(data_vws)
        data_vws.close()
        print("monthly_mean_vws done (pooled)")

if os.path.exists(rh_nc) and len(oni_table) > 0:
    potential_intensity.build_phase_specific_env_climatologies(
        oni_df=oni_table,
        nc_path=rh_nc,
        varname="r",
        out_stem="Monthly_mean_RH600",
        out_dir=__location__,
        pressure_level_idx=0,  # first pressure level = 600 hPa
    )
    print("Phase-specific RH600 climatologies done")
else:
    print("Skipping phase-specific RH600: file or ONI table not available")
    if os.path.exists(rh_nc):
        data_rh = xr.open_dataset(rh_nc)
        environmental.monthly_mean_rh(data_rh)
        data_rh.close()
        print("monthly_mean_rh done (pooled)")

# ============================================================================
# Fix 2: Build thermodynamic PI fields (phase-specific)
# ============================================================================
print("********************************")
print("Building phase-specific Potential Intensity fields...")

sst_nc = os.path.join(__location__, "Monthly_mean_SST.nc")
mslp_nc = os.path.join(__location__, "Monthly_mean_MSLP.nc")
t_nc = os.path.join(
    __location__, "Monthly_mean_T.nc"
)  # ERA5 temperature on pressure levels
q_nc = os.path.join(
    __location__, "Monthly_mean_Q.nc"
)  # ERA5 specific humidity on pressure levels

era5_paths = {}
if os.path.exists(sst_nc):
    era5_paths["sst"] = sst_nc
if os.path.exists(mslp_nc):
    era5_paths["mslp"] = mslp_nc
if os.path.exists(t_nc):
    era5_paths["t"] = t_nc
if os.path.exists(q_nc):
    era5_paths["q"] = q_nc

if "sst" in era5_paths and "mslp" in era5_paths and len(oni_table) > 0:
    potential_intensity.build_phase_specific_pi_climatologies(
        oni_df=oni_table,
        era5_paths=era5_paths,
        out_dir=__location__,
    )
    print("Phase-specific PI fields done")
else:
    print("Skipping PI fields: SST/MSLP .nc or ONI table not available")
    print("  Will fall back to empirical MPI in pressure coefficient fitting")

# ============================================================================
# Load VWS/RH fields for TC_variables (now phase-aware if built above)
# ============================================================================
print("********************************")
vws_fields = {}
rh_fields = {}
for m in range(1, 13):
    vws_path = os.path.join(__location__, f"Monthly_mean_VWS_{m}.txt")
    rh_path = os.path.join(__location__, f"Monthly_mean_RH600_{m}.txt")
    if os.path.exists(vws_path):
        vws_fields[m] = np.loadtxt(vws_path)
    if os.path.exists(rh_path):
        rh_fields[m] = np.loadtxt(rh_path)
latitudes = longitudes = None
try:
    ds_env = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
    latitudes = ds_env.latitude.values
    longitudes = ds_env.longitude.values
    ds_env.close()
except Exception:
    pass
preprocessing.TC_variables(
    nyear,
    months,
    oni_table=oni_table,
    vws_fields=vws_fields,
    rh_fields=rh_fields,
    latitudes=latitudes,
    longitudes=longitudes,
)
print("TC_variables done")

print("********************************")
coefficients.track_coefficients()
print("track_coefficients done")

print("********************************")
environmental.wind_pressure_relationship(idx_basin, months)
print("wind_pressure_relationship done")

# Note: calculate_MPI_fields is still run as fallback for basins/months
# where PI fields may not exist (e.g., missing ERA5 profile data).
# The pressure_coefficients function will prefer PI over MPI when available.
print("********************************")
environmental.calculate_MPI_fields(idx_basin, months, months_for_coef_MPI, mpi_bounds)
print("calculate_MPI_fields done")

print("********************************")
environmental.pressure_coefficients(idx_basin, months, months_for_coef_PRESS)
print("pressure_coefficients done")

print("********************************")
genesis_matrix.Change_genesis_locations(idx_basin, months)
print("genesis_matrix done")
