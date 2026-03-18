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
# Load VWS/RH fields for TC_variables — phase-aware keying
# The dict is keyed by (month, phase_str) so that TC_variables can pick
# the correct field for each storm's ENSO phase.
# ============================================================================
print("********************************")
vws_fields = {}
rh_fields = {}
for m in range(1, 13):
    # Pooled
    vws_path = os.path.join(__location__, f"Monthly_mean_VWS_{m}.txt")
    if os.path.exists(vws_path):
        vws_fields[(m, None)] = np.loadtxt(vws_path)
    rh_path = os.path.join(__location__, f"Monthly_mean_RH600_{m}.txt")
    if os.path.exists(rh_path):
        rh_fields[(m, None)] = np.loadtxt(rh_path)
    # Phase-specific
    for phase in ["LN", "NEU", "EN"]:
        vp = os.path.join(__location__, f"Monthly_mean_VWS_{m}_{phase}.txt")
        if os.path.exists(vp):
            vws_fields[(m, phase)] = np.loadtxt(vp)
        rp = os.path.join(__location__, f"Monthly_mean_RH600_{m}_{phase}.txt")
        if os.path.exists(rp):
            rh_fields[(m, phase)] = np.loadtxt(rp)
print(
    f"Loaded VWS fields: {len(vws_fields)} entries, RH fields: {len(rh_fields)} entries"
)
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
