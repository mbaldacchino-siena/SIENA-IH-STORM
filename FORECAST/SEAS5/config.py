"""Configuration for SEAS5 bias correction pipeline.

Edit the values in this file to change domain, periods, variables, or paths.
All other modules import from here.
"""

from pathlib import Path

# =============================================================================
# Reference periods
# =============================================================================
# 1993-2016 matches the default SEAS5 hindcast climatology used by the
# CDS anomaly products. Do NOT change this without also using a custom
# SEAS5 hindcast climatology — the anomaly reference period MUST match
# the ERA5 climatology period, otherwise bias correction is inconsistent.
CLIMATOLOGY_START_YEAR = 1993
CLIMATOLOGY_END_YEAR = 2016

# ENSO CLIMATOLOGY
ENSO_CLIMATOLOGY_START_YEAR = 1991
ENSO_CLIMATOLOGY_END_YEAR = 2020

# SEAS5 went operational in November 2017. Use 2017 onwards for operational
# forecast archive. Update FORECAST_END_YEAR as needed.
FORECAST_START_YEAR = 2017
FORECAST_END_YEAR = 2026

# =============================================================================
# Spatial domain
# =============================================================================
# IMPORTANT: longitude convention
# ERA5 and SEAS5 on CDS return DIFFERENT longitude conventions depending on
# whether the `area` parameter is set:
#   - No `area` (global request)        -> longitudes in [0, 360)   <- our pick
#   - `area = [N, W, S, E]` with W < 0  -> longitudes in [-180, 180]
#
# The rest of the SIENA-IH-STORM codebase (training-side env files,
# Monthly_mean_*.nc, the storm-generator sampling code in
# CODE/SAMPLE_TC_MOVEMENT.py and CODE/SAMPLE_TC_PRESSURE.py which uses
# `lon % 360.0`) operates in [0, 360). Mixing conventions silently produces
# NaN over the western hemisphere after regridding (the symptom: missing
# values across the whole Atlantic and Americas in saved env_yearly files).
#
# We therefore commit to [0, 360) at the data-load boundary. To get this
# from CDS, set DOMAIN["global"] = True below (omit the `area` parameter)
# OR, if you need a regional download, set lon_min/lon_max explicitly in
# [0, 360] form (e.g. lon_min=260, lon_max=20 for the Atlantic, with
# lon_max < lon_min meaning "wraps the dateline").
DOMAIN = {
    "global": True,  # omit `area` from CDS request, get full [0, 360)
    "lat_max": 90,  # used only if global=False
    "lat_min": -90,
    "lon_min": 0,
    "lon_max": 360,
}

# =============================================================================
# Pressure levels
# =============================================================================
# These are the pressure levels available for SEAS5 on CDS.
# ERA5 has more levels but we restrict to SEAS5 levels for consistent
# vertical profiles between datasets.
# Note: 600 hPa is NOT in the SEAS5 pressure level list. RH600 for GPI
# must be interpolated between 500 and 700 hPa downstream of this pipeline.
SEAS5_PRESSURE_LEVELS = [10, 50, 100, 200, 300, 400, 500, 700, 850, 925, 1000]

# =============================================================================
# Variables
# Mapping: {CDS variable name: short name used in filenames}
# =============================================================================
# Single-level anomaly products on `seasonal-postprocessed-single-levels`
# append "_anomaly" to the variable name at download time (handled in
# cds_download.py). Here we list the base names.
SINGLE_LEVEL_VARS = {
    "sea_surface_temperature": "sst",
    "mean_sea_level_pressure": "msl",
}

# Pressure-level anomaly products on `seasonal-postprocessed-pressure-levels`.
# NOTE: relative_humidity is NOT available as an anomaly on SEAS5. We download
# specific_humidity and temperature; RH at any target level can be computed
# downstream from corrected q and T using the Clausius-Clapeyron relation.
PRESSURE_LEVEL_VARS = {
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}

# =============================================================================
# Forecast sampling
# =============================================================================
# Initialization months to retrieve for SEAS5 forecasts. All 12 is flexible
# but costly. For NH hurricane season, April-June inits are most useful.
INIT_MONTHS = list(range(1, 13))

# Lead times in months. For instantaneous monthly means (T, U, V, SST,
# MSLP) on the `seasonal-monthly-*` and `seasonal-postprocessed-*` CDS
# datasets, leadtime_month=1 is the INIT MONTH itself, so a 6-lead
# request from an April 1 init covers April-September. SEAS5 provides
# up to 7 lead months. See FORECAST/SEAS5/enso.py docstring for sources.
LEADTIME_MONTHS = [1, 2, 3, 4, 5, 6]

# =============================================================================
# Paths
# =============================================================================
DATA_DIR = Path("./forecast_data")
ERA5_RAW_DIR = DATA_DIR / "era5_raw"  # raw monthly ERA5 downloads
ERA5_CLIM_DIR = DATA_DIR / "era5_climatology"  # ERA5 monthly climatology (reusable)
SEAS5_ANOMALY_DIR = DATA_DIR / "seas5_anomaly"  # SEAS5 anomaly downloads from CDS
CORRECTED_DIR = DATA_DIR / "seas5_corrected"  # X_corrected output

for _d in (ERA5_RAW_DIR, ERA5_CLIM_DIR, SEAS5_ANOMALY_DIR, CORRECTED_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================
def cds_area(domain: dict = DOMAIN):
    """Return the CDS `area` value for a request.

    Returns
    -------
    None if domain['global'] is True (caller should omit the `area` key
    from the CDS request, which yields a full-globe grid in [0, 360)
    longitude convention).

    Otherwise, returns [N, W, S, E] for use as the CDS `area` parameter.
    """
    if domain.get("global", False):
        return None
    return [domain["lat_max"], domain["lon_min"], domain["lat_min"], domain["lon_max"]]
