"""Delta bias correction.

    X_corrected(init, lead, member, lat, lon, [level])
        = ERA5_climatology(valid_month, lat, lon, [level])
        + SEAS5_anomaly(init, lead, member, lat, lon, [level])

The valid month is computed from init_month and leadtime_month.

SEAS5 on CDS uses the convention where `leadtime_month = 1` corresponds to
the first FULL month after initialization date. For an April 1 init, lead 1
would then be May (verify this in your specific download — see ECMWF
documentation). If your download uses lead 1 = init month instead, set
LEAD_OFFSET = 0 below.
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import xarray as xr

import FORECAST.SEAS5.config as config
import FORECAST.SEAS5.regrid as regrid

logger = logging.getLogger(__name__)

# SEAS5 leadtime_month convention:
#   LEAD_OFFSET = 0  -> lead 1 = init month itself
LEAD_OFFSET = 0


# =============================================================================
# Valid-month computation
# =============================================================================
def compute_valid_month(
    init_time: xr.DataArray,
    leadtime_month: xr.DataArray,
    offset: int = LEAD_OFFSET,
) -> xr.DataArray:
    """Compute the valid (target) calendar month for each (init, lead) pair.

    valid_month = ((init_month - 1) + (lead - 1) + offset) % 12 + 1
    """
    init_month = init_time.dt.month
    # Broadcast to a (init, lead) grid
    lead_minus_1 = leadtime_month - 1
    valid = ((init_month - 1) + lead_minus_1 + offset) % 12 + 1
    valid.name = "valid_month"
    valid.attrs["description"] = "Target calendar month (1-12) for each (init, lead)"
    return valid


# =============================================================================
# Delta correction
# =============================================================================
def apply_delta_correction(
    era5_clim: xr.DataArray,
    seas5_anomaly: xr.DataArray,
    init_coord: str = "forecast_reference_time",
    lead_coord: str = "forecastMonth",
) -> xr.DataArray:
    """Apply delta correction to a single variable.

    Parameters
    ----------
    era5_clim : xr.DataArray
        ERA5 monthly climatology with a `month` dimension (1-12).
        Must already be on the SEAS5 grid (call `regrid.regrid_like` first).
    seas5_anomaly : xr.DataArray
        SEAS5 anomaly with dims including init time, leadtime_month,
        member number, and lat/lon (+ pressure_level if applicable).
    init_coord : str
        Name of the initialization time coordinate in seas5_anomaly.
    lead_coord : str
        Name of the lead-time coordinate in seas5_anomaly.

    Returns
    -------
    xr.DataArray
        Bias-corrected forecast, same shape as seas5_anomaly.
    """

    # Compute valid month for each (init, lead) pair
    valid_month = compute_valid_month(
        seas5_anomaly[init_coord], seas5_anomaly[lead_coord]
    )

    # Select the climatology value matching each valid month.
    # This uses advanced indexing: the result has dims (init, lead) x lat x lon
    era5_matched = era5_clim.sel(month=valid_month)

    # Broadcast add over member, (init, lead), lat, lon, level
    corrected = era5_matched + seas5_anomaly

    corrected.name = (
        seas5_anomaly.name.replace("_anomaly", "") if seas5_anomaly.name else None
    )
    corrected.attrs = {
        **seas5_anomaly.attrs,
        "bias_correction": "delta (ERA5_clim + SEAS5_anomaly)",
        "reference_period": f"{config.CLIMATOLOGY_START_YEAR}-{config.CLIMATOLOGY_END_YEAR}",
        "lead_offset": LEAD_OFFSET,
    }
    return corrected


def correct_dataset(
    era5_clim_file: Path,
    seas5_anomaly_file: Path,
    output_file: Path,
    overwrite: bool = False,
) -> Path:
    """Full correction pipeline for one variable.

    1. Load ERA5 climatology.
    2. Load SEAS5 anomaly.
    3. Regrid ERA5 climatology onto SEAS5 grid.
    4. Apply delta correction.
    5. Save result as NetCDF.
    """
    if output_file.exists() and not overwrite:
        logger.info("Corrected file exists, skipping: %s", output_file.name)
        return output_file

    logger.info("Correcting: %s + %s", era5_clim_file.name, seas5_anomaly_file.name)

    era5 = xr.open_dataset(era5_clim_file)
    seas5 = xr.open_dataset(seas5_anomaly_file)

    # Regrid ERA5 climatology to SEAS5 grid
    target_grid = regrid.get_target_grid_from_seas5(seas5_anomaly_file)
    era5_regridded = regrid.regrid_like(era5, target_grid)

    # Identify the (single) data variable in each file
    era5_var = _main_data_var(era5_regridded)
    seas5_var = _main_data_var(seas5)

    corrected = apply_delta_correction(
        era5_clim=era5_regridded[era5_var],
        seas5_anomaly=seas5[seas5_var],
    )

    # Wrap into dataset and save
    out_ds = corrected.to_dataset(name=era5_var)
    out_ds.to_netcdf(output_file)
    logger.info("Saved corrected field: %s", output_file)

    era5.close()
    seas5.close()
    return output_file


def _main_data_var(ds: xr.Dataset) -> str:
    """Pick the primary data variable from a dataset.

    Filters out coordinate-like and bounds variables.
    """
    candidates = [
        v
        for v in ds.data_vars
        if not v.endswith("_bnds")
        and not v.startswith("lon_")
        and not v.startswith("lat_")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError(f"No data variables found in {list(ds.data_vars)}")
    # If multiple, prefer the one with most dimensions
    return max(candidates, key=lambda v: len(ds[v].dims))
