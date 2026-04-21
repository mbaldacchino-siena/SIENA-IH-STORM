"""Build and save ERA5 monthly climatologies over the reference period.

The climatology has a `month` dimension (1-12), lat, lon, and (for pressure-
level variables) pressure_level. It is saved as NetCDF for reuse — you only
need to build it once per reference period and variable.

These climatology files are the `X_ERA5_clim` in the delta-correction formula:
    X_corrected = X_ERA5_clim(valid_month) + X_SEAS5_anomaly
"""

from __future__ import annotations
import logging
from pathlib import Path

import xarray as xr

import config

logger = logging.getLogger(__name__)


def _time_coord(ds: xr.Dataset) -> str:
    """Find the time coordinate (CDS uses 'time' or 'valid_time')."""
    for candidate in ("time", "valid_time"):
        if candidate in ds.coords:
            return candidate
    raise KeyError(f"No time coordinate found in dataset. Coords: {list(ds.coords)}")


def compute_monthly_climatology(
    ds: xr.Dataset,
    start_year: int = config.CLIMATOLOGY_START_YEAR,
    end_year: int = config.CLIMATOLOGY_END_YEAR,
) -> xr.Dataset:
    """Compute monthly climatology over [start_year, end_year].

    Returns a dataset with a `month` dimension (1-12) replacing time.
    """
    t = _time_coord(ds)

    # Restrict to reference window
    ds_ref = ds.sel({t: slice(f"{start_year}-01-01", f"{end_year}-12-31")})
    n_months = ds_ref.sizes[t]
    expected = (end_year - start_year + 1) * 12
    if n_months != expected:
        logger.warning(
            "Climatology window: expected %d months, got %d. Check data completeness.",
            expected,
            n_months,
        )

    clim = ds_ref.groupby(f"{t}.month").mean(t, keep_attrs=True)

    clim.attrs["climatology_start_year"] = start_year
    clim.attrs["climatology_end_year"] = end_year
    clim.attrs["climatology_n_months"] = n_months
    clim.attrs["source"] = "ERA5 monthly means"
    return clim


def build_era5_climatology(
    input_file: Path,
    output_file: Path,
    overwrite: bool = False,
) -> Path:
    """Build and save monthly climatology from an ERA5 monthly-means file."""
    if output_file.exists() and not overwrite:
        logger.info("Climatology exists, skipping: %s", output_file.name)
        return output_file

    logger.info("Building climatology: %s -> %s", input_file.name, output_file.name)
    with xr.open_dataset(input_file) as ds:
        clim = compute_monthly_climatology(ds)
        # Load into memory before writing to avoid keeping handle on input
        clim = clim.load()

    clim.to_netcdf(output_file)
    logger.info("Saved climatology: %s", output_file)
    return output_file
