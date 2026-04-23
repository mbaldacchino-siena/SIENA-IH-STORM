"""Loader for bias-corrected SEAS5 fields.

Translates the per-variable NetCDF files produced by the bias-correction
pipeline (one file per variable, per forecast year range) into the two
datasets (`ds_pl`, `ds_sfc`) that `MASTER_forecast_fields.process_member`
consumes.

This is the integration seam: MASTER can swap its direct CDS download
for a call to `prepare_corrected_for_init()` and otherwise continue
unchanged. The returned datasets have:

  ds_pl:  variables u, v, t, q  with dims (time, forecastMonth, number,
          pressure_level, latitude, longitude)
  ds_sfc: variables sst, msl    with dims (time, forecastMonth, number,
          latitude, longitude)

Units and variable naming match what the raw `seasonal-monthly-*` CDS
dataset returns, so downstream code doesn't need to know whether it's
seeing raw or bias-corrected data.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import xarray as xr

from . import config
from . import pipeline

logger = logging.getLogger(__name__)


# =============================================================================
# Climatology presence check
# =============================================================================
def climatology_is_built() -> bool:
    """Check whether all ERA5 climatology files exist for the configured period."""
    for _, short in config.SINGLE_LEVEL_VARS.items():
        if not pipeline.era5_clim_path(short, is_pressure=False).exists():
            return False
    for _, short in config.PRESSURE_LEVEL_VARS.items():
        if not pipeline.era5_clim_path(short, is_pressure=True).exists():
            return False
    return True


def ensure_climatology(overwrite: bool = False) -> None:
    """Build ERA5 climatology if missing. Idempotent — safe to call every run."""
    if climatology_is_built() and not overwrite:
        logger.info("ERA5 climatology already built, skipping steps 1+2")
        return
    logger.info("ERA5 climatology missing — running download + build")
    pipeline.step1_download_era5(overwrite=overwrite)
    pipeline.step2_build_climatology(overwrite=overwrite)


# =============================================================================
# Bias correction for a specific init date
# =============================================================================
def prepare_corrected_for_init(
    init_date: str,
    lead_months: int = 6,
    overwrite_correction: bool = False,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Download SEAS5 anomalies + apply bias correction + return merged datasets.

    Parameters
    ----------
    init_date : str, "YYYY-MM-DD"
    lead_months : int, number of lead months to retrieve (1-6)
    overwrite_correction : bool, re-run steps 3+4 even if output files exist

    Returns
    -------
    ds_pl, ds_sfc : xr.Dataset
        Pressure-level and surface datasets on SEAS5 native grid, with
        bias-corrected values on the ERA5 absolute scale (NOT anomalies).
        Variable names: {u, v, t, q} for ds_pl and {sst, msl} for ds_sfc.
        Dimensions match what MASTER_forecast_fields.process_member expects.
    """
    year = int(init_date[:4])
    month = int(init_date[5:7])
    years = [year]
    init_months = [month]
    leadtime_months = list(range(1, lead_months + 1))

    ensure_climatology()

    pipeline.step3_download_seas5(
        years=years,
        init_months=init_months,
        leadtime_months=leadtime_months,
        overwrite=overwrite_correction,
    )
    pipeline.step4_apply_correction(years=years, overwrite=overwrite_correction)

    ds_pl = _load_merged_pressure_level(years)
    ds_sfc = _load_merged_single_level(years)

    # Filter to just this init date (in case pipeline produced multi-year file)
    ds_pl = _select_init(ds_pl, init_date)
    ds_sfc = _select_init(ds_sfc, init_date)

    return ds_pl, ds_sfc


# =============================================================================
# Helpers to load and merge per-variable files
# =============================================================================
def _load_merged_pressure_level(years: List[int]) -> xr.Dataset:
    """Merge per-variable pressure-level corrected files into one dataset."""
    pieces = []
    for cds_name, short in config.PRESSURE_LEVEL_VARS.items():
        p = pipeline.corrected_path(short, is_pressure=True, years=years)
        if not p.exists():
            raise FileNotFoundError(f"Corrected file missing: {p}")
        ds = xr.open_dataset(p)
        # Normalise variable name to short form
        ds = _rename_to_short(ds, short)
        pieces.append(ds[[short]])
    merged = xr.merge(pieces, compat="no_conflicts")
    return merged


def _load_merged_single_level(years: List[int]) -> xr.Dataset:
    """Merge per-variable single-level corrected files into one dataset."""
    pieces = []
    for cds_name, short in config.SINGLE_LEVEL_VARS.items():
        p = pipeline.corrected_path(short, is_pressure=False, years=years)
        if not p.exists():
            raise FileNotFoundError(f"Corrected file missing: {p}")
        ds = xr.open_dataset(p)
        ds = _rename_to_short(ds, short)
        pieces.append(ds[[short]])
    merged = xr.merge(pieces, compat="no_conflicts")
    return merged


def _rename_to_short(ds: xr.Dataset, short: str) -> xr.Dataset:
    """Ensure the main data variable is named `short`.

    CDS may have returned the variable under a long name, `_anomaly` suffix,
    or the canonical short name. We normalise in one place here so MASTER
    always sees the expected names (u, v, t, q, sst, msl).

    Also strips any generic `_anomaly` suffix from variable names, since
    after bias correction the values are absolute, not anomalies.
    """
    # If the short name is already present, nothing to do.
    if short in ds.data_vars:
        return ds

    # Try common alternatives.
    candidates = [
        f"{short}_anomaly",
        {
            "sst": "sea_surface_temperature",
            "msl": "mean_sea_level_pressure",
            "u": "u_component_of_wind",
            "v": "v_component_of_wind",
            "t": "temperature",
            "q": "specific_humidity",
        }.get(short, None),
    ]
    for c in candidates:
        if c and c in ds.data_vars:
            ds = ds.rename({c: short})
            return ds

    # Fallback: pick the data variable with the most dims and rename.
    real_vars = [v for v in ds.data_vars if not v.endswith("_bnds")]
    if len(real_vars) == 1:
        return ds.rename({real_vars[0]: short})
    if real_vars:
        primary = max(real_vars, key=lambda v: len(ds[v].dims))
        return ds.rename({primary: short})
    raise ValueError(
        f"Cannot identify main data variable in dataset (looking for '{short}')"
    )


def _select_init(ds: xr.Dataset, init_date: str) -> xr.Dataset:
    """Select the single init date from a multi-init dataset (if needed)."""
    # Find the init time coord (CDS varies: 'time' or 'forecast_reference_time')
    init_dim = None
    for cand in ("forecast_reference_time", "time"):
        if cand in ds.dims:
            init_dim = cand
            break
    if init_dim is None:
        return ds

    # Match on year-month (safer than exact datetime matching)
    target_year = int(init_date[:4])
    target_month = int(init_date[5:7])
    init_vals = ds[init_dim]
    init_pd = init_vals.to_pandas()
    matches = (init_pd.dt.year == target_year) & (init_pd.dt.month == target_month)
    if matches.sum() == 0:
        raise ValueError(
            f"No init in file matches {init_date} (available: {init_pd.tolist()})"
        )
    return ds.isel({init_dim: np.where(matches)[0]})


# =============================================================================
# Utility: load ERA5 SST climatology for ENSO computation
# =============================================================================
def load_era5_sst_climatology() -> xr.DataArray:
    """Load the ERA5 SST monthly climatology produced by step2.

    Used by the ENSO module to compute Niño 3.4 anomalies from corrected SST
    (the same climatology baseline, so anomalies are consistent).
    """
    p = pipeline.era5_clim_path("sst", is_pressure=False)
    if not p.exists():
        raise FileNotFoundError(
            f"ERA5 SST climatology missing: {p}. Run ensure_climatology() first."
        )
    ds = xr.open_dataset(p)
    # Pick the SST variable regardless of CDS-assigned name
    for cand in ("sst", "sea_surface_temperature"):
        if cand in ds.data_vars:
            return ds[cand]
    # Fallback: primary data var
    real_vars = [v for v in ds.data_vars if not v.endswith("_bnds")]
    return ds[real_vars[0]]
