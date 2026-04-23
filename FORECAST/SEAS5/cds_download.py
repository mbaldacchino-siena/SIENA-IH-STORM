"""CDS API wrappers for ERA5 and SEAS5 downloads.

Requires:
    - `cdsapi` Python package (`pip install cdsapi`)
    - `~/.cdsapirc` configured with your CDS API key
      (https://cds.climate.copernicus.eu/how-to-api)

All functions are idempotent: if the target file already exists, the download
is skipped unless `overwrite=True`.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import cdsapi

import FORECAST.SEAS5.config as config

logger = logging.getLogger(__name__)


def _client() -> cdsapi.Client:
    """Return a CDS client. Requires ~/.cdsapirc."""
    return cdsapi.Client(quiet=False)


# =============================================================================
# ERA5 downloads (raw monthly means, for building climatology)
# =============================================================================
def download_era5_single_level(
    variable: str,
    years: List[int],
    months: List[int],
    output_path: Path,
    domain: dict = config.DOMAIN,
    overwrite: bool = False,
) -> Path:
    """Download ERA5 monthly-mean single-level data.

    Dataset: `reanalysis-era5-single-levels-monthly-means`
    """
    if output_path.exists() and not overwrite:
        logger.info("ERA5 file exists, skipping: %s", output_path.name)
        return output_path

    logger.info("Downloading ERA5 %s (%d years)", variable, len(years))
    _client().retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variable,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "time": "00:00",
            "area": config.cds_area(domain),
            "format": "netcdf",
        },
        str(output_path),
    )
    return output_path


def download_era5_pressure_level(
    variable: str,
    pressure_levels: List[int],
    years: List[int],
    months: List[int],
    output_path: Path,
    domain: dict = config.DOMAIN,
    overwrite: bool = False,
) -> Path:
    """Download ERA5 monthly-mean pressure-level data.

    Dataset: `reanalysis-era5-pressure-levels-monthly-means`
    """
    if output_path.exists() and not overwrite:
        logger.info("ERA5 file exists, skipping: %s", output_path.name)
        return output_path

    logger.info(
        "Downloading ERA5 %s at %d levels (%d years)",
        variable,
        len(pressure_levels),
        len(years),
    )
    _client().retrieve(
        "reanalysis-era5-pressure-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variable,
            "pressure_level": [str(p) for p in pressure_levels],
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in months],
            "time": "00:00",
            "area": config.cds_area(domain),
            "format": "netcdf",
        },
        str(output_path),
    )
    return output_path


# =============================================================================
# SEAS5 anomaly downloads
# =============================================================================
# The CDS `seasonal-postprocessed-*` datasets provide anomalies referenced
# against the SEAS5 1993-2016 hindcast climatology, stratified by
# (initialization month, lead time). This is exactly what we need.


def download_seas5_single_level_anomaly(
    variable: str,
    years: List[int],
    init_months: List[int],
    leadtime_months: List[int],
    output_path: Path,
    domain: dict = config.DOMAIN,
    overwrite: bool = False,
) -> Path:
    """Download SEAS5 monthly single-level anomaly.

    Dataset: `seasonal-postprocessed-single-levels`
    Reference period: 1993-2016 (fixed by CDS product design).

    Notes
    -----
    The CDS catalog expects the variable with `_anomaly` suffix for this
    dataset (e.g. `sea_surface_temperature_anomaly`). We append it here.
    """
    if output_path.exists() and not overwrite:
        logger.info("SEAS5 anomaly file exists, skipping: %s", output_path.name)
        return output_path

    anomaly_var = f"{variable}_anomaly"
    logger.info(
        "Downloading SEAS5 anomaly %s (years=%s, inits=%s, leads=%s)",
        anomaly_var,
        years,
        init_months,
        leadtime_months,
    )
    _client().retrieve(
        "seasonal-postprocessed-single-levels",
        {
            "originating_centre": "ecmwf",
            "system": "51",
            "variable": anomaly_var,
            "product_type": "monthly_mean",
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in init_months],
            "leadtime_month": [str(lt) for lt in leadtime_months],
            "area": config.cds_area(domain),
            "format": "netcdf",
        },
        str(output_path),
    )
    return output_path


def download_seas5_pressure_level_anomaly(
    variable: str,
    pressure_levels: List[int],
    years: List[int],
    init_months: List[int],
    leadtime_months: List[int],
    output_path: Path,
    domain: dict = config.DOMAIN,
    overwrite: bool = False,
) -> Path:
    """Download SEAS5 monthly pressure-level anomaly.

    Dataset: `seasonal-postprocessed-pressure-levels`
    Reference period: 1993-2016.
    """
    if output_path.exists() and not overwrite:
        logger.info("SEAS5 anomaly file exists, skipping: %s", output_path.name)
        return output_path

    anomaly_var = f"{variable}_anomaly"
    logger.info(
        "Downloading SEAS5 anomaly %s at %d levels",
        anomaly_var,
        len(pressure_levels),
    )
    _client().retrieve(
        "seasonal-postprocessed-pressure-levels",
        {
            "originating_centre": "ecmwf",
            "system": "51",
            "variable": anomaly_var,
            "pressure_level": [str(p) for p in pressure_levels],
            "product_type": "monthly_mean",
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in init_months],
            "leadtime_month": [str(lt) for lt in leadtime_months],
            "area": config.cds_area(domain),
            "format": "netcdf",
        },
        str(output_path),
    )
    return output_path
