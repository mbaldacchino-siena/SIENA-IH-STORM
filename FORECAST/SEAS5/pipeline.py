"""Main pipeline orchestrator for SEAS5 bias correction.

Workflow:
    1. Download ERA5 monthly means over climatology period (1993-2016).
    2. Build ERA5 monthly climatology (saved for reuse).
    3. Download SEAS5 anomalies for requested forecast years/inits/leads.
    4. Regrid ERA5 climatology to SEAS5 grid and apply delta correction.

Run with defaults:
    python pipeline.py

Or import and call steps individually (see README.md).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import FORECAST.SEAS5.config as config
import FORECAST.SEAS5.cds_download as cds_download
import FORECAST.SEAS5.climatology as climatology
import FORECAST.SEAS5.bias_correction as bias_correction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("pipeline")


# =============================================================================
# File naming conventions (centralized to keep steps coherent)
# =============================================================================
def era5_raw_path(
    short_name: str, is_pressure: bool, year_range: None | tuple(int, int) = None
) -> Path:
    suffix = "_pl" if is_pressure else ""
    if year_range is None:
        return config.ERA5_RAW_DIR / (
            f"era5_{short_name}{suffix}_"
            f"{config.CLIMATOLOGY_START_YEAR}-{config.CLIMATOLOGY_END_YEAR}.nc"
        )
    else:
        return config.ERA5_RAW_DIR / (
            f"era5_{short_name}{suffix}_{year_range[0]}-{year_range[1]}.nc"
        )


def era5_clim_path(short_name: str, is_pressure: bool) -> Path:
    suffix = "_pl" if is_pressure else ""
    return config.ERA5_CLIM_DIR / (
        f"era5_clim_{short_name}{suffix}_"
        f"{config.CLIMATOLOGY_START_YEAR}-{config.CLIMATOLOGY_END_YEAR}.nc"
    )


def seas5_anomaly_path(
    short_name: str, is_pressure: bool, years: List[int], months: List[int]
) -> Path:
    suffix = "_pl" if is_pressure else ""
    return config.SEAS5_ANOMALY_DIR / (
        f"seas5anom_{short_name}{suffix}_{min(months)}_{min(years)}-{max(months)}_{max(years)}.nc"
    )


def corrected_path(
    short_name: str, is_pressure: bool, years: List[int], months: List[int]
) -> Path:
    suffix = "_pl" if is_pressure else ""
    return config.CORRECTED_DIR / (
        f"seas5corrected_{short_name}{suffix}_{min(months)}_{min(years)}-{max(months)}_{max(years)}.nc"
    )


# =============================================================================
# Step 1: Download ERA5
# =============================================================================
def step1_download_era5(
    overwrite: bool = False, year_range: None | tuple[int, int] = None
):
    """Download ERA5 monthly means over the climatology period."""
    if year_range is None:
        years = list(
            range(config.CLIMATOLOGY_START_YEAR, config.CLIMATOLOGY_END_YEAR + 1)
        )
    else:
        years = list(range(year_range[0], year_range[1]))

    months = list(range(1, 13))

    logger.info("=== Step 1: Download ERA5 (%d-%d) ===", min(years), max(years))

    for cds_name, short in config.SINGLE_LEVEL_VARS.items():
        cds_download.download_era5_single_level(
            variable=cds_name,
            years=years,
            months=months,
            output_path=era5_raw_path(short, is_pressure=False, year_range=year_range),
            overwrite=overwrite,
        )

    for cds_name, short in config.PRESSURE_LEVEL_VARS.items():
        cds_download.download_era5_pressure_level(
            variable=cds_name,
            pressure_levels=config.SEAS5_PRESSURE_LEVELS,
            years=years,
            months=months,
            output_path=era5_raw_path(short, is_pressure=True, year_range=year_range),
            overwrite=overwrite,
        )


def step1b_download_era5_for_oni(overwrite=False):
    """Download ERA5 SST 1991-2020 specifically for ONI-style classification.
    This is a separate climatology from the 1993-2016 used for bias correction."""

    cds_download.download_era5_single_level(
        variable="sea_surface_temperature",
        years=list(range(1991, 2021)),
        months=list(range(1, 13)),
        output_path=config.ERA5_RAW_DIR / "era5_sst_oni_1991-2020.nc",
        overwrite=overwrite,
    )

    climatology.build_era5_climatology(
        input_file=config.ERA5_RAW_DIR / "era5_sst_oni_1991-2020.nc",
        output_file=config.ERA5_CLIM_DIR / "era5_clim_sst_oni_1991-2020.nc",
        overwrite=overwrite,
    )


# =============================================================================
# Step 2: Build ERA5 climatology
# =============================================================================
def step2_build_climatology(overwrite: bool = False):
    """Compute and save ERA5 monthly climatology (reusable)."""
    logger.info("=== Step 2: Build ERA5 climatology ===")

    for _, short in config.SINGLE_LEVEL_VARS.items():
        climatology.build_era5_climatology(
            input_file=era5_raw_path(short, is_pressure=False),
            output_file=era5_clim_path(short, is_pressure=False),
            overwrite=overwrite,
        )

    for _, short in config.PRESSURE_LEVEL_VARS.items():
        climatology.build_era5_climatology(
            input_file=era5_raw_path(short, is_pressure=True),
            output_file=era5_clim_path(short, is_pressure=True),
            overwrite=overwrite,
        )


# =============================================================================
# Step 3: Download SEAS5 anomalies
# =============================================================================
def step3_download_seas5(
    years: List[int] = None,
    init_months: List[int] = None,
    leadtime_months: List[int] = None,
    overwrite: bool = False,
):
    """Download SEAS5 anomalies for the requested years/inits/leads."""
    if years is None:
        years = list(range(config.FORECAST_START_YEAR, config.FORECAST_END_YEAR + 1))
    if init_months is None:
        init_months = config.INIT_MONTHS
    if leadtime_months is None:
        leadtime_months = config.LEADTIME_MONTHS

    logger.info(
        "=== Step 3: Download SEAS5 anomalies (years=%d-%d, inits=%s, leads=%s) ===",
        min(years),
        max(years),
        init_months,
        leadtime_months,
    )

    for cds_name, short in config.SINGLE_LEVEL_VARS.items():
        cds_download.download_seas5_single_level_anomaly(
            variable=cds_name,
            years=years,
            init_months=init_months,
            leadtime_months=leadtime_months,
            output_path=seas5_anomaly_path(
                short, is_pressure=False, years=years, months=init_months
            ),
            overwrite=overwrite,
        )

    for cds_name, short in config.PRESSURE_LEVEL_VARS.items():
        cds_download.download_seas5_pressure_level_anomaly(
            variable=cds_name,
            pressure_levels=config.SEAS5_PRESSURE_LEVELS,
            years=years,
            init_months=init_months,
            leadtime_months=leadtime_months,
            output_path=seas5_anomaly_path(
                short, is_pressure=True, years=years, months=init_months
            ),
            overwrite=overwrite,
        )


# =============================================================================
# Step 4: Apply bias correction
# =============================================================================
def step4_apply_correction(
    years: List[int] = None,
    months: List[int] = None,
    overwrite: bool = False,
):
    """Regrid ERA5 climatology to SEAS5 grid and apply delta correction."""
    if years is None:
        years = list(range(config.FORECAST_START_YEAR, config.FORECAST_END_YEAR + 1))

    logger.info("=== Step 4: Apply delta correction ===")

    for _, short in config.SINGLE_LEVEL_VARS.items():
        bias_correction.correct_dataset(
            era5_clim_file=era5_clim_path(short, is_pressure=False),
            seas5_anomaly_file=seas5_anomaly_path(
                short,
                is_pressure=False,
                years=years,
                months=months,
            ),
            output_file=corrected_path(short, is_pressure=False, years=years, months=months),
            overwrite=overwrite,
        )

    for _, short in config.PRESSURE_LEVEL_VARS.items():
        bias_correction.correct_dataset(
            era5_clim_file=era5_clim_path(short, is_pressure=True),
            seas5_anomaly_file=seas5_anomaly_path(
                short, is_pressure=True, years=years, months=months
            ),
            output_file=corrected_path(short, is_pressure=True, years=years, months=months),
            overwrite=overwrite,
        )


# =============================================================================
# Full pipeline
# =============================================================================
def run_all(
    forecast_years: List[int] = None,
    init_months: List[int] = None,
    leadtime_months: List[int] = None,
    overwrite: bool = False,
):
    """Run the complete pipeline end-to-end."""
    step1_download_era5(overwrite=overwrite)
    step2_build_climatology(overwrite=overwrite)
    step3_download_seas5(
        years=forecast_years,
        init_months=init_months,
        leadtime_months=leadtime_months,
        overwrite=overwrite,
    )
    step4_apply_correction(years=forecast_years, overwrite=overwrite)
    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run_all()
