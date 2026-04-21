"""Example usage patterns for the SEAS5 bias correction framework.

These examples assume you have a working CDS API key in ~/.cdsapirc.
"""

import pipeline
import config


# =============================================================================
# Example 1: Full default pipeline (climatology + all years 2017-2026)
# =============================================================================
def example_full_default():
    """Run everything end-to-end with defaults from config.py."""
    pipeline.run_all()


# =============================================================================
# Example 2: Only 2026 April-initialized forecast (your whitepaper use case)
# =============================================================================
def example_2026_hurricane_season():
    """Download/correct the April 2026 initialization, leads 1-6 (Apr-Sep)."""
    # Steps 1 + 2 only need to run ONCE per climatology period.
    # They create ERA5 climatology files that get reused.
    pipeline.step1_download_era5()
    pipeline.step2_build_climatology()

    # Step 3 + 4: just for the 2026 forecast
    pipeline.step3_download_seas5(
        years=[2026],
        init_months=[4],  # April init
        leadtime_months=[1, 2, 3, 4, 5, 6],
    )
    pipeline.step4_apply_correction(years=[2026])


# =============================================================================
# Example 3: Backtest over operational archive 2017-2025
# =============================================================================
def example_backtest_archive():
    """Correct the full operational archive for a backtest experiment."""
    # Skip steps 1-2 if climatology is already built
    pipeline.step3_download_seas5(
        years=list(range(2017, 2026)),
        init_months=[4, 5, 6],  # Apr/May/Jun inits covering NH season
        leadtime_months=[1, 2, 3, 4, 5, 6],
    )
    pipeline.step4_apply_correction(years=list(range(2017, 2026)))


# =============================================================================
# Example 4: Load corrected output and inspect
# =============================================================================
def example_load_corrected():
    """Load corrected SST and check dimensions."""
    import xarray as xr

    path = pipeline.corrected_path("sst", is_pressure=False, years=[2026])
    ds = xr.open_dataset(path)
    print(ds)
    # Expected dims: (time/init, forecastMonth/lead, number/member, lat, lon)
    # Feed this directly into your TC hazard generator.


if __name__ == "__main__":
    # Pick the example you want to run
    example_2026_hurricane_season()
