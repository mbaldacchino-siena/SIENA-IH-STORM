"""ENSO phase schedule construction for forecast runs.

Builds a {month_int: phase} dict covering all 12 months of the target
forecast year by combining three monthly streams of Niño 3.4 anomaly:

  1. Observed monthly anomalies from NOAA PSL (ERSST V5, 1981-2010 base)
     — months strictly before the SEAS5 init month.
  2. SEAS5-derived monthly anomalies from the bias-corrected SST,
     for the lead horizon (init month + 5 following months for a 6-lead
     forecast under the standard SEAS5 convention).
  3. Persistence of the most recently classified phase for any remaining
     gap months (operationally standard for ENSO).

A single 3-month centered ONI is then computed from the merged monthly
stream, classified using the ±0.5 K threshold, and persistence applied
where a centered window is incomplete.

Why monthly (PSL) instead of pre-averaged 3-mo ONI (CPC):
    With monthly anomalies on both the observed and the SEAS5 side, we
    can compute one consistent centered 3-month average that mixes the
    two sources at the boundary (e.g. for an April-init forecast, ONI
    for April = mean(observed Mar + SEAS5 Apr + SEAS5 May)). The legacy
    code used CPC's already-3-mo-averaged ONI alongside monthly SEAS5
    anomalies, which silently applied 5-month smoothing to the observed
    side.

SEAS5 lead convention:
    For instantaneous monthly means (T, U, V, SST, MSLP) on the CDS
    `seasonal-monthly-*` and `seasonal-postprocessed-*` datasets,
    `leadtime_month=1` corresponds to the INIT MONTH ITSELF, not the
    following month. Sources:
      - C3S official tutorial https://ecmwf-projects.github.io/copernicus-training-c3s/sf-anomalies.html
        (May 2021 init with leadtime 1-6 -> "May to October")
      - Pichelli et al. 2021 (MDPI Climate, doi:10.3390/cli9120181):
        "6 lead times (i.e., the predictions provided for the month of
         initialization and the following 5 months)"

    Therefore LEAD_OFFSET = 0 in the valid-month formula:
        valid_month = ((init_month - 1) + (lead - 1) + LEAD_OFFSET) % 12 + 1

    Note: GRIB `valid_time` for accumulated quantities (precipitation)
    points to end-of-period, which is misleading. For instantaneous
    monthly means it does not apply.

For April init (month=4) with leads 1-6: valid_months = [4, 5, 6, 7, 8, 9].
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# Niño 3.4 region (5°S-5°N, 170°W-120°W = 190-240°E)
NINO34_LAT = (-5.0, 5.0)
NINO34_LON_0_360 = (190.0, 240.0)
NINO34_LON_M180_180 = (-170.0, -120.0)

# Default threshold for EN/LN classification
DEFAULT_THRESHOLD = 0.5

# SEAS5 convention: lead 1 = init month itself for instantaneous monthly
# means. See module docstring for citations.
DEFAULT_LEAD_OFFSET = 0


# =============================================================================
# Niño 3.4 area mean
# =============================================================================
def nino34_area_mean(
    sst_field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> float:
    """Area-weighted SST mean over Niño 3.4 (5°S-5°N, 170°W-120°W).

    Automatically handles lon in [0, 360] or [-180, 180].
    """
    lat_mask = (lats >= NINO34_LAT[0]) & (lats <= NINO34_LAT[1])

    if np.max(lons) > 180:
        lo, hi = NINO34_LON_0_360
    else:
        lo, hi = NINO34_LON_M180_180
    lon_mask = (lons >= lo) & (lons <= hi)

    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        return np.nan

    region = sst_field[np.ix_(lat_mask, lon_mask)]
    w = np.cos(np.deg2rad(lats[lat_mask]))[:, None]
    w = np.broadcast_to(w, region.shape)
    valid = np.isfinite(region)
    if valid.sum() == 0:
        return np.nan
    return float(np.nansum(region * w * valid) / np.nansum(w * valid))


# =============================================================================
# Observed monthly Niño 3.4 anomaly
# =============================================================================
def load_observed_monthly(
    target_year: int,
    init_month: int,
    source: str = "psl",
    csv_path: Optional[str] = None,
) -> Dict[int, float]:
    """Observed monthly Niño 3.4 anomalies for months strictly < init_month.

    Default source: NOAA PSL (ERSST V5, 1981-2010 baseline). PSL provides
    raw monthly anomalies — exactly what we need to combine with SEAS5
    monthly anomalies under a single centered 3-month average. Also
    includes Dec of (target_year - 1) keyed as month=0 to allow a
    January-centered window.

    Parameters
    ----------
    target_year : int
    init_month : int
    source : {'psl', 'csv'}
        - 'psl': fetch from NOAA PSL (cached 24h on disk).
        - 'csv': read a local CSV with columns [year, month, value].
    csv_path : str or None
        Required when source='csv'. Also used as fallback if 'psl' fails.

    Returns
    -------
    dict : {month_int: monthly_anomaly_K}  (month_int may include 0 = Dec prev)
    """
    if source == "psl":
        try:
            from . import psl_oni
        except ImportError:  # pragma: no cover
            import psl_oni
        try:
            return psl_oni.load_observed_monthly_dict(target_year, init_month)
        except Exception as ex:
            if csv_path:
                logger.warning(
                    "PSL Nino 3.4 fetch failed (%s); falling back to CSV %s",
                    ex,
                    csv_path,
                )
                source = "csv"
            else:
                raise

    if source == "csv":
        if not csv_path:
            logger.warning("No csv_path provided; returning empty observed anomalies")
            return {}
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            logger.warning(
                "CSV not found at %s; returning empty observed anomalies", csv_path
            )
            return {}
        # Accept either 'value' or 'climate_index' column for back-compat.
        val_col = "value" if "value" in df.columns else "climate_index"
        obs = {}
        cur = df[(df["year"] == target_year) & (df["month"] < init_month)]
        for _, row in cur.iterrows():
            v = row.get(val_col, np.nan)
            if np.isfinite(v):
                obs[int(row["month"])] = float(v)
        prev = df[(df["year"] == target_year - 1) & (df["month"] == 12)]
        if len(prev) and np.isfinite(prev[val_col].iloc[0]):
            obs[0] = float(prev[val_col].iloc[0])
        logger.info(
            "Observed monthly anomaly (csv): %d months (%s)",
            len(obs),
            sorted(obs.keys()),
        )
        return obs

    raise ValueError(f"Unknown source: {source!r} (use 'psl' or 'csv')")


# Back-compat alias — older callers expected this name. New code should use
# `load_observed_monthly`. The old function returned 3-mo-centered ONI from
# CPC; this version returns monthly anomalies, which has the SAME interface
# but the values are NOT pre-averaged. Existing call sites that simply pass
# the result through to `build_phase_schedule` continue to work because the
# new build_phase_schedule expects monthly anomalies.
load_observed_oni = load_observed_monthly


# =============================================================================
# SEAS5-derived monthly Niño 3.4 anomaly
# =============================================================================
def seas5_monthly_nino34_anomaly(
    corrected_sst: xr.DataArray,
    era5_sst_clim: xr.DataArray,
    init_month: int,
    lead_offset: int = DEFAULT_LEAD_OFFSET,
) -> Dict[int, float]:
    """Monthly Niño 3.4 anomaly from bias-corrected SEAS5 SST.

    Subtracts the ERA5 monthly climatology at the Niño 3.4 area-mean level.
    Result is a raw monthly anomaly (NOT yet a 3-month mean).

    Parameters
    ----------
    corrected_sst : xr.DataArray
        Bias-corrected SST, dims include (forecastMonth, latitude, longitude).
        Already reduced to a single init time and single member.
    era5_sst_clim : xr.DataArray
        ERA5 monthly SST climatology with `month` dim (1-12), on SAME grid.
    init_month : int
    lead_offset : int
        SEAS5 lead convention. 0 (default) for instantaneous monthly means
        on the `seasonal-monthly-*` and `seasonal-postprocessed-*` CDS
        datasets — see module docstring.

    Returns
    -------
    dict : {valid_month: monthly_anomaly_K}
    """
    if "latitude" in corrected_sst.coords:
        lats = corrected_sst.latitude.values
        lons = corrected_sst.longitude.values
    else:
        lats = corrected_sst.lat.values
        lons = corrected_sst.lon.values

    n_leads = int(corrected_sst.sizes["forecastMonth"])
    anomalies: Dict[int, float] = {}

    for lead_idx in range(n_leads):
        lead = lead_idx + 1
        valid_month = ((init_month - 1) + (lead - 1) + lead_offset) % 12 + 1

        sst_slice = corrected_sst.isel(forecastMonth=lead_idx).values
        clim_slice = era5_sst_clim.sel(month=valid_month).values

        a_fcst = nino34_area_mean(sst_slice, lats, lons)
        a_clim = nino34_area_mean(clim_slice, lats, lons)
        if np.isfinite(a_fcst) and np.isfinite(a_clim):
            anomalies[valid_month] = a_fcst - a_clim

    return anomalies


# =============================================================================
# Merge observed + SEAS5 monthly streams, then 3-mo centered ONI
# =============================================================================
def merge_monthly_streams(
    observed: Dict[int, float],
    seas5: Dict[int, float],
) -> Dict[int, float]:
    """Combine observed and SEAS5 monthly anomaly dicts.

    Observed values take precedence where both sources exist. The result
    can include negative or zero keys (e.g. month=0 = Dec of previous year)
    used to support a January-centered 3-month ONI window.

    Note on baseline consistency: PSL observed anomalies use 1981-2010 and
    SEAS5 (post bias correction) effectively uses ERA5 1993-2016. The two
    baselines differ by ~0.1 K of warming trend, but both are FIXED so
    their combination is stable year-on-year. Document in the paper as a
    minor known offset; do not try to harmonise.
    """
    merged: Dict[int, float] = dict(seas5)
    merged.update(observed)  # observed wins
    return merged


def centered_oni_from_merged(
    monthly: Dict[int, float],
) -> Dict[int, float]:
    """3-month centered ONI from a merged monthly anomaly dict.

    For each target month m in 1..12, compute mean of monthly[m-1],
    monthly[m], monthly[m+1] if all three are present. Uses keys 0
    (= Dec previous year) and 13 (= Jan next year) at the calendar
    boundaries when available.
    """
    oni: Dict[int, float] = {}
    for m in range(1, 13):
        prev_key = m - 1  # may be 0 = Dec previous year
        next_key = m + 1  # may be 13 = Jan next year (rarely available)
        if all(k in monthly for k in (prev_key, m, next_key)):
            oni[m] = (monthly[prev_key] + monthly[m] + monthly[next_key]) / 3.0
    return oni


# Back-compat alias for callers that haven't moved to the merged-stream API.
# The old function only worked on a SEAS5-only dict; the new one is more
# general but still returns the same shape.
seas5_centered_oni_from_monthly = centered_oni_from_merged


# =============================================================================
# Phase classification
# =============================================================================
def classify(oni_val: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Classify ONI value to 'EN' / 'LN' / 'NEU'."""
    if not np.isfinite(oni_val):
        return "NEU"
    if oni_val >= threshold:
        return "EN"
    if oni_val <= -threshold:
        return "LN"
    return "NEU"


# =============================================================================
# Build the 12-month phase schedule
# =============================================================================
def build_phase_schedule(
    monthly: Dict[int, float],
    centered_oni: Dict[int, float],
    sources: Dict[int, str],
    init_month: int,
    threshold: float = DEFAULT_THRESHOLD,
    verbose: bool = True,
) -> Dict[int, str]:
    """Build a {1..12 → phase} dict from a merged monthly stream and centered ONI.

    Per month m:
      1. If a centered 3-month ONI is available for m, classify directly.
      2. Otherwise persist the most recently classified phase (forward fill,
         then backward fill for any leading gap).

    Parameters
    ----------
    monthly : {month: anomaly}
        Merged observed + SEAS5 monthly anomaly stream. Used here only
        for verbose logging — the actual classification reads `centered_oni`.
    centered_oni : {month: ONI_value}
        Centered 3-month means produced by `centered_oni_from_merged`.
    sources : {month: 'obs' | 'seas5' | 'mixed'}
        Per-month tag indicating which source the centered ONI came from.
        'mixed' means at least one of (m-1, m, m+1) was observed and at
        least one was SEAS5. Used for logging only.
    init_month : int
    threshold : float
    verbose : bool
    """
    schedule: Dict[int, str] = {}
    classified_sources: Dict[int, str] = {}

    # Step 1: classify wherever a centered ONI is available
    last_phase: Optional[str] = None
    for m in range(1, 13):
        if m in centered_oni and np.isfinite(centered_oni[m]):
            schedule[m] = classify(centered_oni[m], threshold)
            classified_sources[m] = sources.get(m, "?")
            last_phase = schedule[m]
        else:
            schedule[m] = None  # filled below

    # Step 2: forward-fill persistence
    last_phase = None
    for m in range(1, 13):
        if schedule[m] is not None:
            last_phase = schedule[m]
        elif last_phase is not None:
            schedule[m] = last_phase
            classified_sources[m] = "persist"

    # Step 3: backward-fill any leading gap (rare, but possible if no
    # centered ONI is available for early months)
    next_phase = None
    for m in range(12, 0, -1):
        if schedule[m] is not None:
            next_phase = schedule[m]
        elif next_phase is not None:
            schedule[m] = next_phase
            classified_sources[m] = "persist"

    # If everything is None (no data at all), default to NEU silently
    schedule = {
        m: (schedule[m] if schedule[m] is not None else "NEU") for m in range(1, 13)
    }

    if verbose:
        logger.info(
            "ENSO schedule (init_month=%d, threshold=%.2f):", init_month, threshold
        )
        for m in range(1, 13):
            oni_val = centered_oni.get(m, np.nan)
            src = classified_sources.get(m, "none")
            logger.info(
                "  month %2d: ONI=%+.2f  phase=%s  source=%-7s",
                m,
                oni_val,
                schedule[m],
                src,
            )

    return schedule


def label_centered_oni_sources(
    observed: Dict[int, float],
    seas5: Dict[int, float],
    centered_oni: Dict[int, float],
) -> Dict[int, str]:
    """Tag each centered-ONI month by which monthly streams it draws from."""
    sources: Dict[int, str] = {}
    for m in centered_oni:
        in_obs = sum(1 for k in (m - 1, m, m + 1) if k in observed)
        in_s5 = sum(1 for k in (m - 1, m, m + 1) if k in seas5 and k not in observed)
        if in_obs == 3:
            sources[m] = "obs"
        elif in_s5 == 3:
            sources[m] = "seas5"
        else:
            sources[m] = "mixed"
    return sources


# =============================================================================
# Top-level convenience function
# =============================================================================
def phase_schedule_from_corrected(
    corrected_sst: xr.DataArray,
    era5_sst_clim: xr.DataArray,
    init_date: str,
    csv_path: Optional[str] = None,
    observed_source: str = "psl",
    threshold: float = DEFAULT_THRESHOLD,
    lead_offset: int = DEFAULT_LEAD_OFFSET,
    verbose: bool = True,
) -> Dict[int, str]:
    """Build 12-month phase schedule from bias-corrected SEAS5 + observed.

    Pipeline:
      observed monthly anomalies (NOAA PSL, before init month)
        + SEAS5 monthly anomalies (from bias-corrected SST, lead horizon)
        -> merge into single monthly stream
        -> 3-month centered ONI per target month
        -> classify (EN/NEU/LN at ±0.5 K)
        -> persistence for any uncovered months

    Parameters
    ----------
    corrected_sst : xr.DataArray
        Bias-corrected SST, single member, single init time.
        Dims: (forecastMonth, lat, lon).
    era5_sst_clim : xr.DataArray
        ERA5 SST climatology with `month` dim, on same grid as corrected_sst.
    init_date : str, "YYYY-MM-DD"
    csv_path : str or None
        Optional CSV with columns [year, month, value]. Used as fallback
        if PSL fetch fails, or as the primary source if observed_source='csv'.
    observed_source : {'psl', 'csv'}
        Default 'psl' (NOAA PSL ERSST V5 monthly anomalies, online).
    threshold : float
    lead_offset : int
        SEAS5 lead-month convention. Default 0 (lead 1 = init month) for
        instantaneous monthly means. Set to 1 only if you've verified your
        downloaded files use the alternative convention.
    verbose : bool
    """
    target_year = int(init_date[:4])
    init_month = int(init_date[5:7])

    observed = load_observed_monthly(
        target_year,
        init_month,
        source=observed_source,
        csv_path=csv_path,
    )
    seas5 = seas5_monthly_nino34_anomaly(
        corrected_sst,
        era5_sst_clim,
        init_month,
        lead_offset=lead_offset,
    )

    if verbose:
        logger.info(
            "Observed PSL Niño 3.4 anomalies: %s",
            {m: round(v, 2) for m, v in sorted(observed.items())},
        )
        logger.info(
            "SEAS5 Niño 3.4 anomalies: %s",
            {m: round(v, 2) for m, v in sorted(seas5.items())},
        )

    merged = merge_monthly_streams(observed, seas5)
    centered = centered_oni_from_merged(merged)
    sources = label_centered_oni_sources(observed, seas5, centered)

    if verbose:
        logger.info(
            "Centered 3-mo ONI (per source): %s",
            {m: (round(centered[m], 2), sources[m]) for m in sorted(centered)},
        )

    return build_phase_schedule(
        merged, centered, sources, init_month, threshold=threshold, verbose=verbose
    )
