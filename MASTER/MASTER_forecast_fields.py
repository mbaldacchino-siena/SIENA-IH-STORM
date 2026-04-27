"""
Download and prepare CDS SEAS5 seasonal forecast fields for SIENA-IH-STORM.

CDS dataset structure (from xr.open_dataset):
  Dimensions:
    number: 51                    <- ensemble members (always all 51)
    forecast_reference_time: 1    <- init date
    forecastMonth: 6              <- lead time months
    pressure_level: 12            <- standard SEAS5 levels
    latitude: 180, longitude: 360 <- 1 deg grid
  Data variables: u, v, t, q (pressure levels)
  Surface file:   sst, msl       (single level, same member/time dims)

Workflow:
  1. Download SEAS5 monthly-mean fields (single request -> all 51 members)
  2. For each ensemble member:
     a. Regrid to 0.25 deg to match ERA5 model fields
     b. Compute VWS, RH600
     c. Compute PI via tcpyPI from full T/Q profile
     d. Derive ONI 3.4 from member's Nino 3.4 SST
     e. Save as env_yearly/{STEM}_{env_year}_{month}.npy
     f. Optionally write forecast_config.json with auto-derived phases

Usage:
  # Download once + process all members + generate configs:
  python MASTER_forecast_fields.py \
      --init-date 2026-04-01 \
      --lead-months 6 \
      --generate-config \
      --active-months 6 7 8 9 10 11

  # Process specific members only (from already-downloaded files):
  python MASTER_forecast_fields.py \
      --init-date 2026-04-01 \
      --lead-months 6 \
      --member 0 12 37 \
      --skip-download \
      --generate-config
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import time
import multiprocessing as mp
from functools import partial
from scipy.interpolate import RegularGridInterpolator

try:
    import cdsapi
except ImportError:
    cdsapi = None

try:
    from CODE.potential_intensity import (
        compute_pi_field_tcpyPI,
        compute_pi_field_simplified,
        _nanfill_nearest,
        _upscale_to_target,
        _coarsen_to_match,
        HAS_TCPYPI,
    )
except ImportError:
    HAS_TCPYPI = False

from CODE.siena_utils import (
    save_yearly_field,
    _env_yearly_dir,
    compute_relative_vorticity_spherical,
)

# Bias-correction pipeline (replaces raw SEAS5 download)
from FORECAST.SEAS5 import loader as seas5_loader
from FORECAST.SEAS5 import enso as seas5_enso

__location__ = os.path.realpath(os.getcwd())


# =========================================================================
# SEAS5 preparation — delegates to the bias-correction pipeline.
# This replaces the previous direct-download approach: we now pull SEAS5
# anomalies from the CDS `seasonal-postprocessed-*` datasets, regrid the
# ERA5 1993-2016 climatology, and reconstruct absolute fields via
#       X_corrected = ERA5_clim(valid_month) + SEAS5_anomaly
# before any downstream processing. See FORECAST/SEAS5/README.md for the
# scientific rationale and the reference-period mismatch caveat.
# =========================================================================


def prepare_seas5_bias_corrected(init_date, lead_months, overwrite=False):
    """Run bias-correction pipeline for this init and return merged datasets.

    Idempotent: skips ERA5 download + climatology steps if files already
    exist on disk. Only the forecast-specific steps (download SEAS5
    anomalies + apply correction) re-run when the target init changes.

    Returns
    -------
    ds_pl : xr.Dataset  (u, v, t, q on SEAS5 pressure levels)
    ds_sfc : xr.Dataset (sst, msl)
        Both on SEAS5 native grid (~1°), values on the ERA5 absolute scale.
    """
    seas5_loader.ensure_climatology()
    ds_pl, ds_sfc = seas5_loader.prepare_corrected_for_init(
        init_date=init_date,
        lead_months=lead_months,
        overwrite_correction=overwrite,
    )
    return ds_pl, ds_sfc


# =========================================================================
# Regridding
# =========================================================================


def regrid_to_025(field, src_lats, src_lons, dst_lats=None, dst_lons=None):
    """Bilinear regrid from source grid to 0.25 deg global grid."""
    if dst_lats is None:
        dst_lats = np.arange(90, -90.25, -0.25)
    if dst_lons is None:
        dst_lons = np.arange(0, 360, 0.25)

    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        field = field[::-1, :]

    interp = RegularGridInterpolator(
        (src_lats, src_lons),
        field,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    dst_lat_grid, dst_lon_grid = np.meshgrid(dst_lats, dst_lons, indexing="ij")
    return interp((dst_lat_grid, dst_lon_grid))


# =========================================================================
# Derived variable computation
# =========================================================================


def compute_vws(u200, u850, v200, v850):
    """Vertical wind shear magnitude (200-850 hPa)."""
    return np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)


def compute_rh_from_q_t(q, t, pressure_pa):
    """Relative humidity (%) from specific humidity q (kg/kg), temperature T (K),
    and ambient pressure P (Pa).

    Uses the Bolton (1980) formula for saturation vapour pressure over water:
        e_s(T) = 611.2 * exp(17.67 * T_c / (T_c + 243.5))     [Pa]
    with T_c = T - 273.15.

    Mixing ratios:
        w   = q / (1 - q)
        w_s = 0.622 * e_s / (P - e_s)
        RH  = 100 * w / w_s

    Consistency requirement: q, T, and pressure_pa MUST refer to the same
    pressure level. Feeding (q at 500, T at 500, P=60000) gives garbage
    because w_s is evaluated at the wrong pressure. Callers should
    interpolate q and T to the target level (e.g. via `interp_to_pressure`)
    BEFORE calling this function, and pass the matching pressure.

    Clipped to [5, 99] rather than [0, 100]: monthly-mean RH at 0 or 100
    is almost always an interpolation artifact, and the GPI formula
    (H/50)^3 reacts catastrophically to both extremes.
    """
    t_c = t - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    w = q / (1.0 - q)
    ws = 0.622 * es / (pressure_pa - es)
    rh = 100.0 * w / ws
    return np.clip(rh, 5.0, 99.0)


def interp_to_pressure(field_on_levels, levels_hPa, target_hPa):
    """Linear-in-log-pressure interpolation to a target level.

    Parameters
    ----------
    field_on_levels : array, shape (n_levels, ..., lat, lon)
        Monotonic along axis 0 in the usual atmospheric sense is NOT
        required — we find the bracketing levels explicitly.
    levels_hPa : array-like, shape (n_levels,)
        Pressure of each level in hPa (any order).
    target_hPa : float

    Returns
    -------
    array, shape (..., lat, lon)

    Notes
    -----
    Standard practice in atmospheric science is to interpolate in
    log-pressure because the hydrostatic vertical coordinate is
    logarithmic in pressure (z ~ -H ln(p/p0)). Linear-in-p would
    systematically over- or under-weight one side depending on where
    target_hPa sits relative to the bracket midpoint.

    If target_hPa is exactly equal to one of the available levels,
    that slice is returned directly (no interpolation).

    If target_hPa is outside the range of available levels, raises
    ValueError rather than silently extrapolating.
    """
    levels = np.asarray(levels_hPa, dtype=float)
    target = float(target_hPa)

    # Exact match — no interpolation
    exact = np.where(levels == target)[0]
    if exact.size:
        return field_on_levels[int(exact[0])]

    below = levels[levels < target]
    above = levels[levels > target]
    if below.size == 0 or above.size == 0:
        raise ValueError(
            f"Target pressure {target} hPa outside available levels {sorted(levels)}"
        )
    p_lo = below.max()  # larger p = closer to surface ... but we want the
    p_hi = above.min()  # two levels that bracket the target numerically
    i_lo = int(np.where(levels == p_lo)[0][0])
    i_hi = int(np.where(levels == p_hi)[0][0])

    log_lo = np.log(p_lo)
    log_hi = np.log(p_hi)
    log_tg = np.log(target)
    frac = (log_tg - log_lo) / (log_hi - log_lo)

    return field_on_levels[i_lo] + frac * (
        field_on_levels[i_hi] - field_on_levels[i_lo]
    )


# =========================================================================
# Per-member field extraction and processing
# =========================================================================


def _nearest_level_idx(levels, target):
    """Index of nearest pressure level to target (hPa)."""
    return int(np.abs(levels - target).argmin())


def process_member(
    ds_pl, ds_sfc, member_idx, init_date, lead_months, env_year, dst_lats, dst_lons
):
    """
    Extract, regrid, derive, and save fields for a single ensemble member.

    Parameters
    ----------
    ds_pl : xarray.Dataset, pressure-level data (all members, already open)
    ds_sfc : xarray.Dataset, surface data (all members, already open)
    member_idx : int, index into the 'number' dimension (0-50)
    init_date : str, "YYYY-MM-DD"
    lead_months : int
    env_year : int, synthetic year label for storage (e.g. 10012 for member 12)
    dst_lats, dst_lons : 1D arrays, target 0.25 deg grid
    """
    init_month = int(init_date[5:7])

    # Subset to this member, squeeze out forecast_reference_time
    pl = ds_pl.isel(number=member_idx)
    sfc = ds_sfc.isel(number=member_idx)
    if "forecast_reference_time" in pl.dims:
        pl = pl.squeeze("forecast_reference_time")
    if "forecast_reference_time" in sfc.dims:
        sfc = sfc.squeeze("forecast_reference_time")

    # Source grid
    src_lats = pl.latitude.values
    src_lons = pl.longitude.values
    native_shape = (len(src_lats), len(src_lons))
    fine_shape = (len(dst_lats), len(dst_lons))

    # Pressure levels
    p_levels = pl.pressure_level.values.astype(float)
    idx_200 = _nearest_level_idx(p_levels, 200)
    idx_850 = _nearest_level_idx(p_levels, 850)
    # 600 hPa is not in the SEAS5 pressure level list [10, 50, 100, 200,
    # 300, 400, 500, 700, 850, 925, 1000]. We log-pressure interpolate
    # between 500 and 700 in `process_member`'s lead loop (needs per-lead
    # slicing). The nearest-index approach would silently have returned
    # 500 (first equidistant) and mismatched the downstream RH formula.

    n_leads = min(int(pl.sizes["forecastMonth"]), lead_months)

    for t_idx in range(n_leads):
        # SEAS5 lead 1 (t_idx=0) = init month itself for instantaneous monthly
        # means. valid_month = ((init_month - 1) + (lead - 1)) % 12 + 1.
        valid_month = ((init_month - 1) + t_idx) % 12 + 1

        # Pressure-level fields: (pressure_level, latitude, longitude)
        pl_t = pl.isel(forecastMonth=t_idx)

        u200 = regrid_to_025(
            pl_t["u"].isel(pressure_level=idx_200).values,
            src_lats,
            src_lons,
            dst_lats,
            dst_lons,
        )
        u850 = regrid_to_025(
            pl_t["u"].isel(pressure_level=idx_850).values,
            src_lats,
            src_lons,
            dst_lats,
            dst_lons,
        )
        v200 = regrid_to_025(
            pl_t["v"].isel(pressure_level=idx_200).values,
            src_lats,
            src_lons,
            dst_lats,
            dst_lons,
        )
        v850 = regrid_to_025(
            pl_t["v"].isel(pressure_level=idx_850).values,
            src_lats,
            src_lons,
            dst_lats,
            dst_lons,
        )
        # t and q at 600 hPa: log-pressure interpolation between the
        # bracketing SEAS5 levels (500 and 700 hPa). Interpolate at native
        # resolution FIRST (so we don't interpolate on resampled fields),
        # then regrid to 0.25°.
        t_native = pl_t["t"].values  # (pressure_level, lat, lon)
        q_native = pl_t["q"].values
        t600_native = interp_to_pressure(t_native, p_levels, 600.0)
        q600_native = interp_to_pressure(q_native, p_levels, 600.0)
        t600 = regrid_to_025(t600_native, src_lats, src_lons, dst_lats, dst_lons)
        q600 = regrid_to_025(q600_native, src_lats, src_lons, dst_lats, dst_lons)

        # Surface fields
        sfc_t = sfc.isel(forecastMonth=t_idx)
        sfc_lats = sfc_t.latitude.values
        sfc_lons = sfc_t.longitude.values

        sst = regrid_to_025(
            sfc_t["sst"].values,
            sfc_lats,
            sfc_lons,
            dst_lats,
            dst_lons,
        )
        mslp = regrid_to_025(
            sfc_t["msl"].values,
            sfc_lats,
            sfc_lons,
            dst_lats,
            dst_lons,
        )
        mslp *= 0.01  # Pa -> hPa

        # Derived
        vws = compute_vws(u200, u850, v200, v850)
        rh600 = compute_rh_from_q_t(q600, t600, pressure_pa=60000.0)
        vort850 = compute_relative_vorticity_spherical(u850, v850, dst_lats, dst_lons)

        # Thermodynamic PI
        if HAS_TCPYPI:
            t_profile = pl_t["t"].values  # (pressure_level, lat, lon)
            q_profile = pl_t["q"].values

            sst_native = sfc_t["sst"].values
            mslp_native = sfc_t["msl"].values * 0.01
            if sst_native.shape != native_shape:
                sst_native = _coarsen_to_match(sst_native, native_shape)
                mslp_native = _coarsen_to_match(mslp_native, native_shape)

            pmin, vmax = compute_pi_field_tcpyPI(
                sst_native,
                mslp_native,
                t_profile,
                q_profile,
                p_levels,
            )
            pmin = _nanfill_nearest(pmin)
            vmax = _nanfill_nearest(vmax)
            if pmin.shape != fine_shape:
                pmin = _upscale_to_target(pmin, fine_shape)
            if vmax.shape != fine_shape:
                vmax = _upscale_to_target(vmax, fine_shape)
        else:
            pmin, vmax = compute_pi_field_simplified(sst, mslp)

        # Save
        save_yearly_field(__location__, "VWS", env_year, valid_month, vws)
        save_yearly_field(__location__, "RH600", env_year, valid_month, rh600)
        save_yearly_field(__location__, "MSLP", env_year, valid_month, mslp)
        save_yearly_field(__location__, "SST", env_year, valid_month, sst)
        save_yearly_field(__location__, "PI", env_year, valid_month, pmin)
        save_yearly_field(__location__, "VMAX_PI", env_year, valid_month, vmax)
        save_yearly_field(__location__, "VORT850", env_year, valid_month, vort850)

        print(f"    month {valid_month} (lead {t_idx + 1}) saved")


# =========================================================================
# ONI 3.4 derivation from SEAS5 SST + observed climate_index.csv
# =========================================================================

NINO34_LAT = (-5.0, 5.0)
NINO34_LON = (190.0, 240.0)  # 170W-120W in 0-360 convention
ONI_CLIM_PERIOD = (1991, 2020)


def compute_nino34_sst(sst_field, lats, lons):
    """Area-weighted mean SST over Nino 3.4 region."""
    lat_mask = (lats >= NINO34_LAT[0]) & (lats <= NINO34_LAT[1])
    lon_mask = (lons >= NINO34_LON[0]) & (lons <= NINO34_LON[1])
    region = sst_field[np.ix_(lat_mask, lon_mask)]
    region_lats = lats[lat_mask]
    weights = np.cos(np.deg2rad(region_lats))[:, None]
    weights = np.broadcast_to(weights, region.shape)
    valid = np.isfinite(region)
    if valid.sum() == 0:
        return np.nan
    return float(np.nansum(region * weights) / np.nansum(weights * valid))


def load_oni_climatology():
    """
    Per-month baseline Nino 3.4 SST from ERA5 Monthly_mean_SST.nc
    over the 1991-2020 climatology period.
    """
    sst_nc = os.path.join(__location__, "Monthly_mean_SST.nc")
    if not os.path.exists(sst_nc):
        return None

    ds = xr.open_dataset(sst_nc)
    var = "sst" if "sst" in ds else list(ds.data_vars)[0]
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    lats = ds.latitude.values
    lons = ds.longitude.values
    times = pd.to_datetime(ds[time_dim].values)

    monthly_vals = {m: [] for m in range(1, 13)}
    for t_idx in range(len(times)):
        yr, mo = times[t_idx].year, times[t_idx].month
        if ONI_CLIM_PERIOD[0] <= yr <= ONI_CLIM_PERIOD[1]:
            sst = ds[var].isel({time_dim: t_idx}).values
            val = compute_nino34_sst(sst, lats, lons)
            if np.isfinite(val):
                monthly_vals[mo].append(val)
    ds.close()

    return {
        m: float(np.mean(monthly_vals[m])) if monthly_vals[m] else np.nan
        for m in range(1, 13)
    }


def compute_oni_from_member(
    ds_sfc, member_idx, init_date, lead_months, sst_climatology
):
    """
    Compute monthly Nino 3.4 SST anomalies from one SEAS5 member.
    Works directly on the already-open ds_sfc dataset (all members).
    Returns raw monthly anomalies (not 3-month running mean).
    """
    sfc = ds_sfc.isel(number=member_idx)
    if "forecast_reference_time" in sfc.dims:
        sfc = sfc.squeeze("forecast_reference_time")

    lats = sfc.latitude.values
    lons = sfc.longitude.values
    init_month = int(init_date[5:7])

    anomalies = {}
    n_leads = min(int(sfc.sizes["forecastMonth"]), lead_months)
    for t_idx in range(n_leads):
        # SEAS5 lead 1 (t_idx=0) = init month itself.
        valid_month = ((init_month - 1) + t_idx) % 12 + 1
        sst_field = sfc["sst"].isel(forecastMonth=t_idx).values
        nino34 = compute_nino34_sst(sst_field, lats, lons)
        baseline = sst_climatology.get(valid_month, np.nan)
        if np.isfinite(nino34) and np.isfinite(baseline):
            anomalies[valid_month] = nino34 - baseline
    return anomalies


def build_phase_schedule_from_seas5(
    ds_sfc,
    member_idx,
    init_date,
    lead_months,
    climate_index_path=None,
    observed_source="psl",
    threshold=0.5,
):
    """
    Build a 12-month phase_schedule for one SEAS5 member.

    Delegates to FORECAST.SEAS5.enso.phase_schedule_from_corrected. By
    default the observed monthly Niño 3.4 anomalies are fetched from NOAA
    PSL (ERSST V5, 1981-2010 baseline) and cached for 24h on disk under
    FORECAST/SEAS5/data/psl_nina34.txt.

    The PSL route gives MONTHLY anomalies (not pre-averaged ONI), which
    lets us compute one consistent centered 3-month average that mixes
    observed and SEAS5 monthly values across the init-month boundary.
    Falls back to a CSV (columns: year, month, value) if the network call
    fails and `climate_index_path` is provided.

    Parameters
    ----------
    ds_sfc : xr.Dataset, bias-corrected surface dataset (contains 'sst' + 'msl')
    member_idx : int
    init_date : str "YYYY-MM-DD"
    lead_months : int  (kept for API compatibility; inferred from ds_sfc)
    climate_index_path : str or None
        Required only if observed_source='csv'. Also used as a fallback
        if 'psl' is requested and the network call fails.
    observed_source : {'psl', 'csv'}
    threshold : float
    """
    # Extract this member's bias-corrected SST; squeeze any singleton dims.
    sst = ds_sfc["sst"].isel(number=member_idx)
    for dim in ("forecast_reference_time", "time"):
        if dim in sst.dims and sst.sizes[dim] == 1:
            sst = sst.squeeze(dim)

    era5_sst_clim = seas5_loader.load_era5_sst_climatology()

    csv_full_path = (
        os.path.join(__location__, climate_index_path) if climate_index_path else None
    )

    return seas5_enso.phase_schedule_from_corrected(
        corrected_sst=sst,
        era5_sst_clim=era5_sst_clim,
        init_date=init_date,
        csv_path=csv_full_path,
        observed_source=observed_source,
        threshold=threshold,
    )


# =========================================================================
# Config generation
# =========================================================================


def generate_forecast_config(
    init_date,
    lead_months,
    member_idx,
    env_year,
    active_months,
    phase_schedule,
    observed_months=None,
    out_path="forecast_config.json",
):
    """Write forecast_config.json for one ensemble member."""
    init_year = int(init_date[:4])
    init_month = int(init_date[5:7])

    if observed_months is None:
        observed_months = list(range(1, init_month))

    # SEAS5 lead 1 = init month itself (instantaneous monthly means).
    # For April init with lead_months=6: forecast_months = [4, 5, 6, 7, 8, 9].
    forecast_months = [((init_month - 1) + i) % 12 + 1 for i in range(lead_months)]

    config = {
        "mode": "seasonal_forecast",
        "base_year": init_year,
        "ensemble_member": member_idx,
        "init_date": init_date,
        "months": {},
    }

    for m in range(1, 13):
        phase = phase_schedule.get(m, "NEU")
        if m in observed_months:
            config["months"][str(m)] = {
                "source": "observed",
                "phase": phase,
                "env_year": init_year,
            }
        elif m in forecast_months:
            config["months"][str(m)] = {
                "source": "forecast",
                "phase": phase,
                "env_year": env_year,
            }
        else:
            config["months"][str(m)] = {
                "source": "historical",
                "phase": phase,
            }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config written: {out_path}")
    return config


def _run_single_member(
    job,
    init_date,
    lead_months,
    generate_config=False,
    active_months=None,
    oni_threshold=0.5,
):
    """
    Per-member processing. The bias-correction pipeline must have been
    run already (typically once in __main__); this function only opens
    the already-corrected NetCDFs and extracts fields for one member.

    Parameters
    ----------
    job : member index (0-based) within the SEAS5 ensemble
    init_date : str, "YYYY-MM-DD"
    lead_months : int
    generate_config : bool
    active_months : list of int
    oni_threshold : float
    """
    # Open the merged bias-corrected datasets (all 51 members, single init).
    # These are produced by prepare_seas5_bias_corrected() in __main__ and
    # cached on disk under FORECAST/SEAS5/data/seas5_corrected/.
    ds_pl, ds_sfc = seas5_loader.prepare_corrected_for_init(
        init_date=init_date,
        lead_months=lead_months,
        overwrite_correction=False,  # idempotent; will not re-download/recorrect
    )

    # Target grid for the 0.25° regrid step inside process_member
    try:
        ds_ref = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
        dst_lats = ds_ref.latitude.values
        dst_lons = ds_ref.longitude.values
        ds_ref.close()
    except Exception:
        dst_lats = np.arange(90, -90.25, -0.25)
        dst_lons = np.arange(0, 360, 0.25)

    print(f"Processing member {job}, {lead_months} lead months (bias-corrected)")
    print(f"  PI: {'tcpyPI (full thermodynamic)' if HAS_TCPYPI else 'simplified'}")

    env_year = 10000 + job
    print(f"\n  Member {job} -> env_year {env_year}")

    process_member(
        ds_pl,
        ds_sfc,
        job,
        init_date,
        lead_months,
        env_year,
        dst_lats,
        dst_lons,
    )

    if generate_config:
        phase_schedule = build_phase_schedule_from_seas5(
            ds_sfc,
            job,
            init_date,
            lead_months,
            threshold=oni_threshold,
        )
        generate_forecast_config(
            init_date=init_date,
            lead_months=lead_months,
            member_idx=job,
            env_year=env_year,
            active_months=active_months or [6, 7, 8, 9, 10, 11],
            phase_schedule=phase_schedule,
            out_path=f"forecast_configs/config_m{job}.json",
        )

    ds_pl.close()
    ds_sfc.close()
    print(f"\nDone: member {job} processed.")


# =========================================================================
# Main pipeline: download once, process N members
# =========================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare bias-corrected SEAS5 forecast fields for SIENA-IH-STORM"
    )
    parser.add_argument(
        "--init-date", required=True, help="Forecast init date YYYY-MM-DD"
    )
    parser.add_argument("--lead-months", type=int, default=6)
    parser.add_argument(
        "--member",
        type=int,
        nargs="*",
        default=51,
        help="Member indices to process (default: all 51). E.g. --member 0 12 37",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate forecast_config.json per member with ONI-derived phases",
    )
    parser.add_argument(
        "--active-months",
        type=int,
        nargs="+",
        default=[6, 7, 8, 9, 10, 11],
        help="Active season months for the target basin",
    )
    parser.add_argument("--oni-threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip CDS download AND bias correction (use existing corrected files)",
    )
    parser.add_argument(
        "--force-recorrect",
        action="store_true",
        help="Re-run bias correction even if corrected files exist",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of cores used to parallelize"
    )

    args = parser.parse_args()

    start_time = time.time()

    # ---- Step A: Ensure bias-corrected fields are ready on disk ----
    # The bias-correction pipeline:
    #   1. Downloads ERA5 monthly means (1993-2016) — first run only
    #   2. Builds ERA5 monthly climatology — first run only
    #   3. Downloads SEAS5 anomalies for this init — cached per init
    #   4. Applies X_corrected = ERA5_clim + SEAS5_anomaly — cached per init
    #
    # Subsequent runs for the same init are ~free (idempotent skip).
    if not args.skip_download:
        print("=" * 72)
        print("Preparing bias-corrected SEAS5 fields")
        print(f"  init_date: {args.init_date}, lead_months: {args.lead_months}")
        print("=" * 72)
        ds_pl_check, ds_sfc_check = prepare_seas5_bias_corrected(
            init_date=args.init_date,
            lead_months=args.lead_months,
            overwrite=args.force_recorrect,
        )
        ds_pl_check.close()
        ds_sfc_check.close()
    else:
        print("--skip-download: assuming bias-corrected files already exist on disk")

    # ---- Step B: Dispatch per-member processing ----
    # Workers open the already-corrected NetCDFs read-only. No download
    # happens in workers; they only extract, regrid, derive, and save.
    jobs = [m for m in range(*args.member)]
    n_workers = args.workers or min(len(jobs), mp.cpu_count())

    worker_fn = partial(
        _run_single_member,
        init_date=args.init_date,
        lead_months=args.lead_months,
        generate_config=args.generate_config,
        active_months=args.active_months,
        oni_threshold=args.oni_threshold,
    )

    if n_workers == 1:
        # Sequential mode (useful for debugging)
        results = [worker_fn(job) for job in jobs]
    else:
        # imap_unordered yields results as each job finishes, so progress
        # is visible immediately rather than waiting on the slowest worker.
        results = []
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_fn, jobs)):
                results.append(result)
                elapsed_so_far = time.time() - start_time
                print(
                    f"  [{i + 1}/{len(jobs)} done] "
                    f"{os.path.basename(result) if result else '(empty)'} "
                    f"  ({elapsed_so_far / 60:.1f} min elapsed)"
                )
