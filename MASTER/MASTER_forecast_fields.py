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

from CODE.siena_utils import save_yearly_field, _env_yearly_dir, compute_relative_vorticity_spherical

__location__ = os.path.realpath(os.getcwd())


# =========================================================================
# CDS download — single request, all 51 members included
# =========================================================================


def download_seas5(init_date, lead_months, out_dir):
    """
    Download SEAS5 monthly forecast from CDS.
    CDS always returns ALL 51 ensemble members in a single file,
    regardless of any 'member' parameter in the request.

    Returns dict: {"pl": path, "sfc": path}
    """
    if cdsapi is None:
        raise ImportError(
            "cdsapi not installed. Install with: pip install cdsapi\n"
            "Configure ~/.cdsapirc with your CDS credentials."
        )

    os.makedirs(out_dir, exist_ok=True)
    client = cdsapi.Client()

    year, month, _ = init_date.split("-")
    leadtime_months = list(range(1, lead_months + 1))

    pl_file = os.path.join(out_dir, "seas5_pl.nc")
    if not os.path.exists(pl_file):
        print("Downloading SEAS5 pressure-level fields (all 51 members)...")
        client.retrieve(
            "seasonal-monthly-pressure-levels",
            {
                "originating_centre": "ecmwf",
                "system": "51",
                "variable": [
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "temperature",
                    "specific_humidity",
                ],
                "pressure_level": "all",
                "product_type": "monthly_mean",
                "year": year,
                "month": month.lstrip("0"),
                "leadtime_month": leadtime_months,
                "data_format": "netcdf",
            },
            pl_file,
        )
        print(f"  Saved: {pl_file}")
    else:
        print(f"  Skipping download (exists): {pl_file}")

    sfc_file = os.path.join(out_dir, "seas5_sfc.nc")
    if not os.path.exists(sfc_file):
        print("Downloading SEAS5 surface fields (all 51 members)...")
        client.retrieve(
            "seasonal-monthly-single-levels",
            {
                "originating_centre": "ecmwf",
                "system": "51",
                "variable": [
                    "sea_surface_temperature",
                    "mean_sea_level_pressure",
                ],
                "product_type": "monthly_mean",
                "year": year,
                "month": month.lstrip("0"),
                "leadtime_month": leadtime_months,
                "data_format": "netcdf",
            },
            sfc_file,
        )
        print(f"  Saved: {sfc_file}")
    else:
        print(f"  Skipping download (exists): {sfc_file}")

    return {"pl": pl_file, "sfc": sfc_file}


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


def compute_rh_from_q_t(q, t, pressure_pa=60000.0):
    """Relative humidity from specific humidity and temperature."""
    t_c = t - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    w = q / (1.0 - q)
    ws = 0.622 * es / (pressure_pa - es)
    rh = 100.0 * w / ws
    return np.clip(rh, 0, 100)


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
    idx_600 = _nearest_level_idx(p_levels, 600)
    idx_850 = _nearest_level_idx(p_levels, 850)

    n_leads = min(int(pl.sizes["forecastMonth"]), lead_months)

    for t_idx in range(n_leads):
        valid_month = ((init_month - 1) + t_idx + 1) % 12 + 1

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
        t600 = regrid_to_025(
            pl_t["t"].isel(pressure_level=idx_600).values,
            src_lats,
            src_lons,
            dst_lats,
            dst_lons,
        )
        q600 = regrid_to_025(
            pl_t["q"].isel(pressure_level=idx_600).values,
            src_lats,
            src_lons,
            dst_lats,
            dst_lons,
        )

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
        valid_month = ((init_month - 1) + t_idx + 1) % 12 + 1
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
    climate_index_path="climate_index.csv",
    threshold=0.5,
):
    """
    Build a 12-month phase_schedule by combining:
      1. Observed ONI from climate_index.csv (months before init)
      2. SEAS5-derived ONI from this member's SST (forecast months)
      3. Persistence of last forecast phase (beyond SEAS5 horizon)

    Parameters
    ----------
    ds_sfc : xarray.Dataset (all members, already open)
    member_idx : int
    init_date : str
    lead_months : int
    climate_index_path : str
    threshold : float

    Returns
    -------
    dict : {month_int: "LN"|"NEU"|"EN"} for months 1-12
    """
    init_year = int(init_date[:4])
    init_month = int(init_date[5:7])

    # 1. Observed ONI
    from siena_utils import load_climate_index_table

    oni_df = load_climate_index_table(os.path.join(__location__, climate_index_path))
    observed_oni = {}
    for _, row in oni_df.iterrows():
        y, m = int(row["year"]), int(row["month"])
        if y == init_year and m < init_month:
            observed_oni[m] = float(row["climate_index"])
        if y == init_year - 1 and m >= 11:
            observed_oni[m - 12] = float(row["climate_index"])

    # 2. SEAS5-derived Nino 3.4 anomaly
    sst_climatology = load_oni_climatology()
    if sst_climatology is None:
        print("WARNING: Cannot compute SST climatology. Defaulting to NEU.")
        return {m: "NEU" for m in range(1, 13)}

    forecast_oni = compute_oni_from_member(
        ds_sfc,
        member_idx,
        init_date,
        lead_months,
        sst_climatology,
    )

    # 3. Merge
    monthly_anomaly = {}
    for m in range(1, 13):
        if m in observed_oni:
            monthly_anomaly[m] = observed_oni[m]
        elif m in forecast_oni:
            monthly_anomaly[m] = forecast_oni[m]

    # 4. 3-month running mean ONI
    oni_3m = {}
    for m in range(1, 13):
        vals = []
        for offset in [-1, 0, 1]:
            adj = m + offset
            if adj in monthly_anomaly:
                vals.append(monthly_anomaly[adj])
            elif adj <= 0 and adj in observed_oni:
                vals.append(observed_oni[adj])
            elif adj == 13 and 1 in monthly_anomaly:
                vals.append(monthly_anomaly[1])
        if vals:
            oni_3m[m] = float(np.nanmean(vals))

    # 5. Classify + persistence
    def _classify(v):
        return "EN" if v >= threshold else ("LN" if v <= -threshold else "NEU")

    phase_schedule = {}
    last_phase = "NEU"
    for m in range(1, 13):
        if m in oni_3m and np.isfinite(oni_3m[m]):
            phase_schedule[m] = _classify(oni_3m[m])
            last_phase = phase_schedule[m]
        else:
            phase_schedule[m] = last_phase

    print(f"  Phase schedule (member {member_idx}):")
    for m in range(1, 13):
        src = (
            "obs" if m in observed_oni else "seas5" if m in forecast_oni else "persist"
        )
        oni_val = oni_3m.get(m, np.nan)
        print(
            f"    month {m:2d}: {phase_schedule[m]:3s}  "
            f"(ONI={oni_val:+.2f}, source={src})"
        )

    return phase_schedule


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

    forecast_months = [((init_month - 1) + i + 1) % 12 + 1 for i in range(lead_months)]

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
    Full pipeline: download once -> loop for one member.

    Parameters
    ----------
    job : member of SEAS5 ensemble to process over
    init_date : str, "YYYY-MM-DD"
    lead_months : int
    generate_config : bool
    active_months : list of int
    oni_threshold : float
    skip_download : bool
    """

    raw_dir = os.path.join(__location__, "forecast_raw")

    files = {
        "pl": os.path.join(raw_dir, "seas5_pl.nc"),
        "sfc": os.path.join(raw_dir, "seas5_sfc.nc"),
    }
    for k, p in files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f" {p} not found. Please check.")

    # Open datasets (shared across all members, read-only)
    ds_pl = xr.open_dataset(files["pl"])
    ds_sfc = xr.open_dataset(files["sfc"])

    # Target grid
    try:
        ds_ref = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
        dst_lats = ds_ref.latitude.values
        dst_lons = ds_ref.longitude.values
        ds_ref.close()
    except Exception:
        dst_lats = np.arange(90, -90.25, -0.25)
        dst_lons = np.arange(0, 360, 0.25)

    print(f"Processing {job} member, {lead_months} lead months")
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
    print(f"\nDone: {job} member processed.")


# =========================================================================
# Main pipeline: download once, process N members
# =========================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare SEAS5 forecast fields for SIENA-IH-STORM"
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
        help="Skip CDS download (use existing files)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of cores used to parallelize"
    )

    args = parser.parse_args()

    start_time = time.time()

    raw_dir = os.path.join(__location__, "forecast_raw")
    if not args.skip_download:
        files = download_seas5(args.init_date, args.lead_months, raw_dir)
    else:
        files = {
            "pl": os.path.join(raw_dir, "seas5_pl.nc"),
            "sfc": os.path.join(raw_dir, "seas5_sfc.nc"),
        }
        for k, p in files.items():
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"--skip-download but {p} not found. "
                    f"Run without --skip-download first."
                )

    # Build job list: all (members) of SEAS5
    jobs = [m for m in range(args.member)]
    n_workers = args.workers or min(len(jobs), mp.cpu_count())

    # ---- Run in parallel ----

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
        # imap_unordered yields results as each job finishes, so you see
        # progress immediately instead of waiting for ALL jobs to complete.
        # pool.map blocked on the slowest worker — if one storm entered an
        # infinite loop, no results were returned and no output was visible.
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
