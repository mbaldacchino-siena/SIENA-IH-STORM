"""
Download and prepare CDS SEAS5 seasonal forecast fields for SIENA-IH-STORM.

Workflow:
  1. Download SEAS5 monthly-mean fields for one ensemble member
     (U200, U850, V200, V850, T600, Q600, SST, MSLP)
  2. Regrid to 0.25° (bilinear) to match ERA5-derived model fields
  3. Compute derived variables:
       VWS  = sqrt((u200-u850)^2 + (v200-v850)^2)
       RH600 from Q600 + T600 via Clausius-Clapeyron
  4. Compute PI from SST, MSLP, T-profile, Q-profile using tcpyPI
  5. Save as env_yearly/{STEM}_{env_year}_{month}.npy

Usage:
  python MASTER_forecast_fields.py \
      --init-date 2026-04-01 \
      --lead-months 6 \
      --member 1 \
      --env-year 9999

  # Loop over all 51 members:
  for m in $(seq 0 50); do
      python MASTER_forecast_fields.py \
          --init-date 2026-04-01 --lead-months 6 \
          --member $m --env-year $((10000 + m))
  done
"""

import argparse
import os
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

# Conditional imports — fail gracefully if not installed
try:
    import cdsapi
except ImportError:
    cdsapi = None

try:
    from CODE.potential_intensity import compute_pi_field

    HAS_PI = True
except ImportError:
    HAS_PI = False

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

from CODE.siena_utils import save_yearly_field, _env_yearly_dir

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# =========================================================================
# CDS download
# =========================================================================


def download_seas5(init_date, lead_months, member, out_dir):
    """
    Download SEAS5 monthly forecast from CDS.

    Parameters
    ----------
    init_date : str, e.g. "2026-04-01"
    lead_months : int, number of months to download
    member : int, ensemble member index (0-50)
    out_dir : str, directory for raw downloads

    Returns
    -------
    dict : {variable_name: path_to_downloaded_nc}
    """
    if cdsapi is None:
        raise ImportError(
            "cdsapi not installed. Install with: pip install cdsapi\n"
            "Configure ~/.cdsapirc with your CDS credentials."
        )

    os.makedirs(out_dir, exist_ok=True)
    client = cdsapi.Client()

    year, month, _ = init_date.split("-")
    leadtime_hours = [str(h) for h in range(0, lead_months * 730, 730)][:lead_months]
    # CDS uses leadtime_hour for monthly means; compute month offsets
    leadtime_months = list(range(1, lead_months + 1))

    # --- Pressure-level variables ---
    # Download the FULL profile (all standard levels) for T and Q so
    # that tcpyPI can compute thermodynamic PI from the complete
    # atmospheric column.  U/V only needed at 200+850 hPa for VWS,
    # but CDS returns all requested levels in one file, so we request
    # all standard levels and subset at processing time.
    #
    # SEAS5 standard pressure levels (hPa):
    SEAS5_LEVELS = [
        "1",
        "2",
        "3",
        "5",
        "7",
        "10",
        "20",
        "30",
        "50",
        "70",
        "100",
        "125",
        "150",
        "175",
        "200",
        "225",
        "250",
        "300",
        "350",
        "400",
        "450",
        "500",
        "550",
        "600",
        "650",
        "700",
        "750",
        "775",
        "800",
        "825",
        "850",
        "875",
        "900",
        "925",
        "950",
        "975",
        "1000",
    ]

    pl_file = os.path.join(out_dir, f"seas5_pl_m{member}.nc")
    if not os.path.exists(pl_file):
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
                "pressure_level": SEAS5_LEVELS,
                "product_type": "monthly_mean",
                "year": year,
                "month": month.lstrip("0"),
                "leadtime_month": leadtime_months,
                "data_format": "netcdf",
                "member": str(member),
            },
            pl_file,
        )
    else:
        print(f"  Skipping download (exists): {pl_file}")

    # --- Single-level variables (SST, MSLP) ---
    sfc_file = os.path.join(out_dir, f"seas5_sfc_m{member}.nc")
    if not os.path.exists(sfc_file):
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
                "member": str(member),
            },
            sfc_file,
        )
    else:
        print(f"  Skipping download (exists): {sfc_file}")

    return {"pl": pl_file, "sfc": sfc_file}


# =========================================================================
# Regridding
# =========================================================================


def regrid_to_025(field, src_lats, src_lons, dst_lats=None, dst_lons=None):
    """
    Bilinear regrid from source grid to 0.25° global grid.

    Parameters
    ----------
    field : 2D array (lat, lon)
    src_lats, src_lons : 1D arrays, source coordinates
    dst_lats, dst_lons : 1D arrays, target coordinates (default: 0.25° global)

    Returns
    -------
    2D array on the target grid
    """
    if dst_lats is None:
        dst_lats = np.arange(90, -90.25, -0.25)
    if dst_lons is None:
        dst_lons = np.arange(0, 360, 0.25)

    # Ensure monotonically increasing for RegularGridInterpolator
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
    """
    Relative humidity from specific humidity and temperature at a given
    pressure level.

    Parameters
    ----------
    q : array, specific humidity (kg/kg)
    t : array, temperature (K)
    pressure_pa : float, pressure level in Pa (default 600 hPa)

    Returns
    -------
    array, relative humidity in %
    """
    # Saturation vapour pressure via Tetens (WMO)
    t_c = t - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))  # Pa
    # Mixing ratio from specific humidity
    w = q / (1.0 - q)
    ws = 0.622 * es / (pressure_pa - es)
    rh = 100.0 * w / ws
    return np.clip(rh, 0, 100)


# =========================================================================
# Main pipeline
# =========================================================================


def process_forecast(init_date, lead_months, member, env_year):
    """
    Full pipeline: download → regrid → derive → save.

    Parameters
    ----------
    init_date : str, "YYYY-MM-DD"
    lead_months : int
    member : int, ensemble member (0-50)
    env_year : int, synthetic year label for storage (e.g. 9999, or 10000+member)
    """
    raw_dir = os.path.join(__location__, "forecast_raw", f"member_{member}")
    files = download_seas5(init_date, lead_months, member, raw_dir)

    # ── Load pressure-level data ──
    ds_pl = xr.open_dataset(files["pl"])
    ds_sfc = xr.open_dataset(files["sfc"])

    # Identify coordinate names (CDS varies between datasets)
    lat_name = "latitude" if "latitude" in ds_pl.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_pl.dims else "lon"
    time_name = "forecast_reference_time"
    for candidate in ["forecastMonth", "time", "valid_time", "step"]:
        if candidate in ds_pl.dims:
            time_name = candidate
            break

    plev_name = "pressure_level" if "pressure_level" in ds_pl.dims else "level"
    p_lev_all = ds_pl[plev_name].values.astype(float)  # all levels in hPa

    src_lats = ds_pl[lat_name].values
    src_lons = ds_pl[lon_name].values

    # SEAS5 native grid (~1°) — this IS the PI computation grid.
    # VWS/RH/MSLP/SST get regridded to 0.25°; PI is computed at native
    # resolution and then upscaled, matching the ERA5 pipeline.
    native_shape = (len(src_lats), len(src_lons))

    # Target grid: load from an existing ERA5-derived field to ensure consistency
    try:
        ds_ref = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
        dst_lats = ds_ref.latitude.values
        dst_lons = ds_ref.longitude.values
        ds_ref.close()
    except Exception:
        dst_lats = np.arange(90, -90.25, -0.25)
        dst_lons = np.arange(0, 360, 0.25)

    fine_shape = (len(dst_lats), len(dst_lons))

    # Parse init date to determine forecast valid months
    init_year, init_month = int(init_date[:4]), int(init_date[5:7])

    # Check if tcpyPI is available
    try:
        from CODE.potential_intensity import (
            HAS_TCPYPI,
            compute_pi_field_tcpyPI,
            compute_pi_field_simplified,
            _nanfill_nearest,
            _upscale_to_target,
        )
    except ImportError:
        HAS_TCPYPI = False

    print(f"Processing member {member}, env_year={env_year}")
    if HAS_TCPYPI:
        print(
            f"  PI: full thermodynamic (tcpyPI) at native {native_shape}, "
            f"upscaling to {fine_shape}"
        )
    else:
        print(f"  PI: simplified SST-based (install tcpyPI for full PI)")

    n_times = ds_pl.sizes.get(time_name, lead_months)
    for t_idx in range(min(n_times, lead_months)):
        # Valid month for this lead time
        valid_month = ((init_month - 1) + t_idx + 1) % 12 + 1

        def _sel(ds, var, plev=None):
            """Select time step and optionally pressure level."""
            arr = ds[var].isel({time_name: t_idx})
            if plev is not None:
                # Find nearest level
                levels = arr[plev_name].values
                li = int(np.abs(levels - plev).argmin())
                arr = arr.isel({plev_name: li})
            return arr.values

        def _sel_profile(ds, var):
            """Select time step, keep all pressure levels → (level, lat, lon)."""
            return ds[var].isel({time_name: t_idx}).values

        # ── Extract and regrid VWS components (only 200+850 needed) ──
        u200 = regrid_to_025(
            _sel(ds_pl, "u", 200), src_lats, src_lons, dst_lats, dst_lons
        )
        u850 = regrid_to_025(
            _sel(ds_pl, "u", 850), src_lats, src_lons, dst_lats, dst_lons
        )
        v200 = regrid_to_025(
            _sel(ds_pl, "v", 200), src_lats, src_lons, dst_lats, dst_lons
        )
        v850 = regrid_to_025(
            _sel(ds_pl, "v", 850), src_lats, src_lons, dst_lats, dst_lons
        )

        # ── Extract 600 hPa T/Q for RH computation (regrid to 0.25°) ──
        t600 = regrid_to_025(
            _sel(ds_pl, "t", 600), src_lats, src_lons, dst_lats, dst_lons
        )
        q600 = regrid_to_025(
            _sel(ds_pl, "q", 600), src_lats, src_lons, dst_lats, dst_lons
        )

        # ── Surface fields (regrid to 0.25°) ──
        sfc_lats = ds_sfc[lat_name].values
        sfc_lons = ds_sfc[lon_name].values
        sst = regrid_to_025(
            _sel(ds_sfc, "sst", None), sfc_lats, sfc_lons, dst_lats, dst_lons
        )
        mslp = regrid_to_025(
            _sel(ds_sfc, "msl", None), sfc_lats, sfc_lons, dst_lats, dst_lons
        )
        # MSLP: Pa → hPa
        mslp *= 0.01

        # ── Derived fields ──
        vws = compute_vws(u200, u850, v200, v850)
        rh600 = compute_rh_from_q_t(q600, t600, pressure_pa=60000.0)

        # ── Thermodynamic PI from full T/Q profile ──
        # Compute at SEAS5 native resolution (~1°), then upscale to 0.25°.
        # This mirrors the ERA5 pipeline in climatology.py exactly.
        if HAS_TCPYPI:
            # Full profiles at native resolution: (levels, lat, lon)
            t_profile = _sel_profile(ds_pl, "t")
            q_profile = _sel_profile(ds_pl, "q")

            # SST and MSLP at native resolution (no regrid) for PI computation
            sst_native = _sel(ds_sfc, "sst", None)
            mslp_native = _sel(ds_sfc, "msl", None)
            # Handle grid mismatch: coarsen sfc to match pl if needed
            if sst_native.shape != native_shape:
                from CODE.potential_intensity import _coarsen_to_match

                sst_native = _coarsen_to_match(sst_native, native_shape)
                mslp_native = _coarsen_to_match(mslp_native, native_shape)

            # MSLP to hPa for PI computation
            mslp_native_hPa = mslp_native * 0.01

            pmin, vmax = compute_pi_field_tcpyPI(
                sst_native, mslp_native_hPa, t_profile, q_profile, p_lev_all
            )

            # NaN-fill coastal cells before upscaling
            pmin = _nanfill_nearest(pmin)

            # Upscale from native (~1°) to 0.25°
            if pmin.shape != fine_shape:
                pmin = _upscale_to_target(pmin, fine_shape)
        else:
            # Simplified fallback (already at 0.25° from regridded SST/MSLP)
            pmin, vmax = compute_pi_field_simplified(sst, mslp)

        # ── Save all fields ──
        save_yearly_field(__location__, "VWS", env_year, valid_month, vws)
        save_yearly_field(__location__, "RH600", env_year, valid_month, rh600)
        save_yearly_field(__location__, "MSLP", env_year, valid_month, mslp)
        save_yearly_field(__location__, "SST", env_year, valid_month, sst)
        save_yearly_field(__location__, "PI", env_year, valid_month, pmin)

        print(f"  Saved month {valid_month} (lead {t_idx + 1})")

    ds_pl.close()
    ds_sfc.close()
    print(f"Done: member {member} → env_year {env_year}")


def generate_forecast_config(
    init_date,
    lead_months,
    member,
    env_year,
    active_months,
    phase_schedule,
    observed_months=None,
    out_path="forecast_config.json",
):
    """
    Generate a forecast_config.json from parameters.

    Parameters
    ----------
    init_date : str, "YYYY-MM-DD"
    lead_months : int
    member : int
    env_year : int, synthetic year label for forecast fields
    active_months : list of int, e.g. [6,7,8,9,10,11]
    phase_schedule : dict {month_int: "LN"|"NEU"|"EN"} for ALL 12 months
    observed_months : list of int or None, months with real ERA5 data
        If None, inferred from init_date (all months before init month).
    out_path : str
    """
    init_year = int(init_date[:4])
    init_month = int(init_date[5:7])

    if observed_months is None:
        observed_months = list(range(1, init_month))

    forecast_months = []
    for i in range(lead_months):
        fm = ((init_month - 1) + i + 1) % 12 + 1
        forecast_months.append(fm)

    config = {
        "mode": "seasonal_forecast",
        "base_year": init_year,
        "ensemble_member": member,
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

    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote forecast config: {out_path}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare SEAS5 forecast fields for SIENA-IH-STORM"
    )
    parser.add_argument(
        "--init-date", required=True, help="Forecast init date YYYY-MM-DD"
    )
    parser.add_argument("--lead-months", type=int, default=6)
    parser.add_argument("--member", type=int, default=0, help="Ensemble member (0-50)")
    parser.add_argument(
        "--env-year",
        type=int,
        default=9999,
        help="Synthetic year label for storage (use 10000+member for multi-member runs)",
    )
    args = parser.parse_args()
    process_forecast(args.init_date, args.lead_months, args.member, args.env_year)
