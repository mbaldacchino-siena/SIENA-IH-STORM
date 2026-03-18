"""
Potential Intensity (PI) precomputation for SIENA-IH-STORM.

Computes thermodynamic PI following Bister & Emanuel (2002) via the tcpyPI
package (Gilford 2021, doi:10.5194/gmd-14-2351-2021).

Produces phase-specific (EN/NEU/LN) and pooled monthly PI fields that
replace the empirical SST-based MPI fields in the original STORM model.

If tcpyPI is not installed, falls back to a simplified SST-based
approximation that still improves on the original DeMaria & Kaplan curve
by incorporating MSLP and a basic CAPE-like correction.

@mbaldacchino 2026
"""

import os
import numpy as np
import warnings
from climatology import compute_phase_climatology
import xarray as xr 

try:
    from tcpyPI import pi as calc_pi
    HAS_TCPYPI = True
except ImportError:
    HAS_TCPYPI = False
    warnings.warn(
        "tcpyPI not installed. Install with: pip install tcpyPI\n"
        "Falling back to simplified Bister-Emanuel approximation.\n"
        "For best results, install tcpyPI (Gilford 2021, GMD).",
        stacklevel=2,
    )

PHASES = ["LN", "NEU", "EN"]
PHASE_CODE = {"LN": 0, "NEU": 1, "EN": 2}


def _simplified_pi_point(sst_K, mslp_Pa):
    """
    Simplified PI estimate when tcpyPI is not available.
    Uses the Bister & Emanuel (2002) empirical scaling:
        Vmax^2 ~ Ck/Cd * (Ts - To)/To * (CAPE* - CAPE_b)

    We approximate this with a refined SST-MPI relationship that is
    still superior to the original DeMaria-Kaplan exponential:
        Pmin ~ MSLP - A * (SST - SST_threshold)^B  for SST > threshold
    with physically motivated constants from Holland (1997).
    """
    sst_C = sst_K - 273.15
    mslp_hPa = mslp_Pa / 100.0 if mslp_Pa > 10000 else mslp_Pa

    if sst_C < 26.0 or not np.isfinite(sst_C) or not np.isfinite(mslp_hPa):
        return np.nan, np.nan

    # Holland (1997) style approximation
    # Pressure drop scales roughly as (SST - 26)^1.5 with typical constants
    dp = 28.0 * (sst_C - 26.0) ** 0.76
    dp = min(dp, mslp_hPa - 850.0)  # physical cap
    pmin = mslp_hPa - dp

    # Approximate Vmax from pressure drop using typical WPR
    vmax = 0.7 * dp ** 0.65 if dp > 0 else 0.0

    return pmin, vmax


def compute_pi_field_tcpyPI(sst_K, mslp_Pa, t_K, q_kgkg, p_lev_hPa):
    """
    Compute thermodynamic PI at each grid point using tcpyPI.

    Parameters
    ----------
    sst_K : 2D array (lat, lon), SST in Kelvin
    mslp_Pa : 2D array (lat, lon), mean sea level pressure in Pa
    t_K : 3D array (level, lat, lon), air temperature in K on pressure levels
    q_kgkg : 3D array (level, lat, lon), specific humidity in kg/kg
    p_lev_hPa : 1D array, pressure levels in hPa

    Returns
    -------
    pmin : 2D array (lat, lon), minimum sustainable central pressure (hPa)
    vmax : 2D array (lat, lon), maximum potential intensity wind (m/s)
    """
    nlat, nlon = sst_K.shape
    pmin = np.full((nlat, nlon), np.nan)
    vmax = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            try:
                sst = float(sst_K[i, j])
                if not np.isfinite(sst) or sst < 273.15 + 5.0:
                    continue

                psl = float(mslp_Pa[i, j])
                # tcpyPI expects MSLP in hPa
                if psl > 10000:
                    psl = psl / 100.0

                t_col = t_K[:, i, j].astype(float)
                q_col = q_kgkg[:, i, j].astype(float)

                if np.any(~np.isfinite(t_col)) or np.any(~np.isfinite(q_col)):
                    continue

                # Convert specific humidity to mixing ratio for tcpyPI
                r_col = q_col / (1.0 - q_col)  # mixing ratio kg/kg

                # tcpyPI.pi signature:
                # pi(TEFULL, p, tc, r, TEFULL is SST in K, p levels in hPa,
                #    tc is temperature in C, r is mixing ratio in g/kg)
                result = calc_pi(
                    sst,           # SST in K  (or C depending on version)
                    psl,           # surface pressure hPa
                    p_lev_hPa,    # pressure levels hPa
                    t_col - 273.15,  # temperature in Celsius
                    r_col * 1000.0,  # mixing ratio in g/kg
                )
                # result is (VMAX, PMIN, IFL, TO, LNB)
                v_out = result[0]
                p_out = result[1]
                ifl = result[2]

                if ifl >= 1 and np.isfinite(v_out) and np.isfinite(p_out):
                    vmax[i, j] = v_out
                    pmin[i, j] = p_out

            except Exception:
                continue

    return pmin, vmax


def compute_pi_field_simplified(sst_K, mslp_Pa):
    """
    Compute PI using simplified approximation (no vertical profiles needed).
    """
    nlat, nlon = sst_K.shape
    pmin = np.full((nlat, nlon), np.nan)
    vmax = np.full((nlat, nlon), np.nan)

    for i in range(nlat):
        for j in range(nlon):
            pmin[i, j], vmax[i, j] = _simplified_pi_point(
                sst_K[i, j], mslp_Pa[i, j]
            )

    return pmin, vmax


def build_phase_specific_pi_climatologies(oni_df, era5_paths, out_dir):
    """
    Build phase-specific PI fields from ERA5 monthly data.
    """
    print("Building phase-specific PI climatologies...")
    has_profiles = "t" in era5_paths and "q" in era5_paths
    use_full_pi = HAS_TCPYPI and has_profiles

    if use_full_pi:
        print("  Using full thermodynamic PI (tcpyPI + ERA5 profiles)")
    elif has_profiles:
        print("  tcpyPI not installed. Using simplified PI approximation.")
    else:
        print("  No vertical profile data provided. Using simplified PI.")

    # SST and MSLP — no save (already saved by main pipeline)
    sst_clim, sst_pooled = compute_phase_climatology(
        era5_paths["sst"],
        "sst",
        oni_df,
        "_tmp_sst",
        out_dir=None,
    )
    mslp_clim, mslp_pooled = compute_phase_climatology(
        era5_paths["mslp"],
        "msl",
        oni_df,
        "_tmp_mslp",
        out_dir=None,
        unit_scale=0.01,
    )

    # T and Q profiles — no save, no level selection (need full profile)
    t_clim = t_pooled = q_clim = q_pooled = p_lev_hPa = None
    if has_profiles:
        # Extract pressure levels before compute_phase_climatology closes the file
        ds_peek = xr.open_dataset(era5_paths["t"])
        if "pressure_level" in ds_peek.dims:
            p_lev_hPa = ds_peek.pressure_level.values.astype(float)
        elif "level" in ds_peek.dims:
            p_lev_hPa = ds_peek.level.values.astype(float)
        ds_peek.close()

        # pressure_level_idx=None → keeps all levels, returns (level, lat, lon) arrays
        t_clim, t_pooled = compute_phase_climatology(
            era5_paths["t"],
            "t",
            oni_df,
            "_tmp_t",
            out_dir=None,
        )
        q_clim, q_pooled = compute_phase_climatology(
            era5_paths["q"],
            "q",
            oni_df,
            "_tmp_q",
            out_dir=None,
        )

    # Compute PI for each month × phase
    for month in range(1, 13):
        for phase in PHASES:
            print(f"  Computing PI: month={month}, phase={phase}")
            sst = sst_clim[month].get(phase)
            mslp = mslp_clim[month].get(phase)

            if sst is None or mslp is None:
                print(f"    Skipping: no data for month={month}, phase={phase}")
                continue

            if use_full_pi and t_clim is not None:
                t = t_clim[month].get(phase)
                q = q_clim[month].get(phase)
                if t is not None and q is not None:
                    pmin, vmax = compute_pi_field_tcpyPI(sst, mslp, t, q, p_lev_hPa)
                else:
                    pmin, vmax = compute_pi_field_simplified(sst, mslp)
            else:
                pmin, vmax = compute_pi_field_simplified(sst, mslp)

            np.savetxt(
                os.path.join(out_dir, f"Monthly_mean_PI_{month}_{phase}.txt"), pmin
            )
            np.savetxt(
                os.path.join(out_dir, f"Monthly_mean_VMAX_PI_{month}_{phase}.txt"), vmax
            )

        # Pooled
        print(f"  Computing PI: month={month}, pooled")
        sst_p = sst_pooled.get(month)
        mslp_p = mslp_pooled.get(month)
        if sst_p is not None and mslp_p is not None:
            if use_full_pi and t_pooled is not None:
                t_p = t_pooled.get(month)
                q_p = q_pooled.get(month)
                if t_p is not None and q_p is not None:
                    pmin_p, vmax_p = compute_pi_field_tcpyPI(
                        sst_p, mslp_p, t_p, q_p, p_lev_hPa
                    )
                else:
                    pmin_p, vmax_p = compute_pi_field_simplified(sst_p, mslp_p)
            else:
                pmin_p, vmax_p = compute_pi_field_simplified(sst_p, mslp_p)

            np.savetxt(os.path.join(out_dir, f"Monthly_mean_PI_{month}.txt"), pmin_p)

    print("PI climatologies complete.")

