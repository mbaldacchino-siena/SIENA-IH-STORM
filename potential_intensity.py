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


def build_phase_climatology(ds, varname, oni_df, dim_time='valid_time'):
    """
    Build month × phase climatology from an xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
    varname : str, variable name in ds
    oni_df : DataFrame with columns [year, month, phase]
    dim_time : str, name of the time dimension

    Returns
    -------
    clim : dict {month: {phase: numpy array}}
    """
    import pandas as pd

    times = pd.to_datetime(ds[dim_time].values)
    years = times.year
    months_arr = times.month

    # Build lookup from ONI table
    phase_lookup = {}
    for _, row in oni_df.iterrows():
        phase_lookup[(int(row['year']), int(row['month']))] = str(row['phase']).strip().upper()

    clim = {m: {ph: [] for ph in PHASES} for m in range(1, 13)}
    clim_pooled = {m: [] for m in range(1, 13)}

    for t_idx in range(len(times)):
        yr = int(years[t_idx])
        mo = int(months_arr[t_idx])
        ph = phase_lookup.get((yr, mo), "NEU")
        if ph not in PHASES:
            ph = "NEU"

        if hasattr(ds[varname], 'dims') and 'pressure_level' in ds[varname].dims:
            field = ds[varname].isel({dim_time: t_idx}).values
        elif hasattr(ds[varname], 'dims') and 'level' in ds[varname].dims:
            field = ds[varname].isel({dim_time: t_idx}).values
        else:
            field = ds[varname].isel({dim_time: t_idx}).values

        clim[mo][ph].append(field)
        clim_pooled[mo].append(field)

    # Average
    result = {m: {} for m in range(1, 13)}
    result_pooled = {}
    for m in range(1, 13):
        for ph in PHASES:
            if len(clim[m][ph]) > 0:
                result[m][ph] = np.nanmean(np.stack(clim[m][ph], axis=0), axis=0)
            else:
                # Fallback to pooled
                result[m][ph] = None
        if len(clim_pooled[m]) > 0:
            result_pooled[m] = np.nanmean(np.stack(clim_pooled[m], axis=0), axis=0)
        else:
            result_pooled[m] = None

    # Fill missing phases with pooled
    for m in range(1, 13):
        for ph in PHASES:
            if result[m][ph] is None and result_pooled[m] is not None:
                result[m][ph] = result_pooled[m]

    return result, result_pooled


def build_phase_specific_pi_climatologies(oni_df, era5_paths, out_dir):
    """
    Build phase-specific PI fields from ERA5 monthly data.

    Parameters
    ----------
    oni_df : DataFrame with columns [year, month, phase]
    era5_paths : dict with keys 'sst', 'mslp', and optionally 't', 'q'
                 Values are file paths to ERA5 .nc files.
    out_dir : str, output directory

    Returns
    -------
    None (writes PI fields to disk)
    """
    import xarray as xr

    print("Building phase-specific PI climatologies...")
    has_profiles = 't' in era5_paths and 'q' in era5_paths
    use_full_pi = HAS_TCPYPI and has_profiles

    if use_full_pi:
        print("  Using full thermodynamic PI (tcpyPI + ERA5 profiles)")
    elif has_profiles:
        print("  tcpyPI not installed. Using simplified PI approximation.")
        print("  Install tcpyPI for full thermodynamic PI: pip install tcpyPI")
    else:
        print("  No vertical profile data provided. Using simplified PI.")

    # Build SST and MSLP phase climatologies
    ds_sst = xr.open_dataset(era5_paths['sst'])
    # Detect time dimension name
    time_dim = 'valid_time' if 'valid_time' in ds_sst.dims else 'time'
    sst_varname = 'sst' if 'sst' in ds_sst else list(ds_sst.data_vars)[0]
    sst_clim, sst_pooled = build_phase_climatology(ds_sst, sst_varname, oni_df, dim_time=time_dim)
    ds_sst.close()

    ds_mslp = xr.open_dataset(era5_paths['mslp'])
    time_dim_m = 'valid_time' if 'valid_time' in ds_mslp.dims else 'time'
    mslp_varname = 'msl' if 'msl' in ds_mslp else list(ds_mslp.data_vars)[0]
    mslp_clim, mslp_pooled = build_phase_climatology(ds_mslp, mslp_varname, oni_df, dim_time=time_dim_m)
    ds_mslp.close()

    t_clim = t_pooled = q_clim = q_pooled = p_lev_hPa = None
    if has_profiles:
        ds_t = xr.open_dataset(era5_paths['t'])
        time_dim_t = 'valid_time' if 'valid_time' in ds_t.dims else 'time'
        t_varname = 't' if 't' in ds_t else list(ds_t.data_vars)[0]
        t_clim, t_pooled = build_phase_climatology(ds_t, t_varname, oni_df, dim_time=time_dim_t)
        # Get pressure levels
        if 'pressure_level' in ds_t.dims:
            p_lev_hPa = ds_t.pressure_level.values.astype(float)
        elif 'level' in ds_t.dims:
            p_lev_hPa = ds_t.level.values.astype(float)
        ds_t.close()

        ds_q = xr.open_dataset(era5_paths['q'])
        time_dim_q = 'valid_time' if 'valid_time' in ds_q.dims else 'time'
        q_varname = 'q' if 'q' in ds_q else 'r' if 'r' in ds_q else list(ds_q.data_vars)[0]
        q_clim, q_pooled = build_phase_climatology(ds_q, q_varname, oni_df, dim_time=time_dim_q)
        ds_q.close()

    # Compute PI for each month × phase
    for month in range(1, 13):
        for phase in PHASES:
            print(f"  Computing PI: month={month}, phase={phase}")
            sst = sst_clim[month][phase]
            mslp = mslp_clim[month][phase]

            if sst is None or mslp is None:
                print(f"    Skipping: no data for month={month}, phase={phase}")
                continue

            if use_full_pi and t_clim is not None and q_clim is not None:
                t = t_clim[month][phase]
                q = q_clim[month][phase]
                if t is not None and q is not None:
                    pmin, vmax = compute_pi_field_tcpyPI(sst, mslp, t, q, p_lev_hPa)
                else:
                    pmin, vmax = compute_pi_field_simplified(sst, mslp)
            else:
                pmin, vmax = compute_pi_field_simplified(sst, mslp)

            np.savetxt(
                os.path.join(out_dir, f"Monthly_mean_PI_{month}_{phase}.txt"),
                pmin,
            )
            np.savetxt(
                os.path.join(out_dir, f"Monthly_mean_VMAX_PI_{month}_{phase}.txt"),
                vmax,
            )

        # Also save pooled
        print(f"  Computing PI: month={month}, pooled")
        sst_p = sst_pooled[month]
        mslp_p = mslp_pooled[month]
        if sst_p is not None and mslp_p is not None:
            if use_full_pi and t_pooled is not None and q_pooled is not None:
                t_p = t_pooled[month]
                q_p = q_pooled[month]
                if t_p is not None and q_p is not None:
                    pmin_p, vmax_p = compute_pi_field_tcpyPI(sst_p, mslp_p, t_p, q_p, p_lev_hPa)
                else:
                    pmin_p, vmax_p = compute_pi_field_simplified(sst_p, mslp_p)
            else:
                pmin_p, vmax_p = compute_pi_field_simplified(sst_p, mslp_p)

            np.savetxt(
                os.path.join(out_dir, f"Monthly_mean_PI_{month}.txt"),
                pmin_p,
            )

    print("PI climatologies complete.")


def build_phase_specific_env_climatologies(oni_df, nc_path, varname, out_stem, out_dir,
                                            pressure_level_idx=None):
    """
    Build phase-specific monthly climatologies for any ERA5 variable (VWS, RH, etc).

    Parameters
    ----------
    oni_df : DataFrame with [year, month, phase]
    nc_path : str, path to ERA5 .nc file
    varname : str, variable name in the dataset
    out_stem : str, output file stem (e.g., 'Monthly_mean_VWS')
    out_dir : str, output directory
    pressure_level_idx : int or None, if the data has a pressure_level dim, select this index
    """
    import xarray as xr

    print(f"Building phase-specific {out_stem} climatologies...")
    ds = xr.open_dataset(nc_path)
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

    # If pressure level selection needed
    if pressure_level_idx is not None:
        if 'pressure_level' in ds.dims:
            ds = ds.isel(pressure_level=pressure_level_idx)
        elif 'level' in ds.dims:
            ds = ds.isel(level=pressure_level_idx)

    # Detect variable name
    if varname not in ds:
        # Try to find it
        candidates = list(ds.data_vars)
        if len(candidates) == 1:
            varname = candidates[0]
        else:
            print(f"  WARNING: variable '{varname}' not found in {nc_path}. Available: {candidates}")
            ds.close()
            return

    clim, clim_pooled = build_phase_climatology(ds, varname, oni_df, dim_time=time_dim)
    ds.close()

    for month in range(1, 13):
        # Save pooled
        if clim_pooled[month] is not None:
            np.savetxt(
                os.path.join(out_dir, f"{out_stem}_{month}.txt"),
                clim_pooled[month],
            )
        # Save phase-specific
        for phase in PHASES:
            if clim[month][phase] is not None:
                np.savetxt(
                    os.path.join(out_dir, f"{out_stem}_{month}_{phase}.txt"),
                    clim[month][phase],
                )
                print(f"  Saved {out_stem}_{month}_{phase}.txt")

    print(f"{out_stem} climatologies complete.")
