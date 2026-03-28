"""
SIENA-IH-STORM Diagnostic Script
=================================
Run AFTER MASTER_preprocessing.py has completed.

Measures three specific concerns:
  §4  Exponential term sign convention — are c > 0 and d ≥ 0 everywhere?
  §5  MPI consistency — how much does the runtime PI field differ from the MPI
      used during coefficient fitting?
  §6  VWS/RH centering — how unbalanced is the effective regularization?

Usage:
    python DIAGNOSTIC_pressure_mpi_vws.py

Outputs:
    diagnostic_pressure_coefficients.csv   — per-cell c, d, mu, and flags
    diagnostic_mpi_consistency.csv         — per-cell (fitted MPI) vs (runtime PI)
    diagnostic_vws_rh_centering.csv        — predictor statistics and effective penalty
    diagnostic_summary.txt                 — human-readable summary
"""

import numpy as np
import pandas as pd
import os
import sys

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# ── Basin metadata ──────────────────────────────────────────────────────
BASIN_NAMES = ["EP", "NA", "NI", "SI", "SP", "WP"]
MONTHS_DEFAULT = [
    [6, 7, 8, 9, 10, 11],
    [6, 7, 8, 9, 10, 11],
    [10, 11],
    [1, 2, 3, 4, 11, 12],
    [1, 2, 3, 4, 11, 12],
    [5, 6, 7, 8, 9, 10, 11],
]


def load_input_months():
    """Try to read months from input.dat, fall back to defaults."""
    try:
        import import_data

        _, _, _, _, months, *_ = import_data.input_data("input.dat")
        return months
    except Exception:
        return MONTHS_DEFAULT


# =========================================================================
# §4  EXPONENTIAL TERM: Verify c > 0 and d ≥ 0 across all cells
# =========================================================================
def diagnose_exponential_term():
    """
    The James-Mason pressure model is:
        dp1_hat = a + b*dp0 + c*exp(-d*(P - MPI)+) + env_terms

    Physical constraints:
      • c > 0 : the exponential term must contribute POSITIVE dp
        (intensification restoring force) when the storm is near MPI.
      • d ≥ 0 : the restoring effect must decay (not grow) as the storm
        moves away from MPI.
      • |mu| should be small relative to the typical dp (~2 hPa/step).
        A large |mu| means the mean function is systematically biased,
        and the residual distribution is doing the work the regression
        should be doing.

    Returns a DataFrame with one row per (basin, month, lat_cell, lon_cell).
    """
    coef_path = os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy")
    if not os.path.exists(coef_path):
        print("ERROR: COEFFICIENTS_JM_PRESSURE.npy not found. Run preprocessing first.")
        return None

    coeflist = np.load(coef_path, allow_pickle=True).item()
    months = load_input_months()
    rows = []

    for idx in range(6):
        if idx not in coeflist:
            continue
        for month in months[idx]:
            if month not in coeflist[idx]:
                continue
            cells = coeflist[idx][month]
            if not cells:
                continue
            cell_arr = np.array(cells)
            if cell_arr.ndim != 2:
                continue

            for cell_i in range(cell_arr.shape[0]):
                row = cell_arr[cell_i]
                if len(row) >= 12:
                    c0, c1, c2, c3 = row[0], row[1], row[2], row[3]
                    mu = row[4]
                    std_neg, std_pos = row[5], row[6]
                    mpi_fitted = row[7]
                    c_vws, c_rh = row[8], row[9]
                    c_en, c_ln = row[10], row[11]
                elif len(row) >= 7:
                    c0, c1, c2, c3 = row[0], row[1], row[2], row[3]
                    mu = row[4]
                    std_neg = std_pos = row[5]
                    mpi_fitted = row[6]
                    c_vws = c_rh = c_en = c_ln = 0.0
                else:
                    continue

                rows.append(
                    {
                        "basin": BASIN_NAMES[idx],
                        "basin_idx": idx,
                        "month": month,
                        "cell_idx": cell_i,
                        "a": c0,
                        "b_dp": c1,
                        "c_exp": c2,
                        "d_decay": c3,
                        "mu_residual": mu,
                        "std_neg": std_neg,
                        "std_pos": std_pos,
                        "mpi_fitted": mpi_fitted,
                        "c_vws": c_vws,
                        "c_rh": c_rh,
                        "c_en": c_en,
                        "c_ln": c_ln,
                        # ── Flags ──
                        "FLAG_c_nonpositive": int(c2 <= 0),
                        "FLAG_d_negative": int(c3 < 0),
                        "FLAG_mu_large": int(abs(mu) > 1.0),
                        "FLAG_mu_very_large": int(abs(mu) > 2.0),
                        "FLAG_asymmetry_ratio": std_neg / std_pos
                        if std_pos > 0
                        else np.nan,
                    }
                )

    df = pd.DataFrame(rows)
    return df


# =========================================================================
# §5  MPI CONSISTENCY: Compare fitted MPI vs. runtime PI field
# =========================================================================
def diagnose_mpi_consistency():
    """
    For each (basin, month, spatial cell), compare:
      • mpi_fitted : the MPI value stored in COEFFICIENTS_JM_PRESSURE.npy
        (column 7). This is what was used to compute presmpi = P - MPI
        during coefficient fitting.
      • pi_runtime : the Monthly_mean_PI_{month}.txt field value at the
        same grid cell center. This is what gets loaded at runtime in
        SAMPLE_TC_PRESSURE.py and OVERRIDES mpi_fitted.

    If |mpi_fitted - pi_runtime| is large (say > 10 hPa), the exponential
    term c*exp(-d*(P - MPI)) operates in a different regime at runtime than
    it was trained in.

    The key metric is the "effective shift":
        delta_mpi = pi_runtime - mpi_fitted

    And its impact on the exponential term:
        ratio = exp(+d * delta_mpi)

    If ratio >> 1 (say > 1.5), the exponential brake is much STRONGER at
    runtime than during training → storms under-intensify.
    If ratio << 1 (say < 0.7), the brake is WEAKER → storms over-intensify.
    """
    import preprocessing

    coef_path = os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy")
    if not os.path.exists(coef_path):
        print("ERROR: COEFFICIENTS_JM_PRESSURE.npy not found.")
        return None

    coeflist = np.load(coef_path, allow_pickle=True).item()
    months = load_input_months()

    # Load lat/lon grid for mapping cell indices to coordinates
    try:
        ds = pd.HDFStore  # dummy
        import xarray as xr

        ds_env = xr.open_dataset(os.path.join(__location__, "Monthly_mean_SST.nc"))
        lat_global = ds_env.latitude.values
        lon_global = ds_env.longitude.values
        ds_env.close()
    except Exception:
        lat_global = np.linspace(90, -90, 721)
        lon_global = np.linspace(0, 359.75, 1440)

    rows = []
    step = 5

    for idx in range(6):
        if idx not in coeflist:
            continue
        lat0, lat1, lon0, lon1_raw = preprocessing.BOUNDARIES_BASINS(idx)

        # Match the NA hack in pressure_coefficients
        if idx == 1:
            lon1 = lon1_raw + 5
        else:
            lon1 = lon1_raw

        n_lat_bins = int((lat1 - lat0) / step)
        n_lon_bins = int((lon1 - lon0) / step)

        for month in months[idx]:
            if month not in coeflist[idx]:
                continue

            cells = coeflist[idx][month]
            cell_arr = np.array(cells)
            if cell_arr.ndim != 2 or cell_arr.shape[1] < 8:
                continue

            # ── Load the runtime PI field (what SAMPLE_TC_PRESSURE sees) ──
            pi_path = os.path.join(__location__, f"Monthly_mean_PI_{month}.txt")
            pi_field = None
            pi_source = "NONE"
            if os.path.exists(pi_path):
                pi_field = np.loadtxt(pi_path)
                pi_source = "PI_thermodynamic"

            mpi_col = 7 if cell_arr.shape[1] >= 12 else 6
            d_col = 3  # the decay rate 'd'

            for cell_i in range(cell_arr.shape[0]):
                # Map flat cell index to (lat_bin, lon_bin)
                lat_bin_i = cell_i // n_lon_bins
                lon_bin_i = cell_i % n_lon_bins

                cell_lat = lat0 + lat_bin_i * step + step / 2.0  # cell center
                cell_lon = lon0 + lon_bin_i * step + step / 2.0

                mpi_fitted = float(cell_arr[cell_i, mpi_col])
                d_val = float(cell_arr[cell_i, d_col])

                # ── Look up runtime PI at cell center ──
                pi_runtime = np.nan
                if pi_field is not None:
                    lat_idx = int(np.abs(lat_global - cell_lat).argmin())
                    lon_c = cell_lon % 360.0
                    lon_idx = int(np.abs(lon_global - lon_c).argmin())
                    if (
                        0 <= lat_idx < pi_field.shape[0]
                        and 0 <= lon_idx < pi_field.shape[1]
                    ):
                        pi_runtime = float(pi_field[lat_idx, lon_idx])

                delta_mpi = np.nan
                exp_ratio = np.nan
                if (
                    np.isfinite(pi_runtime)
                    and np.isfinite(mpi_fitted)
                    and pi_runtime > 0
                ):
                    delta_mpi = pi_runtime - mpi_fitted
                    if d_val > 0:
                        # How much the exponential term shifts:
                        # At training: c * exp(-d * (P - mpi_fitted))
                        # At runtime:  c * exp(-d * (P - pi_runtime))
                        #            = c * exp(-d * (P - mpi_fitted - delta_mpi))
                        #            = c * exp(-d * (P - mpi_fitted)) * exp(+d * delta_mpi)
                        # So the multiplicative change is exp(+d * delta_mpi)
                        # ratio > 1 means stronger brake at runtime (storms under-intensify)
                        exp_ratio = np.exp(d_val * delta_mpi)

                rows.append(
                    {
                        "basin": BASIN_NAMES[idx],
                        "basin_idx": idx,
                        "month": month,
                        "cell_idx": cell_i,
                        "cell_lat": cell_lat,
                        "cell_lon": cell_lon,
                        "mpi_fitted": mpi_fitted,
                        "pi_runtime": pi_runtime,
                        "pi_source": pi_source,
                        "delta_mpi": delta_mpi,
                        "d_decay": d_val,
                        "exp_ratio": exp_ratio,
                        # ── Flags ──
                        "FLAG_no_pi": int(
                            not np.isfinite(pi_runtime) or pi_runtime <= 0
                        ),
                        "FLAG_delta_gt_5": int(
                            abs(delta_mpi) > 5 if np.isfinite(delta_mpi) else False
                        ),
                        "FLAG_delta_gt_10": int(
                            abs(delta_mpi) > 10 if np.isfinite(delta_mpi) else False
                        ),
                        "FLAG_delta_gt_20": int(
                            abs(delta_mpi) > 20 if np.isfinite(delta_mpi) else False
                        ),
                        "FLAG_ratio_lt_0.5": int(
                            exp_ratio < 0.5 if np.isfinite(exp_ratio) else False
                        ),
                        "FLAG_ratio_gt_1.5": int(
                            exp_ratio > 1.5 if np.isfinite(exp_ratio) else False
                        ),
                        "FLAG_ratio_gt_2.0": int(
                            exp_ratio > 2.0 if np.isfinite(exp_ratio) else False
                        ),
                    }
                )

    df = pd.DataFrame(rows)
    return df


# =========================================================================
# §6  VWS/RH CENTERING: Quantify scale imbalance and intercept absorption
# =========================================================================
def diagnose_vws_rh_centering():
    """
    The pressure model includes:
        ... + c_vws * VWS + c_rh * RH + ...

    VWS and RH are NOT centered (mean-subtracted) before fitting. This
    means:
      1. The intercept 'a' absorbs the mean contribution:
         a_effective = a + c_vws * mean(VWS) + c_rh * mean(RH)
      2. The ridge penalty (applied to c_en, c_ln) does not penalize
         c_vws and c_rh directly, but their fitted values are affected
         by the different scales of the predictors.

    This diagnostic computes:
      - Mean, std, range of VWS and RH across all observations per basin
      - The fitted c_vws and c_rh across cells
      - The "effective contribution" = |c_vws * std(VWS)| vs |c_rh * std(RH)|
        which tells you how much each predictor actually moves dp1
      - The mean absorption: c_vws * mean(VWS) and c_rh * mean(RH)
    """
    pres_var_path = os.path.join(__location__, "TC_PRESSURE_VARIABLES.npy")
    coef_path = os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy")

    if not os.path.exists(pres_var_path) or not os.path.exists(coef_path):
        print("ERROR: Required .npy files not found. Run preprocessing first.")
        return None

    pres_variables = np.load(pres_var_path, allow_pickle=True).item()
    coeflist = np.load(coef_path, allow_pickle=True).item()
    months = load_input_months()

    rows = []

    for idx in range(6):
        # ── Extract VWS and RH from the observation pool ──
        if 8 not in pres_variables or 9 not in pres_variables:
            print(f"  Basin {BASIN_NAMES[idx]}: No VWS/RH in TC_PRESSURE_VARIABLES")
            continue

        vws_all = np.array(pres_variables[8][idx], dtype=float)
        rh_all = np.array(pres_variables[9][idx], dtype=float)

        # Clean: remove NaN
        vws_clean = vws_all[np.isfinite(vws_all)]
        rh_clean = rh_all[np.isfinite(rh_all)]

        if len(vws_clean) < 10 or len(rh_clean) < 10:
            continue

        vws_mean = float(np.mean(vws_clean))
        vws_std = float(np.std(vws_clean))
        vws_p05 = float(np.percentile(vws_clean, 5))
        vws_p95 = float(np.percentile(vws_clean, 95))

        rh_mean = float(np.mean(rh_clean))
        rh_std = float(np.std(rh_clean))
        rh_p05 = float(np.percentile(rh_clean, 5))
        rh_p95 = float(np.percentile(rh_clean, 95))

        # ── Extract c_vws and c_rh from all cells in this basin ──
        c_vws_list = []
        c_rh_list = []
        intercept_list = []

        if idx not in coeflist:
            continue

        for month in months[idx]:
            if month not in coeflist[idx]:
                continue
            cells = coeflist[idx][month]
            cell_arr = np.array(cells)
            if cell_arr.ndim != 2 or cell_arr.shape[1] < 12:
                continue

            for ci in range(cell_arr.shape[0]):
                c_vws_list.append(float(cell_arr[ci, 8]))
                c_rh_list.append(float(cell_arr[ci, 9]))
                intercept_list.append(float(cell_arr[ci, 0]))

        if not c_vws_list:
            continue

        c_vws_arr = np.array(c_vws_list)
        c_rh_arr = np.array(c_rh_list)
        intercept_arr = np.array(intercept_list)

        # ── Effective contribution of each predictor ──
        # How much does a 1-sigma shift in VWS move dp1?
        eff_vws = np.abs(np.median(c_vws_arr)) * vws_std
        eff_rh = np.abs(np.median(c_rh_arr)) * rh_std

        # How much of the intercept is absorbed by the mean field?
        absorbed_vws = np.median(c_vws_arr) * vws_mean
        absorbed_rh = np.median(c_rh_arr) * rh_mean

        rows.append(
            {
                "basin": BASIN_NAMES[idx],
                "n_obs_vws": len(vws_clean),
                "n_obs_rh": len(rh_clean),
                # ── Predictor statistics ──
                "vws_mean": round(vws_mean, 2),
                "vws_std": round(vws_std, 2),
                "vws_p05": round(vws_p05, 2),
                "vws_p95": round(vws_p95, 2),
                "rh_mean": round(rh_mean, 2),
                "rh_std": round(rh_std, 2),
                "rh_p05": round(rh_p05, 2),
                "rh_p95": round(rh_p95, 2),
                # ── Coefficient statistics ──
                "c_vws_median": round(float(np.median(c_vws_arr)), 5),
                "c_vws_mean": round(float(np.mean(c_vws_arr)), 5),
                "c_vws_std": round(float(np.std(c_vws_arr)), 5),
                "c_rh_median": round(float(np.median(c_rh_arr)), 5),
                "c_rh_mean": round(float(np.mean(c_rh_arr)), 5),
                "c_rh_std": round(float(np.std(c_rh_arr)), 5),
                # ── Effective contribution (hPa/step per 1-sigma predictor shift) ──
                "effective_vws_1sigma": round(eff_vws, 4),
                "effective_rh_1sigma": round(eff_rh, 4),
                "ratio_eff_vws_over_rh": round(eff_vws / eff_rh, 3)
                if eff_rh > 1e-8
                else np.nan,
                # ── Mean absorption into intercept ──
                "absorbed_by_vws_mean": round(absorbed_vws, 4),
                "absorbed_by_rh_mean": round(absorbed_rh, 4),
                "intercept_median": round(float(np.median(intercept_arr)), 4),
                # ── Flags ──
                # If effective contributions differ by > 5×, one predictor
                # is effectively dead or dominant — centering would help.
                "FLAG_scale_imbalance_gt_5x": int(
                    max(eff_vws, eff_rh) / max(min(eff_vws, eff_rh), 1e-10) > 5.0
                ),
                # If absorbed mean > intercept magnitude, intercept is
                # compensating for uncentered predictors.
                "FLAG_vws_absorption_dominant": int(
                    abs(absorbed_vws) > abs(float(np.median(intercept_arr))) * 0.5
                ),
                "FLAG_rh_absorption_dominant": int(
                    abs(absorbed_rh) > abs(float(np.median(intercept_arr))) * 0.5
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df


# =========================================================================
# SUMMARY
# =========================================================================
def write_summary(df_exp, df_mpi, df_vws, fpath):
    lines = []
    lines.append("=" * 72)
    lines.append("SIENA-IH-STORM DIAGNOSTIC SUMMARY")
    lines.append("=" * 72)

    # §4 Exponential term
    lines.append("\n§4  EXPONENTIAL TERM SIGN CONVENTION")
    lines.append("-" * 40)
    if df_exp is not None and len(df_exp) > 0:
        n_cells = len(df_exp)
        n_c_bad = df_exp["FLAG_c_nonpositive"].sum()
        n_d_bad = df_exp["FLAG_d_negative"].sum()
        n_mu_large = df_exp["FLAG_mu_large"].sum()
        n_mu_vlarge = df_exp["FLAG_mu_very_large"].sum()

        lines.append(f"Total cells examined: {n_cells}")
        lines.append(f"Cells with c ≤ 0:    {n_c_bad} ({100 * n_c_bad / n_cells:.1f}%)")
        lines.append(f"  → These cells have no intensification restoring force.")
        lines.append(f"  → Acceptable ONLY if they're at high latitudes where")
        lines.append(f"     storms rarely intensify.")
        lines.append(f"Cells with d < 0:    {n_d_bad} ({100 * n_d_bad / n_cells:.1f}%)")
        lines.append(f"  → These cells have GROWING restoring force with distance")
        lines.append(f"     from MPI — physically wrong.")
        lines.append(
            f"Cells with |mu| > 1: {n_mu_large} ({100 * n_mu_large / n_cells:.1f}%)"
        )
        lines.append(
            f"Cells with |mu| > 2: {n_mu_vlarge} ({100 * n_mu_vlarge / n_cells:.1f}%)"
        )
        lines.append(f"  → mu is the residual mean. If large, the regression")
        lines.append(f"     systematically over/under-predicts dp1 and the")
        lines.append(f"     residual distribution compensates.")

        # Per-basin
        for basin in BASIN_NAMES:
            sub = df_exp[df_exp["basin"] == basin]
            if len(sub) == 0:
                continue
            lines.append(
                f"\n  {basin}: {len(sub)} cells, "
                f"c≤0: {sub['FLAG_c_nonpositive'].sum()}, "
                f"d<0: {sub['FLAG_d_negative'].sum()}, "
                f"|mu|>1: {sub['FLAG_mu_large'].sum()}, "
                f"median(c)={sub['c_exp'].median():.3f}, "
                f"median(d)={sub['d_decay'].median():.4f}, "
                f"median(|mu|)={sub['mu_residual'].abs().median():.3f}"
            )
    else:
        lines.append("  No data available.")

    # §5 MPI consistency
    lines.append("\n\n§5  MPI CONSISTENCY: FITTED MPI vs RUNTIME PI")
    lines.append("-" * 40)
    if df_mpi is not None and len(df_mpi) > 0:
        n_cells = len(df_mpi)
        valid = df_mpi[df_mpi["delta_mpi"].notna() & (df_mpi["FLAG_no_pi"] == 0)]
        n_valid = len(valid)

        lines.append(f"Total cells: {n_cells}")
        lines.append(f"Cells with valid runtime PI: {n_valid}")
        lines.append(f"Cells with NO runtime PI:    {df_mpi['FLAG_no_pi'].sum()}")

        if n_valid > 0:
            lines.append(f"\nΔMPI = PI_runtime - MPI_fitted (hPa):")
            lines.append(f"  mean:   {valid['delta_mpi'].mean():+.1f} hPa")
            lines.append(f"  median: {valid['delta_mpi'].median():+.1f} hPa")
            lines.append(f"  std:    {valid['delta_mpi'].std():.1f} hPa")
            lines.append(f"  p05:    {valid['delta_mpi'].quantile(0.05):+.1f} hPa")
            lines.append(f"  p95:    {valid['delta_mpi'].quantile(0.95):+.1f} hPa")
            lines.append(
                f"\n  |ΔMPI| > 5 hPa:  {valid['FLAG_delta_gt_5'].sum()} cells ({100 * valid['FLAG_delta_gt_5'].mean():.1f}%)"
            )
            lines.append(
                f"  |ΔMPI| > 10 hPa: {valid['FLAG_delta_gt_10'].sum()} cells ({100 * valid['FLAG_delta_gt_10'].mean():.1f}%)"
            )
            lines.append(
                f"  |ΔMPI| > 20 hPa: {valid['FLAG_delta_gt_20'].sum()} cells ({100 * valid['FLAG_delta_gt_20'].mean():.1f}%)"
            )

            lines.append(f"\nExponential ratio = exp(+d × ΔMPI):")
            lines.append(f"  This is the multiplicative change in the exponential")
            lines.append(f"  brake at runtime relative to training.")
            lines.append(
                f"  ratio>1 = brake STRONGER at runtime → storms under-intensify"
            )
            lines.append(f"  ratio<1 = brake WEAKER at runtime → storms over-intensify")
            lines.append(f"  ratio≈1 = consistent (target after fix)")
            ratio_valid = valid[valid["exp_ratio"].notna()]
            if len(ratio_valid) > 0:
                lines.append(f"  mean:   {ratio_valid['exp_ratio'].mean():.3f}")
                lines.append(f"  median: {ratio_valid['exp_ratio'].median():.3f}")
                lines.append(f"  p05:    {ratio_valid['exp_ratio'].quantile(0.05):.3f}")
                lines.append(f"  p95:    {ratio_valid['exp_ratio'].quantile(0.95):.3f}")
                lines.append(
                    f"  ratio > 1.5: {valid['FLAG_ratio_gt_1.5'].sum()} cells"
                    f" — brake 50%+ stronger, storms under-intensify"
                )
                lines.append(
                    f"  ratio > 2.0: {valid['FLAG_ratio_gt_2.0'].sum()} cells"
                    f" — brake 100%+ stronger, significant under-intensification"
                )
                lines.append(
                    f"  ratio < 0.5: {valid['FLAG_ratio_lt_0.5'].sum()} cells"
                    f" — brake 50%+ weaker, storms over-intensify"
                )

            lines.append(f"\nINTERPRETATION:")
            med_delta = valid["delta_mpi"].median()
            med_ratio = (
                ratio_valid["exp_ratio"].median() if len(ratio_valid) > 0 else np.nan
            )
            if abs(med_delta) < 5:
                lines.append(
                    f"  ✓ Median ΔMPI = {med_delta:+.1f} hPa (ratio={med_ratio:.2f}) — GOOD."
                )
                lines.append(
                    f"    The fitted MPI and runtime PI are broadly consistent."
                )
            elif abs(med_delta) < 15:
                lines.append(
                    f"  ⚠ Median ΔMPI = {med_delta:+.1f} hPa (ratio={med_ratio:.2f}) — MODERATE."
                )
                if med_delta > 0:
                    lines.append(
                        f"    PI_runtime > MPI_fitted → exponential brake is STRONGER"
                    )
                    lines.append(f"    at runtime → synthetic storms UNDER-INTENSIFY.")
                else:
                    lines.append(
                        f"    PI_runtime < MPI_fitted → exponential brake is WEAKER"
                    )
                    lines.append(f"    at runtime → synthetic storms OVER-INTENSIFY.")
                lines.append(
                    f"    Consider changing nanmin → nanmedian in MPI aggregation."
                )
            else:
                lines.append(
                    f"  ✗ Median ΔMPI = {med_delta:+.1f} hPa (ratio={med_ratio:.2f}) — SERIOUS."
                )
                lines.append(
                    f"    The regression was trained on a substantially different"
                )
                lines.append(f"    MPI than what's used at runtime.")
                lines.append(f"    FIX: Change nanmin → nanmedian in environmental.py,")
                lines.append(f"    or remove the PI override in SAMPLE_TC_PRESSURE.py.")

            # Per-basin
            for basin in BASIN_NAMES:
                sub = valid[valid["basin"] == basin]
                if len(sub) == 0:
                    continue
                lines.append(
                    f"\n  {basin}: median ΔMPI={sub['delta_mpi'].median():+.1f}, "
                    f"|Δ|>10: {sub['FLAG_delta_gt_10'].sum()}/{len(sub)}, "
                    f"median exp_ratio={sub['exp_ratio'].median():.3f}"
                )
    else:
        lines.append("  No data available.")

    # §6 VWS/RH centering
    lines.append("\n\n§6  VWS / RH CENTERING DIAGNOSTIC")
    lines.append("-" * 40)
    if df_vws is not None and len(df_vws) > 0:
        for _, row in df_vws.iterrows():
            lines.append(
                f"\n  {row['basin']} (n_obs VWS={row['n_obs_vws']}, RH={row['n_obs_rh']}):"
            )
            lines.append(
                f"    VWS: mean={row['vws_mean']:.1f}, std={row['vws_std']:.1f}, "
                f"range=[{row['vws_p05']:.1f}, {row['vws_p95']:.1f}]"
            )
            lines.append(
                f"    RH:  mean={row['rh_mean']:.1f}, std={row['rh_std']:.1f}, "
                f"range=[{row['rh_p05']:.1f}, {row['rh_p95']:.1f}]"
            )
            lines.append(
                f"    c_vws median={row['c_vws_median']:.5f}, "
                f"c_rh median={row['c_rh_median']:.5f}"
            )
            lines.append(
                f"    Effective 1σ contribution: "
                f"VWS→{row['effective_vws_1sigma']:.4f} hPa/step, "
                f"RH→{row['effective_rh_1sigma']:.4f} hPa/step, "
                f"ratio={row['ratio_eff_vws_over_rh']}"
            )
            lines.append(
                f"    Mean absorbed by intercept: "
                f"VWS→{row['absorbed_by_vws_mean']:.4f}, "
                f"RH→{row['absorbed_by_rh_mean']:.4f}, "
                f"intercept={row['intercept_median']:.4f}"
            )

            if row["FLAG_scale_imbalance_gt_5x"]:
                lines.append(f"    ⚠ SCALE IMBALANCE > 5×: One predictor dominates.")
                lines.append(f"      Consider standardizing VWS and RH before fitting.")
            if row["FLAG_vws_absorption_dominant"]:
                lines.append(f"    ⚠ VWS mean absorption > 50% of intercept magnitude.")
            if row["FLAG_rh_absorption_dominant"]:
                lines.append(f"    ⚠ RH mean absorption > 50% of intercept magnitude.")

        lines.append(f"\n  INTERPRETATION:")
        lines.append(f"    If ratio_eff_vws_over_rh is close to 1.0, both predictors")
        lines.append(
            f"    contribute equally per 1σ shift — centering is not critical."
        )
        lines.append(f"    If the ratio is > 5 or < 0.2, one predictor is effectively")
        lines.append(f"    dead and centering + potentially different regularization")
        lines.append(f"    would help recover its signal.")
        lines.append(f"\n    If both effective contributions are < 0.1 hPa/step,")
        lines.append(f"    neither VWS nor RH meaningfully affects dp1 and they")
        lines.append(f"    could be removed without loss. If they're > 0.5 hPa/step,")
        lines.append(f"    they matter and centering is important for stability.")
    else:
        lines.append("  No data available.")

    lines.append("\n" + "=" * 72)

    text = "\n".join(lines)
    with open(fpath, "w") as f:
        f.write(text)
    print(text)


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    print("Running SIENA-IH-STORM diagnostics...\n")

    print("§4  Exponential term sign convention...")
    df_exp = diagnose_exponential_term()
    if df_exp is not None:
        out4 = os.path.join(__location__, "diagnostic_pressure_coefficients.csv")
        df_exp.to_csv(out4, index=False)
        print(f"  Saved: {out4} ({len(df_exp)} cells)")

    print("§5  MPI consistency...")
    df_mpi = diagnose_mpi_consistency()
    if df_mpi is not None:
        out5 = os.path.join(__location__, "diagnostic_mpi_consistency.csv")
        df_mpi.to_csv(out5, index=False)
        print(f"  Saved: {out5} ({len(df_mpi)} cells)")

    print("§6  VWS/RH centering...")
    df_vws = diagnose_vws_rh_centering()
    if df_vws is not None:
        out6 = os.path.join(__location__, "diagnostic_vws_rh_centering.csv")
        df_vws.to_csv(out6, index=False)
        print(f"  Saved: {out6} ({len(df_vws)} basins)")

    summary_path = os.path.join(__location__, "diagnostic_summary.txt")
    write_summary(df_exp, df_mpi, df_vws, summary_path)
    print(f"\nSummary: {summary_path}")
