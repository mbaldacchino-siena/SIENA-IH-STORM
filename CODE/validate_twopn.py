"""
H3 Diagnostic: Two-piece normal vs symmetric normal fit validation.

Run after MASTER_preprocessing.py has completed.
Exports residual_distribution_diagnostic.csv with per-cell KS statistics.

Usage:
    python validate_twopn.py
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import kstest, skew, norm
from scipy.special import erf
from scipy.optimize import least_squares

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def _twopn_cdf(x, mu, std_neg, std_pos):
    """CDF of two-piece normal distribution."""
    z = x - mu
    p_left = std_neg / (std_neg + std_pos)
    out = np.empty_like(x, dtype=float)
    left = z < 0
    out[left] = p_left * (1.0 + erf(z[left] / (std_neg * np.sqrt(2.0))))
    out[~left] = p_left + (1.0 - p_left) * erf(z[~left] / (std_pos * np.sqrt(2.0)))
    return out


def validate_residuals():
    """
    For each basin × month × spatial cell, compute the pressure model
    residuals and compare the two-piece normal vs symmetric normal fit.
    """
    import CODE.preprocessing as preprocessing
    import CODE.environmental as environmental

    pres_variables = np.load(
        os.path.join(__location__, "TC_PRESSURE_VARIABLES.npy"), allow_pickle=True
    ).item()
    coef_jm = np.load(
        os.path.join(__location__, "COEFFICIENTS_JM_PRESSURE.npy"), allow_pickle=True
    ).item()

    basin_names = ["EP", "NA", "NI", "SI", "SP", "WP"]
    results = []

    for idx in range(6):
        if idx not in coef_jm or not coef_jm[idx]:
            continue
        for month, cell_list in coef_jm[idx].items():
            if not cell_list:
                continue

            # Build the observation dataframe for this basin-month
            dp0_all = np.array(pres_variables[0][idx])
            dp1_all = np.array(pres_variables[1][idx])
            pres_all = np.array(pres_variables[2][idx])
            lat_all = np.array(pres_variables[3][idx])
            lon_all = np.array(pres_variables[4][idx])
            month_all = np.array(pres_variables[5][idx])

            mask = (
                (month_all == month)
                & (pres_all > 0)
                & (dp0_all > -10000)
                & (dp1_all > -10000)
            )
            if mask.sum() < 20:
                continue

            dp0 = dp0_all[mask]
            dp1 = dp1_all[mask]
            pres = pres_all[mask]

            # For each cell, extract the fitted coefficients and compute residuals
            cell_arr = np.array(cell_list)
            if cell_arr.ndim != 2 or cell_arr.shape[1] < 8:
                continue

            # Aggregate residuals across all cells for this basin-month
            # (cells share the same observation pool via the 3x3 neighborhood pooling)
            # Use the basin-month-level aggregate for a cleaner test
            if cell_arr.shape[1] >= 12:
                c0, c1, c2, c3 = (
                    cell_arr[:, 0].mean(),
                    cell_arr[:, 1].mean(),
                    cell_arr[:, 2].mean(),
                    cell_arr[:, 3].mean(),
                )
                mu_stored = cell_arr[:, 4].mean()
                std_neg_stored = cell_arr[:, 5].mean()
                std_pos_stored = cell_arr[:, 6].mean()
                mpi_vals = cell_arr[:, 7]
            else:
                c0, c1, c2, c3 = (
                    cell_arr[:, 0].mean(),
                    cell_arr[:, 1].mean(),
                    cell_arr[:, 2].mean(),
                    cell_arr[:, 3].mean(),
                )
                mu_stored = cell_arr[:, 4].mean()
                std_neg_stored = cell_arr[:, 5].mean()
                std_pos_stored = std_neg_stored
                mpi_vals = cell_arr[:, 6]

            mpi_mean = (
                np.nanmean(mpi_vals[np.isfinite(mpi_vals)])
                if np.any(np.isfinite(mpi_vals))
                else 900.0
            )
            presmpi = np.maximum(0.0, pres - mpi_mean)

            # Predicted dp1 from the mean function
            pred = c0 + c1 * dp0 + c2 * np.exp(-c3 * presmpi)
            resid = dp1 - pred

            # Remove extreme outliers for a cleaner test
            q01, q99 = np.percentile(resid, [1, 99])
            clean = resid[(resid > q01) & (resid < q99)]
            if len(clean) < 30:
                continue

            # Fit symmetric normal
            mu_sym, std_sym = norm.fit(clean)
            ks_sym, p_sym = kstest(clean, "norm", args=(mu_sym, std_sym))

            # Fit two-piece normal
            mu_twopn = np.mean(clean)
            centered = clean - mu_twopn
            neg = centered[centered < 0]
            pos = centered[centered >= 0]
            std_neg = float(np.sqrt(np.mean(neg**2))) if len(neg) > 1 else std_sym
            std_pos = float(np.sqrt(np.mean(pos**2))) if len(pos) > 1 else std_sym

            # KS test against fitted two-piece normal CDF
            ks_twopn, p_twopn = kstest(
                clean, lambda x: _twopn_cdf(x, mu_twopn, std_neg, std_pos)
            )

            results.append(
                {
                    "basin": basin_names[idx],
                    "basin_idx": idx,
                    "month": month,
                    "n_obs": len(clean),
                    "skewness": float(skew(clean)),
                    "mu_symmetric": mu_sym,
                    "std_symmetric": std_sym,
                    "ks_symmetric": ks_sym,
                    "p_symmetric": p_sym,
                    "mu_twopn": mu_twopn,
                    "std_neg": std_neg,
                    "std_pos": std_pos,
                    "ks_twopn": ks_twopn,
                    "p_twopn": p_twopn,
                    "delta_ks": ks_sym - ks_twopn,  # positive = two-piece better
                    "twopn_wins": int(ks_twopn < ks_sym),
                }
            )

    df = pd.DataFrame(results)
    out_path = os.path.join(__location__, "residual_distribution_diagnostic.csv")
    df.to_csv(out_path, index=False)
    print(f"\nH3 Diagnostic saved to {out_path}")
    print(f"Total basin-month combos tested: {len(df)}")

    if len(df) > 0:
        print(f"\n--- Summary ---")
        print(
            f"Two-piece normal wins in {df['twopn_wins'].sum()}/{len(df)} "
            f"({100 * df['twopn_wins'].mean():.1f}%) of basin-months"
        )
        print(
            f"Mean ΔKS (sym - twopn, positive = twopn better): {df['delta_ks'].mean():.4f}"
        )
        print(f"Mean skewness of residuals: {df['skewness'].mean():.3f}")

        # Per-basin summary
        for basin in basin_names:
            sub = df[df["basin"] == basin]
            if len(sub) > 0:
                print(
                    f"  {basin}: twopn wins {sub['twopn_wins'].sum()}/{len(sub)}, "
                    f"mean skew={sub['skewness'].mean():.3f}, "
                    f"mean ΔKS={sub['delta_ks'].mean():.4f}"
                )

    return df


