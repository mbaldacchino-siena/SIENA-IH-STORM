import os
import numpy as np
import pandas as pd

PHASE_TO_CODE = {"LN": 0, "NEU": 1, "EN": 2}
CODE_TO_PHASE = {v: k for k, v in PHASE_TO_CODE.items()}


def normalize_phase(phase):
    if phase is None:
        return None
    phase = str(phase).strip().upper()
    if phase in {"N", "NONE", "ALL", "POOLED"}:
        return None
    if phase not in PHASE_TO_CODE:
        raise ValueError(f"Unsupported phase: {phase}")
    return phase


def phase_code(phase):
    phase = normalize_phase(phase)
    return None if phase is None else PHASE_TO_CODE[phase]


def phase_from_index(value):
    try:
        value = int(value)
    except Exception:
        return "NEU"
    return CODE_TO_PHASE.get(value, "NEU")


def file_with_phase(base_dir, stem, month, phase=None, ext="txt"):
    phase = normalize_phase(phase)
    if phase is not None:
        candidate = os.path.join(base_dir, f"{stem}_{month}_{phase}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_dir, f"{stem}_{month}.{ext}")


def load_monthly_field(base_dir, stem, month, phase=None):
    return np.loadtxt(file_with_phase(base_dir, stem, month, phase=phase, ext="txt"))


def load_climate_index_table(path="climate_index.csv"):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["year", "month", "climate_index", "phase"])
    df = pd.read_csv(path)
    if "phase" not in df.columns:
        values = df["climate_index"].astype(float)
        df["phase"] = np.where(
            values >= 0.5, "EN", np.where(values <= -0.5, "LN", "NEU")
        )
    return df


def build_phase_lookup(df):
    lookup = {}
    if df is None or len(df) == 0:
        return lookup
    for _, row in df.iterrows():
        lookup[(int(row["year"]), int(row["month"]))] = phase_code(row["phase"])
    return lookup


# =========================================================================
# C4 FIX: Exposure-based Poisson rate correction
# =========================================================================


def count_phase_months(oni_df, monthsall, threshold=0.5):
    """
    Count the number of active-season months classified as each ENSO phase,
    per basin, across all years.  This is the "exposure" denominator for the
    Poisson rate estimator.

    The correct exposure-based rate is:

        rate_ph = storms_ph × L / M_ph

    where L = len(monthsall[idx]) (season length) and M_ph = total active-
    season months classified as phase ph across all years.

    This avoids the double-counting problem: a mixed year contributes
    fractionally to each phase denominator, proportional to how many of
    its active months were actually in that phase.

    Parameters
    ----------
    oni_df : DataFrame with columns [year, month, climate_index, phase]
    monthsall : list of 6 lists, active-season months per basin
    threshold : float (used only if 'phase' column is missing)

    Returns
    -------
    phase_month_counts : dict {basin_idx: {phase_code: int}}
        Total active-season months per phase per basin.
    """
    # Ensure phase column exists
    if "phase" not in oni_df.columns:
        oni_df = oni_df.copy()
        vals = oni_df["climate_index"].astype(float)
        oni_df["phase"] = np.where(
            vals >= threshold, "EN", np.where(vals <= -threshold, "LN", "NEU")
        )

    phase_month_counts = {idx: {0: 0, 1: 0, 2: 0} for idx in range(6)}

    for idx in range(6):
        season_months = set(monthsall[idx])
        for _, row in oni_df.iterrows():
            m = int(row["month"])
            if m not in season_months:
                continue
            ph = PHASE_TO_CODE.get(str(row["phase"]).strip().upper(), 1)
            phase_month_counts[idx][ph] += 1

    return phase_month_counts


def verify_phase_rates(
    poisson_phase_rate, pooled_rate, phase_month_counts, season_length, idx
):
    """
    Sanity check: weighted sum of phase rates (weighted by fraction of
    active-season time in each phase) must recover the pooled rate.

    rate_pooled ≈ Σ_ph  rate_ph × (M_ph / total_active_months)

    Parameters
    ----------
    poisson_phase_rate : dict {phase_code: rate}
    pooled_rate : float
    phase_month_counts : dict {phase_code: int} for this basin
    season_length : int, active months per year for this basin
    idx : basin index (for logging)
    """
    total_months = sum(phase_month_counts.values())
    if total_months == 0 or pooled_rate == 0:
        return

    weighted = sum(
        poisson_phase_rate.get(ph, 0) * phase_month_counts[ph] / total_months
        for ph in [0, 1, 2]
    )
    ratio = weighted / pooled_rate if pooled_rate > 0 else float("nan")
    print(
        f"  Basin {idx} sanity check: "
        f"weighted phase sum = {weighted:.2f}, pooled rate = {pooled_rate:.1f}, "
        f"ratio = {ratio:.4f} (should be ~1.0)"
    )


def nearest_env_value(field, latitudes, longitudes, lat, lon):
    lat = float(lat)
    lon = float(lon)
    if lon < 0:
        lon += 360.0
    lat_idx = int(np.abs(latitudes - lat).argmin())
    lon_idx = int(np.abs(longitudes - lon).argmin())
    return float(field[lat_idx, lon_idx])


def solve_ridge(X, y, penalty_cols=None, alpha=0.0, add_intercept=True):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    if penalty_cols is None:
        penalty_cols = []
    if alpha and penalty_cols:
        P = np.zeros((len(penalty_cols), X.shape[1]))
        for i, col in enumerate(penalty_cols):
            P[i, col] = np.sqrt(alpha)
        X_aug = np.vstack([X, P])
        y_aug = np.concatenate([y, np.zeros(P.shape[0])])
    else:
        X_aug, y_aug = X, y
    beta, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    return beta


# =========================================================================
# H1 FIX: Leave-one-year-out cross-validation for ridge lambda
# =========================================================================


def select_lambda_cv(
    X, y, years, penalty_cols, lambda_grid, add_intercept=True, min_fold_size=10
):
    """
    Leave-one-year-out cross-validation for ridge penalty selection.

    Parameters
    ----------
    X : 2D array (n_obs, n_features), predictor matrix (NO year column)
    y : 1D array (n_obs,), response
    years : 1D array (n_obs,), year labels for fold splitting
    penalty_cols : list of int, column indices in the AUGMENTED X
                   (after intercept prepend) to penalize
    lambda_grid : list of float, candidate lambda values
    add_intercept : bool
    min_fold_size : int, skip folds with fewer observations

    Returns
    -------
    best_lambda : float
    best_mse : float
    all_results : list of (lambda, mean_mse) for diagnostics
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    years = np.asarray(years)
    unique_years = np.unique(years)

    all_results = []
    best_lambda = lambda_grid[0]
    best_mse = np.inf

    for lam in lambda_grid:
        mse_folds = []
        for hold_year in unique_years:
            train_mask = years != hold_year
            test_mask = years == hold_year
            if test_mask.sum() < min_fold_size:
                continue

            beta = solve_ridge(
                X[train_mask],
                y[train_mask],
                penalty_cols=penalty_cols,
                alpha=lam,
                add_intercept=add_intercept,
            )

            X_test = X[test_mask]
            if add_intercept:
                X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])
            pred = X_test @ beta
            mse_folds.append(float(np.mean((y[test_mask] - pred) ** 2)))

        if mse_folds:
            mean_mse = np.mean(mse_folds)
            all_results.append((lam, mean_mse))
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_lambda = lam

    return best_lambda, best_mse, all_results
