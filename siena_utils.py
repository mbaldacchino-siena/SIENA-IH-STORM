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


# =========================================================================
# Month-level environmental field resampling
# =========================================================================
# Instead of loading phase-mean climatological fields (which suppress
# interannual variability and compress the intensity tail), we store
# individual year-month fields and draw a random historical year per
# active-season month. The pool is keyed by (phase, month): for an LN
# September, only historical years where September was actually LN are
# eligible. This guarantees phase-consistent resampling while preserving
# the joint VWS-RH-PI covariance that drives extreme events.
#
# Storage layout:
#   env_yearly/
#     VWS_{year}_{month}.npy
#     RH600_{year}_{month}.npy
#     MSLP_{year}_{month}.npy
#     PI_{year}_{month}.npy
#     env_pool.json          ← {phase: {month: [years]}}
#
# For seasonal forecasts: place forecast fields in the same directory
# with a synthetic "year" label (e.g. 9999) and run with --env-year 9999.
# =========================================================================

import json

ENV_YEARLY_DIR = "env_yearly"


def _env_yearly_dir(base_dir):
    """Return the env_yearly directory path, creating it if needed."""
    d = os.path.join(base_dir, ENV_YEARLY_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def save_env_pool(base_dir, env_pool):
    """
    Save the environment pool: {phase: {month_str: [year1, year2, ...]}}.

    Each (phase, month) maps to historical years where that specific month
    was in that ENSO phase. This guarantees phase-consistent resampling.

    Parameters
    ----------
    base_dir : str, project directory
    env_pool : dict, e.g. {"LN": {"6": [1988, 1999], "9": [1995]}, ...}
    """
    path = os.path.join(_env_yearly_dir(base_dir), "env_pool.json")
    pool = {}
    for ph, months in env_pool.items():
        if isinstance(months, dict):
            pool[ph] = {str(m): [int(y) for y in yrs] for m, yrs in months.items()}
        else:
            pool[ph] = [int(y) for y in months]
    with open(path, "w") as f:
        json.dump(pool, f, indent=2)


def load_env_pool(base_dir):
    """
    Load the environment pool: {phase: {month_str: [year1, ...]}}.
    """
    path = os.path.join(_env_yearly_dir(base_dir), "env_pool.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def sample_env_year(env_pool, phase, month):
    """
    Draw a random historical year for a specific (phase, month).

    The pool is structured as {"LN": {"6": [1988, 1999], ...}, ...}.
    This guarantees the returned year actually had that month in the
    target ENSO phase.

    Parameters
    ----------
    env_pool : dict from load_env_pool()
    phase : str, "LN" / "NEU" / "EN"
    month : int, calendar month (1-12)

    Returns
    -------
    int : a historical year, or None if pool is empty
    """
    phase = normalize_phase(phase)
    mo_key = str(int(month))

    if phase is not None:
        phase_pool = env_pool.get(phase, {})
        years = phase_pool.get(mo_key, [])
        if years:
            return int(np.random.choice(years))

    # Fallback: pool across all phases for this month
    all_years = []
    for ph_pool in env_pool.values():
        if isinstance(ph_pool, dict):
            all_years.extend(ph_pool.get(mo_key, []))
        elif isinstance(ph_pool, list):
            # Backward compat with old year-level pool format
            all_years.extend(ph_pool)
    if all_years:
        return int(np.random.choice(all_years))
    return None


def draw_env_years_for_season(env_pool, phase, active_months):
    """
    Draw one historical year per active-season month for a simulated year.

    All storms born in the same month share the same environmental context
    (within-month coherence), while each month independently comes from a
    year where that month was actually in the target ENSO phase.

    Parameters
    ----------
    env_pool : dict from load_env_pool()
    phase : str, "LN" / "NEU" / "EN"
    active_months : list of int, e.g. [6, 7, 8, 9, 10, 11] for NA

    Returns
    -------
    dict : {month: historical_year} e.g. {6: 1988, 7: 1999, ...}
           Returns None values for months with empty pools.
    """
    env_years = {}
    for m in active_months:
        env_years[m] = sample_env_year(env_pool, phase, m)
    return env_years


def save_yearly_field(base_dir, stem, year, month, field):
    """Save a single year-month environmental field as .npy."""
    d = _env_yearly_dir(base_dir)
    path = os.path.join(d, f"{stem}_{year}_{month}.npy")
    np.save(path, field.astype(np.float32))


def load_yearly_field(base_dir, stem, year, month):
    """
    Load a year-specific field. Returns None if not found.
    Falls back to phase-mean or pooled field if yearly file is missing.
    """
    d = _env_yearly_dir(base_dir)
    path = os.path.join(d, f"{stem}_{year}_{month}.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.float64)
    return None


def load_field_with_year_fallback(base_dir, stem, month, phase=None, env_year=None):
    """
    Load an environmental field with year → phase-mean → pooled fallback.

    Priority:
      1. Year-specific field (if env_year is set and file exists)
      2. Phase-specific mean field (Monthly_mean_{stem}_{month}_{phase}.txt)
      3. Pooled mean field (Monthly_mean_{stem}_{month}.txt)
    """
    if env_year is not None:
        yearly = load_yearly_field(base_dir, stem, env_year, month)
        if yearly is not None:
            return yearly
    # Fall back to phase-mean or pooled
    try:
        return load_monthly_field(base_dir, f"Monthly_mean_{stem}", month, phase=phase)
    except Exception:
        return None


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
