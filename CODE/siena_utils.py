import os
import numpy as np
import pandas as pd

PHASE_TO_CODE = {"LN": 0, "NEU": 1, "EN": 2}
CODE_TO_PHASE = {v: k for k, v in PHASE_TO_CODE.items()}
TS_THRESHOLD_MS = 18.0 * 0.88 # ~ 15.84 m/s (~34 kt 1-min after 0.88 conv)
C1_THRESHOLD_MS = 33* 0.88
C2_THRESHOLD_MS = 43*0.88
C3_THRESHOLD_MS = 50*0.88
C4_THRESHOLD_MS = 58*0.88
C5_THRESHOLD_MS = 70*0.88

def normalize_phase(phase):
    if phase is None:
        return None
    phase = str(phase).strip().upper()
    if phase in {"N", "NONE", "ALL", "POOLED", "FCST"}:
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



# =========================================================================
# Forecast mode: blended-rate genesis with month-level ENSO phases
# =========================================================================


def compute_monthly_genesis_weights(genesis_months_phase, idx, active_months):
    """
    Compute per-month genesis fraction w_m for each phase, from the
    historical genesis month lists.

    Parameters
    ----------
    genesis_months_phase : dict, {basin_idx: {"LN": [months...], "NEU": [...], "EN": [...]}}
        As stored in GENESIS_MONTHS_PHASE.npy
    idx : int, basin index
    active_months : list of int

    Returns
    -------
    dict : {phase_str: {month: float}}
        Normalised weights (sum to 1 over active_months for each phase).
        Falls back to uniform if a phase has no data for a month.
    """
    weights = {}
    for phase in ["LN", "NEU", "EN"]:
        month_list = genesis_months_phase[idx].get(phase, [])
        if len(month_list) == 0:
            # No data: uniform over active months
            w = {m: 1.0 / len(active_months) for m in active_months}
        else:
            counts = {m: 0 for m in active_months}
            for m in month_list:
                if m in counts:
                    counts[m] += 1
            total = sum(counts.values())
            if total == 0:
                w = {m: 1.0 / len(active_months) for m in active_months}
            else:
                w = {m: c / total for m, c in counts.items()}
        weights[phase] = w
    return weights


def blended_genesis(
    poisson_phase_rate,
    genesis_months_phase,
    idx,
    active_months,
    month_phases,
):
    """
    Draw storm count and genesis months for a synthetic year with
    month-level ENSO phases (forecast mode).

    Algorithm:
      1. Compute blended annual rate:
         λ_blend = Σ_m  λ_{ph(m)} × w_{m|ph(m)}
         where w_{m|ph} is the historical fraction of phase-ph storms
         born in month m.

      2. Draw total storms from Poisson(λ_blend).

      3. Distribute storms across months via multinomial with
         probabilities p_m = λ_{ph(m)} × w_{m|ph(m)} / λ_blend.
         The multinomial naturally anti-correlates months given a fixed
         total — no artificial month independence.

    Parameters
    ----------
    poisson_phase_rate : dict, {basin_idx: {phase_code: float}}
        Annual Poisson rate per phase, from POISSON_GENESIS_PARAMETERS_PHASE.npy
    genesis_months_phase : dict, from GENESIS_MONTHS_PHASE.npy
    idx : int, basin index
    active_months : list of int, e.g. [6,7,8,9,10,11]
    month_phases : dict, {month_int: "LN"|"NEU"|"EN"}
        The ENSO phase assigned to each month in this forecast year.

    Returns
    -------
    storms_per_year : int
    genesis_month_list : list of int (one entry per storm)
    """
    from siena_utils import PHASE_TO_CODE

    # Step 0: historical monthly weights per phase
    monthly_weights = compute_monthly_genesis_weights(
        genesis_months_phase, idx, active_months
    )

    # Step 1: blended rate
    lambda_blend = 0.0
    monthly_contributions = {}  # {month: λ_{ph(m)} * w_{m|ph(m)}}
    for m in active_months:
        ph_str = month_phases.get(m, "NEU")
        ph_code = PHASE_TO_CODE[ph_str]
        lam_ph = poisson_phase_rate[idx].get(ph_code, 0)
        w_m = monthly_weights[ph_str].get(m, 0)
        contrib = lam_ph * w_m
        monthly_contributions[m] = contrib
        lambda_blend += contrib

    if lambda_blend <= 0:
        return 0, []

    # Step 2: draw total count
    storms_per_year = int(np.random.poisson(lambda_blend))
    if storms_per_year == 0:
        return 0, []

    # Step 3: multinomial distribution across months
    probs = np.array([monthly_contributions[m] for m in active_months])
    probs = probs / probs.sum()  # normalise (should already sum to 1)
    counts = np.random.multinomial(storms_per_year, probs)

    genesis_month_list = []
    for m, c in zip(active_months, counts):
        genesis_month_list.extend([m] * c)

    return storms_per_year, genesis_month_list



def compute_relative_vorticity_spherical(u, v, lat_deg, lon_deg, radius=6.371e6):
    """
    Relative vorticity on a regular lat-lon grid:

        zeta = 1/(a cos(phi)) * dv/dlambda - 1/a * du/dphi + u tan(phi)/a

    Parameters
    ----------
    u, v : 2D arrays (lat, lon), m s-1
    lat_deg, lon_deg : 1D arrays, degrees
    radius : float, Earth radius in m

    Returns
    -------
    zeta : 2D array, s-1
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))

    coslat = np.cos(lat_rad)
    tanlat = np.tan(lat_rad)
    coslat = np.where(np.abs(coslat) < 1e-10, np.nan, coslat)

    # d()/dphi with the true latitude coordinate
    dudphi = np.gradient(u, lat_rad, axis=0, edge_order=2)

    # d()/dlambda with cyclic longitude handling
    dlon = float(np.mean(np.diff(lon_rad)))
    dvdlambda = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dlon)

    zeta = (
        dvdlambda / (radius * coslat[:, None])
        - dudphi / radius
        + u * tanlat[:, None] / radius
    )
    return zeta