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


def file_with_phase(base_dir, stem, month, phase=None, ext='txt'):
    phase = normalize_phase(phase)
    if phase is not None:
        candidate = os.path.join(base_dir, f"{stem}_{month}_{phase}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(base_dir, f"{stem}_{month}.{ext}")


def load_monthly_field(base_dir, stem, month, phase=None):
    return np.loadtxt(file_with_phase(base_dir, stem, month, phase=phase, ext='txt'))


def load_climate_index_table(path='climate_index.csv'):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['year', 'month', 'climate_index', 'phase'])
    df = pd.read_csv(path)
    if 'phase' not in df.columns:
        values = df['climate_index'].astype(float)
        df['phase'] = np.where(values >= 0.5, 'EN', np.where(values <= -0.5, 'LN', 'NEU'))
    return df


def build_phase_lookup(df):
    lookup = {}
    if df is None or len(df) == 0:
        return lookup
    for _, row in df.iterrows():
        lookup[(int(row['year']), int(row['month']))] = phase_code(row['phase'])
    return lookup


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
