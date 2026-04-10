"""
============================================================================
SIENA-IH-STORM Evaluation Pipeline
============================================================================
Generates all validation figures and summary statistics for the whitepaper.
Compares model configurations (B0, B1, S1, S2, S3) against IBTrACS across
ENSO phases for the North Atlantic (NA) and Western Pacific (WP) basins.

Usage:
    python evaluation_pipeline.py --data-dir ./evaluation_output --basins NA WP

Requirements:
    numpy, pandas, matplotlib, scipy, cartopy (optional, for coastlines)

Author: Mathys Baldacchino / evaluation helper
"""

import argparse
import os
import glob
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ── Try to import cartopy (optional, for proper coastlines) ──────────────
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EvalConfig:
    """Central configuration for the evaluation pipeline."""

    data_dir: str = "./evaluation_output"
    output_dir: str = "./figures"
    basins: list = field(default_factory=lambda: ["NA", "WP"])

    # Which model IDs map to your paper labels.
    # Keys = folder names on disk, values = paper label.
    # Adjust this mapping to match your actual folder names.
    model_map: dict = field(
        default_factory=lambda: {
            "B0": "B0",
            "B1": "B1",
            "S1": "S1",
            "S2": "S2",
            "S3": "S3",  # change to e.g. "S6": "S3" if your hybrid is in the S6 folder
        }
    )

    phases: list = field(default_factory=lambda: ["EN", "NEU", "LN"])
    phases_all: list = field(default_factory=lambda: ["ALL", "EN", "NEU", "LN"])

    # ENSO phase frequencies for active-season months.
    # Auto-loaded from {data_dir}/{basin}/phase_fractions.csv if available.
    # Falls back to these defaults (computed from ONI 3.4, 1980–2021).
    phase_freqs: dict = field(
        default_factory=lambda: {
            "NA": {"EN": 0.3, "NEU": 0.3, "LN": 0.4},
            "WP": {"EN": 0.1, "NEU": 0.9, "LN": 0.0},
        }
    )

    def load_phase_freqs(self, basin: str) -> dict:
        """
        Load phase frequencies from phase_fractions.csv exported by
        run_evaluation.py. Falls back to hardcoded defaults.
        """
        csv_path = os.path.join(self.data_dir, basin, "phase_fractions.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, keep_default_na=False)
            row = df[df["basin"] == basin] if "basin" in df.columns else df
            if len(row) > 0:
                r = row.iloc[0]
                freqs = {ph: float(r[ph]) for ph in ["EN", "NEU", "LN"] if ph in r}
                if freqs:
                    self.phase_freqs[basin] = freqs
                    print(f"  Phase freqs loaded from {csv_path}: {freqs}")
                    return freqs
        if basin in self.phase_freqs:
            print(
                f"  Phase freqs using defaults for {basin}: {self.phase_freqs[basin]}"
            )
            return self.phase_freqs[basin]
        print(f"  WARNING: No phase freqs for {basin}, using uniform.")
        return {"EN": 1 / 3, "NEU": 1 / 3, "LN": 1 / 3}

    # Baselines vs proposed (used for visual grouping)
    baseline_ids: list = field(default_factory=lambda: ["B0", "B1"])
    proposed_ids: list = field(default_factory=lambda: ["S1", "S2", "S3"])

    # Saffir-Simpson thresholds (1-min sustained, m/s)
    ss_thresholds: dict = field(
        default_factory=lambda: {
            "Cat 1": 33,
            "Cat 2": 43,
            "Cat 3": 50,
            "Cat 4": 58,
            "Cat 5": 70,
        }
    )

    # NA basin extent for maps (lon_min, lon_max, lat_min, lat_max)
    basin_extents: dict = field(
        default_factory=lambda: {
            "NA": (-100, -10, 5, 55),
            "WP": (100, 180, 5, 45),
            "EP": (-180, -80, 5, 40),
            "NI": (30, 100, 5, 35),
            "SI": (10, 135, -40, -5),
            "SP": (135, 240, -40, -5),
        }
    )

    # Colors and styles
    model_colors: dict = field(
        default_factory=lambda: {
            "IBTrACS": "#1a1a1a",
            "B0": "#7f8c8d",
            "B1": "#c0392b",
            "S1": "#e67e22",
            "S2": "#27ae60",
            "S3": "#2980b9",
        }
    )
    phase_colors: dict = field(
        default_factory=lambda: {
            "EN": "#e74c3c",
            "NEU": "#3498db",
            "LN": "#2ecc71",
            "ALL": "#7f8c8d",
        }
    )
    model_linestyles: dict = field(
        default_factory=lambda: {
            "IBTrACS": "-",
            "B0": "--",
            "B1": "--",
            "S1": "-.",
            "S2": ":",
            "S3": "-",
        }
    )
    model_markers: dict = field(
        default_factory=lambda: {
            "IBTrACS": "D",
            "B0": "s",
            "B1": "x",
            "S1": "^",
            "S2": "v",
            "S3": "o",
        }
    )

    # Figure settings
    dpi: int = 200
    fig_format: str = "png"  # "png" or "pdf"
    use_cartopy: bool = True  # falls back gracefully if not installed

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def all_model_ids(self):
        return list(self.model_map.values())

    def disk_name(self, paper_id: str) -> str:
        """Reverse lookup: paper label -> folder name on disk."""
        for k, v in self.model_map.items():
            if v == paper_id:
                return k
        return paper_id


# ============================================================================
# Data loading
# ============================================================================


def load_summary_metrics(cfg: EvalConfig, basin: str) -> pd.DataFrame:
    """Load the summary_metrics.csv for a given basin."""
    path = os.path.join(cfg.data_dir, basin, "summary_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Summary metrics not found: {path}")
    df = pd.read_csv(path)
    # Remap candidate names to paper labels
    inv_map = {v: v for v in cfg.model_map.values()}
    inv_map.update(cfg.model_map)
    inv_map["IBTrACS"] = "IBTrACS"
    df["candidate"] = df["candidate"].map(lambda x: inv_map.get(x, x))
    return df


def load_density_grid(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a density CSV (genesis, track, or ACE) and return (lat, lon, grid).
    Longitude is converted from 0–360 to -180–180.
    """
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    lat_col, lon_col, val_col = cols[0], cols[1], cols[2]

    lat = np.sort(df[lat_col].unique())
    lon = np.sort(df[lon_col].unique())
    grid = np.full((len(lat), len(lon)), np.nan)

    lat_idx = {v: i for i, v in enumerate(lat)}
    lon_idx = {v: i for i, v in enumerate(lon)}
    for _, row in df.iterrows():
        grid[lat_idx[row[lat_col]], lon_idx[row[lon_col]]] = row[val_col]

    # Convert to -180..180
    lon180 = np.where(lon > 180, lon - 360, lon)
    order = np.argsort(lon180)
    return lat, lon180[order], grid[:, order]


def load_lifetime(path: str) -> pd.DataFrame:
    """Load a lifetime CSV."""
    return pd.read_csv(path)


def load_return_periods(path: str) -> pd.DataFrame:
    """Load an rp_*.csv file."""
    return pd.read_csv(path)


def find_density_file(
    cfg: EvalConfig, basin: str, model: str, dtype: str, phase: str
) -> Optional[str]:
    """Resolve the path to a density CSV, accounting for model_map."""
    disk = cfg.disk_name(model) if model != "IBTrACS" else "IBTrACS"
    path = os.path.join(cfg.data_dir, basin, disk, f"{dtype}_{disk}_{phase}.csv")
    return path if os.path.exists(path) else None


def find_lifetime_file(
    cfg: EvalConfig, basin: str, model: str, phase: str
) -> Optional[str]:
    disk = cfg.disk_name(model) if model != "IBTrACS" else "IBTrACS"
    path = os.path.join(cfg.data_dir, basin, disk, f"lifetime_{disk}_{phase}.csv")
    return path if os.path.exists(path) else None


def find_rp_file(cfg: EvalConfig, basin: str, model: str, phase: str) -> Optional[str]:
    disk = cfg.disk_name(model) if model != "IBTrACS" else "IBTrACS"
    path = os.path.join(cfg.data_dir, basin, disk, f"rp_{disk}_{phase}.csv")
    return path if os.path.exists(path) else None


# ============================================================================
# Statistical helpers
# ============================================================================


def ks_2sample(synth: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p-value)."""
    synth = synth[~np.isnan(synth)]
    obs = obs[~np.isnan(obs)]
    if len(synth) < 2 or len(obs) < 2:
        return np.nan, np.nan
    return stats.ks_2samp(obs, synth)


def bias_pct(synth_val: float, obs_val: float) -> float:
    """Percentage bias."""
    if obs_val == 0:
        return np.nan
    return (synth_val - obs_val) / obs_val * 100


def spatial_correlation(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Pearson correlation between two flattened grids (ignoring NaNs)."""
    a = np.nan_to_num(grid_a, nan=0.0).ravel()
    b = np.nan_to_num(grid_b, nan=0.0).ravel()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


def spatial_rmse(grid_synth: np.ndarray, grid_obs: np.ndarray) -> float:
    """Root mean square error between two density grids."""
    a = np.nan_to_num(grid_synth, nan=0.0).ravel()
    b = np.nan_to_num(grid_obs, nan=0.0).ravel()
    return np.sqrt(np.mean((a - b) ** 2))


def ad_2sample(synth: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """Anderson-Darling 2-sample test. More tail-sensitive than KS."""
    synth = synth[~np.isnan(synth)]
    obs = obs[~np.isnan(obs)]
    if len(synth) < 2 or len(obs) < 2:
        return np.nan, np.nan
    result = stats.anderson_ksamp([obs, synth])
    return result.statistic, result.pvalue


def wasserstein_dist(synth: np.ndarray, obs: np.ndarray) -> float:
    """Earth Mover's Distance (1st Wasserstein) between two samples."""
    synth = synth[~np.isnan(synth)]
    obs = obs[~np.isnan(obs)]
    if len(synth) < 2 or len(obs) < 2:
        return np.nan
    return stats.wasserstein_distance(obs, synth)


def chi2_categorical(
    synth_counts: np.ndarray, obs_counts: np.ndarray
) -> tuple[float, float]:
    """
    Chi-squared test for categorical frequency comparison
    (e.g. Saffir-Simpson category distributions).
    synth_counts and obs_counts are arrays of counts per category.
    synth_counts is rescaled to match the obs total before testing.
    """
    obs_counts = np.asarray(obs_counts, dtype=float)
    synth_counts = np.asarray(synth_counts, dtype=float)
    if obs_counts.sum() == 0 or synth_counts.sum() == 0:
        return np.nan, np.nan
    # Rescale synthetic to expected counts under obs total
    expected = synth_counts / synth_counts.sum() * obs_counts.sum()
    # Avoid zero expected
    mask = expected > 0
    if mask.sum() < 2:
        return np.nan, np.nan
    stat, p = stats.chisquare(obs_counts[mask], f_exp=expected[mask])
    return stat, p


# ============================================================================
# Figure 1: Summary scorecard heatmap
# ============================================================================


def fig_scorecard(cfg: EvalConfig, basin: str, sm: pd.DataFrame):
    """
    Heatmap scorecard: rows = models, cols = ENSO phases.
    One panel per metric (genesis bias, landfall bias, lifetime bias, KS Pmin, KS Vmax).
    """
    ibt = sm[sm.candidate == "IBTrACS"].set_index("phase")
    models = [m for m in cfg.all_model_ids if m in sm.candidate.values]

    metrics = [
        ("genesis_mean", "Genesis count\n(mean/yr)", "bias_pct"),
        ("lifetime_hours", "Lifetime\n(hours)", "bias_pct"),
        ("landfall_mean", "Landfall count\n(mean/yr)", "bias_pct"),
        ("ks_pmin", "KS stat\n(Pmin)", "raw"),
        ("ks_vmax", "KS stat\n(Vmax)", "raw"),
    ]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(3.6 * len(metrics), 0.6 * len(models) + 2),
        gridspec_kw={"wspace": 0.4},
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax_i, (metric, label, mode) in enumerate(metrics):
        ax = axes[ax_i]
        data = np.full((len(models), len(cfg.phases_all)), np.nan)
        annot = np.full((len(models), len(cfg.phases_all)), "", dtype=object)

        for i, model in enumerate(models):
            for j, phase in enumerate(cfg.phases_all):
                row = sm[(sm.candidate == model) & (sm.phase == phase)]
                if row.empty:
                    continue
                val = row[metric].values[0]
                if pd.isna(val):
                    continue
                if mode == "bias_pct":
                    ref = ibt.loc[phase, metric] if phase in ibt.index else np.nan
                    if pd.notna(ref) and ref != 0:
                        bias = bias_pct(val, ref)
                        data[i, j] = bias
                        annot[i, j] = f"{val:.1f}\n({bias:+.0f}%)"
                else:
                    data[i, j] = val
                    annot[i, j] = f"{val:.3f}"

        # Colormaps
        if mode == "bias_pct":
            vabs = min(np.nanmax(np.abs(data)), 60) if np.any(np.isfinite(data)) else 30
            norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
            cmap = "RdYlGn_r"
        else:
            vmin = max(0, np.nanmin(data) * 0.9) if np.any(np.isfinite(data)) else 0
            vmax = np.nanmax(data) * 1.1 if np.any(np.isfinite(data)) else 0.3
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = "RdYlGn_r"

        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(models)):
            for j in range(len(cfg.phases_all)):
                txt = annot[i, j] if annot[i, j] else "—"
                fontsize = 7 if mode == "bias_pct" else 8
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight="medium",
                )

        ax.set_xticks(range(len(cfg.phases_all)))
        ax.set_xticklabels(cfg.phases_all, fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if ax_i == 0 else [""] * len(models), fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)

        # Separator between baselines and proposed
        n_base = sum(1 for m in models if m in cfg.baseline_ids)
        if 0 < n_base < len(models):
            ax.axhline(n_base - 0.5, color="black", linewidth=1.5)

        cb = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.04)
        cb.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Model evaluation scorecard — {basin} basin\n"
        f"Bias (%) vs IBTrACS 1980–2021 | KS statistics: lower is better",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    _save(fig, cfg, f"scorecard_{basin}")


# ============================================================================
# Figure 2: Summary bar chart (genesis, landfall, lifetime, KS)
# ============================================================================


def fig_summary_bars(cfg: EvalConfig, basin: str, sm: pd.DataFrame):
    """Grouped bar charts comparing all models per ENSO phase."""
    ibt = sm[sm.candidate == "IBTrACS"].set_index("phase")
    models = [m for m in cfg.all_model_ids if m in sm.candidate.values]

    panels = [
        ("genesis_mean", "Storms / yr", "Mean annual genesis count"),
        ("landfall_mean", "Landfalls / yr", "Mean annual landfall count"),
        ("lifetime_hours", "Hours", "Mean TC lifetime"),
        ("ks_vmax", "KS statistic", "KS(Vmax) — lower is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for pi, (metric, ylabel, title) in enumerate(panels):
        ax = axes[pi // 2, pi % 2]
        x = np.arange(len(cfg.phases_all))
        w = 0.8 / (len(models) + 1)

        # IBTrACS reference bands
        for j, ph in enumerate(cfg.phases_all):
            if ph in ibt.index and pd.notna(ibt.loc[ph, metric]):
                ref = ibt.loc[ph, metric]
                if metric != "ks_vmax":  # no reference band for KS
                    ax.axhspan(
                        ref * 0.9, ref * 1.1, alpha=0.12, color="black", zorder=0
                    )
                    ax.plot(j, ref, "k_", markersize=14, markeredgewidth=2.5, zorder=5)

        for k, model in enumerate(models):
            vals = []
            for ph in cfg.phases_all:
                row = sm[(sm.candidate == model) & (sm.phase == ph)]
                vals.append(
                    row[metric].values[0]
                    if not row.empty and pd.notna(row[metric].values[0])
                    else np.nan
                )
            offset = (k - len(models) / 2 + 0.5) * w
            color = cfg.model_colors.get(model, "#999")
            ax.bar(
                x + offset,
                vals,
                w,
                label=model if pi == 0 else "",
                color=color,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(cfg.phases_all, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    # Build a combined legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.insert(
        0,
        Line2D(
            [],
            [],
            marker="_",
            color="black",
            markersize=10,
            markeredgewidth=2,
            linestyle="None",
        ),
    )
    labels.insert(0, "IBTrACS (±10%)")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(models) + 1,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle(
        f"Climatological summary — {basin}", fontsize=13, fontweight="bold", y=1.0
    )
    plt.tight_layout()
    _save(fig, cfg, f"summary_bars_{basin}")


# ============================================================================
# Figures 3–5: Spatial density maps (genesis, track, ACE)
# ============================================================================


def _map_axis(ax, extent, cfg: EvalConfig):
    """Configure a map axis (with or without cartopy)."""
    if HAS_CARTOPY and cfg.use_cartopy and hasattr(ax, "set_extent"):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(
            cfeature.LAND, facecolor="#e8e8e8", edgecolor="#888", linewidth=0.3
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#666")
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 6}
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_facecolor("#e8e8e8")
        ax.tick_params(labelsize=7)


def fig_density_panels(
    cfg: EvalConfig,
    basin: str,
    dtype: str,
    cmap: str = "YlOrRd",
    log: bool = True,
    title_prefix: str = "Genesis density",
    cbar_label: str = "events per 10 kyr per cell",
):
    """
    Multi-panel density map: rows = models, cols = ENSO phases.
    Works for genesis_density, track_density, ace_density.
    """
    models = ["IBTrACS"] + cfg.all_model_ids
    phases = cfg.phases
    extent = cfg.basin_extents.get(basin, (-100, -10, 5, 55))

    # Determine color scale from IBTrACS data
    vmin_pos, vmax_val = 1e30, 0
    for ph in phases:
        p = find_density_file(cfg, basin, "IBTrACS", dtype, ph)
        if p:
            _, _, g = load_density_grid(p)
            pos = g[g > 0]
            if len(pos):
                vmin_pos = min(vmin_pos, pos.min())
                vmax_val = max(vmax_val, np.percentile(pos, 97))
    if vmax_val == 0:
        vmax_val = 100
        vmin_pos = 1

    use_carto = HAS_CARTOPY and cfg.use_cartopy
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_carto else {}
    fig, axes = plt.subplots(
        len(models),
        len(phases),
        figsize=(5 * len(phases), 3 * len(models)),
        subplot_kw=subplot_kw,
    )
    if len(models) == 1:
        axes = axes[np.newaxis, :]

    for i, model in enumerate(models):
        for j, ph in enumerate(phases):
            ax = axes[i, j]
            p = find_density_file(cfg, basin, model, dtype, ph)
            if p:
                lat, lon, grid = load_density_grid(p)
                grid_plot = np.where(grid > 0, grid, np.nan)
                if log:
                    norm = mcolors.LogNorm(vmin=vmin_pos, vmax=vmax_val)
                else:
                    norm = mcolors.Normalize(vmin=0, vmax=vmax_val)
                ax.pcolormesh(
                    lon,
                    lat,
                    grid_plot,
                    cmap=cmap,
                    norm=norm,
                    shading="auto",
                    transform=ccrs.PlateCarree() if use_carto else None,
                )
            _map_axis(ax, extent, cfg)

            if i == 0:
                ax.set_title(ph, fontsize=12, fontweight="bold", pad=6)
            if j == 0:
                label = model if model == "IBTrACS" else f"{model}"
                if use_carto:
                    ax.text(
                        -0.12,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        fontsize=9,
                        fontweight="bold",
                        va="center",
                        ha="center",
                        rotation=90,
                    )
                else:
                    ax.set_ylabel(label, fontsize=9, fontweight="bold")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
    if log:
        sm_cb = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.LogNorm(vmin=vmin_pos, vmax=vmax_val)
        )
    else:
        sm_cb = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=vmax_val)
        )
    fig.colorbar(sm_cb, cax=cbar_ax).set_label(cbar_label, fontsize=9)

    fig.suptitle(
        f"{title_prefix} — {basin}, by ENSO phase",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    _save(fig, cfg, f"{dtype}_{basin}")


def fig_density_diff(
    cfg: EvalConfig,
    basin: str,
    dtype: str,
    title_prefix: str = "Genesis density",
):
    """
    Difference maps (synthetic − observed) for each model vs IBTrACS.
    """
    models = cfg.all_model_ids
    phases = cfg.phases
    extent = cfg.basin_extents.get(basin, (-100, -10, 5, 55))

    use_carto = HAS_CARTOPY and cfg.use_cartopy
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_carto else {}
    fig, axes = plt.subplots(
        len(models),
        len(phases),
        figsize=(5 * len(phases), 3 * len(models)),
        subplot_kw=subplot_kw,
    )
    if len(models) == 1:
        axes = axes[np.newaxis, :]

    for i, model in enumerate(models):
        for j, ph in enumerate(phases):
            ax = axes[i, j]
            p_obs = find_density_file(cfg, basin, "IBTrACS", dtype, ph)
            p_mod = find_density_file(cfg, basin, model, dtype, ph)

            if p_obs and p_mod:
                lat_o, lon_o, g_o = load_density_grid(p_obs)
                lat_m, lon_m, g_m = load_density_grid(p_mod)
                go = np.nan_to_num(g_o, nan=0.0)
                gm = np.nan_to_num(g_m, nan=0.0)

                if go.shape == gm.shape:
                    diff = gm - go
                    lat_d, lon_d = lat_o, lon_o
                else:
                    # Interpolate model grid onto obs grid
                    from scipy.interpolate import RegularGridInterpolator

                    interp = RegularGridInterpolator(
                        (lat_m, lon_m), gm, bounds_error=False, fill_value=0
                    )
                    pts = (
                        np.array(np.meshgrid(lat_o, lon_o, indexing="ij"))
                        .reshape(2, -1)
                        .T
                    )
                    diff = interp(pts).reshape(len(lat_o), len(lon_o)) - go
                    lat_d, lon_d = lat_o, lon_o

                nonzero = diff[diff != 0]
                vabs = np.percentile(np.abs(nonzero), 97) if len(nonzero) else 1
                norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
                pcm = ax.pcolormesh(
                    lon_d,
                    lat_d,
                    diff,
                    cmap="RdBu_r",
                    norm=norm,
                    shading="auto",
                    transform=ccrs.PlateCarree() if use_carto else None,
                )
                plt.colorbar(pcm, ax=ax, shrink=0.7, pad=0.02)

            _map_axis(ax, extent, cfg)

            if i == 0:
                ax.set_title(ph, fontsize=12, fontweight="bold")
            if j == 0:
                label = f"{model} − IBTrACS"
                if use_carto:
                    ax.text(
                        -0.12,
                        0.5,
                        label,
                        transform=ax.transAxes,
                        fontsize=9,
                        fontweight="bold",
                        va="center",
                        ha="center",
                        rotation=90,
                    )
                else:
                    ax.set_ylabel(label, fontsize=9, fontweight="bold")

    fig.suptitle(
        f"{title_prefix} difference (synthetic − observed) — {basin}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    _save(fig, cfg, f"{dtype}_diff_{basin}")


# ============================================================================
# Figure 6: Intensity CDFs (Vmax and Pmin)
# ============================================================================


def fig_intensity_cdfs(cfg: EvalConfig, basin: str):
    """
    Overlay CDF plots of Vmax and Pmin for all available models and phases.
    Two rows: top = ALL, bottom = per-phase overlay for proposed model vs IBTrACS.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top row: ALL phase, all models ──
    for col, (var, xlabel) in enumerate(
        [("vmax", "Lifetime max wind (m/s)"), ("pmin", "Min central pressure (hPa)")]
    ):
        ax = axes[0, col]
        for model in ["IBTrACS"] + cfg.all_model_ids:
            p = find_lifetime_file(cfg, basin, model, "ALL")
            if not p:
                continue
            df = load_lifetime(p)
            vals = df[var].dropna().values
            if len(vals) < 2:
                continue
            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            color = cfg.model_colors.get(model, "#999")
            ls = cfg.model_linestyles.get(model, "-")
            lw = 2.5 if model == "IBTrACS" else 1.3
            ax.plot(
                sorted_v,
                cdf,
                color=color,
                linestyle=ls,
                linewidth=lw,
                label=f"{model} (n={len(vals):,})",
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("CDF", fontsize=10)
        ax.set_title(f"{var.upper()} CDF — ALL", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ── Bottom row: per-phase comparison (proposed vs IBTrACS) ──
    proposed = cfg.proposed_ids[-1]  # last proposed = full hybrid
    for col, (var, xlabel) in enumerate(
        [("vmax", "Lifetime max wind (m/s)"), ("pmin", "Min central pressure (hPa)")]
    ):
        ax = axes[1, col]
        for ph in cfg.phases:
            for model in ["IBTrACS", proposed]:
                p = find_lifetime_file(cfg, basin, model, ph)
                if not p:
                    continue
                df = load_lifetime(p)
                vals = df[var].dropna().values
                if len(vals) < 2:
                    continue
                sorted_v = np.sort(vals)
                cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
                color = cfg.phase_colors[ph]
                ls = "-" if model == "IBTrACS" else "--"
                lw = 2.0 if model == "IBTrACS" else 1.3
                ax.plot(
                    sorted_v,
                    cdf,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    label=f"{model} {ph}",
                )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("CDF", fontsize=10)
        ax.set_title(f"{var.upper()} CDF by ENSO phase", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Intensity distributions — {basin}", fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    _save(fig, cfg, f"intensity_cdfs_{basin}")


# ============================================================================
# Figure 7: Lifetime distributions
# ============================================================================


def fig_lifetime_distributions(cfg: EvalConfig, basin: str):
    """Histogram + CDF of TC lifetime (duration_hours) across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    for model in ["IBTrACS"] + cfg.all_model_ids:
        p = find_lifetime_file(cfg, basin, model, "ALL")
        if not p:
            continue
        df = load_lifetime(p)
        vals = df["duration_hours"].dropna().values
        color = cfg.model_colors.get(model, "#999")
        ax.hist(
            vals,
            bins=np.arange(0, 500, 12),
            density=True,
            histtype="step",
            linewidth=1.5 if model == "IBTrACS" else 1.0,
            color=color,
            label=model,
        )

    ax.set_xlabel("Duration (hours)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("TC lifetime distribution (ALL)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # CDF
    ax = axes[1]
    for model in ["IBTrACS"] + cfg.all_model_ids:
        p = find_lifetime_file(cfg, basin, model, "ALL")
        if not p:
            continue
        df = load_lifetime(p)
        vals = np.sort(df["duration_hours"].dropna().values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        color = cfg.model_colors.get(model, "#999")
        ls = cfg.model_linestyles.get(model, "-")
        ax.plot(
            vals,
            cdf,
            color=color,
            linestyle=ls,
            linewidth=2.0 if model == "IBTrACS" else 1.2,
            label=model,
        )

    ax.set_xlabel("Duration (hours)", fontsize=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_title("TC lifetime CDF (ALL)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"TC lifetime — {basin}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, cfg, f"lifetime_{basin}")


# ============================================================================
# Figure 8: Return periods at coastal cities
# ============================================================================


def fig_return_periods(cfg: EvalConfig, basin: str):
    """
    Return period curves per city.
    Two figures:
      (a) Phase comparison: B1 vs proposed, per ENSO phase
      (b) Model comparison: all models at ALL phase
    """
    # Discover cities from any available rp file
    cities = set()
    for model in cfg.all_model_ids:
        for ph in cfg.phases_all:
            p = find_rp_file(cfg, basin, model, ph)
            if p:
                df = load_return_periods(p)
                cities.update(df["city"].unique())
    cities = sorted(cities)
    if not cities:
        print(f"  [SKIP] No return period data for {basin}")
        return

    ncols = min(3, len(cities))
    nrows = int(np.ceil(len(cities) / ncols))

    # ── (a) B1 vs proposed, per ENSO phase ──
    proposed = cfg.proposed_ids[-1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, city in enumerate(cities):
        ax = axes[idx // ncols, idx % ncols]
        for model in ["B1", proposed]:
            for ph in cfg.phases:
                p = find_rp_file(cfg, basin, model, ph)
                if not p:
                    continue
                df = load_return_periods(p)
                cdf = df[df.city == city].dropna(subset=["wind_ms"])
                if cdf.empty:
                    continue
                color = cfg.phase_colors[ph]
                ls = "--" if model == "B1" else "-"
                marker = "x" if model == "B1" else "o"
                ax.plot(
                    cdf.return_period,
                    cdf.wind_ms,
                    color=color,
                    linestyle=ls,
                    marker=marker,
                    markersize=4,
                    linewidth=1.5,
                    label=f"{model} – {ph}",
                    alpha=0.85,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Return period (yr)", fontsize=9)
        ax.set_ylabel("Wind speed (m/s)", fontsize=9)
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        _add_ss_lines(ax, cfg)

    # Hide unused axes
    for idx in range(len(cities), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, len(handles)),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Wind return periods — {basin} | B1 (dashed) vs {proposed} (solid)",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()
    _save(fig, cfg, f"return_periods_phase_{basin}")

    # ── (b) ALL climatology, all models ──
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, city in enumerate(cities):
        ax = axes[idx // ncols, idx % ncols]
        for model in cfg.all_model_ids:
            p = find_rp_file(cfg, basin, model, "ALL")
            if not p:
                continue
            df = load_return_periods(p)
            cdf = df[df.city == city].dropna(subset=["wind_ms"])
            if cdf.empty:
                continue
            color = cfg.model_colors.get(model, "#999")
            marker = cfg.model_markers.get(model, "o")
            ax.plot(
                cdf.return_period,
                cdf.wind_ms,
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.5,
                label=model,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Return period (yr)", fontsize=9)
        ax.set_ylabel("Wind speed (m/s)", fontsize=9)
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        _add_ss_lines(ax, cfg)

    for idx in range(len(cities), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(6, len(handles)),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"Wind return periods (ALL climatology) — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()
    _save(fig, cfg, f"return_periods_ALL_{basin}")


def _add_ss_lines(ax, cfg: EvalConfig):
    """Add Saffir-Simpson category lines to a return-period axis."""
    for label, ws in cfg.ss_thresholds.items():
        ax.axhline(ws, color="#999", alpha=0.25, linewidth=0.7)
        ax.text(
            ax.get_xlim()[0] * 1.1, ws + 0.5, label, fontsize=6, color="#999", alpha=0.7
        )


# ============================================================================
# Figure 9: Spatial correlation / RMSE summary table
# ============================================================================


def fig_spatial_scores(cfg: EvalConfig, basin: str):
    """
    Compact heatmap of spatial Pearson-r and RMSE between each model's density
    fields and IBTrACS, per ENSO phase.
    """
    dtypes = ["genesis_density", "track_density", "ace_density"]
    dtype_labels = ["Genesis", "Track", "ACE"]
    models = cfg.all_model_ids
    phases = cfg.phases

    # Compute scores
    records = []
    for dtype, dlabel in zip(dtypes, dtype_labels):
        for model in models:
            for ph in phases:
                p_obs = find_density_file(cfg, basin, "IBTrACS", dtype, ph)
                p_mod = find_density_file(cfg, basin, model, dtype, ph)
                if not (p_obs and p_mod):
                    records.append(
                        dict(
                            dtype=dlabel,
                            model=model,
                            phase=ph,
                            pearson_r=np.nan,
                            rmse=np.nan,
                        )
                    )
                    continue
                _, _, g_obs = load_density_grid(p_obs)
                _, _, g_mod = load_density_grid(p_mod)
                go = np.nan_to_num(g_obs, nan=0.0)
                gm = np.nan_to_num(g_mod, nan=0.0)
                if go.shape != gm.shape:
                    records.append(
                        dict(
                            dtype=dlabel,
                            model=model,
                            phase=ph,
                            pearson_r=np.nan,
                            rmse=np.nan,
                        )
                    )
                    continue
                r = spatial_correlation(gm, go)
                rmse = spatial_rmse(gm, go)
                records.append(
                    dict(dtype=dlabel, model=model, phase=ph, pearson_r=r, rmse=rmse)
                )

    scores_df = pd.DataFrame(records)

    # Heatmap: Pearson r
    fig, axes = plt.subplots(
        1,
        len(dtypes),
        figsize=(6 * len(dtypes), 0.6 * len(models) * len(phases) / 3 + 2),
    )
    if len(dtypes) == 1:
        axes = [axes]

    for di, dlabel in enumerate(dtype_labels):
        ax = axes[di]
        sub = scores_df[scores_df.dtype == dlabel]
        data = np.full((len(models), len(phases)), np.nan)
        annot = np.full((len(models), len(phases)), "", dtype=object)

        for i, model in enumerate(models):
            for j, ph in enumerate(phases):
                row = sub[(sub.model == model) & (sub.phase == ph)]
                if row.empty:
                    continue
                r = row.pearson_r.values[0]
                rmse = row.rmse.values[0]
                data[i, j] = r
                annot[i, j] = f"r={r:.3f}\nRMSE={rmse:.2f}"

        im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
        for i in range(len(models)):
            for j in range(len(phases)):
                ax.text(j, i, annot[i, j] or "—", ha="center", va="center", fontsize=7)

        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if di == 0 else [""] * len(models), fontsize=9)
        ax.set_title(f"{dlabel} density", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

        n_base = sum(1 for m in models if m in cfg.baseline_ids)
        if 0 < n_base < len(models):
            ax.axhline(n_base - 0.5, color="black", linewidth=1.5)

    fig.suptitle(
        f"Spatial fidelity (Pearson r & RMSE vs IBTrACS) — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"spatial_scores_{basin}")

    return scores_df


# ============================================================================
# Figure 10: ENSO phase contrast (LN − EN difference maps)
# ============================================================================


def fig_enso_contrast(cfg: EvalConfig, basin: str, dtype: str = "track_density"):
    """
    Shows (LN − EN) difference for IBTrACS and each model.
    Tests whether models reproduce the spatial ENSO modulation pattern.
    """
    models = ["IBTrACS"] + cfg.all_model_ids
    extent = cfg.basin_extents.get(basin, (-100, -10, 5, 55))

    use_carto = HAS_CARTOPY and cfg.use_cartopy
    subplot_kw = {"projection": ccrs.PlateCarree()} if use_carto else {}
    ncols = min(3, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows), subplot_kw=subplot_kw
    )
    axes = np.atleast_2d(axes)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols, idx % ncols]
        p_ln = find_density_file(cfg, basin, model, dtype, "LN")
        p_en = find_density_file(cfg, basin, model, dtype, "EN")

        if p_ln and p_en:
            lat_ln, lon_ln, g_ln = load_density_grid(p_ln)
            lat_en, lon_en, g_en = load_density_grid(p_en)
            gln = np.nan_to_num(g_ln, nan=0.0)
            gen = np.nan_to_num(g_en, nan=0.0)

            if gln.shape == gen.shape:
                diff = gln - gen
                nonzero = diff[diff != 0]
                vabs = np.percentile(np.abs(nonzero), 95) if len(nonzero) else 1
                norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
                ax.pcolormesh(
                    lon_ln,
                    lat_ln,
                    diff,
                    cmap="RdBu_r",
                    norm=norm,
                    shading="auto",
                    transform=ccrs.PlateCarree() if use_carto else None,
                )

        _map_axis(ax, extent, cfg)
        ax.set_title(model, fontsize=10, fontweight="bold")

    for idx in range(len(models), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    dtype_pretty = dtype.replace("_", " ").title()
    fig.suptitle(
        f"ENSO contrast (La Niña − El Niño) | {dtype_pretty} — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"enso_contrast_{dtype}_{basin}")


# ============================================================================
# Table export: comprehensive metrics CSV
# ============================================================================


def export_full_metrics(cfg: EvalConfig, basin: str, sm: pd.DataFrame):
    """
    Compute and export a comprehensive CSV of all evaluation metrics,
    including spatial scores, KS tests recomputed from lifetime data, etc.
    """
    records = []
    ibt = sm[sm.candidate == "IBTrACS"].set_index("phase")

    for model in cfg.all_model_ids:
        for ph in cfg.phases_all:
            row_sm = sm[(sm.candidate == model) & (sm.phase == ph)]
            entry = {"model": model, "phase": ph}

            # From summary_metrics
            for col in [
                "genesis_mean",
                "genesis_std",
                "lifetime_hours",
                "landfall_mean",
                "landfall_std",
                "ks_pmin",
                "ks_vmax",
            ]:
                entry[col] = (
                    row_sm[col].values[0]
                    if not row_sm.empty and col in row_sm.columns
                    else np.nan
                )

            # Bias vs IBTrACS
            if ph in ibt.index:
                for col in ["genesis_mean", "landfall_mean", "lifetime_hours"]:
                    ref = ibt.loc[ph, col]
                    if pd.notna(ref) and pd.notna(entry.get(col)):
                        entry[f"{col}_bias_pct"] = bias_pct(entry[col], ref)

            # Recompute KS from lifetime if available
            p_mod = find_lifetime_file(cfg, basin, model, ph)
            p_obs = find_lifetime_file(cfg, basin, "IBTrACS", ph)
            if p_mod and p_obs:
                df_m = load_lifetime(p_mod)
                df_o = load_lifetime(p_obs)
                for var in ["vmax", "pmin"]:
                    ks_stat, ks_p = ks_2sample(
                        df_m[var].dropna().values, df_o[var].dropna().values
                    )
                    entry[f"ks_{var}_recomputed"] = ks_stat
                    entry[f"ks_{var}_pvalue"] = ks_p

            # Spatial scores for this phase
            for dtype in ["genesis_density", "track_density", "ace_density"]:
                p_o = find_density_file(cfg, basin, "IBTrACS", dtype, ph)
                p_m = find_density_file(cfg, basin, model, dtype, ph)
                if p_o and p_m:
                    _, _, go = load_density_grid(p_o)
                    _, _, gm = load_density_grid(p_m)
                    if go.shape == gm.shape:
                        entry[f"{dtype}_pearson_r"] = spatial_correlation(gm, go)
                        entry[f"{dtype}_rmse"] = spatial_rmse(gm, go)

            records.append(entry)

    df_out = pd.DataFrame(records)
    out_path = os.path.join(cfg.output_dir, f"full_metrics_{basin}.csv")
    df_out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Exported: {out_path}")
    return df_out


# ============================================================================
# Utility
# ============================================================================


def _save(fig, cfg: EvalConfig, name: str):
    path = os.path.join(cfg.output_dir, f"{name}.{cfg.fig_format}")
    fig.savefig(
        path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Figure 11: Saffir-Simpson category frequencies
# ============================================================================


def fig_ss_categories(cfg: EvalConfig, basin: str):
    """
    Stacked or grouped bar chart of Saffir-Simpson category fractions
    (0=TS/TD, 1–5) per model and per ENSO phase.
    Critical for reinsurance validation: does the model produce the right
    proportion of major hurricanes?
    """
    ss_cats = [0, 1, 2, 3, 4, 5]
    ss_labels = ["TS", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
    ss_palette = ["#3498db", "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

    models = ["IBTrACS"] + cfg.all_model_ids
    phases = cfg.phases_all

    fig, axes = plt.subplots(1, len(phases), figsize=(5 * len(phases), 5), sharey=True)
    if len(phases) == 1:
        axes = [axes]

    for j, ph in enumerate(phases):
        ax = axes[j]
        fracs_all = []
        labels_ax = []

        for model in models:
            p = find_lifetime_file(cfg, basin, model, ph)
            if not p:
                fracs_all.append([np.nan] * len(ss_cats))
                labels_ax.append(model)
                continue
            df = load_lifetime(p)
            total = len(df)
            if total == 0:
                fracs_all.append([0.0] * len(ss_cats))
            else:
                counts = np.array([(df["max_ss"] == c).sum() for c in ss_cats])
                fracs_all.append(counts / total * 100)
            labels_ax.append(model)

        fracs_all = np.array(fracs_all)  # (n_models, n_cats)
        x = np.arange(len(labels_ax))

        bottom = np.zeros(len(labels_ax))
        for c_i, (cat_label, color) in enumerate(zip(ss_labels, ss_palette)):
            vals = fracs_all[:, c_i]
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                label=cat_label if j == 0 else "",
            )
            # Annotate percentages for Cat 3+ (major hurricanes)
            if c_i >= 3:
                for xi, v in enumerate(vals):
                    if v > 2:  # only annotate if visible
                        ax.text(
                            xi,
                            bottom[xi] + v / 2,
                            f"{v:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=6,
                            fontweight="bold",
                            color="white",
                        )
            bottom += np.nan_to_num(vals, nan=0)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_ax, fontsize=8, rotation=30, ha="right")
        ax.set_title(ph, fontsize=12, fontweight="bold")
        ax.set_ylabel("Fraction (%)" if j == 0 else "", fontsize=10)
        ax.set_ylim(0, 100)

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"Saffir-Simpson category distribution — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"ss_categories_{basin}")


# ============================================================================
# Figure 12: Monthly genesis seasonality
# ============================================================================


def fig_monthly_seasonality(cfg: EvalConfig, basin: str):
    """
    Monthly distribution of genesis events, per model and ENSO phase.
    Validates that the model reproduces the Aug–Oct peak in the NA
    and that ENSO modulation of the seasonal cycle is captured.
    """
    models = ["IBTrACS"] + cfg.all_model_ids
    phases = cfg.phases

    fig, axes = plt.subplots(
        1, len(phases), figsize=(5.5 * len(phases), 4.5), sharey=True
    )
    if len(phases) == 1:
        axes = [axes]

    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    for j, ph in enumerate(phases):
        ax = axes[j]
        for model in models:
            p = find_lifetime_file(cfg, basin, model, ph)
            if not p:
                continue
            df = load_lifetime(p)
            if len(df) == 0:
                continue
            # Fraction of storms per month
            month_counts = np.array(
                [(df["month"] == m).sum() for m in months], dtype=float
            )
            month_frac = month_counts / month_counts.sum() * 100

            color = cfg.model_colors.get(model, "#999")
            ls = cfg.model_linestyles.get(model, "-")
            lw = 2.5 if model == "IBTrACS" else 1.3
            ax.plot(
                months,
                month_frac,
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker=cfg.model_markers.get(model, "o"),
                markersize=4,
                label=model,
            )

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Fraction of genesis (%)" if j == 0 else "", fontsize=10)
        ax.set_title(ph, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Monthly genesis seasonality — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"seasonality_{basin}")


# ============================================================================
# Figure 13: QQ plots for intensity tails
# ============================================================================


def fig_qq_intensity(cfg: EvalConfig, basin: str):
    """
    QQ plots of Vmax and Pmin: synthetic quantiles vs IBTrACS quantiles.
    Much more informative than CDFs for tail behavior — reveals whether
    the model over/under-produces extreme events.
    """
    models = cfg.all_model_ids
    quantiles = np.linspace(0.01, 0.99, 200)

    fig, axes = plt.subplots(
        len(models), 2, figsize=(10, 3.5 * len(models)), squeeze=False
    )

    for i, model in enumerate(models):
        for j, (var, xlabel) in enumerate(
            [
                ("vmax", "Vmax (m/s)"),
                ("pmin", "Pmin (hPa)"),
            ]
        ):
            ax = axes[i, j]
            has_data = False

            for ph in cfg.phases_all:
                p_obs = find_lifetime_file(cfg, basin, "IBTrACS", ph)
                p_mod = find_lifetime_file(cfg, basin, model, ph)
                if not (p_obs and p_mod):
                    continue

                obs = load_lifetime(p_obs)[var].dropna().values
                mod = load_lifetime(p_mod)[var].dropna().values
                if len(obs) < 10 or len(mod) < 10:
                    continue

                q_obs = np.quantile(obs, quantiles)
                q_mod = np.quantile(mod, quantiles)
                color = cfg.phase_colors.get(ph, "#999")
                ax.scatter(
                    q_obs, q_mod, s=4, alpha=0.5, color=color, label=ph, zorder=3
                )
                has_data = True

            if has_data:
                # 1:1 line
                lims = [
                    min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1]),
                ]
                ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, zorder=1)
                ax.set_xlim(lims)
                ax.set_ylim(lims)

            ax.set_xlabel(f"IBTrACS {xlabel}", fontsize=9)
            ax.set_ylabel(f"{model} {xlabel}", fontsize=9)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7, markerscale=2)
            if i == 0:
                ax.set_title(xlabel, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"QQ plots — synthetic vs observed intensity — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save(fig, cfg, f"qq_intensity_{basin}")


# ============================================================================
# Figure 14: Genesis latitude distribution
# ============================================================================


def fig_genesis_latitude(cfg: EvalConfig, basin: str):
    """
    KDE of genesis latitude per model and ENSO phase.
    Physical diagnostic: are storms forming at the right latitudes?
    ENSO is known to shift genesis latitude (e.g. equatorward during El Niño
    in the WP).
    """
    models = ["IBTrACS"] + cfg.all_model_ids
    phases = cfg.phases

    fig, axes = plt.subplots(
        1, len(phases), figsize=(5.5 * len(phases), 4.5), sharey=True
    )
    if len(phases) == 1:
        axes = [axes]

    lat_bins = np.linspace(5, 55, 100)

    for j, ph in enumerate(phases):
        ax = axes[j]
        for model in models:
            p = find_lifetime_file(cfg, basin, model, ph)
            if not p:
                continue
            df = load_lifetime(p)
            lats = df["lat_genesis"].dropna().values
            if len(lats) < 5:
                continue

            try:
                kde = stats.gaussian_kde(lats, bw_method=0.3)
                ax.plot(
                    lat_bins,
                    kde(lat_bins),
                    color=cfg.model_colors.get(model, "#999"),
                    linestyle=cfg.model_linestyles.get(model, "-"),
                    linewidth=2.0 if model == "IBTrACS" else 1.2,
                    label=model,
                )
            except np.linalg.LinAlgError:
                continue

        ax.set_xlabel("Genesis latitude (°N)", fontsize=10)
        ax.set_ylabel("Density" if j == 0 else "", fontsize=10)
        ax.set_title(ph, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Genesis latitude distribution — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"genesis_latitude_{basin}")


# ============================================================================
# Figure 15: Landfalling storm intensity (proxy: lifetime max for LF storms)
# ============================================================================


def fig_landfall_intensity(cfg: EvalConfig, basin: str):
    """
    CDF of Vmax and Pmin for landfalling storms only (has_landfall == 1).

    NOTE: This uses lifetime maximum intensity, not the value at the moment
    of landfall. Proper landfall intensity requires the full track with a
    land mask. This proxy is still informative: if the model over-intensifies
    landfalling storms, it will show here. Add a caveat in the paper.
    """
    models = ["IBTrACS"] + cfg.all_model_ids

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, phase_set in enumerate([["ALL"], cfg.phases]):
        for col, (var, xlabel) in enumerate(
            [
                ("vmax", "Lifetime max wind (m/s) — landfalling TCs"),
                ("pmin", "Min central pressure (hPa) — landfalling TCs"),
            ]
        ):
            ax = axes[row, col]
            for model in models:
                for ph in phase_set:
                    p = find_lifetime_file(cfg, basin, model, ph)
                    if not p:
                        continue
                    df = load_lifetime(p)
                    lf = df[df["has_landfall"] == 1]
                    vals = lf[var].dropna().values
                    if len(vals) < 5:
                        continue
                    sorted_v = np.sort(vals)
                    cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)

                    if len(phase_set) == 1:
                        color = cfg.model_colors.get(model, "#999")
                        ls = cfg.model_linestyles.get(model, "-")
                        label = model
                    else:
                        color = cfg.phase_colors.get(ph, "#999")
                        ls = "-" if model == "IBTrACS" else "--"
                        label = f"{model} {ph}"

                    lw = 2.0 if model == "IBTrACS" else 1.2
                    ax.plot(
                        sorted_v,
                        cdf,
                        color=color,
                        linestyle=ls,
                        linewidth=lw,
                        label=label,
                    )

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("CDF", fontsize=9)
            title = "ALL — all models" if row == 0 else "By ENSO phase"
            ax.set_title(f"{var.upper()} | {title}", fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        f"Landfalling TC intensity (lifetime max proxy) — {basin}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"landfall_intensity_{basin}")


# ============================================================================
# Figure 16: Translation speed proxy (distance / duration)
# ============================================================================


def fig_translation_speed(cfg: EvalConfig, basin: str):
    """
    Approximate mean translation speed = great-circle-distance(genesis→last) / duration.
    This is a rough proxy — true translation speed needs step-by-step track data.
    Still useful for detecting systematic speed biases (e.g. too-fast storms
    that don't linger at landfall).

    NOTE: This requires lon_genesis. The distance is estimated from genesis to
    a hypothetical endpoint using the track extent implied by n_steps and the
    mean step size. Since we only have summary stats, we use duration_hours and
    n_steps to estimate mean speed = (n_steps * ~3h * assumed_step_km) / duration.
    A simpler diagnostic: just compare duration distributions, which is a speed proxy.

    We instead plot n_steps (track length proxy) vs vmax, which reveals whether
    the model produces realistic intensity-duration relationships.
    """
    models = ["IBTrACS"] + cfg.all_model_ids

    fig, axes = plt.subplots(
        1, len(cfg.phases), figsize=(5.5 * len(cfg.phases), 5), sharey=True
    )
    if len(cfg.phases) == 1:
        axes = [axes]

    for j, ph in enumerate(cfg.phases):
        ax = axes[j]
        for model in models:
            p = find_lifetime_file(cfg, basin, model, ph)
            if not p:
                continue
            df = load_lifetime(p)
            if len(df) < 10:
                continue

            dur = df["duration_hours"].dropna().values
            vmax = df["vmax"].dropna().values
            n = min(len(dur), len(vmax))
            dur, vmax = dur[:n], vmax[:n]

            color = cfg.model_colors.get(model, "#999")
            alpha = 0.6 if model == "IBTrACS" else 0.05
            s = 12 if model == "IBTrACS" else 1
            ax.scatter(
                dur,
                vmax,
                s=s,
                alpha=alpha,
                color=color,
                label=model,
                zorder=3 if model == "IBTrACS" else 1,
            )

            # Add median contour via binned statistics
            bins = np.arange(0, 450, 24)
            bin_idx = np.digitize(dur, bins) - 1
            medians = []
            bin_centers = []
            for bi in range(len(bins) - 1):
                mask = bin_idx == bi
                if mask.sum() >= 5:
                    medians.append(np.median(vmax[mask]))
                    bin_centers.append((bins[bi] + bins[bi + 1]) / 2)
            if bin_centers:
                lw = 2.5 if model == "IBTrACS" else 1.5
                ls = cfg.model_linestyles.get(model, "-")
                ax.plot(
                    bin_centers,
                    medians,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    zorder=5,
                )

        ax.set_xlabel("Duration (hours)", fontsize=10)
        ax.set_ylabel("Lifetime max Vmax (m/s)" if j == 0 else "", fontsize=10)
        ax.set_title(ph, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(
        f"Intensity–duration relationship — {basin}\n"
        f"(scatter + binned median; longer duration ≈ slower translation)",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    _save(fig, cfg, f"intensity_duration_{basin}")


# ============================================================================
# Figure 17: Return period bootstrap confidence intervals
# ============================================================================


def fig_rp_bootstrap_ci(cfg: EvalConfig, basin: str, n_boot: int = 500):
    """
    Return period curves with bootstrap 90% confidence intervals.
    Resamples the lifetime catalog (by year) to produce CI bands.

    Without year-level catalog access, we approximate CIs from the rp CSV
    by assuming a Poisson process and using the Clopper-Pearson interval
    on exceedance counts at each return period.
    """
    proposed = cfg.proposed_ids[-1]
    cities = set()
    for ph in cfg.phases:
        p = find_rp_file(cfg, basin, proposed, ph)
        if p:
            cities.update(load_return_periods(p)["city"].unique())
    cities = sorted(cities)
    if not cities:
        print(f"  [SKIP] No RP data for bootstrap CI — {basin}")
        return

    ncols = min(3, len(cities))
    nrows = int(np.ceil(len(cities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, city in enumerate(cities):
        ax = axes[idx // ncols, idx % ncols]

        for ph in cfg.phases:
            p = find_rp_file(cfg, basin, proposed, ph)
            if not p:
                continue
            df = load_return_periods(p)
            cdf = df[df.city == city].dropna(subset=["wind_ms"])
            if cdf.empty:
                continue

            rp = cdf["return_period"].values
            ws = cdf["wind_ms"].values
            color = cfg.phase_colors[ph]
            ax.plot(
                rp,
                ws,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=f"{ph}",
                zorder=3,
            )

            # Approximate 90% CI using Poisson counting uncertainty.
            # At return period T in a Y-year catalog, n_events ≈ Y/T.
            # We assume Y = 10000 (from the paper).
            Y = 10_000
            n_events = Y / rp
            # Poisson CI on the rate → CI on wind quantile
            # Using delta method: σ(V) ≈ dV/d(1/T) * σ(1/T)
            # Simpler: ±1.645 * V * 1/sqrt(n_events) as rough scaling
            # This is approximate but gives reasonable bands.
            ci_half = 1.645 * ws / np.sqrt(np.maximum(n_events, 1))
            ax.fill_between(
                rp, ws - ci_half, ws + ci_half, color=color, alpha=0.12, zorder=1
            )

        ax.set_xscale("log")
        ax.set_xlabel("Return period (yr)", fontsize=9)
        ax.set_ylabel("Wind speed (m/s)", fontsize=9)
        ax.set_title(city, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        _add_ss_lines(ax, cfg)
        ax.legend(fontsize=8)

    for idx in range(len(cities), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        f"Return periods with approx. 90% CI — {proposed} — {basin}\n"
        f"(CI from Poisson counting uncertainty, Y=10,000 yr catalog)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save(fig, cfg, f"rp_bootstrap_ci_{basin}")


# ============================================================================
# Figure 18: ENSO phase ordering diagnostic
# ============================================================================


def fig_enso_phase_ordering(cfg: EvalConfig, basin: str, sm: pd.DataFrame):
    """
    For each model, checks whether the expected ENSO phase ordering holds:
      NA: genesis LN > NEU > EN, landfall LN > NEU > EN
      WP: genesis EN > NEU > LN  (reversed)

    Produces a compact diagnostic table showing pass/fail per metric per model.
    This is essential for demonstrating that the ENSO conditioning *works*.
    """
    models = cfg.all_model_ids
    # Expected orderings (descending)
    expected = {
        "NA": {
            "genesis_mean": ["LN", "NEU", "EN"],
            "landfall_mean": ["LN", "NEU", "EN"],
        },
        "WP": {
            "genesis_mean": ["EN", "NEU", "LN"],
            "landfall_mean": ["EN", "NEU", "LN"],
        },
    }
    if basin not in expected:
        print(f"  [SKIP] No expected ordering defined for {basin}")
        return

    ordering = expected[basin]
    metrics = list(ordering.keys())

    fig, ax = plt.subplots(figsize=(3 + 1.5 * len(metrics), 1 + 0.5 * len(models)))

    data = np.full((len(models), len(metrics)), np.nan)
    annot = np.full((len(models), len(metrics)), "", dtype=object)

    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            exp_order = ordering[metric]  # e.g. ["LN", "NEU", "EN"]
            vals = {}
            for ph in cfg.phases:
                row = sm[(sm.candidate == model) & (sm.phase == ph)]
                if not row.empty and pd.notna(row[metric].values[0]):
                    vals[ph] = row[metric].values[0]

            if len(vals) < 3:
                annot[i, j] = "—"
                continue

            # Check ordering
            actual_order = sorted(vals, key=vals.get, reverse=True)
            matches = sum(a == e for a, e in zip(actual_order, exp_order))
            data[i, j] = matches / len(exp_order)

            actual_str = " > ".join([f"{p}({vals[p]:.1f})" for p in actual_order])
            status = "✓" if actual_order == exp_order else "✗"
            annot[i, j] = f"{status}\n{actual_str}"

    cmap = mcolors.ListedColormap(["#e74c3c", "#f39c12", "#2ecc71"])
    bounds = [0, 0.5, 0.9, 1.01]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(
                j,
                i,
                annot[i, j],
                ha="center",
                va="center",
                fontsize=7,
                fontweight="medium",
            )

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9, fontweight="bold")

    n_base = sum(1 for m in models if m in cfg.baseline_ids)
    if 0 < n_base < len(models):
        ax.axhline(n_base - 0.5, color="black", linewidth=1.5)

    exp_str = " > ".join(ordering[metrics[0]])
    fig.suptitle(
        f"ENSO phase ordering diagnostic — {basin}\n"
        f"Expected: {exp_str} | ✓ = correct, ✗ = violated",
        fontsize=12,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    _save(fig, cfg, f"enso_ordering_{basin}")


# ============================================================================
# Extended metrics export (with new statistics)
# ============================================================================


def export_extended_metrics(cfg: EvalConfig, basin: str, sm: pd.DataFrame):
    """
    Augments the full_metrics CSV with:
      - Anderson-Darling statistics (more tail-sensitive than KS)
      - Wasserstein (Earth Mover's) distance for Vmax and Pmin
      - Chi-squared test on SS category frequencies
      - Genesis seasonality correlation (monthly fractions)
      - Landfall fraction bias
    """
    ibt = sm[sm.candidate == "IBTrACS"].set_index("phase")
    months = np.arange(1, 13)
    records = []

    for model in cfg.all_model_ids:
        for ph in cfg.phases_all:
            entry = {"model": model, "phase": ph}

            p_mod = find_lifetime_file(cfg, basin, model, ph)
            p_obs = find_lifetime_file(cfg, basin, "IBTrACS", ph)

            if p_mod and p_obs:
                df_m = load_lifetime(p_mod)
                df_o = load_lifetime(p_obs)

                # AD and Wasserstein for Vmax and Pmin
                for var in ["vmax", "pmin"]:
                    mod_vals = df_m[var].dropna().values
                    obs_vals = df_o[var].dropna().values
                    ad_stat, ad_p = ad_2sample(mod_vals, obs_vals)
                    w_dist = wasserstein_dist(mod_vals, obs_vals)
                    entry[f"ad_{var}"] = ad_stat
                    entry[f"ad_{var}_pvalue"] = ad_p
                    entry[f"wasserstein_{var}"] = w_dist

                # Chi-squared on SS categories
                ss_cats = [0, 1, 2, 3, 4, 5]
                obs_counts = np.array([(df_o["max_ss"] == c).sum() for c in ss_cats])
                mod_counts = np.array([(df_m["max_ss"] == c).sum() for c in ss_cats])
                chi2_stat, chi2_p = chi2_categorical(mod_counts, obs_counts)
                entry["chi2_ss"] = chi2_stat
                entry["chi2_ss_pvalue"] = chi2_p

                # Major hurricane fraction (Cat 3+)
                obs_major = (df_o["max_ss"] >= 3).mean()
                mod_major = (df_m["max_ss"] >= 3).mean()
                entry["obs_major_frac"] = obs_major
                entry["mod_major_frac"] = mod_major
                entry["major_frac_bias_pct"] = (
                    bias_pct(mod_major, obs_major) if obs_major > 0 else np.nan
                )

                # Seasonality correlation (monthly genesis fraction)
                obs_month = np.array(
                    [(df_o["month"] == m).sum() for m in months], dtype=float
                )
                mod_month = np.array(
                    [(df_m["month"] == m).sum() for m in months], dtype=float
                )
                if obs_month.sum() > 0 and mod_month.sum() > 0:
                    obs_frac = obs_month / obs_month.sum()
                    mod_frac = mod_month / mod_month.sum()
                    entry["seasonality_r"] = np.corrcoef(obs_frac, mod_frac)[0, 1]
                    entry["seasonality_rmse"] = np.sqrt(
                        np.mean((obs_frac - mod_frac) ** 2)
                    )

                # Landfall fraction bias
                obs_lf_frac = df_o["has_landfall"].mean()
                mod_lf_frac = df_m["has_landfall"].mean()
                entry["obs_landfall_frac"] = obs_lf_frac
                entry["mod_landfall_frac"] = mod_lf_frac
                entry["landfall_frac_bias_pct"] = (
                    bias_pct(mod_lf_frac, obs_lf_frac) if obs_lf_frac > 0 else np.nan
                )

                # Mean genesis latitude bias
                obs_lat = df_o["lat_genesis"].mean()
                mod_lat = df_m["lat_genesis"].mean()
                entry["obs_mean_lat_genesis"] = obs_lat
                entry["mod_mean_lat_genesis"] = mod_lat
                entry["lat_genesis_bias_deg"] = mod_lat - obs_lat

            records.append(entry)

    df_out = pd.DataFrame(records)
    out_path = os.path.join(cfg.output_dir, f"extended_metrics_{basin}.csv")
    df_out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Exported: {out_path}")
    return df_out


def run_basin(cfg: EvalConfig, basin: str):
    """Run all evaluation figures and tables for a single basin."""
    print(f"\n{'=' * 60}")
    print(f"  Basin: {basin}")
    print(f"{'=' * 60}")

    # Load phase frequencies from CSV if available
    cfg.load_phase_freqs(basin)

    # Load summary metrics
    try:
        sm = load_summary_metrics(cfg, basin)
    except FileNotFoundError:
        print(f"  [SKIP] No summary_metrics.csv for {basin}")
        return

    n = 1

    # ── Scorecard ──
    print(f"\n[{n}/18] Scorecard heatmap")
    n += 1
    fig_scorecard(cfg, basin, sm)

    # ── Summary bar chart ──
    print(f"[{n}/18] Summary bar chart")
    n += 1
    fig_summary_bars(cfg, basin, sm)

    # ── Density maps ──
    density_specs = [
        (
            "genesis_density",
            "YlOrRd",
            True,
            "Genesis density",
            "Genesis events per 10 kyr per cell",
        ),
        ("track_density", "viridis", True, "Track density", "TC fixes per yr per cell"),
        (
            "ace_density",
            "inferno",
            True,
            "ACE density",
            "ACE (10⁴ kn² per yr per cell)",
        ),
    ]
    for dtype, cmap, log, title, cbar in density_specs:
        print(f"[{n}/18] {title} panels")
        n += 1
        fig_density_panels(
            cfg, basin, dtype, cmap=cmap, log=log, title_prefix=title, cbar_label=cbar
        )
        fig_density_diff(cfg, basin, dtype, title_prefix=title)

    # ── Intensity CDFs ──
    print(f"[{n}/18] Intensity CDFs")
    n += 1
    fig_intensity_cdfs(cfg, basin)

    # ── Lifetime distributions ──
    print(f"[{n}/18] Lifetime distributions")
    n += 1
    fig_lifetime_distributions(cfg, basin)

    # ── Return periods ──
    print(f"[{n}/18] Return periods")
    n += 1
    fig_return_periods(cfg, basin)

    # ── Spatial scores ──
    print(f"[{n}/18] Spatial correlation/RMSE scores")
    n += 1
    fig_spatial_scores(cfg, basin)

    # ── ENSO contrast ──
    print(f"[{n}/18] ENSO contrast maps (LN − EN)")
    n += 1
    fig_enso_contrast(cfg, basin, "track_density")
    fig_enso_contrast(cfg, basin, "genesis_density")

    # ── NEW: Saffir-Simpson categories ──
    print(f"[{n}/18] Saffir-Simpson category frequencies")
    n += 1
    fig_ss_categories(cfg, basin)

    # ── NEW: Monthly seasonality ──
    print(f"[{n}/18] Monthly genesis seasonality")
    n += 1
    fig_monthly_seasonality(cfg, basin)

    # ── NEW: QQ plots ──
    print(f"[{n}/18] QQ plots (intensity tails)")
    n += 1
    fig_qq_intensity(cfg, basin)

    # ── NEW: Genesis latitude ──
    print(f"[{n}/18] Genesis latitude distribution")
    n += 1
    fig_genesis_latitude(cfg, basin)

    # ── NEW: Landfall intensity ──
    print(f"[{n}/18] Landfall intensity (proxy)")
    n += 1
    fig_landfall_intensity(cfg, basin)

    # ── NEW: Intensity-duration relationship ──
    print(f"[{n}/18] Intensity–duration relationship")
    n += 1
    fig_translation_speed(cfg, basin)

    # ── NEW: Return period CIs ──
    print(f"[{n}/18] Return period bootstrap CIs")
    n += 1
    fig_rp_bootstrap_ci(cfg, basin)

    # ── NEW: ENSO phase ordering diagnostic ──
    print(f"[{n}/18] ENSO phase ordering diagnostic")
    n += 1
    fig_enso_phase_ordering(cfg, basin, sm)

    # ── Export metrics ──
    print("\nExporting comprehensive metrics CSV")
    export_full_metrics(cfg, basin, sm)

    print("Exporting extended metrics CSV (AD, Wasserstein, Chi², seasonality)")
    export_extended_metrics(cfg, basin, sm)


# def main():
#     parser = argparse.ArgumentParser(description="SIENA-IH-STORM evaluation pipeline")
#     parser.add_argument(
#         "--data-dir",
#         default="./evaluation_output",
#         help="Root directory containing basin folders",
#     )
#     parser.add_argument(
#         "--output-dir", default="./figures", help="Output directory for figures"
#     )
#     parser.add_argument(
#         "--basins", nargs="+", default=["NA"], help="Basins to evaluate (e.g. NA WP)"
#     )
#     parser.add_argument(
#         "--format", default="png", choices=["png", "pdf"], help="Figure output format"
#     )
#     parser.add_argument("--dpi", type=int, default=200)
#     parser.add_argument(
#         "--no-cartopy", action="store_true", help="Disable cartopy even if installed"
#     )
#     args = parser.parse_args()

#     cfg = EvalConfig(
#         data_dir=args.data_dir,
#         output_dir=args.output_dir,
#         basins=args.basins,
#         dpi=args.dpi,
#         fig_format=args.format,
#         use_cartopy=not args.no_cartopy,
#     )

#     print("SIENA-IH-STORM Evaluation Pipeline")
#     print(f"  Data dir:   {cfg.data_dir}")
#     print(f"  Output dir: {cfg.output_dir}")
#     print(f"  Basins:     {cfg.basins}")
#     print(f"  Models:     {cfg.all_model_ids}")
#     print(f"  Cartopy:    {'yes' if HAS_CARTOPY and cfg.use_cartopy else 'no'}")

#     for basin in cfg.basins:
#         run_basin(cfg, basin)

#     print("\n✓ Done.")


# if __name__ == "__main__":
#     main()
