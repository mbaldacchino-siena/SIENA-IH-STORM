"""
visualization.py  —  Publication-quality spatial plots for SIENA-IH-STORM
==========================================================================

Merges and fixes VALIDATION_spatial.py + VALIDATION_difference.py.

Plot types
----------
  1. Absolute density maps     — genesis, track, ACE  (per phase + IBTrACS)
  2. Difference maps           — new−IBTrACS, new−old  (3-row triplet)
  3. LMI distribution          — overlaid per-storm Vmax histograms
  4. Wind-pressure scatter     — with WPR reference curves
  5. Intensity CDFs            — Pmin & Vmax per phase

Uses evaluation.py's `load_catalog()` for data loading — works with both
synthetic catalogs and IBTrACS converted via ibtracs_to_storm.py.

Wind convention
---------------
STORM files store 10-min sustained winds (m/s).  This module keeps that
convention internally.  For plots where 1-min thresholds matter (SS
categories on LMI axes), conversions are applied *only at display time*
and clearly labeled.

Requires: numpy, pandas, matplotlib
Optional: cartopy (for coastlines / land shading — strongly recommended)

Author: SIENA-IH-STORM evaluation framework
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    warnings.warn(
        "cartopy not installed — maps will not show coastlines. "
        "Install with: pip install cartopy"
    )

# Import evaluation.py helpers
from POST_PROCESSING.utils.evaluation import (
    load_catalog,
    BASIN_BOUNDS as _EVAL_BOUNDS,
    BASIN_ID_MAP,
    _per_storm_agg,
)

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

# Basin bounds: (lat0, lat1, lon0, lon1) — matching preprocessing.py
BASIN_BOUNDS = {
    "EP": (5, 60, 180, 285),
    "NA": (5, 60, 255, 360),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}

# Slightly expanded plot extents for cartopy
BASIN_PLOT_EXTENT = {
    "EP": [175, 290, 0, 65],
    "NA": [250, 365, 0, 65],
    "NI": [25, 105, 0, 40],
    "SI": [5, 140, -65, 0],
    "SP": [130, 245, -65, 0],
    "WP": [95, 185, 0, 65],
}

PHASE_COLORS = {
    "EN": "#E53935",
    "NEU": "#212121",
    "LN": "#1E88E5",
    "ALL": "#4CAF50",
    "IBTrACS": "#F9A825",
}
PHASE_LABELS = {
    "EN": "El Niño",
    "NEU": "Neutral",
    "LN": "La Niña",
    "ALL": "All phases",
    "IBTrACS": "IBTrACS",
}

# Saffir-Simpson thresholds in 10-min m/s (for annotation on plots)
SS_THRESH_10MIN = {"Cat 1": 33, "Cat 2": 43, "Cat 3": 50, "Cat 4": 58, "Cat 5": 70}
# ... and in 1-min knots (for axis labeling when converting)
SS_THRESH_1MIN_KN = {
    "TS": 34,
    "Cat 1": 64,
    "Cat 2": 83,
    "Cat 3": 96,
    "Cat 4": 113,
    "Cat 5": 137,
}


# ═══════════════════════════════════════════════════════════════════════
# GRID COMPUTATION
# ═══════════════════════════════════════════════════════════════════════


def compute_density_grid(
    df: pd.DataFrame,
    basin: str,
    mode: str = "track",
    resolution: float = 1.0,
    n_years: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin catalog data onto a regular lat/lon grid.

    Parameters
    ----------
    df : catalog DataFrame (output of evaluation.load_catalog)
    basin : basin code
    mode : "track" (all points), "genesis" (first timestep per storm), "ace"
    resolution : grid cell size in degrees
    n_years : normalization factor (total simulated years)

    Returns
    -------
    grid : 2D array (n_lat_cells × n_lon_cells), normalized per year
    lat_edges, lon_edges : 1D edge arrays
    """
    lat0, lat1, lon0, lon1 = BASIN_BOUNDS[basin]
    lat_edges = np.arange(lat0, lat1 + resolution, resolution)
    lon_edges = np.arange(lon0, lon1 + resolution, resolution)

    if mode == "genesis":
        # First timestep of each storm
        if "timestep" in df.columns:
            sub = df[df["timestep"] == 0]
        elif "global_storm_uid" in df.columns:
            sub = df.drop_duplicates("global_storm_uid", keep="first")
        else:
            sub = df.drop_duplicates("storm_id", keep="first")
        lats, lons = sub["lat"].values, sub["lon"].values
        grid, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])

    elif mode == "ace":
        lats, lons = df["lat"].values, df["lon"].values
        wind_col = "wind" if "wind" in df.columns else "wind_ms"
        wind_kt = df[wind_col].values * 1.94384
        weights = wind_kt**2 * 1e-4  # standard ACE units
        grid, _, _ = np.histogram2d(
            lats, lons, bins=[lat_edges, lon_edges], weights=weights
        )

    else:  # track density
        lats, lons = df["lat"].values, df["lon"].values
        grid, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])

    grid = grid / max(n_years, 1.0)
    return grid, lat_edges, lon_edges


# ═══════════════════════════════════════════════════════════════════════
# MAP HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _make_map_axes(n_panels, basin, figwidth=16):
    """Create a row of map subplots with optional cartopy."""
    ncols = min(4, n_panels) if n_panels > 2 else n_panels
    nrows = int(np.ceil(n_panels / ncols))
    height = figwidth / ncols * 0.6 * nrows

    kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figwidth, height), subplot_kw=kw, squeeze=False
    )
    axes_flat = axes.flatten()
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)
    return fig, axes_flat[:n_panels]


def _dress_map_axis(ax, basin):
    """Add coastlines, land, and set extent."""
    extent = BASIN_PLOT_EXTENT.get(basin)
    if HAS_CARTOPY:
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
    elif extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])


def _pcolormesh(ax, grid, lat_edges, lon_edges, **kwargs):
    """Plot pcolormesh with or without cartopy transform."""
    grid_plot = np.ma.masked_where(np.abs(grid) < 1e-12, grid)
    transform_kw = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    return ax.pcolormesh(
        lon_edges, lat_edges, grid_plot, shading="flat", **transform_kw, **kwargs
    )


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1: ABSOLUTE DENSITY MAPS  (genesis / track / ACE)
# ═══════════════════════════════════════════════════════════════════════


def plot_density_panels(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    n_years_dict: Dict[str, float],
    mode: str = "track",
    resolution: float = 1.0,
    outdir: str = ".",
    cmap: Optional[str] = None,
    figwidth: float = 16,
) -> plt.Figure:
    """
    Side-by-side density maps for multiple datasets/phases.

    Parameters
    ----------
    datasets : {"EN": df, "NEU": df, "LN": df, "IBTrACS": df, ...}
    n_years_dict : {"EN": 10000, "IBTrACS": 42, ...}
    mode : "genesis", "track", or "ace"
    """
    if cmap is None:
        cmap = {"genesis": "YlOrRd", "track": "YlGnBu", "ace": "hot_r"}.get(
            mode, "YlOrRd"
        )
    unit = {
        "genesis": "storms/yr/cell",
        "track": "fixes/yr/cell",
        "ace": "ACE/yr/cell",
    }.get(mode, "")
    mode_label = {
        "genesis": "Genesis Density",
        "track": "Track Density",
        "ace": "ACE Density",
    }.get(mode, mode)

    n = len(datasets)
    fig, axes = _make_map_axes(n, basin, figwidth)

    grids = {}
    for label, df in datasets.items():
        g, lat_e, lon_e = compute_density_grid(
            df,
            basin,
            mode=mode,
            resolution=resolution,
            n_years=n_years_dict.get(label, 1.0),
        )
        grids[label] = g

    # Common vmax from 95th percentile of non-zero values
    all_vals = np.concatenate([g[g > 0] for g in grids.values() if np.any(g > 0)])
    vmax = np.percentile(all_vals, 95) if len(all_vals) > 0 else 1.0

    for ax, label in zip(axes, datasets.keys()):
        _dress_map_axis(ax, basin)
        im = _pcolormesh(ax, grids[label], lat_e, lon_e, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(PHASE_LABELS.get(label, label), fontsize=11)

    plt.colorbar(im, ax=list(axes), shrink=0.6, label=unit, pad=0.02)
    fig.suptitle(f"{mode_label} — {basin} ({resolution}° grid)", fontsize=14, y=1.02)
    try:
        fig.tight_layout()
    except Exception:
        pass

    outpath = os.path.join(outdir, f"validation_{mode}_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2: DIFFERENCE MAPS  (3-row triplet)
# ═══════════════════════════════════════════════════════════════════════


def plot_difference_triplet(
    datasets_new: Dict[str, pd.DataFrame],
    dataset_ref: pd.DataFrame,
    basin: str,
    n_years_new: float,
    n_years_ref: float,
    mode: str = "track",
    resolution: float = 2.0,
    datasets_old: Optional[Dict[str, pd.DataFrame]] = None,
    n_years_old: Optional[float] = None,
    outdir: str = ".",
    diff_cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Three-row publication figure:
      Row 1: Absolute panels  (IBTrACS + per-phase new [+ per-phase old])
      Row 2: New − IBTrACS    (per phase)
      Row 3: New − Old        (per phase, if old provided)

    Parameters
    ----------
    datasets_new : {"EN": df, "NEU": df, "LN": df}
    dataset_ref : IBTrACS catalog DataFrame
    datasets_old : optional old model catalogs {"EN": df, ...}
    mode : "genesis", "track", or "ace"
    """
    mode_label = {
        "genesis": "Genesis Density",
        "track": "Track Density",
        "ace": "ACE Density",
    }.get(mode, mode)
    abs_cmap = {"genesis": "YlOrRd", "track": "YlGnBu", "ace": "hot_r"}.get(
        mode, "YlOrRd"
    )
    unit = {
        "genesis": "storms/yr/cell",
        "track": "fixes/yr/cell",
        "ace": "ACE/yr/cell",
    }.get(mode, "")

    phases = [ph for ph in ["EN", "NEU", "LN"] if ph in datasets_new]
    has_old = datasets_old is not None and len(datasets_old) > 0

    # ── Compute all grids ──
    grids = {}
    g_ref, lat_e, lon_e = compute_density_grid(
        dataset_ref, basin, mode, resolution, n_years_ref
    )
    grids["ibtracs"] = g_ref

    for ph in phases:
        g, _, _ = compute_density_grid(
            datasets_new[ph], basin, mode, resolution, n_years_new
        )
        grids[f"new_{ph}"] = g

    if has_old:
        for ph in phases:
            if ph in datasets_old:
                g, _, _ = compute_density_grid(
                    datasets_old[ph], basin, mode, resolution, n_years_old
                )
                grids[f"old_{ph}"] = g

    # ── Layout ──
    ncols = max(1 + len(phases), len(phases))  # ibtracs + phases
    if has_old:
        ncols = max(ncols, 1 + 2 * len(phases))  # also old panels
    nrows = 2 + (1 if has_old else 0)

    kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.2 * nrows + 0.8),
        subplot_kw=kw,
        squeeze=False,
    )

    # ── Row 0: Absolute panels ──
    abs_panels = [("ibtracs", "IBTrACS")]
    for ph in phases:
        abs_panels.append((f"new_{ph}", f"SIENA {PHASE_LABELS.get(ph, ph)}"))
    if has_old:
        for ph in phases:
            if f"old_{ph}" in grids:
                abs_panels.append((f"old_{ph}", f"Old {PHASE_LABELS.get(ph, ph)}"))

    abs_grids = [grids[k] for k, _ in abs_panels if k in grids]
    all_pos = (
        np.concatenate([g[g > 0] for g in abs_grids]) if abs_grids else np.array([1])
    )
    vmax_abs = np.percentile(all_pos, 99) if len(all_pos) > 0 else 1.0

    for i, (key, label) in enumerate(abs_panels):
        if i >= ncols:
            break
        _dress_map_axis(axes[0, i], basin)
        if key in grids:
            im0 = _pcolormesh(
                axes[0, i],
                grids[key],
                lat_e,
                lon_e,
                cmap=abs_cmap,
                vmin=0,
                vmax=vmax_abs,
            )
        axes[0, i].set_title(label, fontsize=9)
    plt.colorbar(im0, ax=axes[0, :].tolist(), shrink=0.6, label=unit, pad=0.02)
    for i in range(len(abs_panels), ncols):
        axes[0, i].set_visible(False)

    # ── Row 1: New − IBTrACS ──
    diff1 = []
    for ph in phases:
        key = f"new_{ph}"
        if key in grids:
            shape = (
                min(grids[key].shape[0], g_ref.shape[0]),
                min(grids[key].shape[1], g_ref.shape[1]),
            )
            d = grids[key][: shape[0], : shape[1]] - g_ref[: shape[0], : shape[1]]
            diff1.append((d, f"SIENA−IBTrACS ({PHASE_LABELS.get(ph, ph)})"))

    vlim1 = (
        max(
            np.percentile(np.abs(d[d != 0]), 99) if np.any(d != 0) else 0.01
            for d, _ in diff1
        )
        if diff1
        else 0.01
    )
    vlim1 = max(vlim1, 0.01)

    for i, (d, label) in enumerate(diff1):
        _dress_map_axis(axes[1, i], basin)
        im1 = _pcolormesh(
            axes[1, i],
            d,
            lat_e,
            lon_e,
            cmap=diff_cmap,
            norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim1, vmax=vlim1),
        )
        axes[1, i].set_title(label, fontsize=9)
    plt.colorbar(im1, ax=axes[1, :].tolist(), shrink=0.6, label=f"Δ {unit}", pad=0.02)
    for i in range(len(diff1), ncols):
        axes[1, i].set_visible(False)

    # ── Row 2: New − Old ──
    if has_old:
        diff2 = []
        for ph in phases:
            k_new, k_old = f"new_{ph}", f"old_{ph}"
            if k_new in grids and k_old in grids:
                shape = (
                    min(grids[k_new].shape[0], grids[k_old].shape[0]),
                    min(grids[k_new].shape[1], grids[k_old].shape[1]),
                )
                d = (
                    grids[k_new][: shape[0], : shape[1]]
                    - grids[k_old][: shape[0], : shape[1]]
                )
                diff2.append((d, f"SIENA−Old ({PHASE_LABELS.get(ph, ph)})"))

        if diff2:
            vlim2 = max(
                np.percentile(np.abs(d[d != 0]), 99) if np.any(d != 0) else 0.01
                for d, _ in diff2
            )
            vlim2 = max(vlim2, 0.01)
            for i, (d, label) in enumerate(diff2):
                _dress_map_axis(axes[2, i], basin)
                im2 = _pcolormesh(
                    axes[2, i],
                    d,
                    lat_e,
                    lon_e,
                    cmap=diff_cmap,
                    norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim2, vmax=vlim2),
                )
                axes[2, i].set_title(label, fontsize=9)
            plt.colorbar(
                im2, ax=axes[2, :].tolist(), shrink=0.6, label=f"Δ {unit}", pad=0.02
            )
        for i in range(len(diff2), ncols):
            axes[2, i].set_visible(False)

    fig.suptitle(f"{mode_label} — {basin} ({resolution}° grid)", fontsize=14, y=1.01)
    try:
        fig.tight_layout()
    except Exception:
        pass
    safe = mode_label.lower().replace(" ", "_")
    outpath = os.path.join(outdir, f"diff_{safe}_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3: LMI DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════


def plot_lmi_distribution(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    outdir: str = ".",
    wind_convention: str = "10min_ms",
) -> plt.Figure:
    """
    Per-storm lifetime maximum intensity (Vmax) histogram, overlaid.

    Parameters
    ----------
    datasets : {"EN": df, "IBTrACS": df, ...}
    wind_convention : "10min_ms" (STORM default) or "1min_kn"
                      If 10min_ms, SS thresholds are annotated in 10-min units.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    wind_col = "wind" if "wind" in list(datasets.values())[0].columns else "wind_ms"
    bins = np.arange(15, 95, 2.5)

    for label, df in datasets.items():
        if "global_storm_uid" in df.columns:
            lmi = df.groupby("global_storm_uid")[wind_col].max().dropna()
        elif "storm_uid" in df.columns:
            lmi = df.groupby("storm_uid")[wind_col].max().dropna()
        else:
            storms = _per_storm_agg(df)
            lmi = storms["vmax"].dropna()

        color = PHASE_COLORS.get(label, "gray")
        display = PHASE_LABELS.get(label, label)

        ax.hist(
            lmi.values,
            bins=bins,
            density=True,
            alpha=0.35,
            label=display,
            color=color,
            edgecolor=color,
            linewidth=0.8,
        )
        # Step-line overlay
        counts, edges = np.histogram(lmi.values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, color=color, linewidth=2)

    # SS category annotations
    for cat_label, v in SS_THRESH_10MIN.items():
        ax.axvline(v, color="gray", ls=":", alpha=0.5, lw=0.8)
        ax.text(
            v + 0.5,
            ax.get_ylim()[1] * 0.95,
            cat_label,
            fontsize=7,
            color="gray",
            va="top",
            rotation=90,
        )

    ax.set_xlabel("Lifetime Maximum Intensity [m/s, 10-min sustained]")
    ax.set_ylabel("Probability density")
    ax.set_title(f"LMI Distribution — {basin}")
    ax.legend(frameon=False)
    ax.set_xlim(15, 85)
    ax.grid(True, alpha=0.2)

    try:
        fig.tight_layout()
    except Exception:
        pass
    outpath = os.path.join(outdir, f"validation_lmi_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4: WIND-PRESSURE SCATTER
# ═══════════════════════════════════════════════════════════════════════


def plot_wind_pressure_scatter(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    outdir: str = ".",
    max_points: int = 50_000,
) -> plt.Figure:
    """
    Scatter plot of (pressure, wind) for all track points.
    Overlays standard wind-pressure relationship (WPR) curves.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    wind_col = "wind" if "wind" in list(datasets.values())[0].columns else "wind_ms"

    for label, df in datasets.items():
        mask = (df["pressure"] > 850) & (df["pressure"] < 1020) & (df[wind_col] > 10)
        sub = df[mask]
        if len(sub) > max_points:
            sub = sub.sample(max_points, random_state=42)

        color = PHASE_COLORS.get(label, "gray")
        display = PHASE_LABELS.get(label, label)
        ax.scatter(
            sub["pressure"],
            sub[wind_col],
            s=1,
            alpha=0.08,
            color=color,
            rasterized=True,
        )
        ax.scatter([], [], s=30, color=color, label=display, alpha=0.8)

    # Reference WPR curves
    dp = np.linspace(1, 120, 200)
    for a, b, ls, wpr_lbl in [
        (0.7, 0.62, "-", "WPR: V=0.70·ΔP^{0.62}"),
        (0.6, 0.65, "--", "WPR: V=0.60·ΔP^{0.65}"),
    ]:
        ax.plot(
            1013 - dp, a * dp**b, ls, color="black", lw=1.5, alpha=0.6, label=wpr_lbl
        )

    ax.set_xlabel("Central pressure [hPa]")
    ax.set_ylabel("Maximum wind speed [m/s, 10-min]")
    ax.set_title(f"Wind-Pressure Relationship — {basin}")
    ax.set_xlim(880, 1020)
    ax.set_ylim(10, 85)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    try:
        fig.tight_layout()
    except Exception:
        pass
    outpath = os.path.join(outdir, f"validation_wpr_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5: INTENSITY CDFs  (Pmin & Vmax per phase)
# ═══════════════════════════════════════════════════════════════════════


def plot_intensity_cdfs(
    datasets: Dict[str, pd.DataFrame],
    basin: str,
    outdir: str = ".",
) -> plt.Figure:
    """
    ECDF of lifetime-minimum pressure and lifetime-maximum wind,
    one curve per dataset/phase.
    """
    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(14, 5))

    wind_col = "wind" if "wind" in list(datasets.values())[0].columns else "wind_ms"

    for label, df in datasets.items():
        if "global_storm_uid" in df.columns:
            storms = df.groupby("global_storm_uid").agg(
                pmin=("pressure", "min"), vmax=(wind_col, "max")
            )
        elif "storm_uid" in df.columns:
            storms = df.groupby("storm_uid").agg(
                pmin=("pressure", "min"), vmax=(wind_col, "max")
            )
        else:
            storms = _per_storm_agg(df)

        color = PHASE_COLORS.get(label, "gray")
        display = PHASE_LABELS.get(label, label)

        pmin = np.sort(storms["pmin"].dropna().values)
        vmax = np.sort(storms["vmax"].dropna().values)
        ecdf = lambda x: np.arange(1, len(x) + 1) / len(x)

        ax_p.plot(pmin, ecdf(pmin), color=color, label=display, lw=1.5)
        ax_v.plot(vmax, ecdf(vmax), color=color, label=display, lw=1.5)

    ax_p.set_xlabel("Minimum central pressure (hPa)")
    ax_p.set_ylabel("CDF")
    ax_p.set_title(f"Lifetime Pmin — {basin}")
    ax_p.legend(frameon=False, fontsize=8)
    ax_p.grid(True, alpha=0.2)

    ax_v.set_xlabel("Maximum wind speed (m/s, 10-min)")
    ax_v.set_ylabel("CDF")
    ax_v.set_title(f"Lifetime Vmax — {basin}")
    ax_v.legend(frameon=False, fontsize=8)
    ax_v.grid(True, alpha=0.2)

    try:
        fig.tight_layout()
    except Exception:
        pass
    outpath = os.path.join(outdir, f"validation_cdfs_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: RUN ALL PLOTS FOR ONE BASIN
# ═══════════════════════════════════════════════════════════════════════


def run_all_plots(
    datasets: Dict[str, pd.DataFrame],
    n_years_dict: Dict[str, float],
    basin: str,
    outdir: str = "validation_plots",
    resolution_density: float = 2.0,
    resolution_track: float = 1.0,
    dataset_ref: Optional[pd.DataFrame] = None,
    n_years_ref: Optional[float] = None,
    datasets_old: Optional[Dict[str, pd.DataFrame]] = None,
    n_years_old: Optional[float] = None,
):
    """
    Generate the full validation suite for a single basin.

    Parameters
    ----------
    datasets : {"EN": df, "NEU": df, "LN": df} — new model catalogs
    n_years_dict : {"EN": 10000, "NEU": 10000, "LN": 10000}
    basin : basin code
    outdir : output directory
    dataset_ref : IBTrACS reference catalog (optional)
    n_years_ref : IBTrACS years (e.g. 42)
    datasets_old : old model catalogs for difference plots (optional)
    n_years_old : old model years

    Example
    -------
    >>> from evaluation import load_catalog
    >>> s3_en = load_catalog("/data/S3/", "NA", "EN")
    >>> s3_neu = load_catalog("/data/S3/", "NA", "NEU")
    >>> s3_ln = load_catalog("/data/S3/", "NA", "LN")
    >>> ref = load_catalog("/data/ibtracs_ref/", "NA")
    >>> run_all_plots(
    ...     datasets={"EN": s3_en, "NEU": s3_neu, "LN": s3_ln},
    ...     n_years_dict={"EN": 10000, "NEU": 10000, "LN": 10000},
    ...     basin="NA",
    ...     dataset_ref=ref, n_years_ref=42,
    ... )
    """
    os.makedirs(outdir, exist_ok=True)

    # Include IBTrACS in the datasets dict for side-by-side plots
    all_ds = dict(datasets)
    all_ny = dict(n_years_dict)
    if dataset_ref is not None:
        all_ds["IBTrACS"] = dataset_ref
        all_ny["IBTrACS"] = n_years_ref or 42

    # ── Absolute density maps ──
    for mode, res in [
        ("genesis", resolution_density),
        ("track", resolution_track),
        ("ace", resolution_density),
    ]:
        print(f"\n  Plotting {mode} density...")
        plot_density_panels(
            all_ds, basin, all_ny, mode=mode, resolution=res, outdir=outdir
        )

    # ── Difference maps (if reference provided) ──
    if dataset_ref is not None:
        for mode, res in [
            ("genesis", resolution_density),
            ("track", resolution_track),
            ("ace", resolution_density),
        ]:
            print(f"\n  Plotting {mode} difference maps...")
            plot_difference_triplet(
                datasets,
                dataset_ref,
                basin,
                n_years_new=list(n_years_dict.values())[0],
                n_years_ref=n_years_ref or 42,
                mode=mode,
                resolution=res,
                datasets_old=datasets_old,
                n_years_old=n_years_old,
                outdir=outdir,
            )

    # ── Non-spatial plots ──
    print("\n  Plotting LMI distribution...")
    plot_lmi_distribution(all_ds, basin, outdir=outdir)

    print("  Plotting wind-pressure scatter...")
    plot_wind_pressure_scatter(all_ds, basin, outdir=outdir)

    print("  Plotting intensity CDFs...")
    plot_intensity_cdfs(all_ds, basin, outdir=outdir)

    print(f"\n  All plots saved to: {outdir}/")
