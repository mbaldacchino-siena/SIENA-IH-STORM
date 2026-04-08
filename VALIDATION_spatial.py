"""
SIENA-IH-STORM Spatial Validation Plots
========================================
Produces publication-quality maps comparing synthetic TC output against IBTrACS.

Generates:
  1. Genesis density maps (per phase + IBTrACS)
  2. Track density maps
  3. ACE density maps
  4. Lifetime Maximum Intensity (LMI) distributions
  5. Wind-pressure scatter with WPR overlay

Usage:
    python VALIDATION_spatial.py --basins NA --ibtracs_period 1980 2020

Requires:
    numpy, pandas, matplotlib, cartopy

All synthetic data is read from STORM output files in the working directory.
IBTrACS data is read from the preprocessed .npy files.

Author: Generated for SIENA-IH-STORM validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import sys
import glob
import argparse

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: cartopy not installed. Maps will use plain lat/lon axes.")

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# ── Basin domains ──
BASIN_BOUNDS = {
    "EP": (5, 60, 180, 285),
    "NA": (5, 60, 255, 360),
    "NI": (5, 60, 30, 100),
    "SI": (-60, -5, 10, 135),
    "SP": (-60, -5, 135, 240),
    "WP": (5, 60, 100, 180),
}

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
    "IBTrACS": "#F9A825",
}
PHASE_LABELS = {
    "EN": "El Niño",
    "NEU": "Neutral",
    "LN": "La Niña",
    "IBTrACS": "IBTrACS",
}


# =========================================================================
# DATA LOADING
# =========================================================================


def load_storm_output(basin, phase, pattern=None):
    """
    Load STORM output files for a given basin and phase.
    Returns a DataFrame with columns:
        year, month, storm_id, timestep, basin_idx, lat, lon,
        pressure, wind, rmax, category, landfall, dist_coast
    """
    if pattern is None:
        pattern = os.path.join(
            __location__, f"STORM_DATA_IBTRACS_{basin}_{phase}_*_YEARS_*.txt"
        )
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files found for {basin}/{phase}: {pattern}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            arr = np.loadtxt(f, delimiter=",")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Extract loop index from filename for unique storm IDs
            loop_idx = os.path.basename(f).split("_")[-1].replace(".txt", "")
            df = pd.DataFrame(
                arr[:, :13],
                columns=[
                    "year",
                    "month",
                    "storm_id",
                    "timestep",
                    "basin_idx",
                    "lat",
                    "lon",
                    "pressure",
                    "wind",
                    "rmax",
                    "category",
                    "landfall",
                    "dist_coast",
                ],
            )
            # Make storm_id unique across loops
            df["storm_uid"] = (
                df["storm_id"].astype(int).astype(str)
                + f"_L{loop_idx}_Y"
                + df["year"].astype(int).astype(str)
            )
            df["wind"] *= 1/0.88
            frames.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ibtracs_processed():
    """Load preprocessed IBTrACS data from .npy files."""
    try:
        latlist = np.load(
            os.path.join(__location__, "LATLIST_INTERP.npy"), allow_pickle=True
        ).item()
        lonlist = np.load(
            os.path.join(__location__, "LONLIST_INTERP.npy"), allow_pickle=True
        ).item()
        windlist = np.load(
            os.path.join(__location__, "WINDLIST_INTERP.npy"), allow_pickle=True
        ).item()
        preslist = np.load(
            os.path.join(__location__, "PRESLIST_INTERP.npy"), allow_pickle=True
        ).item()
        monthlist = np.load(
            os.path.join(__location__, "MONTHLIST_INTERP.npy"), allow_pickle=True
        ).item()
        basinlist = np.load(
            os.path.join(__location__, "BASINLIST_INTERP.npy"), allow_pickle=True
        ).item()
    except FileNotFoundError:
        print("IBTrACS .npy files not found. Run MASTER_preprocessing.py first.")
        return pd.DataFrame()

    rows = []
    for i in range(len(latlist)):
        if len(latlist[i]) > 0:
            for j in range(len(latlist[i])):
                rows.append(
                    {
                        "storm_uid": f"IB_{i}",
                        "lat": latlist[i][j],
                        "lon": lonlist[i][j],
                        "wind": windlist[i][j] if j < len(windlist[i]) else np.nan,
                        "pressure": preslist[i][j] if j < len(preslist[i]) else np.nan,
                        "month": monthlist[i][0] if len(monthlist[i]) > 0 else np.nan,
                        "basin_idx": basinlist[i][0]
                        if len(basinlist[i]) > 0
                        else np.nan,
                        "timestep": j,
                    }
                )
    return pd.DataFrame(rows)


def count_effective_years(df):
    """Count unique (year, storm_id) combinations to estimate simulation years."""
    if "year" in df.columns:
        return len(df["year"].unique())
    return 1


# =========================================================================
# GRID COMPUTATION
# =========================================================================


def compute_density_grid(lats, lons, basin, resolution=1.0, weights=None):
    """
    Bin lat/lon points onto a regular grid. Optionally weight by a value (e.g., ACE).
    Returns: grid (2D array), lat_edges, lon_edges
    """
    lat0, lat1, lon0, lon1 = BASIN_BOUNDS[basin]
    lat_edges = np.arange(lat0, lat1 + resolution, resolution)
    lon_edges = np.arange(lon0, lon1 + resolution, resolution)

    if weights is not None:
        grid, _, _ = np.histogram2d(
            lats, lons, bins=[lat_edges, lon_edges], weights=weights
        )
    else:
        grid, _, _ = np.histogram2d(lats, lons, bins=[lat_edges, lon_edges])

    return grid, lat_edges, lon_edges


# =========================================================================
# PLOT 1: GENESIS DENSITY
# =========================================================================


def plot_genesis_density(datasets, basin, n_years_dict, resolution=2.0, outdir="."):
    """
    Plot genesis point density per year for each dataset.
    datasets: dict {label: DataFrame}
    n_years_dict: dict {label: float} effective years for normalization
    """
    n = len(datasets)
    fig, axes = _make_map_subplots(n, basin)

    vmax = 0
    grids = {}
    for label, df in datasets.items():
        # Genesis = first timestep of each storm
        if "timestep" in df.columns:
            genesis = df[df["timestep"] == 0].copy()
        else:
            genesis = df.drop_duplicates("storm_uid", keep="first").copy()

        grid, lat_e, lon_e = compute_density_grid(
            genesis["lat"].values, genesis["lon"].values, basin, resolution=resolution
        )
        grid = grid / n_years_dict.get(label, 1.0)
        grids[label] = (grid, lat_e, lon_e)
        vmax = max(vmax, np.percentile(grid[grid > 0], 95) if np.any(grid > 0) else 1)

    for ax, (label, (grid, lat_e, lon_e)) in zip(axes, grids.items()):
        _plot_density_panel(
            ax,
            grid,
            lat_e,
            lon_e,
            basin,
            label,
            vmax,
            cmap="YlOrRd",
            unit="storms/yr/cell",
        )

    fig.suptitle(f"Genesis Density — {basin} ({resolution}° grid)", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(outdir, f"validation_genesis_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# =========================================================================
# PLOT 2: TRACK DENSITY
# =========================================================================


def plot_track_density(datasets, basin, n_years_dict, resolution=1.0, outdir="."):
    """Track point density per year."""
    n = len(datasets)
    fig, axes = _make_map_subplots(n, basin)

    vmax = 0
    grids = {}
    for label, df in datasets.items():
        grid, lat_e, lon_e = compute_density_grid(
            df["lat"].values, df["lon"].values, basin, resolution=resolution
        )
        grid = grid / n_years_dict.get(label, 1.0)
        grids[label] = (grid, lat_e, lon_e)
        vmax = max(vmax, np.percentile(grid[grid > 0], 95) if np.any(grid > 0) else 1)

    for ax, (label, (grid, lat_e, lon_e)) in zip(axes, grids.items()):
        _plot_density_panel(
            ax,
            grid,
            lat_e,
            lon_e,
            basin,
            label,
            vmax,
            cmap="YlGnBu",
            unit="track pts/yr/cell",
        )

    fig.suptitle(f"Track Density — {basin} ({resolution}° grid)", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(outdir, f"validation_track_density_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# =========================================================================
# PLOT 3: ACE DENSITY
# =========================================================================


def plot_ace_density(datasets, basin, n_years_dict, resolution=2.0, outdir="."):
    """
    ACE density: sum of Vmax^2 per cell, normalized by years.
    ACE units: 10^4 kt^2 per cell per year (standard NOAA convention).
    We compute in m/s then convert.
    """
    n = len(datasets)
    fig, axes = _make_map_subplots(n, basin)

    vmax_val = 0
    grids = {}
    for label, df in datasets.items():
        wind_ms = df["wind"].values
        # ACE contribution per 3-hour timestep: Vmax^2 (in kt^2) × 10^-4
        # 1 m/s = 1.94384 kt
        wind_kt = wind_ms * 1.94384
        ace_per_point = wind_kt**2 * 1e-4  # standard ACE units

        grid, lat_e, lon_e = compute_density_grid(
            df["lat"].values,
            df["lon"].values,
            basin,
            resolution=resolution,
            weights=ace_per_point,
        )
        grid = grid / n_years_dict.get(label, 1.0)
        grids[label] = (grid, lat_e, lon_e)
        vmax_val = max(
            vmax_val, np.percentile(grid[grid > 0], 95) if np.any(grid > 0) else 1
        )

    for ax, (label, (grid, lat_e, lon_e)) in zip(axes, grids.items()):
        _plot_density_panel(
            ax,
            grid,
            lat_e,
            lon_e,
            basin,
            label,
            vmax_val,
            cmap="hot_r",
            unit="ACE/yr/cell",
        )

    fig.suptitle(f"ACE Density — {basin} ({resolution}° grid)", fontsize=14, y=1.02)
    fig.tight_layout()
    outpath = os.path.join(outdir, f"validation_ace_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# =========================================================================
# PLOT 4: LIFETIME MAXIMUM INTENSITY DISTRIBUTION
# =========================================================================


def plot_lmi_distribution(datasets, basin, outdir="."):
    """
    Histogram of per-storm maximum wind speed (LMI).
    Normalized to probability density for fair comparison across
    datasets with different numbers of storms.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.arange(15, 90, 2.5)
    for label, df in datasets.items():
        lmi = df.groupby("storm_uid")["wind"].max().dropna()
        color = PHASE_COLORS.get(label, "gray")
        display_label = PHASE_LABELS.get(label, label)
        ax.hist(
            lmi.values,
            bins=bins,
            density=True,
            alpha=0.35,
            label=display_label,
            color=color,
            edgecolor=color,
            linewidth=0.8,
        )
        # Overlay KDE-style line
        counts, edges = np.histogram(lmi.values, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centers, counts, color=color, linewidth=2)

    # Category thresholds
    for v, cat in [
        (33, "Cat 1"),
        (43, "Cat 2"),
        (50, "Cat 3"),
        (58, "Cat 4"),
        (70, "Cat 5"),
    ]:
        ax.axvline(v, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(
            v + 0.5,
            ax.get_ylim()[1] * 0.95,
            cat,
            fontsize=7,
            color="gray",
            va="top",
            rotation=90,
        )

    ax.set_xlabel("Lifetime Maximum Intensity [m/s]")
    ax.set_ylabel("Probability density")
    ax.set_title(f"LMI Distribution — {basin}")
    ax.legend(frameon=False)
    ax.set_xlim(15, 85)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    outpath = os.path.join(outdir, f"validation_lmi_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# =========================================================================
# PLOT 5: WIND-PRESSURE SCATTER
# =========================================================================


def plot_wind_pressure_scatter(datasets, basin, outdir="."):
    """
    Scatter plot of (pressure, wind) for all track points, with
    the fitted WPR curve overlaid.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, df in datasets.items():
        mask = (df["pressure"] > 850) & (df["pressure"] < 1020) & (df["wind"] > 10)
        sub = df[mask]
        if len(sub) > 50000:
            sub = sub.sample(50000, random_state=42)
        color = PHASE_COLORS.get(label, "gray")
        display_label = PHASE_LABELS.get(label, label)
        ax.scatter(
            sub["pressure"], sub["wind"], s=1, alpha=0.08, color=color, rasterized=True
        )
        # Dummy point for legend
        ax.scatter([], [], s=30, color=color, label=display_label, alpha=0.8)

    # Overlay standard WPR curves for reference
    dp = np.linspace(1, 120, 200)
    for a, b, style, wpr_label in [
        (0.7, 0.62, "-", "WPR: V=0.70·ΔP^0.62"),
        (0.6, 0.65, "--", "WPR: V=0.60·ΔP^0.65"),
    ]:
        vmax = a * dp**b
        pres = 1013 - dp
        ax.plot(
            pres, vmax, style, color="black", linewidth=1.5, alpha=0.6, label=wpr_label
        )

    ax.set_xlabel("Central pressure [hPa]")
    ax.set_ylabel("Maximum wind speed [m/s]")
    ax.set_title(f"Wind-Pressure Relationship — {basin}")
    ax.set_xlim(880, 1020)
    ax.set_ylim(10, 85)
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.2)
    ax.invert_xaxis()

    fig.tight_layout()
    outpath = os.path.join(outdir, f"validation_wpr_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# =========================================================================
# MAP HELPERS
# =========================================================================


def _make_map_subplots(n_panels, basin, figwidth=16):
    """Create a row of map subplots with optional cartopy projection."""
    if n_panels <= 2:
        ncols = n_panels
    elif n_panels <= 4:
        ncols = 2
    else:
        ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    height = figwidth / ncols * 0.6 * nrows

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figwidth, height),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(figwidth, height), squeeze=False
        )

    axes = axes.flatten()
    # Hide extra axes
    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    return fig, axes[:n_panels]


def _plot_density_panel(ax, grid, lat_edges, lon_edges, basin, title, vmax, cmap, unit):
    """Plot a single density panel on a map axis."""
    extent = BASIN_PLOT_EXTENT.get(
        basin, [lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]]
    )

    # Mask zeros for clean plotting
    grid_plot = np.ma.masked_where(grid <= 0, grid)

    display_label = PHASE_LABELS.get(title, title)

    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        im = ax.pcolormesh(
            lon_edges,
            lat_edges,
            grid_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            shading="flat",
        )
    else:
        im = ax.pcolormesh(
            lon_edges,
            lat_edges,
            grid_plot,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            shading="flat",
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    ax.set_title(display_label, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.7, label=unit, pad=0.02)


# =========================================================================
# MAIN
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="SIENA-IH-STORM spatial validation")
    parser.add_argument(
        "--basins", nargs="*", default=["NA"], help="Basins to validate"
    )
    parser.add_argument(
        "--ibtracs_years", type=float, default=41, help="IBTrACS period length"
    )
    parser.add_argument(
        "--sim_years",
        type=float,
        default=5000,
        help="Effective simulation years per phase",
    )
    parser.add_argument("--outdir", default="validation_plots", help="Output directory")
    parser.add_argument(
        "--storm_pattern",
        default=None,
        help="Override glob pattern for STORM files (use {basin} and {phase} placeholders)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    basin_idx_map = {"EP": 0, "NA": 1, "NI": 2, "SI": 3, "SP": 4, "WP": 5}

    for basin in args.basins:
        print(f"\n{'=' * 60}")
        print(f"Validation for basin: {basin}")
        print(f"{'=' * 60}")

        idx = basin_idx_map[basin]

        # ── Load datasets ──
        datasets = {}
        n_years = {}

        # IBTrACS
        print("  Loading IBTrACS...")
        ibt = load_ibtracs_processed()
        if len(ibt) > 0:
            ibt = ibt[ibt["basin_idx"] == idx]
            if len(ibt) > 0:
                datasets["IBTrACS"] = ibt
                n_years["IBTrACS"] = args.ibtracs_years

        # Synthetic phases
        for phase in ["EN", "NEU", "LN"]:
            print(f"  Loading {phase}...")
            if args.storm_pattern:
                pattern = args.storm_pattern.format(basin=basin, phase=phase)
            else:
                pattern = None
            df = load_storm_output(basin, phase, pattern=pattern)
            if len(df) > 0:
                datasets[phase] = df
                n_years[phase] = args.sim_years

        if not datasets:
            print(f"  No data found for {basin}. Skipping.")
            continue

        print(f"  Datasets loaded: {list(datasets.keys())}")
        for label, df in datasets.items():
            n_storms = df["storm_uid"].nunique()
            n_pts = len(df)
            print(f"    {label}: {n_storms:,} storms, {n_pts:,} track points")

        # ── Generate plots ──
        print("\n  Plotting genesis density...")
        plot_genesis_density(
            datasets, basin, n_years, resolution=2.0, outdir=args.outdir
        )

        print("  Plotting track density...")
        plot_track_density(datasets, basin, n_years, resolution=1.0, outdir=args.outdir)

        print("  Plotting ACE density...")
        plot_ace_density(datasets, basin, n_years, resolution=2.0, outdir=args.outdir)

        print("  Plotting LMI distribution...")
        plot_lmi_distribution(datasets, basin, outdir=args.outdir)

        print("  Plotting wind-pressure scatter...")
        plot_wind_pressure_scatter(datasets, basin, outdir=args.outdir)

    print(f"\nAll plots saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
