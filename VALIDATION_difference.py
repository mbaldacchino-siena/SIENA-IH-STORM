"""
SIENA-IH-STORM Difference Maps
================================
Plots spatial differences in genesis, track density, and ACE between:
  1. SIENA (new) vs IBTrACS  — model bias
  2. SIENA (new) vs STORM (old) — improvement from new physics

Usage:
    python VALIDATION_difference.py \
      --basins NA \
      --new_pattern "/path/to/STORM_DATA_IBTRACS_{basin}_{phase}_*_YEARS_*.txt" \
      --old_pattern "/path/to/old/STORM_DATA_IBTRACS_{basin}_1000_YEARS_*.txt" \
      --sim_years_new 5000 \
      --sim_years_old 5000 \
      --ibtracs_years 41 \
      --outdir validation_diff_plots

If --old_pattern is not given, only SIENA-vs-IBTrACS panels are produced.

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

dir_path = os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

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
PHASE_LABELS = {
    "EN": "El Niño",
    "NEU": "Neutral",
    "LN": "La Niña",
    "ALL": "All phases",
    "IBTrACS": "IBTrACS",
}


# =========================================================================
# DATA LOADING (reused from VALIDATION_spatial.py)
# =========================================================================


def load_storm_files(pattern):
    """Load STORM output from a glob pattern into a single DataFrame."""
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            arr = np.loadtxt(f, delimiter=",")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
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
            df["storm_uid"] = (
                df["storm_id"].astype(int).astype(str)
                + f"_L{loop_idx}_Y"
                + df["year"].astype(int).astype(str)
            )
            df["wind"] *= 1 / 0.88
            frames.append(df)
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_ibtracs_processed(basin_idx):
    """Load preprocessed IBTrACS .npy files, filter to basin."""
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
        print("IBTrACS .npy files not found.")
        return pd.DataFrame()
    rows = []
    for i in range(len(latlist)):
        if (
            len(latlist[i]) > 0
            and len(basinlist[i]) > 0
            and basinlist[i][0] == basin_idx
        ):
            for j in range(len(latlist[i])):
                rows.append(
                    {
                        "storm_uid": f"IB_{i}",
                        "lat": latlist[i][j],
                        "lon": lonlist[i][j],
                        "wind": windlist[i][j] if j < len(windlist[i]) else np.nan,
                        "pressure": preslist[i][j] if j < len(preslist[i]) else np.nan,
                        "timestep": j,
                    }
                )
    return pd.DataFrame(rows)


# =========================================================================
# GRID COMPUTATION
# =========================================================================


def compute_grid(df, basin, resolution, mode="count"):
    """
    Compute a 2D grid from track data.
    mode: 'count' (track density), 'genesis' (first timestep only), 'ace'
    """
    lat0, lat1, lon0, lon1 = BASIN_BOUNDS[basin]
    lat_edges = np.arange(lat0, lat1 + resolution, resolution)
    lon_edges = np.arange(lon0, lon1 + resolution, resolution)

    if mode == "genesis":
        if "timestep" in df.columns:
            sub = df[df["timestep"] == 0]
        else:
            sub = df.drop_duplicates("storm_uid", keep="first")
        grid, _, _ = np.histogram2d(
            sub["lat"].values, sub["lon"].values, bins=[lat_edges, lon_edges]
        )
    elif mode == "ace":
        wind_kt = df["wind"].values * 1.94384
        weights = wind_kt**2 * 1e-4
        grid, _, _ = np.histogram2d(
            df["lat"].values,
            df["lon"].values,
            bins=[lat_edges, lon_edges],
            weights=weights,
        )
    else:  # count / track density
        grid, _, _ = np.histogram2d(
            df["lat"].values, df["lon"].values, bins=[lat_edges, lon_edges]
        )

    return grid, lat_edges, lon_edges


# =========================================================================
# PLOTTING
# =========================================================================


def _make_diff_figure(n_rows, n_cols, basin, figwidth=16):
    """Create a figure with map subplots."""
    height = figwidth / n_cols * 0.55 * n_rows
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(figwidth, height),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(figwidth, height), squeeze=False
        )
    return fig, axes


def _plot_absolute_panel(ax, grid, lat_e, lon_e, basin, title, vmax, cmap):
    """Plot an absolute density panel."""
    extent = BASIN_PLOT_EXTENT.get(basin)
    grid_plot = np.ma.masked_where(grid <= 0, grid)
    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        im = ax.pcolormesh(
            lon_e,
            lat_e,
            grid_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            shading="flat",
        )
    else:
        im = ax.pcolormesh(
            lon_e, lat_e, grid_plot, cmap=cmap, vmin=0, vmax=vmax, shading="flat"
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=10)
    return im


def _plot_diff_panel(ax, grid, lat_e, lon_e, basin, title, vlim, cmap="RdBu_r"):
    """Plot a difference panel with diverging colormap centered on zero."""
    extent = BASIN_PLOT_EXTENT.get(basin)
    grid_plot = np.ma.masked_where(grid == 0, grid)
    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=0)
        im = ax.pcolormesh(
            lon_e,
            lat_e,
            grid_plot,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            # vmin=-vlim,
            # vmax=vlim,
            shading="flat",
            norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim),
        )
    else:
        im = ax.pcolormesh(
            lon_e,
            lat_e,
            grid_plot,
            cmap=cmap,
            # vmin=-vlim,
            # vmax=vlim,
            shading="flat",
            norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim),
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_title(title, fontsize=10)
    return im


def plot_difference_triplet(
    grids_dict,
    lat_e,
    lon_e,
    basin,
    metric_name,
    unit,
    outdir=".",
    abs_cmap="YlOrRd",
    diff_cmap="RdBu_r",
):
    """
    Plot a 3-row figure:
      Row 1: Absolute maps (IBTrACS, SIENA_new, STORM_old)  — or 2 panels if no old
      Row 2: SIENA_new minus IBTrACS  (per phase if available, or pooled)
      Row 3: SIENA_new minus STORM_old (per phase if available, or pooled)

    grids_dict keys: 'ibtracs', 'new_EN', 'new_NEU', 'new_LN', 'new_ALL',
                     'old_EN', 'old_NEU', 'old_LN', 'old_ALL'
    """
    has_old = any(k.startswith("old_") for k in grids_dict)
    has_phases = "new_EN" in grids_dict

    # ── Row 1: Absolute panels ──
    if has_phases:
        abs_labels = ["IBTrACS", "SIENA El Niño", "SIENA Neutral", "SIENA La Niña"]
        abs_keys = ["ibtracs", "new_EN", "new_NEU", "new_LN"]
        if has_old:
            abs_labels += ["Old El Niño", "Old Neutral", "Old La Niña"]
            abs_keys += ["old_EN", "old_NEU", "old_LN"]
    else:
        abs_labels = ["IBTrACS", "SIENA (new)"]
        abs_keys = ["ibtracs", "new_ALL"]
        if has_old:
            abs_labels.append("STORM (old)")
            abs_keys.append("old_ALL")

    n_abs = len(abs_keys)

    # ── Row 2: new minus IBTrACS ──
    if has_phases:
        diff1_labels = [
            "SIENA−IBTrACS (EN)",
            "SIENA−IBTrACS (NEU)",
            "SIENA−IBTrACS (LN)",
        ]
        diff1_pairs = [
            ("new_EN", "ibtracs"),
            ("new_NEU", "ibtracs"),
            ("new_LN", "ibtracs"),
        ]
    else:
        diff1_labels = ["SIENA − IBTrACS"]
        diff1_pairs = [("new_ALL", "ibtracs")]

    # ── Row 3: new minus old ──
    diff2_labels = []
    diff2_pairs = []
    if has_old:
        if has_phases:
            for ph, ph_label in [("EN", "EN"), ("NEU", "NEU"), ("LN", "LN")]:
                diff2_labels.append(f"SIENA−Old ({ph_label})")
                diff2_pairs.append((f"new_{ph}", f"old_{ph}"))
        else:
            diff2_labels = ["SIENA − Old"]
            diff2_pairs = [("new_ALL", "old_ALL")]

    ncols = max(n_abs, len(diff1_labels), len(diff2_labels) if diff2_labels else 1)
    nrows = 2 + (1 if diff2_labels else 0)

    fig, axes = _make_diff_figure(nrows, ncols, basin, figwidth=5 * ncols)

    # Compute vmax for absolute panels
    all_abs = [grids_dict[k] for k in abs_keys if k in grids_dict]
    vmax_abs = max(np.percentile(g[g > 0], 99) if np.any(g > 0) else 1 for g in all_abs)

    # ── Plot Row 1: Absolute ──
    for i, (key, label) in enumerate(zip(abs_keys, abs_labels)):
        if key in grids_dict:
            im = _plot_absolute_panel(
                axes[0, i],
                grids_dict[key],
                lat_e,
                lon_e,
                basin,
                label,
                vmax_abs,
                abs_cmap,
            )
    # Colorbar for row 1
    plt.colorbar(im, ax=axes[0, :].tolist(), shrink=0.6, label=unit, pad=0.02)
    # Hide unused axes in row 1
    for i in range(n_abs, ncols):
        axes[0, i].set_visible(False)

    # ── Plot Row 2: new minus IBTrACS ──
    diff_grids_1 = []
    for i, (new_key, ref_key) in enumerate(diff1_pairs):
        if new_key in grids_dict and ref_key in grids_dict:
            # Ensure same shape
            g_new = grids_dict[new_key]
            g_ref = grids_dict[ref_key]
            shape = (
                min(g_new.shape[0], g_ref.shape[0]),
                min(g_new.shape[1], g_ref.shape[1]),
            )
            diff = g_new[: shape[0], : shape[1]] - g_ref[: shape[0], : shape[1]]
            diff_grids_1.append(diff)

    if diff_grids_1:
        vlim1 = max(
            np.percentile(np.abs(d[d != 0]), 99) if np.any(d != 0) else 0.1
            for d in diff_grids_1
        )
        vlim1 = max(vlim1, 0.01)
        for i, (diff, label) in enumerate(zip(diff_grids_1, diff1_labels)):
            im2 = _plot_diff_panel(
                axes[1, i], diff, lat_e, lon_e, basin, label, vlim1, diff_cmap
            )
        plt.colorbar(
            im2, ax=axes[1, :].tolist(), shrink=0.6, label=f"Δ {unit}", pad=0.02
        )
    for i in range(len(diff1_labels), ncols):
        axes[1, i].set_visible(False)

    # ── Plot Row 3: new minus old ──
    if diff2_labels:
        diff_grids_2 = []
        for new_key, old_key in diff2_pairs:
            if new_key in grids_dict and old_key in grids_dict:
                g_new = grids_dict[new_key]
                g_old = grids_dict[old_key]
                shape = (
                    min(g_new.shape[0], g_old.shape[0]),
                    min(g_new.shape[1], g_old.shape[1]),
                )
                diff = g_new[: shape[0], : shape[1]] - g_old[: shape[0], : shape[1]]
                diff_grids_2.append(diff)

        if diff_grids_2:
            vlim2 = max(
                np.percentile(np.abs(d[d != 0]), 99) if np.any(d != 0) else 0.1
                for d in diff_grids_2
            )
            vlim2 = max(vlim2, 0.01)
            for i, (diff, label) in enumerate(zip(diff_grids_2, diff2_labels)):
                im3 = _plot_diff_panel(
                    axes[2, i], diff, lat_e, lon_e, basin, label, vlim2, diff_cmap
                )
            plt.colorbar(
                im3, ax=axes[2, :].tolist(), shrink=0.6, label=f"Δ {unit}", pad=0.02
            )
        for i in range(len(diff2_labels), ncols):
            axes[2, i].set_visible(False)

    fig.suptitle(f"{metric_name} — {basin}", fontsize=14, y=1.01)
    fig.tight_layout()
    safe_name = metric_name.lower().replace(" ", "_")
    outpath = os.path.join(outdir, f"diff_{safe_name}_{basin}.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"  Saved: {outpath}")
    return fig


# =========================================================================
# MAIN
# =========================================================================


def main():
    parser = argparse.ArgumentParser(description="SIENA-IH-STORM difference maps")
    parser.add_argument("--basins", nargs="*", default=["NA"])
    parser.add_argument(
        "--new_pattern",
        default=None,
        help="Glob pattern for new SIENA files. Use {basin} and {phase} placeholders.",
    )
    parser.add_argument(
        "--old_pattern",
        default=None,
        help="Glob pattern for old STORM files. Use {basin} and {phase} placeholders. "
        "If the old model had no phase separation, use {basin} only.",
    )
    parser.add_argument(
        "--old_has_phases",
        action="store_true",
        help="Set if old STORM files are phase-separated (separate EN/NEU/LN files)",
    )
    parser.add_argument("--sim_years_new", type=float, default=5000)
    parser.add_argument("--sim_years_old", type=float, default=5000)
    parser.add_argument("--ibtracs_years", type=float, default=41)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--outdir", default="validation_diff_plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    basin_idx_map = {"EP": 0, "NA": 1, "NI": 2, "SI": 3, "SP": 4, "WP": 5}

    for basin in args.basins:
        print(f"\n{'=' * 60}")
        print(f"Difference maps for: {basin}")
        print(f"{'=' * 60}")

        idx = basin_idx_map[basin]
        res = args.resolution

        # ── Load IBTrACS ──
        print("  Loading IBTrACS...")
        ibt = load_ibtracs_processed(idx)
        if len(ibt) == 0:
            print("  No IBTrACS data. Skipping.")
            continue

        # ── Load new SIENA (per phase) ──
        new_data = {}
        for phase in ["EN", "NEU", "LN"]:
            if args.new_pattern:
                pat = args.new_pattern.format(basin=basin, phase=phase)
            else:
                pat = os.path.join(
                    __location__, f"STORM_DATA_IBTRACS_{basin}_{phase}_*_YEARS_*.txt"
                )
            df = load_storm_files(pat)
            if len(df) > 0:
                new_data[phase] = df
                print(f"  New {phase}: {df['storm_uid'].nunique():,} storms")

        # ── Pool new phases for comparison with IBTrACS ──
        if new_data:
            new_all = pd.concat(new_data.values(), ignore_index=True)
        else:
            print("  No new SIENA data found. Skipping.")
            continue

        # ── Load old STORM ──
        old_data = {}
        if args.old_pattern:
            if args.old_has_phases:
                for phase in ["EN", "NEU", "LN"]:
                    # Try with phase tag: ELNINO, LANINA, ENSO_NEUTRAL or EN/NEU/LN
                    pat = args.old_pattern.format(basin=basin, phase=phase)
                    df = load_storm_files(pat)
                    if len(df) > 0:
                        old_data[phase] = df
                        print(f"  Old {phase}: {df['storm_uid'].nunique():,} storms")
            else:
                pat = args.old_pattern.format(basin=basin, phase="ALL")
                df = load_storm_files(pat)
                if len(df) > 0:
                    old_data["ALL"] = df
                    print(f"  Old (pooled): {df['storm_uid'].nunique():,} storms")

        # ── Compute grids for each metric ──
        for mode, metric_name, unit, abs_cmap in [
            ("genesis", "Genesis Density", "storms/yr/cell", "YlOrRd"),
            ("count", "Track Density", "track pts/yr/cell", "YlGnBu"),
            ("ace", "ACE Density", "ACE/yr/cell", "hot_r"),
        ]:
            print(f"\n  Computing {metric_name}...")
            grids = {}

            # IBTrACS
            g, lat_e, lon_e = compute_grid(ibt, basin, res, mode=mode)
            grids["ibtracs"] = g / args.ibtracs_years

            # New SIENA per phase
            # Each phase has sim_years_new effective years
            for phase in ["EN", "NEU", "LN"]:
                if phase in new_data:
                    g, _, _ = compute_grid(new_data[phase], basin, res, mode=mode)
                    grids[f"new_{phase}"] = g / args.sim_years_new

            # New SIENA pooled (average of phases, weighted equally)
            phase_grids = [
                grids[f"new_{ph}"] for ph in ["EN", "NEU", "LN"] if f"new_{ph}" in grids
            ]
            if phase_grids:
                # For comparison with unconditional IBTrACS: average the phase rates
                grids["new_ALL"] = np.mean(phase_grids, axis=0)

            # Old STORM
            if old_data:
                if "ALL" in old_data:
                    g, _, _ = compute_grid(old_data["ALL"], basin, res, mode=mode)
                    grids["old_ALL"] = g / args.sim_years_old
                    # Replicate for phase-level diffs
                    for ph in ["EN", "NEU", "LN"]:
                        grids[f"old_{ph}"] = grids["old_ALL"]
                else:
                    for phase in ["EN", "NEU", "LN"]:
                        if phase in old_data:
                            g, _, _ = compute_grid(
                                old_data[phase], basin, res, mode=mode
                            )
                            grids[f"old_{phase}"] = g / args.sim_years_old
                    old_phase_grids = [
                        grids[f"old_{ph}"]
                        for ph in ["EN", "NEU", "LN"]
                        if f"old_{ph}" in grids
                    ]
                    if old_phase_grids:
                        grids["old_ALL"] = np.mean(old_phase_grids, axis=0)

            plot_difference_triplet(
                grids,
                lat_e,
                lon_e,
                basin,
                metric_name,
                unit,
                outdir=args.outdir,
                abs_cmap=abs_cmap,
            )

    print(f"\nAll plots saved to: {args.outdir}/")


if __name__ == "__main__":
    main()
