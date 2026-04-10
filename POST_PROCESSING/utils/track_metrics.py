"""
track_metrics.py  —  Track-level metric extraction
====================================================

Supplements evaluation.py with metrics that require 3-hourly step data:

  1. Landfall intensity at first landfall (not lifetime max)
  2. Translation speed distribution (from consecutive lat/lon)
  3. Pressure tendency rate (Δp per 3h step)
  4. Recurvature detection (latitude of maximum poleward extent before
     equatorward turn)
  5. Coastline-binned landfall density
  6. Enhanced lifetime export (adds lf_pressure, lf_wind, mean_speed_kmh,
     lat_lmi, lon_lmi)

All functions take the raw STORM catalog DataFrame (as returned by
evaluation.load_catalog) and return DataFrames or dicts suitable for
CSV export.

Usage in run_evaluation.py:
    from track_metrics import (
        enhanced_lifetime_distribution,
        translation_speed_distribution,
        pressure_tendency_distribution,
        recurvature_stats,
        coastline_landfall_density,
    )

Author: Mathys Baldacchino / track metric extensions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════


def _haversine_vec(lon1, lat1, lon2, lat2):
    """
    Vectorized great-circle distance (km) between consecutive points.
    All inputs in decimal degrees.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _per_storm_agg_extended(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Extended one-row-per-storm summary. Adds to evaluation._per_storm_agg:
      - lf_pressure, lf_wind (at first landfall timestep)
      - lat_lmi, lon_lmi (location of lifetime maximum intensity)
      - mean_speed_kmh (mean translation speed)
      - max_dp_3h (largest 3h pressure drop = fastest intensification)
      - has_recurvature (boolean)
      - lat_recurvature (latitude where recurvature occurs)
    """
    g = catalog.groupby("global_storm_uid", sort=False)

    # Basic aggregation
    agg = g.agg(
        year=("global_year", "first"),
        month=("month", "first"),
        n_steps=("timestep", "count"),
        lat_genesis=("lat", "first"),
        lon_genesis=("lon", "first"),
        pmin=("pressure", "min"),
        vmax=("wind", "max"),
        max_ss=("ss_cat", "max"),
        has_landfall=("landfall", "max"),
    )
    agg["duration_hours"] = agg["n_steps"] * 3

    # ── Location of lifetime maximum intensity (LMI) ──
    idx_lmi = g["wind"].idxmax()
    lmi_rows = catalog.loc[idx_lmi, ["lat", "lon", "global_storm_uid"]].set_index(
        "global_storm_uid"
    )
    agg["lat_lmi"] = lmi_rows["lat"]
    agg["lon_lmi"] = lmi_rows["lon"]

    # ── First landfall intensity ──
    lf_mask = catalog["landfall"] == 1
    if lf_mask.any():
        lf_df = catalog[lf_mask].groupby("global_storm_uid", sort=False)
        first_lf = lf_df.first()[["pressure", "wind", "lat", "lon"]].rename(
            columns={
                "pressure": "lf_pressure",
                "wind": "lf_wind",
                "lat": "lf_lat",
                "lon": "lf_lon",
            }
        )
        agg = agg.join(first_lf, how="left")
    else:
        for col in ["lf_pressure", "lf_wind", "lf_lat", "lf_lon"]:
            agg[col] = np.nan

    # ── Translation speed (mean over lifetime) ──
    # Compute per-step distance and speed, then aggregate
    catalog_sorted = catalog.sort_values(["global_storm_uid", "timestep"]).reset_index(
        drop=True
    )

    # Shift within each storm to get consecutive-point distances
    cat_shift = catalog_sorted.groupby("global_storm_uid", sort=False)[
        ["lat", "lon"]
    ].shift(1)
    dist_km = _haversine_vec(
        cat_shift["lon"].values,
        cat_shift["lat"].values,
        catalog_sorted["lon"].values,
        catalog_sorted["lat"].values,
    )
    catalog_sorted["dist_km"] = dist_km
    catalog_sorted["speed_kmh"] = dist_km / 3.0  # 3-hourly steps

    speed_agg = catalog_sorted.groupby("global_storm_uid", sort=False).agg(
        mean_speed_kmh=("speed_kmh", "mean"),
        median_speed_kmh=("speed_kmh", "median"),
        total_dist_km=("dist_km", "sum"),
    )
    agg = agg.join(speed_agg, how="left")

    # ── Pressure tendency (Δp per 3h step) ──
    p_shift = catalog_sorted.groupby("global_storm_uid", sort=False)["pressure"].shift(
        1
    )
    catalog_sorted["dp_3h"] = catalog_sorted["pressure"] - p_shift
    # Negative dp = intensification, positive = weakening

    dp_agg = catalog_sorted.groupby("global_storm_uid", sort=False).agg(
        max_intensification_3h=(
            "dp_3h",
            "min",
        ),  # most negative = fastest intensification
        max_weakening_3h=("dp_3h", "max"),
        mean_dp_3h=("dp_3h", "mean"),
    )
    agg = agg.join(dp_agg, how="left")

    # ── Recurvature detection ──
    # A storm "recurves" if its latitude first increases (poleward movement)
    # then decreases, OR vice versa in the SH. Simplified: detect a local
    # maximum in absolute latitude along the track.
    def _detect_recurvature(sub):
        lats = sub["lat"].values
        if len(lats) < 6:
            return pd.Series({"has_recurvature": False, "lat_recurvature": np.nan})
        # Use absolute latitude for both hemispheres
        abs_lat = np.abs(lats)
        # Smooth to avoid noise-induced false detections
        if len(abs_lat) >= 5:
            kernel = np.ones(5) / 5
            smooth = np.convolve(abs_lat, kernel, mode="same")
        else:
            smooth = abs_lat
        # Find the index of maximum poleward extent
        peak_idx = np.argmax(smooth)
        # Recurvature if peak is not at the start or end (with margin)
        margin = max(2, len(lats) // 5)
        if margin < peak_idx < len(lats) - margin:
            # Check that there's meaningful equatorward motion after peak
            post_peak = smooth[peak_idx:]
            if len(post_peak) > 2 and (smooth[peak_idx] - post_peak[-1]) > 1.0:
                return pd.Series(
                    {"has_recurvature": True, "lat_recurvature": lats[peak_idx]}
                )
        return pd.Series({"has_recurvature": False, "lat_recurvature": np.nan})

    recurv = catalog_sorted.groupby("global_storm_uid", sort=False).apply(
        _detect_recurvature
    )
    agg = agg.join(recurv, how="left")

    agg.index.name = "storm_uid"
    return agg


# ═══════════════════════════════════════════════════════════════════════
# EXPORTABLE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def enhanced_lifetime_distribution(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced per-storm summary — drop-in replacement for
    evaluation.lifetime_distribution() with additional columns:

      lf_pressure, lf_wind, lat_lmi, lon_lmi, mean_speed_kmh,
      median_speed_kmh, total_dist_km, max_intensification_3h,
      max_weakening_3h, has_recurvature, lat_recurvature

    Export to CSV and use in evaluation_pipeline.py for the new figures.
    """
    agg = _per_storm_agg_extended(catalog)
    return agg.reset_index()


def translation_speed_distribution(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-timestep translation speed for every TC in the catalog.
    Returns a DataFrame with columns:
      global_storm_uid, timestep, lat, lon, speed_kmh, dist_km

    Use this for histograms / KDEs of translation speed.
    """
    cat = catalog.sort_values(["global_storm_uid", "timestep"]).reset_index(drop=True)
    shift = cat.groupby("global_storm_uid", sort=False)[["lat", "lon"]].shift(1)
    dist = _haversine_vec(
        shift["lon"].values,
        shift["lat"].values,
        cat["lon"].values,
        cat["lat"].values,
    )
    out = cat[["global_storm_uid", "timestep", "lat", "lon"]].copy()
    out["dist_km"] = dist
    out["speed_kmh"] = dist / 3.0
    # Drop first step of each storm (NaN from shift)
    out = out.dropna(subset=["speed_kmh"])
    return out


def pressure_tendency_distribution(
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-timestep pressure tendency (Δp over 3h) for every TC.
    Returns DataFrame with columns:
      global_storm_uid, timestep, pressure, dp_3h, wind, lat, lon

    Negative dp_3h = intensification, positive = weakening.
    Use this to validate the two-piece normal residual model (Eq. 12).
    """
    cat = catalog.sort_values(["global_storm_uid", "timestep"]).reset_index(drop=True)
    p_prev = cat.groupby("global_storm_uid", sort=False)["pressure"].shift(1)
    out = cat[["global_storm_uid", "timestep", "pressure", "wind", "lat", "lon"]].copy()
    out["dp_3h"] = cat["pressure"] - p_prev
    out = out.dropna(subset=["dp_3h"])
    return out


def recurvature_stats(catalog: pd.DataFrame) -> dict:
    """
    Basin-level recurvature statistics.

    Returns
    -------
    dict with:
      recurvature_fraction : fraction of storms that recurve
      mean_lat_recurvature : mean latitude of recurvature point
      std_lat_recurvature  : std of recurvature latitude
    """
    agg = _per_storm_agg_extended(catalog)
    frac = agg["has_recurvature"].mean()
    recurv_lats = agg.loc[agg["has_recurvature"] == True, "lat_recurvature"]
    return {
        "recurvature_fraction": float(frac),
        "n_recurving": int(agg["has_recurvature"].sum()),
        "n_total": len(agg),
        "mean_lat_recurvature": float(recurv_lats.mean())
        if len(recurv_lats)
        else np.nan,
        "std_lat_recurvature": float(recurv_lats.std()) if len(recurv_lats) else np.nan,
    }


def coastline_landfall_density(
    catalog: pd.DataFrame,
    n_years: float,
    lat_bin_size: float = 2.0,
    lon_bin_size: float = 2.0,
) -> pd.DataFrame:
    """
    Landfall event density binned along the coastline.

    Takes all timesteps where landfall == 1, bins by lat/lon, normalizes
    to events per year per bin. Only the first landfall per storm is counted.

    Returns DataFrame with columns: lat_center, lon_center, landfalls_per_yr
    """
    # First landfall per storm
    lf = catalog[catalog["landfall"] == 1].copy()
    first_lf = lf.groupby("global_storm_uid", sort=False).first().reset_index()

    lat_bins = np.arange(
        np.floor(first_lf["lat"].min()),
        np.ceil(first_lf["lat"].max()) + lat_bin_size,
        lat_bin_size,
    )
    lon_bins = np.arange(
        np.floor(first_lf["lon"].min()),
        np.ceil(first_lf["lon"].max()) + lon_bin_size,
        lon_bin_size,
    )

    H, lon_edges, lat_edges = np.histogram2d(
        first_lf["lon"].values,
        first_lf["lat"].values,
        bins=[lon_bins, lat_bins],
    )
    density = H.T / n_years

    # Build a tidy DataFrame
    rows = []
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    for i, lat_c in enumerate(lat_centers):
        for j, lon_c in enumerate(lon_centers):
            if density[i, j] > 0:
                rows.append(
                    {
                        "lat_center": lat_c,
                        "lon_center": lon_c,
                        "landfalls_per_yr": density[i, j],
                    }
                )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER — add to run_evaluation.py
# ═══════════════════════════════════════════════════════════════════════


def export_track_metrics(
    catalog: pd.DataFrame,
    n_years: float,
    outdir: str,
    label: str,
):
    """
    One-call export of all track-level metric CSVs.
    Meant to be called from run_evaluation.py alongside the existing exports.

    Generates:
      {outdir}/lifetime_enhanced_{label}.csv
      {outdir}/speed_distribution_{label}.csv
      {outdir}/pressure_tendency_{label}.csv
      {outdir}/landfall_density_{label}.csv
      {outdir}/recurvature_{label}.csv

    Example usage in run_evaluation.py:
        from track_metrics import export_track_metrics
        export_track_metrics(cat, n_yrs, str(cand_outdir), f"{cand_name}_{phase_label}")
    """
    import os

    os.makedirs(outdir, exist_ok=True)

    print(f"    Exporting enhanced lifetime for {label}...")
    lt = enhanced_lifetime_distribution(catalog)
    lt.to_csv(os.path.join(outdir, f"lifetime_enhanced_{label}.csv"), index=False)

    print(f"    Exporting translation speed for {label}...")
    # For large catalogs, subsample to avoid giant CSVs
    speed = translation_speed_distribution(catalog)
    if len(speed) > 2_000_000:
        speed = speed.sample(2_000_000, random_state=42)
    speed.to_csv(
        os.path.join(outdir, f"speed_distribution_{label}.csv"),
        index=False,
        float_format="%.2f",
    )

    print(f"    Exporting pressure tendency for {label}...")
    dp = pressure_tendency_distribution(catalog)
    # Export summary stats instead of raw data (too large)
    dp_summary = (
        dp.groupby("global_storm_uid")
        .agg(
            mean_dp=("dp_3h", "mean"),
            min_dp=("dp_3h", "min"),
            max_dp=("dp_3h", "max"),
            std_dp=("dp_3h", "std"),
            n_steps=("dp_3h", "count"),
        )
        .reset_index()
    )
    dp_summary.to_csv(
        os.path.join(outdir, f"pressure_tendency_{label}.csv"),
        index=False,
        float_format="%.3f",
    )

    # Also export the raw dp distribution as a histogram (compact)
    bins = np.arange(-30, 31, 0.5)
    hist, edges = np.histogram(dp["dp_3h"].dropna(), bins=bins, density=True)
    dp_hist = pd.DataFrame(
        {
            "dp_3h_center": (edges[:-1] + edges[1:]) / 2,
            "density": hist,
        }
    )
    dp_hist.to_csv(
        os.path.join(outdir, f"dp_histogram_{label}.csv"),
        index=False,
        float_format="%.6f",
    )

    print(f"    Exporting landfall density for {label}...")
    lf_dens = coastline_landfall_density(catalog, n_years)
    lf_dens.to_csv(
        os.path.join(outdir, f"landfall_density_{label}.csv"),
        index=False,
        float_format="%.4f",
    )

    print(f"    Exporting recurvature stats for {label}...")
    recurv = recurvature_stats(catalog)
    pd.DataFrame([recurv]).to_csv(
        os.path.join(outdir, f"recurvature_{label}.csv"),
        index=False,
        float_format="%.4f",
    )

    print(f"    Done: {label}")
