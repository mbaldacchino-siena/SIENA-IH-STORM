"""Figure 3 - anomaly preservation check.

Delta correction is an additive shift of the climatological mean. The
*anomaly* (deviation from the forecast-source's own climatology) must be
unchanged. This figure verifies that — at every (init, lead, member, cell)
the anomaly in raw == anomaly in corrected, i.e. all points lie on the
1:1 line.

This is what defends the approach: we're not reshaping the forecast, only
re-anchoring its baseline.

Anomaly is defined here as "deviation from the dataset's own mean over
the validation period, per calendar month of the valid date". That is a
locally-defined anomaly: both raw and corrected series are centered using
their own means, so the comparison is apples-to-apples even though raw
and corrected have different absolute offsets.
"""

from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from FORECAST.SEAS5.VALIDATION import datasets


def _anomaly_per_calendar_month(
    forecast: xr.DataArray,
    sample_every: int = 1,
) -> xr.DataArray:
    """Per-cell anomaly relative to this dataset's mean for each calendar valid month.

    For each grid cell and each valid calendar month (1-12), we compute the
    mean over all (init, lead, member) samples that fall on that cell and
    month, then subtract.

    Delta correction adds a per-cell, per-month constant. Subtracting the
    per-cell per-month mean removes that constant, so raw and corrected
    anomalies must match exactly. An earlier version lumped lat/lon into
    the group mean, which only holds if the offset is spatially uniform —
    wrong for delta correction.
    """
    import pandas as pd

    init_pd = pd.to_datetime(forecast["time"].values)
    leads = forecast["forecastMonth"].values
    # SEAS5 lead 1 = init month itself for instantaneous monthly means.
    # valid_month = ((init_month - 1) + (lead - 1)) % 12 + 1
    valid_months = np.array(
        [
            [((it.month - 1) + (int(lead) - 1)) % 12 + 1 for lead in leads]
            for it in init_pd
        ]
    )

    forecast = forecast.assign_coords(
        valid_month=(("time", "forecastMonth"), valid_months)
    )

    # Per-cell mean across (time, forecastMonth, number) grouped by valid_month.
    # Stack (time, forecastMonth) first so groupby can operate along a 1-D axis
    # while keeping lat/lon/number intact.
    stacked = forecast.stack(sample=("time", "forecastMonth"))
    group_mean = stacked.groupby("valid_month").mean(dim=("sample", "number"))
    # group_mean has dims (valid_month, lat, lon)

    anom = stacked.groupby("valid_month") - group_mean
    anom = anom.unstack("sample")

    if sample_every > 1:
        anom = anom.isel(
            time=slice(None, None, sample_every),
            number=slice(None, None, sample_every),
        )
    return anom


def plot_anomaly_scatter(
    short: str,
    years: range,
    level_hPa: Optional[int] = None,
    domain: Optional[dict] = None,
    max_points: int = 100_000,
    ax: Optional[plt.Axes] = None,
    **load_kw,
) -> plt.Axes:
    """Scatter raw anomaly vs corrected anomaly; should sit on the 1:1 line.

    Parameters
    ----------
    max_points : cap on the number of scattered points (hex-binned above).
    """
    raw, corr, _ = datasets.load_trio(
        short, years, level_hPa=level_hPa, domain=domain, **load_kw
    )

    raw_anom = _anomaly_per_calendar_month(raw).values.ravel()
    corr_anom = _anomaly_per_calendar_month(corr).values.ravel()

    # Drop NaN pairs
    mask = np.isfinite(raw_anom) & np.isfinite(corr_anom)
    raw_anom, corr_anom = raw_anom[mask], corr_anom[mask]

    # Subsample for rendering
    if raw_anom.size > max_points:
        idx = np.random.default_rng(0).choice(raw_anom.size, max_points, replace=False)
        raw_anom, corr_anom = raw_anom[idx], corr_anom[idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.hexbin(raw_anom, corr_anom, gridsize=60, bins="log", mincnt=1, cmap="viridis")
    lim = float(np.nanmax(np.abs(np.concatenate([raw_anom, corr_anom]))))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="1:1")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"Raw SEAS5 anomaly ({short})")
    ax.set_ylabel(f"Corrected SEAS5 anomaly ({short})")
    level_label = f" @ {level_hPa} hPa" if level_hPa else ""
    ax.set_title(
        f"Anomaly preservation | {short}{level_label} | {min(years)}-{max(years)}"
    )
    ax.legend(loc="upper left")

    # Numeric check - correlation and slope should be ~1
    corr_coef = np.corrcoef(raw_anom, corr_anom)[0, 1]
    slope = np.polyfit(raw_anom, corr_anom, 1)[0]
    ax.text(
        0.05,
        0.95,
        f"corr={corr_coef:.4f}\nslope={slope:.4f}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    return ax
