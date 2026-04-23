"""Figure 2 - RMSE vs lead time for raw and corrected SEAS5.

Spatial RMSE against ERA5, averaged over init dates and members, plotted
as a function of lead time. The story: raw RMSE grows with lead because
the systematic bias compounds; corrected RMSE is roughly flat because
the systematic component has been removed.
"""

from __future__ import annotations
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from FORECAST.SEAS5.VALIDATION import datasets


def _spatial_rmse(diff: xr.DataArray) -> xr.DataArray:
    """RMSE over spatial dims, leaving (time, lead, member) intact."""
    lat_name = "latitude" if "latitude" in diff.dims else "lat"
    lon_name = "longitude" if "longitude" in diff.dims else "lon"
    # Cosine-latitude weighting for an honest area-weighted RMSE
    w = np.cos(np.deg2rad(diff[lat_name]))
    w2 = (w / w.mean()).astype("float32")
    sq = (diff**2) * w2
    return np.sqrt(sq.mean(dim=[lat_name, lon_name]))


def compute_rmse_by_lead(
    forecast: xr.DataArray,
    era5: xr.DataArray,
) -> xr.DataArray:
    """RMSE as function of lead, averaged across init dates and members."""
    era5_matched = datasets.match_era5_to_forecast(forecast, era5)
    diff = forecast - era5_matched  # (time, lead, number, lat, lon)
    rmse_per_sample = _spatial_rmse(diff)  # (time, lead, number)
    collapse = [d for d in ("time", "number") if d in rmse_per_sample.dims]
    return rmse_per_sample.mean(dim=collapse)


def plot_rmse_vs_lead(
    variables: Iterable[str],
    years: range,
    level_hPa: Optional[int] = None,
    domain: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    **load_kw,
) -> plt.Axes:
    """Plot raw vs corrected RMSE as a function of lead, for each variable.

    Parameters
    ----------
    variables : iterable of short names (e.g. ["sst", "msl"])
        One subplot row per variable when len > 1.
    """
    variables = list(variables)
    if ax is None:
        fig, axes = plt.subplots(
            len(variables), 1, figsize=(7, 3 * len(variables)), squeeze=False
        )
        axes = axes.flatten()
    else:
        axes = [ax]

    for axi, var in zip(axes, variables):
        raw, corr, era5 = datasets.load_trio(
            var, years, level_hPa=level_hPa, domain=domain, **load_kw
        )
        rmse_raw = compute_rmse_by_lead(raw, era5)
        rmse_corr = compute_rmse_by_lead(corr, era5)

        leads = rmse_raw.forecastMonth.values
        axi.plot(leads, rmse_raw.values, "o-", label="Raw SEAS5", color="C3")
        axi.plot(leads, rmse_corr.values, "s-", label="Corrected", color="C0")
        axi.set_xlabel("Lead month")
        axi.set_ylabel(f"RMSE ({var})")
        level_label = f" @ {level_hPa} hPa" if level_hPa else ""
        axi.set_title(f"{var}{level_label}")
        axi.grid(alpha=0.3)
        axi.legend()

    return axes[0] if len(axes) == 1 else axes
