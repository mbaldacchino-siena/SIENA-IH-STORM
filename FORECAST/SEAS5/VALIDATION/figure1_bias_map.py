"""Figure 1 - spatial map of mean raw SEAS5 bias against ERA5.

Shows where SEAS5 is systematically off from ERA5 over the validation
period, to motivate the need for spatially-structured correction.
"""

from __future__ import annotations
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from FORECAST.SEAS5.VALIDATION import datasets


def compute_raw_bias(
    raw: xr.DataArray,
    era5: xr.DataArray,
    lead: Union[int, str, List[int]] = "all",
    reduce: str = "mean",
) -> xr.DataArray:
    """Mean (or median) of (raw - era5) across inits, members, and selected leads.

    Parameters
    ----------
    raw : (time, forecastMonth, number, lat, lon) DataArray
    era5 : (time, lat, lon) DataArray — monthly means for every valid date
    lead : int | 'all' | list of int
        Which leads to include before averaging.
    reduce : 'mean' | 'median'
    """
    era5_matched = datasets.match_era5_to_forecast(raw, era5)
    # era5_matched: (time, forecastMonth, lat, lon) — broadcast against members
    diff = raw - era5_matched

    if lead != "all":
        leads = [lead] if np.isscalar(lead) else list(lead)
        diff = diff.sel(forecastMonth=leads)

    # Collapse time, forecastMonth, number
    collapse_dims = [d for d in ("time", "forecastMonth", "number") if d in diff.dims]
    reducer = {"mean": "mean", "median": "median"}[reduce]
    return getattr(diff, reducer)(dim=collapse_dims)


def plot_bias_map(
    short: str,
    years: range,
    lead: Union[int, str, List[int]] = "all",
    reduce: str = "mean",
    level_hPa: Optional[int] = None,
    domain: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **load_kw,
) -> plt.Axes:
    """Plot the mean raw-bias map for one variable.

    Example
    -------
    >>> plot_bias_map("sst", range(2017, 2026), lead=3,
    ...               domain={"lat_min": 0, "lat_max": 40,
    ...                       "lon_min": -100, "lon_max": 0})
    """
    raw, _, era5 = datasets.load_trio(
        short, years, level_hPa=level_hPa, domain=domain, **load_kw
    )
    bias = compute_raw_bias(raw, era5, lead=lead, reduce=reduce)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Symmetric color range if not specified
    if vmin is None or vmax is None:
        v = float(np.nanmax(np.abs(bias.values)))
        vmin, vmax = -v, v

    bias.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={"label": f"{reduce} bias ({short})"},
    )

    lead_label = "all leads" if lead == "all" else f"lead {lead}"
    level_label = f" @ {level_hPa} hPa" if level_hPa else ""
    ax.set_title(
        f"Raw SEAS5 - ERA5 {reduce} bias | {short}{level_label} | "
        f"{min(years)}-{max(years)} | {lead_label}"
    )
    return ax
