"""Shared data-loading utilities for the validation plots.

Every figure needs three things:
    - X_raw      : uncorrected SEAS5 absolute field
    - X_corr     : bias-corrected SEAS5 (from the pipeline output)
    - X_era5     : the ERA5 reference (monthly means for matching dates)

This module provides a single `load_trio()` function that returns all three
as aligned xarray DataArrays for a given variable and spatial subset. The
plotting modules only have to compute statistics; they don't have to know
where the files live.

ASSUMED FILE LAYOUT
===================
Corrected (produced by the pipeline):
    {CORRECTED_DIR}/seas5corrected_{short}[_pl]_{Y0}-{Y1}.nc

Raw uncorrected SEAS5 (you download separately):
    {RAW_SEAS5_DIR}/seas5raw_{short}[_pl]_{Y0}-{Y1}.nc
    Expected variable names: same short names as corrected files
    Expected dims: (time, forecastMonth, number, latitude, longitude[, pressure_level])

ERA5 monthly means (for the validation period, distinct from the 1993-2016
climatology used by the pipeline):
    {ERA5_VALID_DIR}/era5_{short}[_pl]_{Y0}-{Y1}.nc
    Dims: (time, latitude, longitude[, pressure_level])

If your raw SEAS5 lives elsewhere (e.g. you used the legacy MASTER download
that produced one file per init), override the `raw_path` / `era5_path`
arguments in `load_trio()`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from FORECAST.SEAS5 import config, pipeline, regrid


# =============================================================================
# Default paths (edit to match your layout)
# =============================================================================
RAW_SEAS5_DIR = config.DATA_DIR / "seas5_raw"
ERA5_VALID_DIR = config.DATA_DIR / "era5_validation"
RAW_SEAS5_DIR.mkdir(parents=True, exist_ok=True)
ERA5_VALID_DIR.mkdir(parents=True, exist_ok=True)


def _is_pressure(short: str) -> bool:
    return short in config.PRESSURE_LEVEL_VARS.values()


def _default_path(kind: str, short: str, years) -> Path:
    """Build a default file path for a given data kind."""
    suffix = "_pl" if _is_pressure(short) else ""
    y0, y1 = min(years), max(years)
    if kind == "raw":
        return RAW_SEAS5_DIR / f"seas5raw_{short}{suffix}_{y0}-{y1}.nc"
    if kind == "era5":
        return ERA5_VALID_DIR / f"era5_{short}{suffix}_{y0}-{y1}.nc"
    if kind == "corrected":
        return pipeline.corrected_path(short, _is_pressure(short), list(years))
    raise ValueError(f"Unknown kind: {kind}")


# =============================================================================
# Variable extraction helpers (normalise CDS naming quirks)
# =============================================================================
def _pick_main_var(ds: xr.Dataset, short: str) -> xr.DataArray:
    """Pick the primary DataArray from a dataset, renaming to short form."""
    if short in ds.data_vars:
        return ds[short]

    alternatives = {
        "sst": ["sea_surface_temperature", "sst_anomaly"],
        "msl": ["mean_sea_level_pressure", "msl_anomaly"],
        "u": ["u_component_of_wind", "u_anomaly"],
        "v": ["v_component_of_wind", "v_anomaly"],
        "t": ["temperature", "t_anomaly"],
        "q": ["specific_humidity", "q_anomaly"],
    }
    for alt in alternatives.get(short, []):
        if alt in ds.data_vars:
            return ds[alt]

    real = [v for v in ds.data_vars if not v.endswith("_bnds")]
    if len(real) == 1:
        return ds[real[0]]
    raise KeyError(f"Cannot find variable '{short}' in dataset: {list(ds.data_vars)}")


# =============================================================================
# Spatial subsetting
# =============================================================================
def _apply_domain(da: xr.DataArray, domain: Optional[dict]) -> xr.DataArray:
    """Crop to a lat/lon box. Handles both 0-360 and -180-180 conventions."""
    if domain is None:
        return da

    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    lat_sl = (domain["lat_min"], domain["lat_max"])
    # respect ascending/descending lat
    if da[lat_name].values[0] > da[lat_name].values[-1]:
        lat_sl = (domain["lat_max"], domain["lat_min"])
    da = da.sel({lat_name: slice(*lat_sl)})

    lon_max_in = float(da[lon_name].max())
    w, e = domain["lon_min"], domain["lon_max"]
    if lon_max_in > 180 and (w < 0 or e < 0):
        w = w % 360
        e = e % 360
    if w <= e:
        da = da.sel({lon_name: slice(w, e)})
    else:  # wraps dateline
        da = xr.concat(
            [da.sel({lon_name: slice(w, 360)}), da.sel({lon_name: slice(0, e)})],
            dim=lon_name,
        )
    return da


# =============================================================================
# Grid alignment
# =============================================================================
def _align_to(da: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Regrid da onto reference's lat/lon grid (bilinear) if grids differ."""
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    same_lat = da[lat_name].size == reference[lat_name].size and np.allclose(
        da[lat_name].values, reference[lat_name].values
    )
    same_lon = da[lon_name].size == reference[lon_name].size and np.allclose(
        da[lon_name].values, reference[lon_name].values
    )
    if same_lat and same_lon:
        return da

    ref_ds = xr.Dataset(
        coords={lat_name: reference[lat_name], lon_name: reference[lon_name]}
    )
    return regrid.regrid_like(da.to_dataset(name=da.name or "_tmp"), ref_ds)[
        da.name or "_tmp"
    ]


# =============================================================================
# Public API
# =============================================================================
def load_trio(
    short: str,
    years: range,
    level_hPa: Optional[int] = None,
    domain: Optional[dict] = None,
    raw_path: Optional[Union[str, Path]] = None,
    era5_path: Optional[Union[str, Path]] = None,
    corrected_path: Optional[Union[str, Path]] = None,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load (X_raw, X_corr, X_era5) for one variable, aligned on the same grid.

    Parameters
    ----------
    short : str
        Short variable name (sst, msl, t, u, v, q).
    years : iterable of int
        Years covered by the file range (must match the file naming).
    level_hPa : int, optional
        For pressure-level variables, select a single level after loading.
    domain : dict with lat_min/lat_max/lon_min/lon_max, optional
        Spatial subset applied after loading. Defaults to full file extent.
    raw_path, era5_path, corrected_path : path-like, optional
        Override default paths if your files live elsewhere.

    Returns
    -------
    raw, corr, era5 : xr.DataArray
        Values on the SEAS5 grid (corrected file's grid).
        raw and corr have dims (time, forecastMonth, number, lat, lon);
        era5 has dims (time, lat, lon) — one value per month, not per init.
    """
    rp = Path(raw_path) if raw_path else _default_path("raw", short, years)
    cp = (
        Path(corrected_path)
        if corrected_path
        else _default_path("corrected", short, years)
    )
    ep = Path(era5_path) if era5_path else _default_path("era5", short, years)

    for p, kind in [(rp, "raw"), (cp, "corrected"), (ep, "ERA5 validation")]:
        if not p.exists():
            raise FileNotFoundError(f"{kind} file not found: {p}")

    raw = _pick_main_var(xr.open_dataset(rp), short)
    corr = _pick_main_var(xr.open_dataset(cp), short)
    era5 = _pick_main_var(xr.open_dataset(ep), short)

    if level_hPa is not None:
        for da in (raw, corr, era5):
            if "pressure_level" not in da.dims:
                raise ValueError(
                    f"level_hPa requested but {da.name} has no pressure_level"
                )
        sel = {"pressure_level": level_hPa}
        raw = raw.sel(sel)
        corr = corr.sel(sel)
        era5 = era5.sel(sel)

    # Align ERA5 grid to the corrected grid (the canonical one)
    era5 = _align_to(era5, corr)
    # raw should already be on SEAS5 native grid, but enforce for safety
    raw = _align_to(raw, corr)

    raw = _apply_domain(raw, domain)
    corr = _apply_domain(corr, domain)
    era5 = _apply_domain(era5, domain)

    return raw, corr, era5


# =============================================================================
# Index utilities — match (init, lead) to ERA5 calendar months
# =============================================================================
def match_era5_to_forecast(
    forecast: xr.DataArray,
    era5: xr.DataArray,
    init_coord: str = "time",
    lead_coord: str = "forecastMonth",
    lead_offset: int = 0,
) -> xr.DataArray:
    """For every (init, lead) in forecast, pick the matching ERA5 monthly mean.

    Returns an array broadcastable against forecast with dims (init, lead, lat, lon).

    The matching calendar date is
        valid_date = init_date + (lead - 1 + lead_offset) months

    For SEAS5 instantaneous monthly means, lead_offset=0 (lead 1 = init month).
    """
    import pandas as pd

    init_times = forecast[init_coord].values
    leads = forecast[lead_coord].values

    # Map (init, lead) -> valid_time
    init_pd = pd.to_datetime(init_times)
    valid_times = np.empty((len(init_pd), len(leads)), dtype="datetime64[ns]")
    for i, it in enumerate(init_pd):
        for j, lead in enumerate(leads):
            offset_months = int(lead) - 1 + lead_offset
            valid = (it + pd.DateOffset(months=offset_months)).replace(day=1)
            valid_times[i, j] = np.datetime64(valid)

    # ERA5 uses 'time' typically; handle 'valid_time' too
    era5_time_coord = None
    for cand in ("time", "valid_time"):
        if cand in era5.coords:
            era5_time_coord = cand
            break
    if era5_time_coord is None:
        raise KeyError(f"No time coord in ERA5 (have {list(era5.coords)})")

    # Normalise ERA5 time stamps to month-start
    era5_times = pd.to_datetime(era5[era5_time_coord].values)
    era5_month_start = np.array(
        [np.datetime64(pd.Timestamp(t.year, t.month, 1)) for t in era5_times]
    )
    era5 = era5.assign_coords({era5_time_coord: era5_month_start})

    # Build selector
    flat_valid = valid_times.reshape(-1)
    try:
        picked = era5.sel({era5_time_coord: flat_valid})
    except KeyError as ex:
        raise KeyError(
            f"Some valid times are missing from ERA5 file. "
            f"ERA5 covers: {era5[era5_time_coord].values.min()}..{era5[era5_time_coord].values.max()}. "
            f"Requested range: {flat_valid.min()}..{flat_valid.max()}"
        ) from ex

    # Reshape back to (init, lead, lat, lon)
    lat_name = "latitude" if "latitude" in picked.dims else "lat"
    lon_name = "longitude" if "longitude" in picked.dims else "lon"
    out = picked.values.reshape(
        len(init_pd), len(leads), picked.sizes[lat_name], picked.sizes[lon_name]
    )
    return xr.DataArray(
        out,
        dims=(init_coord, lead_coord, lat_name, lon_name),
        coords={
            init_coord: forecast[init_coord],
            lead_coord: forecast[lead_coord],
            lat_name: picked[lat_name],
            lon_name: picked[lon_name],
        },
        name=era5.name,
    )
