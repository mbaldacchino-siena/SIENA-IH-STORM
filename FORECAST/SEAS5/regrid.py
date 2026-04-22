"""Regridding utilities to align ERA5 (0.25 deg) to SEAS5 (1 deg) grid.

For monthly-mean fields at continental scales, bilinear interpolation is
standard practice and preserves large-scale features. We use xarray's
`interp_like` (scipy backend) — no ESMF dependency required.

If you need conservative regridding for precipitation-like fields, swap in
xESMF here — the interface stays the same.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def _lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    """Return (lat_name, lon_name) used in the dataset."""
    lat = "latitude" if "latitude" in ds.coords else "lat"
    lon = "longitude" if "longitude" in ds.coords else "lon"
    if lat not in ds.coords or lon not in ds.coords:
        raise KeyError(f"Could not find lat/lon coords. Available: {list(ds.coords)}")
    return lat, lon


def _normalize_longitudes(ds: xr.Dataset, target_ds: xr.Dataset) -> xr.Dataset:
    """Align longitude conventions (0-360 vs -180-180) between datasets."""
    src_lat, src_lon = _lat_lon_names(ds)
    tgt_lat, tgt_lon = _lat_lon_names(target_ds)

    src_lon_max = float(ds[src_lon].max())
    tgt_lon_max = float(target_ds[tgt_lon].max())
    src_lon_min = float(ds[src_lon].min())
    tgt_lon_min = float(target_ds[tgt_lon].min())

    source_is_0_360 = src_lon_max > 180
    target_is_0_360 = tgt_lon_max > 180

    if source_is_0_360 and not target_is_0_360:
        logger.info("Converting source longitude: 0-360 -> -180-180")
        ds = ds.assign_coords({src_lon: (((ds[src_lon] + 180) % 360) - 180)}).sortby(
            src_lon
        )
    elif not source_is_0_360 and target_is_0_360:
        logger.info("Converting source longitude: -180-180 -> 0-360")
        ds = ds.assign_coords({src_lon: (ds[src_lon] % 360)}).sortby(src_lon)

    return ds


def get_target_grid_from_seas5(seas5_file: Path) -> xr.Dataset:
    """Load a SEAS5 file and return an empty dataset carrying its grid.

    The returned object has lat/lon coords matching SEAS5 native grid.
    Use it as the target in `regrid_like()`.
    """
    ds = xr.open_dataset(seas5_file)
    lat, lon = _lat_lon_names(ds)
    grid = xr.Dataset(coords={lat: ds[lat], lon: ds[lon]})
    grid.attrs["source"] = str(seas5_file.name)
    return grid


def regrid_like(
    source: xr.Dataset,
    target: xr.Dataset,
    method: str = "linear",
) -> xr.Dataset:
    """Regrid source dataset onto target grid using bilinear interpolation.

    Handles longitude convention mismatches (0-360 vs -180-180) automatically.
    """
    source = _normalize_longitudes(source, target)
    tgt_lat, tgt_lon = _lat_lon_names(target)

    logger.info(
        "Regridding: source %dx%d -> target %dx%d (method=%s)",
        source.sizes.get(_lat_lon_names(source)[0], 0),
        source.sizes.get(_lat_lon_names(source)[1], 0),
        target.sizes[tgt_lat],
        target.sizes[tgt_lon],
        method,
    )

    regridded = source.interp(
        {tgt_lat: target[tgt_lat], tgt_lon: target[tgt_lon]},
        method=method,
    )
    return regridded
