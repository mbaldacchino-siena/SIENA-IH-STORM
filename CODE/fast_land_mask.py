"""
Fast land mask — drop-in replacement for climada.util.coordinates.coord_on_land

CLIMADA's coord_on_land loads Natural Earth shapefiles and runs shapely
point-in-polygon tests on every call.  This module rasterizes the land
polygons once into a boolean numpy array, then resolves queries via pure
array indexing — typically 100-1000x faster for repeated or bulk lookups.

Usage
-----
    from fast_land_mask import FastLandMask

    mask = FastLandMask()                        # one-time cost (~1-5 s)
    on_land = mask.is_land(lat_array, lon_array)  # microseconds per point

    # or use the module-level drop-in:
    from fast_land_mask import coord_on_land
    on_land = coord_on_land(lat_array, lon_array)

Backends
--------
1. "natural_earth" (default) — rasterizes cartopy's Natural Earth 'land'
   shapefile.  Best fidelity, consistent with what CLIMADA uses internally.
   Requires cartopy (already a CLIMADA dependency).
2. "global_land_mask" — uses the global-land-mask package which ships a
   pre-computed ~1 km raster.  Zero init cost, pip-installable, but adds
   an extra dependency.

Disk caching (natural_earth backend)
-------------------------------------
Pass cache_path to save the rasterized mask as a compressed .npz.
Subsequent loads skip rasterization entirely (~0.05 s vs ~2-5 s).

Dependencies: numpy + one of {cartopy+shapely, global-land-mask}.
All except global-land-mask are already present in any CLIMADA environment.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class FastLandMask:
    """Pre-rasterized global land mask for fast coordinate lookups.

    Parameters
    ----------
    resolution : float
        Grid cell size in degrees (natural_earth backend only).
        Default 0.04 deg (~4.4 km).  Use 0.01 for ~1 km precision.
    ne_resolution : str
        Natural Earth vector resolution: '10m', '50m', or '110m'.
        Only used with 'natural_earth' backend.  Default '50m'.
    backend : str
        'natural_earth' or 'global_land_mask'.  Default 'natural_earth'.
    cache_path : str or Path, optional
        Save/load rasterized mask here (natural_earth backend only).
    """

    def __init__(
        self,
        resolution: float = 0.04,
        ne_resolution: str = "50m",
        backend: str = "natural_earth",
        cache_path: Optional[Union[str, Path]] = None,
    ):
        self.resolution = resolution
        self.backend = backend
        self._globe_module = None

        if backend == "global_land_mask":
            self._init_global_land_mask()
        elif backend == "natural_earth":
            self._init_natural_earth(resolution, ne_resolution, cache_path)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_global_land_mask(self):
        from global_land_mask import globe

        self._globe_module = globe
        self._mask = None
        self._n_lat = self._n_lon = None
        logger.info("FastLandMask ready (global-land-mask backend).")

    def _init_natural_earth(self, resolution, ne_resolution, cache_path):
        self.ne_resolution = ne_resolution
        self._n_lon = int(round(360.0 / resolution))
        self._n_lat = int(round(180.0 / resolution))
        self._lon_min = -180.0
        self._lat_max = 90.0

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading cached land mask from %s", cache_path)
                self._mask = np.load(cache_path)["mask"]
                if self._mask.shape != (self._n_lat, self._n_lon):
                    raise ValueError(
                        f"Cached shape {self._mask.shape} != expected "
                        f"({self._n_lat}, {self._n_lon}).  Delete cache and retry."
                    )
                return

        logger.info(
            "Rasterizing land mask at %.4f deg (%dx%d) from NE %s ...",
            resolution,
            self._n_lat,
            self._n_lon,
            ne_resolution,
        )
        self._mask = self._rasterize(ne_resolution)

        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, mask=self._mask)
            logger.info("Saved land mask cache -> %s", cache_path)

    def _rasterize(self, ne_resolution: str) -> np.ndarray:
        from cartopy.io import shapereader
        from shapely.geometry import shape
        from shapely.ops import unary_union
        import shapely.vectorized

        shp_path = shapereader.natural_earth(
            resolution=ne_resolution,
            category="physical",
            name="land",
        )
        reader = shapereader.Reader(shp_path)
        land_geom = unary_union([shape(rec.geometry) for rec in reader.records()])

        lon_centres = np.linspace(
            -180.0 + self.resolution / 2,
            180.0 - self.resolution / 2,
            self._n_lon,
        )
        lat_centres = np.linspace(
            90.0 - self.resolution / 2,
            -90.0 + self.resolution / 2,
            self._n_lat,
        )
        lon_grid, lat_grid = np.meshgrid(lon_centres, lat_centres)
        mask = shapely.vectorized.contains(land_geom, lon_grid, lat_grid)
        return mask.astype(np.bool_)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_land(
        self,
        lat: Union[np.ndarray, float, list],
        lon: Union[np.ndarray, float, list],
    ) -> np.ndarray:
        """Return boolean array: True where (lat, lon) is on land.

        Parameters
        ----------
        lat, lon : array-like
            Latitude(s) and longitude(s) in EPSG:4326.

        Returns
        -------
        np.ndarray of bool, or scalar bool for scalar inputs.
        """
        if self._globe_module is not None:
            return self._globe_module.is_land(
                np.asarray(lat, dtype=np.float64),
                np.asarray(lon, dtype=np.float64),
            )

        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        scalar = lat.ndim == 0
        lat = np.atleast_1d(lat)
        lon = np.atleast_1d(lon)

        lon = (lon + 180.0) % 360.0 - 180.0

        row = ((self._lat_max - lat) / self.resolution).astype(np.intp)
        col = ((lon - self._lon_min) / self.resolution).astype(np.intp)
        np.clip(row, 0, self._n_lat - 1, out=row)
        np.clip(col, 0, self._n_lon - 1, out=col)

        result = self._mask[row, col]
        return bool(result) if scalar else result

    def coord_on_land(
        self,
        lat: Union[np.ndarray, float, list],
        lon: Union[np.ndarray, float, list],
    ) -> np.ndarray:
        """Drop-in replacement for climada.util.coordinates.coord_on_land."""
        return self.is_land(lat, lon)

    @property
    def shape(self):
        if self._mask is not None:
            return self._mask.shape
        return "delegated to global_land_mask"

    def __repr__(self):
        return (
            f"FastLandMask(backend={self.backend!r}, "
            f"resolution={self.resolution} deg, shape={self.shape})"
        )


# ======================================================================
# Module-level singleton
# ======================================================================

_DEFAULT_MASK: Optional[FastLandMask] = None


def get_default_mask(**kwargs) -> FastLandMask:
    """Return (and cache) a module-level FastLandMask singleton."""
    global _DEFAULT_MASK
    if _DEFAULT_MASK is None:
        _DEFAULT_MASK = FastLandMask(**kwargs)
    return _DEFAULT_MASK


def coord_on_land(lat, lon, **mask_kwargs) -> np.ndarray:
    """Module-level drop-in for climada.util.coordinates.coord_on_land.

    First call creates the singleton; all subsequent calls are array lookups.
    """
    return get_default_mask(**mask_kwargs).is_land(lat, lon)


# ======================================================================
# Benchmark
# ======================================================================

if __name__ == "__main__":
    import time

    for backend_name in ["global_land_mask", "natural_earth"]:
        print(f"\n{'=' * 60}")
        print(f"Backend: {backend_name}")
        print("=" * 60)

        try:
            t0 = time.perf_counter()
            mask = FastLandMask(backend=backend_name)
            init_s = time.perf_counter() - t0
            print(f"  Init:  {init_s:.3f} s")
        except Exception as e:
            print(f"  Skipped -- {e}")
            continue

        assert mask.is_land(48.86, 2.35), "Paris should be land"
        assert not mask.is_land(0.0, -150.0), "Mid-Pacific should be ocean"
        assert mask.is_land(-33.80, 151.10), "Sydney inland should be land"
        assert not mask.is_land(60.0, -30.0), "N Atlantic should be ocean"
        print("  Sanity checks passed")

        rng = np.random.default_rng(42)
        for n in [10_000, 100_000, 1_000_000]:
            lats = rng.uniform(-90, 90, n)
            lons = rng.uniform(-180, 180, n)
            t0 = time.perf_counter()
            res = mask.is_land(lats, lons)
            dt = time.perf_counter() - t0
            print(
                f"  {n:>10,} pts -> {dt * 1000:8.1f} ms  "
                f"({dt / n * 1e6:.2f} us/pt, {res.mean() * 100:.1f}% land)"
            )
