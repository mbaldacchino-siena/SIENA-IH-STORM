"""
Tropical Cyclone Forecast Verification Toolkit (North Atlantic) - POOLED
=========================================================================

Compare a synthetic TC model against IBTrACS and CSU April forecasts.

Designed for the structure: each forecast year Y has K stochastic members,
where each member is a single TCTracks containing N simulated seasons of Y.
Each track Dataset has a `sim_year` attribute (or similar) identifying which
of the N seasons it belongs to. This toolkit pools all K * N seasons per
forecast year and reports the resulting distribution.

Key design choices for performance
----------------------------------
1.  Per-storm features (winds, lat/lon, on_land, hour-of-day) are extracted
    ONCE per track via `_extract_track_features`. xarray attribute access
    has substantial Python overhead, so we touch each Dataset's accessors
    exactly once per track.
2.  Per-member metrics use a vectorised per-storm record + pandas groupby
    by sim_year, avoiding Python loops over thousands of seasons.
3.  Members are processed in parallel via ProcessPoolExecutor (`fork` start
    method on Linux). With ~500ms per-member compute (1500 storms @ 0.4ms
    each), pickle/fork overhead is negligible.

Season-level metrics
--------------------
- Counts at TS (>=34 kt), HU (>=64 kt), MH (>=96 kt) - nested.
- ACE in 10^4 kt^2. Two methods:
    * 'synoptic' (default): canonical NHC definition - sum v^2/10^4 only at
      00/06/12/18 UTC samples where v >= 34 kt. Works for 6h best-track AND
      for 3h tracks aligned to synoptic hours.
    * 'scaled': sum v^2/10^4 over ALL samples where v >= 34 kt, multiplied
      by (median_timestep_h / 6). Robust for non-aligned timestamps.
- Total landfall counts at TS/HU/MH intensity.

Landfall probabilities by region
--------------------------------
Four regions x three intensities. Computed as the fraction of pooled seasons
with at least one hit at the given intensity:
  * us_total       - full US coastline (East Coast + Gulf Coast)
  * us_east_coast  - FL peninsula south/east of Cedar Key, FL up to Maine
  * us_gulf_coast  - FL panhandle west of Cedar Key westward to Brownsville
  * caribbean      - "tracking through" the box (10-20 N, 88-60 W) at the
                     given intensity. NOT a landfall criterion - matches
                     CSU's published definition exactly.

CSU values
----------
Season totals from the CSU verification archive (2017-2023) and April press
releases for 2024 / 2025. MH landfall probabilities extracted from each
year's April forecast PDF (page 2 summary box).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from climada.hazard import TCTracks

# from climada.util.coordinates import coord_on_land
from CODE.fast_land_mask import FastLandMask

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TS_KT = 34.0
HU_KT = 64.0
MH_KT = 96.0  # Category 3+

SYNOPTIC_HOURS = (0, 6, 12, 18)


# ---------------------------------------------------------------------------
# CSU April forecasts: season totals
# ---------------------------------------------------------------------------
CSU_APRIL_FORECASTS: dict[int, dict[str, float]] = {
    2017: dict(named_storms=11, hurricanes=4, major_hurricanes=2, ace=75),
    2018: dict(named_storms=14, hurricanes=7, major_hurricanes=3, ace=130),
    2019: dict(named_storms=13, hurricanes=5, major_hurricanes=2, ace=80),
    2020: dict(named_storms=16, hurricanes=8, major_hurricanes=4, ace=150),
    2021: dict(named_storms=17, hurricanes=8, major_hurricanes=4, ace=150),
    2022: dict(named_storms=19, hurricanes=9, major_hurricanes=4, ace=160),
    2023: dict(named_storms=13, hurricanes=6, major_hurricanes=2, ace=100),
    2024: dict(named_storms=23, hurricanes=11, major_hurricanes=5, ace=210),
    2025: dict(named_storms=17, hurricanes=9, major_hurricanes=4, ace=155),
}

CSU_APRIL_LANDFALL_PROBS: dict[int, dict[str, float]] = {
    2017: dict(
        lf_prob_us_total_mh=42,
        lf_prob_us_east_mh=24,
        lf_prob_us_gulf_mh=24,
        lf_prob_caribbean_mh=34,
    ),
    2018: dict(
        lf_prob_us_total_mh=63,
        lf_prob_us_east_mh=39,
        lf_prob_us_gulf_mh=38,
        lf_prob_caribbean_mh=52,
    ),
    2019: dict(
        lf_prob_us_total_mh=48,
        lf_prob_us_east_mh=28,
        lf_prob_us_gulf_mh=28,
        lf_prob_caribbean_mh=39,
    ),
    2020: dict(
        lf_prob_us_total_mh=69,
        lf_prob_us_east_mh=45,
        lf_prob_us_gulf_mh=44,
        lf_prob_caribbean_mh=58,
    ),
    2021: dict(
        lf_prob_us_total_mh=69,
        lf_prob_us_east_mh=45,
        lf_prob_us_gulf_mh=44,
        lf_prob_caribbean_mh=58,
    ),
    2022: dict(
        lf_prob_us_total_mh=71,
        lf_prob_us_east_mh=47,
        lf_prob_us_gulf_mh=46,
        lf_prob_caribbean_mh=60,
    ),
    2023: dict(
        lf_prob_us_total_mh=44,
        lf_prob_us_east_mh=22,
        lf_prob_us_gulf_mh=28,
        lf_prob_caribbean_mh=49,
    ),
    2024: dict(
        lf_prob_us_total_mh=62,
        lf_prob_us_east_mh=34,
        lf_prob_us_gulf_mh=42,
        lf_prob_caribbean_mh=66,
    ),
    2025: dict(
        lf_prob_us_total_mh=51,
        lf_prob_us_east_mh=26,
        lf_prob_us_gulf_mh=33,
        lf_prob_caribbean_mh=56,
    ),
}

LF_PROB_COLS = [
    "lf_prob_us_total_ts",
    "lf_prob_us_total_hu",
    "lf_prob_us_total_mh",
    "lf_prob_us_east_ts",
    "lf_prob_us_east_hu",
    "lf_prob_us_east_mh",
    "lf_prob_us_gulf_ts",
    "lf_prob_us_gulf_hu",
    "lf_prob_us_gulf_mh",
    "lf_prob_caribbean_ts",
    "lf_prob_caribbean_hu",
    "lf_prob_caribbean_mh",
]

# ---------------------------------------------------------------------------
# TSR (Tropical Storm Risk) April forecasts, 2017-2025
# ---------------------------------------------------------------------------
# Source: TSR April PDFs at https://www.tropicalstormrisk.com/for_hurr.html
# TSR uses 'intense_hurricanes' (Cat 3-5) which is the same definition as
# CSU's 'major_hurricanes'. Reproduced as `major_hurricanes` for column
# consistency with CSU.
#
# TSR provides US-landfalling COUNTS (TS, HU, ACE) - these are direct count
# forecasts, NOT probabilities, so they're stored under a different schema
# than CSU's lf_prob_*. They populate the `landfall_ts` and `landfall_hu`
# columns directly (just like the synthetic and observed sources do).
#
# Also: TSR did NOT issue an April forecast for 2025 (their archive jumps
# from December 2024 extended-range to the May 2025 pre-season). The 2025
# entry below uses TSR's December 2024 extended-range forecast - flagged in
# the dict to make the longer lead time visible. If you prefer to drop 2025
# from the TSR comparison, just delete that key.
TSR_APRIL_FORECASTS: dict[int, dict[str, float]] = {
    2017: dict(
        named_storms=11,
        hurricanes=4,
        major_hurricanes=2,
        ace=67,
        landfall_ts=2,
        landfall_hu=0,
        us_landfall_ace=1.0,
    ),
    2018: dict(
        named_storms=12,
        hurricanes=6,
        major_hurricanes=2,
        ace=84,
        landfall_ts=2,
        landfall_hu=1,
        us_landfall_ace=1.5,
    ),
    2019: dict(
        named_storms=12,
        hurricanes=5,
        major_hurricanes=2,
        ace=81,
        landfall_ts=2,
        landfall_hu=1,
        us_landfall_ace=1.3,
    ),
    2020: dict(
        named_storms=16,
        hurricanes=8,
        major_hurricanes=3,
        ace=130,
        landfall_ts=4,
        landfall_hu=2,
        us_landfall_ace=3.2,
    ),
    2021: dict(
        named_storms=17,
        hurricanes=8,
        major_hurricanes=3,
        ace=134,
        landfall_ts=4,
        landfall_hu=2,
        us_landfall_ace=2.5,
    ),
    2022: dict(
        named_storms=18,
        hurricanes=8,
        major_hurricanes=4,
        ace=138,
        landfall_ts=4,
        landfall_hu=2,
        us_landfall_ace=2.8,
    ),
    2023: dict(
        named_storms=12,
        hurricanes=6,
        major_hurricanes=2,
        ace=84,
        landfall_ts=3,
        landfall_hu=1,
        us_landfall_ace=1.7,
    ),
    2024: dict(
        named_storms=23,
        hurricanes=11,
        major_hurricanes=5,
        ace=217,
        landfall_ts=5,
        landfall_hu=3,
        us_landfall_ace=4.6,
    ),
    # 2025: TSR did NOT issue an April forecast - this is the December 2024
    # extended-range forecast. No US landfall forecast was published.
    2025: dict(
        named_storms=15,
        hurricanes=7,
        major_hurricanes=3,
        ace=129,
        landfall_ts=np.nan,
        landfall_hu=np.nan,
        us_landfall_ace=np.nan,
    ),
}


REGIONS = ("us_total", "us_east", "us_gulf", "caribbean")
INTENSITIES = ("ts", "hu", "mh")
INTENSITY_THRESHOLDS = ((TS_KT, "ts"), (HU_KT, "hu"), (MH_KT, "mh"))

# Which key in track.attrs identifies the simulated year. Override at the
# call site if your tracks use a different name.
DEFAULT_SIM_YEAR_ATTR = "sid"


# ---------------------------------------------------------------------------
# IBTrACS loading
# ---------------------------------------------------------------------------
def load_ibtracs_na(
    years: Iterable[int],
    *,
    provider: str | Sequence[str] | None = None,
    rescale_windspeeds: bool = True,
    estimate_missing: bool = False,
    discard_single_points: bool = True,
    file_name: str = "IBTrACS.ALL.v04r01.nc",
) -> TCTracks:
    """Load IBTrACS observed tracks for the North Atlantic basin via CLIMADA."""
    yr_min, yr_max = min(years), max(years)
    LOG.info("Loading IBTrACS for years %d-%d", yr_min, yr_max)
    return TCTracks.from_ibtracs_netcdf(
        provider=provider,
        year_range=(yr_min, yr_max),
        basin=None,
        genesis_basin="NA",
        rescale_windspeeds=rescale_windspeeds,
        estimate_missing=estimate_missing,
        discard_single_points=discard_single_points,
        file_name=file_name,
    )


def split_tracks_by_year(tracks: TCTracks) -> dict[int, TCTracks]:
    """Split a TCTracks into one TCTracks per genesis year (uses timestamps)."""
    by_year: dict[int, list[xr.Dataset]] = {}
    for ds in tracks.data:
        first_time = pd.Timestamp(ds.time.values[0])
        by_year.setdefault(int(first_time.year), []).append(ds)
    out: dict[int, TCTracks] = {}
    for year, ds_list in by_year.items():
        tt = TCTracks()
        tt.data = list(ds_list)
        out[year] = tt
    return out


# ---------------------------------------------------------------------------
# Per-track feature extraction
# ---------------------------------------------------------------------------
def _extract_track_features(track: xr.Dataset, mask: FastLandMask) -> dict:
    """Pull all numpy arrays needed for metrics from an xarray track ONCE.

    Touches each xarray accessor exactly once per track. All downstream
    metric functions read from the returned dict.
    """
    # Wind unit conversion
    tY = time()
    unit = str(track.attrs.get("max_sustained_wind_unit", "kn")).lower().strip()
    v_raw = np.asarray(track["max_sustained_wind"].values, dtype=float)
    if unit in ("kn", "kt", "knots", "knot"):
        v = v_raw
    elif unit in ("m/s", "ms", "meters/s", "meters_per_second"):
        v = v_raw / 0.514444
    elif unit in ("km/h", "kmh", "km h-1"):
        v = v_raw / 1.852
    elif unit in ("mph",):
        v = v_raw / 1.15078
    else:
        raise ValueError(
            f"Unknown wind unit {unit!r} on track "
            f"{track.attrs.get('name', track.attrs.get('sid', '?'))}"
        )

    lat = np.asarray(track.lat.values, dtype=float)
    lon = np.asarray(track.lon.values, dtype=float)
    # Normalize longitudes to the -180..+180 convention. STORM (and several
    # other synthetic generators) emit 0..360. The region predicates and
    # CSU/Caribbean boxes downstream all assume -180..+180, so we reconcile
    # here once. FastLandMask appears to handle either convention, so we
    # pass the normalized array to be safe.
    lon = ((lon + 180.0) % 360.0) - 180.0
    time_raw = track.time.values

    # Vectorised hour/minute extraction without DatetimeIndex construction
    time_ns = time_raw.astype("datetime64[ns]")
    epoch_ns = time_ns.view("int64")
    hour = ((epoch_ns // 3_600_000_000_000) % 24).astype(np.int64)
    minute = ((epoch_ns // 60_000_000_000) % 60).astype(np.int64)
    second = ((epoch_ns // 1_000_000_000) % 60).astype(np.int64)

    # Median timestep in hours
    if len(time_ns) >= 2:
        dt_ns = np.diff(epoch_ns)
        dt_h = float(np.median(dt_ns) / 3.6e12)
    else:
        dt_h = float("nan")

    # Land mask - the expensive call. ONCE per track.
    if len(lat) >= 1:
        on_land = mask.coord_on_land(lat, lon)
    else:
        on_land = np.zeros(0, dtype=bool)

    if v.size and not np.all(np.isnan(v)):
        vmax = float(np.nanmax(v))
    else:
        vmax = float("nan")

    return dict(
        v=v,
        lat=lat,
        lon=lon,
        hour=hour,
        minute=minute,
        second=second,
        dt_h=dt_h,
        on_land=on_land,
        vmax=vmax,
        n=len(v),
    )


# ---------------------------------------------------------------------------
# Per-storm metric record
# ---------------------------------------------------------------------------
def _storm_record_from_features(feat: dict, method: str = "synoptic") -> dict:
    """All per-storm metrics from precomputed features.

    Returns a flat dict suitable for stacking into a DataFrame:
      vmax, ace, is_ts, is_hu, is_mh,
      lf_ts, lf_hu, lf_mh                       (any-region landfall flags)
      lf_us_total_{ts,hu,mh}, lf_us_east_{...}, lf_us_gulf_{...},
      lf_caribbean_{ts,hu,mh}
    """
    v = feat["v"]
    lat = feat["lat"]
    lon = feat["lon"]
    on_land = feat["on_land"]
    n = feat["n"]
    vmax = feat["vmax"]

    # Counts
    is_ts = (not np.isnan(vmax)) and vmax >= TS_KT
    is_hu = (not np.isnan(vmax)) and vmax >= HU_KT
    is_mh = (not np.isnan(vmax)) and vmax >= MH_KT

    # ACE
    if v.size == 0 or not np.any(v >= TS_KT):
        ace = 0.0
    else:
        if method == "synoptic":
            hour = feat["hour"]
            synoptic = (
                ((hour == 0) | (hour == 6) | (hour == 12) | (hour == 18))
                & (feat["minute"] == 0)
                & (feat["second"] == 0)
            )
            mask = (v >= TS_KT) & synoptic
            scale = 1.0
        elif method == "scaled":
            dt_h = feat["dt_h"]
            mask = v >= TS_KT
            scale = (dt_h / 6.0) if np.isfinite(dt_h) else 0.0
        else:
            raise ValueError(f"Unknown ACE method {method!r}")
        if mask.any():
            ace = float(np.nansum(v[mask] ** 2) / 1.0e4) * scale
        else:
            ace = 0.0

    # Landfall flags - initialised False
    rec = dict(
        vmax=vmax,
        ace=ace,
        is_ts=is_ts,
        is_hu=is_hu,
        is_mh=is_mh,
        lf_ts=False,
        lf_hu=False,
        lf_mh=False,
    )
    for r in REGIONS:
        for i in INTENSITIES:
            rec[f"lf_{r}_{i}"] = False

    # Sea -> land transitions (vectorised)
    if n >= 2 and on_land.any():
        transitions = (~on_land[:-1]) & on_land[1:]
        if transitions.any():
            idx = np.flatnonzero(transitions) + 1
            lf_v = v[idx]
            lf_lat = lat[idx]
            lf_lon = lon[idx]
            valid = ~np.isnan(lf_v)
            if valid.any():
                lf_v = lf_v[valid]
                lf_lat = lf_lat[valid]
                lf_lon = lf_lon[valid]

                max_lf_w = float(np.max(lf_v))
                rec["lf_ts"] = max_lf_w >= TS_KT
                rec["lf_hu"] = max_lf_w >= HU_KT
                rec["lf_mh"] = max_lf_w >= MH_KT

                in_east = (
                    (lf_lat >= 24.5)
                    & (lf_lat <= 47.0)
                    & (lf_lon >= -83.0)
                    & (lf_lon <= -66.0)
                )
                in_gulf = (
                    (lf_lat >= 25.85)
                    & (lf_lat <= 32.0)
                    & (lf_lon >= -98.0)
                    & (lf_lon <= -83.0)
                )
                in_us = in_east | in_gulf

                for thr, suf in INTENSITY_THRESHOLDS:
                    at_intensity = lf_v >= thr
                    if (at_intensity & in_us).any():
                        rec[f"lf_us_total_{suf}"] = True
                        if (at_intensity & in_east).any():
                            rec[f"lf_us_east_{suf}"] = True
                        if (at_intensity & in_gulf).any():
                            rec[f"lf_us_gulf_{suf}"] = True

    # Caribbean: any track point in the box at given intensity
    if n >= 1:
        in_carib = (lat >= 10.0) & (lat <= 20.0) & (lon >= -88.0) & (lon <= -60.0)
        with np.errstate(invalid="ignore"):
            for thr, suf in INTENSITY_THRESHOLDS:
                if np.any(in_carib & (v >= thr)):
                    rec[f"lf_caribbean_{suf}"] = True

    return rec


from time import time


# ---------------------------------------------------------------------------
# Per-member: build a flat DataFrame with one row per storm
# ---------------------------------------------------------------------------
def member_storm_records(
    member: TCTracks,
    mask: FastLandMask,
    *,
    ace_method: str = "synoptic",
    sim_year_attr: str = DEFAULT_SIM_YEAR_ATTR,
) -> pd.DataFrame:
    """One-row-per-storm DataFrame for a single member.

    The `sim_year` column comes from `track.attrs[sim_year_attr]`. If a
    track lacks the attribute, it falls back to the calendar year of the
    first timestamp (compatible with single-year observed TCTracks).
    """
    records = []
    for ds in member.data:
        feat = _extract_track_features(track=ds, mask=mask)
        rec = _storm_record_from_features(feat, method=ace_method)
        sy = ds.attrs.get(sim_year_attr)
        if sy is None:
            sy = int(pd.Timestamp(ds.time.values[0]).year)
        rec["sim_year"] = int(sy)
        records.append(rec)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Aggregate to per-season metrics
# ---------------------------------------------------------------------------
# Boolean lf_* columns aggregate via "any" within a season; others sum.
_BOOL_LF_COLS = [f"lf_{r}_{i}" for r in REGIONS for i in INTENSITIES]


def aggregate_storms_to_seasons(
    storm_records: pd.DataFrame,
    *,
    member_id_col: str | None = None,
    expected_sim_years: Mapping | Sequence | None = None,
) -> pd.DataFrame:
    """Group per-storm records by (member_id, sim_year) -> per-season totals.

    Parameters
    ----------
    storm_records : DataFrame
        Output of `member_storm_records`, possibly concatenated across
        members.
    member_id_col : str, optional
        Name of column identifying the member. None = single-member case.
    expected_sim_years : optional
        - If `member_id_col` is None: a sequence of sim_years that should
          appear in the result (missing ones are filled with zero storms).
        - If `member_id_col` is given: a mapping {member_id: list_of_sim_years}
          OR a sequence (treated as the same set for every member).
        Filling in zero-storm seasons is essential for unbiased per-season
        averages: a year with zero NS still counts as 0 in the mean.

    Returns
    -------
    DataFrame indexed by (member_id, sim_year) or sim_year, with columns:
        named_storms, hurricanes, major_hurricanes, ace,
        landfall_ts, landfall_hu, landfall_mh,
        lf_<region>_<intensity>  (booleans).
    """
    group_cols = [member_id_col, "sim_year"] if member_id_col else ["sim_year"]

    if storm_records.empty:
        # Build an empty-but-indexed frame with the right shape
        result_cols = [
            "named_storms",
            "hurricanes",
            "major_hurricanes",
            "ace",
            "landfall_ts",
            "landfall_hu",
            "landfall_mh",
        ] + _BOOL_LF_COLS
        if expected_sim_years is None:
            return pd.DataFrame(columns=result_cols)
        # build the index of expected (member, sim_year)
        if member_id_col and isinstance(expected_sim_years, Mapping):
            tuples = [(m, y) for m, ys in expected_sim_years.items() for y in ys]
            idx = pd.MultiIndex.from_tuples(tuples, names=group_cols)
        elif member_id_col:
            raise ValueError(
                "Need expected_sim_years as mapping when "
                "member_id_col given and storm_records empty."
            )
        else:
            idx = pd.Index(list(expected_sim_years), name="sim_year")
        out = pd.DataFrame(0.0, index=idx, columns=result_cols)
        for c in _BOOL_LF_COLS:
            out[c] = False
        return out

    sum_cols = ["is_ts", "is_hu", "is_mh", "ace", "lf_ts", "lf_hu", "lf_mh"]
    seasons_sum = (
        storm_records.groupby(group_cols)[sum_cols]
        .sum()
        .rename(
            columns={
                "is_ts": "named_storms",
                "is_hu": "hurricanes",
                "is_mh": "major_hurricanes",
                "lf_ts": "landfall_ts",
                "lf_hu": "landfall_hu",
                "lf_mh": "landfall_mh",
            }
        )
    )

    seasons_any = storm_records.groupby(group_cols)[_BOOL_LF_COLS].any()

    out = pd.concat([seasons_sum, seasons_any], axis=1)

    # Fill in zero-storm seasons if we know what to expect
    if expected_sim_years is not None:
        if member_id_col and isinstance(expected_sim_years, Mapping):
            tuples = [(m, y) for m, ys in expected_sim_years.items() for y in ys]
            full_idx = pd.MultiIndex.from_tuples(tuples, names=group_cols)
        elif member_id_col:
            # broadcast same sim_year list across observed members
            members_present = sorted(out.index.get_level_values(member_id_col).unique())
            tuples = [(m, y) for m in members_present for y in expected_sim_years]
            full_idx = pd.MultiIndex.from_tuples(tuples, names=group_cols)
        else:
            full_idx = pd.Index(list(expected_sim_years), name="sim_year")

        out = out.reindex(full_idx)
        # numeric -> 0; bool -> False
        numeric_cols = [
            "named_storms",
            "hurricanes",
            "major_hurricanes",
            "ace",
            "landfall_ts",
            "landfall_hu",
            "landfall_mh",
        ]
        out[numeric_cols] = out[numeric_cols].fillna(0).astype(float)
        for c in _BOOL_LF_COLS:
            out[c] = out[c].fillna(False).astype(bool)

    return out


# ---------------------------------------------------------------------------
# Pooled summary: one forecast year, K members x N seasons
# ---------------------------------------------------------------------------
def pooled_summary(seasons_df: pd.DataFrame) -> dict[str, float]:
    """Pool across all (member, sim_year) rows -> mean + percentiles.

    Returns a flat dict with mean / p05 / p50 / p95 of each metric, and the
    landfall PROBABILITY (% of pooled seasons with at least one hit) for
    each (region, intensity).
    """
    out: dict[str, float] = {}
    metric_cols = [
        "named_storms",
        "hurricanes",
        "major_hurricanes",
        "ace",
        "landfall_ts",
        "landfall_hu",
        "landfall_mh",
    ]
    means = seasons_df[metric_cols].mean()
    p05 = seasons_df[metric_cols].quantile(0.05)
    p50 = seasons_df[metric_cols].quantile(0.50)
    p95 = seasons_df[metric_cols].quantile(0.95)
    for c in metric_cols:
        out[c] = float(means[c])
        out[f"{c}_p05"] = float(p05[c])
        out[f"{c}_p50"] = float(p50[c])
        out[f"{c}_p95"] = float(p95[c])

    # Landfall probabilities (% of seasons with >=1 hit)
    for r in REGIONS:
        for i in INTENSITIES:
            col = f"lf_{r}_{i}"
            out[f"lf_prob_{r}_{i}"] = float(seasons_df[col].mean()) * 100.0
    return out


# ---------------------------------------------------------------------------
# Worker for parallel per-member processing
# ---------------------------------------------------------------------------
def _member_records_worker(args):
    member, ace_method, sim_year_attr, mask = args
    df = member_storm_records(
        member, ace_method=ace_method, sim_year_attr=sim_year_attr, mask=mask
    )
    # Also collect set of sim_years declared by the member's tracks
    declared = sorted(
        {
            int(ds.attrs[sim_year_attr])
            for ds in member.data
            if sim_year_attr in ds.attrs
        }
    )
    return df, declared


def process_year_pooled(
    members: Sequence[TCTracks],
    mask: FastLandMask,
    *,
    ace_method: str = "synoptic",
    sim_year_attr: str = DEFAULT_SIM_YEAR_ATTR,
    n_workers: int | None = None,
    expected_sim_years_per_member: int | Sequence[int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Process all members for one forecast year, return pooled summary.

    Parameters
    ----------
    members : sequence of TCTracks
        One TCTracks per member; each containing many simulated seasons
        (identified by `sim_year_attr` on each track Dataset).
    ace_method, sim_year_attr : passed through.
    n_workers : int or None
        Parallel worker count. None = serial. With ~500ms per-member
        compute typical for 100-year members @ 15 storms/year, parallelism
        scales near-linearly up to your core count.
    expected_sim_years_per_member :
        - If int: assume each member has sim_years 0..N-1 (a sanity check).
        - If sequence of ints: the exact set of sim_years each member
          should have. Members missing some of these years will have
          zero-storm seasons inserted (essential for unbiased averages).
        - If None: sim_years are discovered per-member from the data.
          Members will only contribute their declared sim_years; if some
          members had genuinely zero-storm years that are absent from the
          data, they will be UNDERCOUNTED. Pass an explicit value if you
          know the expected sim_year set.

    Returns
    -------
    (summary_dict, seasons_df)
        summary_dict : pooled mean/percentile/probability metrics.
        seasons_df   : per-(member, sim_year) DataFrame, with zero-storm
                       seasons filled in.
    """
    n = len(members)
    if n == 0:
        return {}, pd.DataFrame()

    use_parallel = n_workers is not None and n_workers > 1 and n >= 4

    args_list = [(m, ace_method, sim_year_attr, mask) for m in members]

    if not use_parallel:
        per_member_results = [_member_records_worker(a) for a in args_list]
    else:
        ctx = None
        if hasattr(os, "fork"):
            import multiprocessing as mp

            ctx = mp.get_context("fork")
        chunksize = max(1, n // (n_workers * 2))
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            per_member_results = list(
                ex.map(
                    _member_records_worker,
                    args_list,
                    chunksize=chunksize,
                )
            )

    # Build expected (member, sim_year) index
    expected_map: dict[int, list[int]] = {}
    for i, (df, declared) in enumerate(per_member_results):
        if expected_sim_years_per_member is None:
            expected_map[i] = declared
        elif isinstance(expected_sim_years_per_member, int):
            expected_map[i] = list(range(expected_sim_years_per_member))
        else:
            expected_map[i] = list(expected_sim_years_per_member)

    # Tag and concat
    parts = []
    for i, (df, _) in enumerate(per_member_results):
        if not df.empty:
            df = df.copy()
            df["member"] = i
            parts.append(df)
    storm_records = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    seasons_df = aggregate_storms_to_seasons(
        storm_records,
        member_id_col="member",
        expected_sim_years=expected_map,
    )
    if seasons_df.empty:
        return {}, pd.DataFrame()
    summary = pooled_summary(seasons_df)
    return summary, seasons_df


# ---------------------------------------------------------------------------
# Top-level comparison
# ---------------------------------------------------------------------------
def compute_climatology(
    years: Iterable[int] = range(1991, 2021),
    *,
    ace_method: str = "synoptic",
    ibtracs_provider: str | Sequence[str] | None = None,
    ibtracs_file_name: str = "IBTrACS.ALL.v04r01.nc",
    mask: "FastLandMask | None" = None,
) -> dict[str, float]:
    """Per-season averages from IBTrACS over a reference period.

    Default 1991-2020 matches the WMO/NOAA/CSU current climate normal.
    Returns a dict in the same column schema as `compute_season_metrics`,
    but values are averages across the reference years (counts and ACE),
    or fractions x 100 for landfall probabilities (% of years with >=1 hit).

    The result is internally consistent with this toolkit's "observed" rows:
    same `coord_on_land`, same ACE convention, same region boxes. Use this
    rather than hard-coded NHC numbers if you want apples-to-apples.

    Note: published NHC numbers (14 NS / 7 HU / 3 MH / ~123 ACE) may differ
    from what this returns by ~10% depending on best-track conventions and
    region definitions. The toolkit's numbers are the right comparator
    because they use the same code path as your other rows.
    """
    if mask is None:
        mask = FastLandMask(cache_path="~/.climada/land_mask.npz")
    yrs = sorted(years)
    LOG.info("Computing climatology over %d-%d", yrs[0], yrs[-1])
    obs_all = load_ibtracs_na(
        yrs, provider=ibtracs_provider, file_name=ibtracs_file_name
    )
    obs_by_year = split_tracks_by_year(obs_all)

    # Build per-season records, one row per year
    rows = []
    for y in yrs:
        tt = obs_by_year.get(y)
        if tt is None:
            continue
        recs = member_storm_records(
            tt,
            ace_method=ace_method,
            sim_year_attr="no_sim_year_attr_force_year_fallback",
            mask=mask,
        )
        if recs.empty:
            # Empty season - all zeros
            row = {
                c: 0.0
                for c in (
                    "named_storms",
                    "hurricanes",
                    "major_hurricanes",
                    "ace",
                    "landfall_ts",
                    "landfall_hu",
                    "landfall_mh",
                )
            }
            for r in REGIONS:
                for i in INTENSITIES:
                    row[f"lf_{r}_{i}"] = False
            row["sim_year"] = y
            rows.append(row)
        else:
            agg = aggregate_storms_to_seasons(recs)
            if len(agg) == 0:
                continue
            # year-summary row
            row = agg.iloc[0].to_dict()
            row["sim_year"] = y
            rows.append(row)

    if not rows:
        return {}
    season_df = pd.DataFrame(rows)

    out: dict[str, float] = {}
    metric_cols = [
        "named_storms",
        "hurricanes",
        "major_hurricanes",
        "ace",
        "landfall_ts",
        "landfall_hu",
        "landfall_mh",
    ]
    for c in metric_cols:
        out[c] = float(season_df[c].mean())
    # Landfall probabilities: fraction of years with >=1 hit, expressed as %
    for r in REGIONS:
        for i in INTENSITIES:
            col = f"lf_{r}_{i}"
            out[f"lf_prob_{r}_{i}"] = float(season_df[col].mean()) * 100.0
    return out


def compare_seasons(
    synthetic_members_by_year: Mapping[int, Sequence[TCTracks]],
    observed_tracks_by_year: Mapping[int, TCTracks] | None = None,
    csu_april_forecasts: Mapping[int, Mapping[str, float]] = CSU_APRIL_FORECASTS,
    csu_april_landfall_probs: Mapping[
        int, Mapping[str, float]
    ] = CSU_APRIL_LANDFALL_PROBS,
    tsr_april_forecasts: Mapping[int, Mapping[str, float]] = TSR_APRIL_FORECASTS,
    years: Iterable[int] | None = None,
    *,
    ibtracs_provider: str | Sequence[str] | None = None,
    ibtracs_file_name: str = "IBTrACS.ALL.v04r01.nc",
    ace_method: str = "synoptic",
    sim_year_attr: str = DEFAULT_SIM_YEAR_ATTR,
    n_workers: int | None = None,
    expected_sim_years_per_member: int | Sequence[int] | None = None,
    return_seasons: bool = False,
    include_climatology: bool = True,
    climatology_years: Iterable[int] = range(1991, 2021),
    mask: FastLandMask | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Compare synthetic ensemble vs observed (IBTrACS) vs forecasts.

    Sources in the output (one row each, indexed by year):
      - 'observed'           : IBTrACS for that year
      - 'csu_april'          : CSU April forecast (NS/HU/MH/ACE +
                               MH-landfall probabilities)
      - 'tsr_april'          : TSR April forecast (NS/HU/IH/ACE +
                               US-landfalling counts; NOT probabilities)
      - 'synthetic_mean'     : pooled mean of your ensemble + percentiles
      - 'climatology_19912020' (one row per year, identical) :
                               IBTrACS 1991-2020 average. Acts as a
                               'no-skill' reference. Computed once and
                               broadcast across all years for plotting.

    Each forecast year `Y` has K members; each is one TCTracks containing
    N simulated seasons of Y. Seasons are pooled across all K * N samples;
    mean, 5th/50th/95th percentiles and landfall probabilities are reported.

    Parameters
    ----------
    expected_sim_years_per_member :
        Pass `100` (or your N) to ensure zero-storm seasons are counted in
        the pooled distribution. STRONGLY RECOMMENDED.
    include_climatology, climatology_years :
        If include_climatology=True (default), compute the IBTrACS average
        over `climatology_years` and add it as a 'climatology_19912020' row
        for each forecast year. Set False to skip (saves one IBTrACS load).

    Returns
    -------
    df : DataFrame
        MultiIndex (year, source). Sources include 'observed', 'csu_april',
        'tsr_april', 'synthetic_mean', and (if requested) the climatology row.
    seasons_by_year (only if return_seasons=True) :
        dict[year -> per-(member, sim_year) DataFrame] for ad-hoc analysis.
    """
    if years is None:
        years = sorted(synthetic_members_by_year.keys())
    years = sorted(years)

    if observed_tracks_by_year is None:
        all_obs = load_ibtracs_na(
            years, provider=ibtracs_provider, file_name=ibtracs_file_name
        )
        observed_tracks_by_year = split_tracks_by_year(all_obs)
    if mask is None:
        mask = FastLandMask(cache_path="~/.climada/land_mask.npz")

    # Compute climatology once
    clim_row: dict[str, float] | None = None
    clim_label: str | None = None
    if include_climatology:
        cy = sorted(climatology_years)
        clim_label = f"climatology_{cy[0]}_{cy[-1]}"
        clim_row = compute_climatology(
            cy,
            ace_method=ace_method,
            ibtracs_provider=ibtracs_provider,
            ibtracs_file_name=ibtracs_file_name,
            mask=mask,
        )
        # Pad in NaN for landfall counts not present in climatology output
        # (we only computed mean counts; no `_p05` etc. - that's fine)

    rows: list[dict] = []
    seasons_by_year: dict[int, pd.DataFrame] = {}
    from time import time

    for year in years:
        print(year)
        # ---- OBSERVED ----
        obs_tt = observed_tracks_by_year.get(year)
        if obs_tt is not None:
            # Treat observed as a single "member" with one sim_year
            obs_records = member_storm_records(
                obs_tt,
                ace_method=ace_method,
                sim_year_attr="no_sim_year_attr",
                mask=mask,
            )
            if not obs_records.empty:
                obs_seasons = aggregate_storms_to_seasons(obs_records)
                # One row per genesis year; for IBTrACS that's just one row
                if len(obs_seasons) >= 1:
                    obs_row = obs_seasons.iloc[0].to_dict()
                    # Promote bool lf_* to landfall probabilities (0 or 100)
                    obs_out = {}
                    for c, v in obs_row.items():
                        if c in _BOOL_LF_COLS:
                            obs_out[f"lf_prob_{c[3:]}"] = 100.0 if v else 0.0
                        else:
                            obs_out[c] = float(v)
                    rows.append(dict(year=year, source="observed", **obs_out))
        else:
            LOG.warning("No observed tracks for year %d", year)
        # ---- CSU APRIL ----
        if year in csu_april_forecasts:
            csu = dict(csu_april_forecasts[year])
            csu.update(landfall_ts=np.nan, landfall_hu=np.nan, landfall_mh=np.nan)
            csu_lf = csu_april_landfall_probs.get(year, {})
            for col in LF_PROB_COLS:
                csu[col] = float(csu_lf[col]) if col in csu_lf else np.nan
            rows.append(dict(year=year, source="csu_april", **csu))

        # ---- TSR APRIL ----
        # TSR provides US-landfall COUNTS (not probabilities). They go
        # directly into landfall_ts / landfall_hu. Landfall probability
        # columns (lf_prob_*) are NaN for TSR - their forecast schema is
        # different from CSU's. The us_landfall_ace column is preserved.
        if year in tsr_april_forecasts:
            tsr = dict(tsr_april_forecasts[year])
            # TSR doesn't forecast major-hurricane US landfalls
            tsr.setdefault("landfall_mh", np.nan)
            for col in LF_PROB_COLS:
                tsr[col] = np.nan
            rows.append(dict(year=year, source="tsr_april", **tsr))

        # ---- CLIMATOLOGY (1991-2020 by default) ----
        if clim_row is not None:
            row = dict(year=year, source=clim_label, **clim_row)
            rows.append(row)

        # ---- SYNTHETIC (pooled) ----
        members = synthetic_members_by_year.get(year)
        if members is None or len(members) == 0:
            continue
        summary, seasons_df = process_year_pooled(
            members=members,
            mask=mask,
            ace_method=ace_method,
            sim_year_attr=sim_year_attr,
            n_workers=n_workers,
            expected_sim_years_per_member=expected_sim_years_per_member,
        )
        if summary:
            rows.append(dict(year=year, source="synthetic_mean", **summary))
        if return_seasons:
            seasons_by_year[year] = seasons_df

    df = pd.DataFrame(rows).set_index(["year", "source"])
    if return_seasons:
        return df, seasons_by_year
    return df


# ---------------------------------------------------------------------------
# Skill summary and Brier scores
# ---------------------------------------------------------------------------
def skill_summary(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    forecast_sources: Sequence[str] = ("csu_april", "tsr_april", "synthetic_mean"),
    truth_source: str = "observed",
) -> pd.DataFrame:
    """Bias / MAE / RMSE / Pearson r per (forecast_source, metric)."""
    if metrics is None:
        metrics = [
            "named_storms",
            "hurricanes",
            "major_hurricanes",
            "ace",
            "lf_prob_us_total_mh",
            "lf_prob_us_east_mh",
            "lf_prob_us_gulf_mh",
            "lf_prob_caribbean_mh",
        ]
    out_rows: list[dict] = []
    avail_sources = df.index.get_level_values("source").unique()
    for src in forecast_sources:
        if src not in avail_sources:
            continue
        truth = df.xs(truth_source, level="source")[list(metrics)]
        fcst = df.xs(src, level="source")[list(metrics)]
        common = truth.index.intersection(fcst.index)
        if len(common) == 0:
            continue
        truth = truth.loc[common]
        fcst = fcst.loc[common]
        for m in metrics:
            if m not in fcst.columns or m not in truth.columns:
                continue
            if fcst[m].isna().all() or truth[m].isna().all():
                continue
            err = fcst[m] - truth[m]
            out_rows.append(
                dict(
                    source=src,
                    metric=m,
                    n=int(err.notna().sum()),
                    bias=float(err.mean()),
                    mae=float(err.abs().mean()),
                    rmse=float(np.sqrt((err**2).mean())),
                    pearson_r=(
                        float(truth[m].corr(fcst[m]))
                        if truth[m].std() > 0 and fcst[m].std() > 0
                        else np.nan
                    ),
                )
            )
    return pd.DataFrame(out_rows).set_index(["source", "metric"])


def landfall_brier_scores(
    df: pd.DataFrame,
    forecast_source: str = "csu_april",
    truth_source: str = "observed",
    intensities: Sequence[str] = ("mh",),
    regions: Sequence[str] = REGIONS,
) -> pd.DataFrame:
    """Brier scores for landfall probability forecasts."""
    out_rows: list[dict] = []
    for region in regions:
        for intensity in intensities:
            col = f"lf_prob_{region}_{intensity}"
            if col not in df.columns:
                continue
            try:
                truth = df.xs(truth_source, level="source")[col] / 100.0
                fcst = df.xs(forecast_source, level="source")[col] / 100.0
            except KeyError:
                continue
            common = truth.index.intersection(fcst.index)
            if len(common) == 0:
                continue
            t = truth.loc[common].dropna()
            f = fcst.loc[common].dropna()
            common_idx = t.index.intersection(f.index)
            if len(common_idx) == 0:
                continue
            t = t.loc[common_idx]
            f = f.loc[common_idx]
            err = (f - t) ** 2
            bs = float(err.mean())
            clim = t.mean()
            ref = float(((clim - t) ** 2).mean())
            bss = float(1 - bs / ref) if ref > 0 else float("nan")
            out_rows.append(
                dict(
                    region=region,
                    intensity=intensity,
                    source=forecast_source,
                    n=int(err.size),
                    brier_score=bs,
                    ref_climatology_bs=ref,
                    brier_skill_score=bss,
                )
            )
    return pd.DataFrame(out_rows).set_index(["region", "intensity"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Explicit per-source styling. Climatology is rendered as a dashed line
# rather than markers - it's a constant reference, not a per-year forecast.
_SOURCE_STYLES = {
    "observed": dict(color="black", marker="o", linestyle="-", zorder=4),
    "csu_april": dict(color="red", marker="s", linestyle="-", zorder=3),
    "tsr_april": dict(color="blue", marker="^", linestyle="-", zorder=3),
    "synthetic_mean": dict(color="green", marker="D", linestyle="-", zorder=3),
    "synthetic_p05": dict(color="green", marker="", linestyle=":", zorder=2),
    "synthetic_p95": dict(color="green", marker="", linestyle=":", zorder=2),
}


def _style_for_source(source: str) -> dict:
    """Return a matplotlib-style kwargs dict for a given source name."""
    if source.startswith("climatology_"):
        return dict(color="grey", marker="", linestyle="--", linewidth=1.8, zorder=1)
    return _SOURCE_STYLES.get(source, dict(marker="o", linestyle="-"))


def plot_metric_comparison(
    df: pd.DataFrame, metric: str, ax=None, sources: Sequence[str] | None = None
):
    """Line plot of one metric across years, one line per source.

    Parameters
    ----------
    sources : optional sequence
        Restrict to these sources (in plotting order). Default: all sources
        present in df. Climatology rows are styled as a horizontal dashed line.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    pivot = df[metric].unstack("source")
    if sources is None:
        # Stable canonical order: data first, then forecasts, then synthetic,
        # then climatology last so it sits behind everything else visually.
        order = [
            "observed",
            "csu_april",
            "tsr_april",
            "synthetic_mean",
            "synthetic_p05",
            "synthetic_p95",
        ]
        sources = [c for c in order if c in pivot.columns]
        sources += [c for c in pivot.columns if c.startswith("climatology_")]
        sources += [c for c in pivot.columns if c not in sources]
    for src in sources:
        if src not in pivot.columns:
            continue
        style = _style_for_source(src)
        ax.plot(pivot.index, pivot[src].values, label=src, **style)
    ax.set_xlabel("Year")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by source")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax


def plot_distribution_per_year(
    seasons_by_year: Mapping[int, pd.DataFrame],
    metric: str,
    df_summary: pd.DataFrame | None = None,
    ax=None,
    overlay_sources: Sequence[str] = ("observed", "csu_april", "tsr_april"),
):
    """Box/violin plot of per-season distributions across years.

    `seasons_by_year` from `compare_seasons(..., return_seasons=True)`.
    Overlays observed/CSU/TSR markers and a horizontal climatology
    reference line (auto-detected from df_summary).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    years = sorted(seasons_by_year.keys())
    data = [seasons_by_year[y][metric].values for y in years]
    parts = ax.violinplot(
        data, positions=range(len(years)), showmedians=True, showextrema=False
    )
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} - synthetic distribution per forecast year")
    ax.grid(True, alpha=0.3)
    if df_summary is not None:
        avail_sources = df_summary.index.get_level_values("source").unique()
        # Markers for forecasts and observed
        for src in overlay_sources:
            if src not in avail_sources:
                continue
            style = _style_for_source(src)
            try:
                vals = df_summary.xs(src, level="source")[metric]
                ys = [vals.get(y, np.nan) for y in years]
            except KeyError:
                continue
            ax.plot(
                range(len(years)),
                ys,
                marker=style.get("marker", "o"),
                linestyle="",
                markersize=8,
                color=style.get("color", "black"),
                label=src,
            )
        # Climatology as a horizontal dashed line
        clim_sources = [s for s in avail_sources if s.startswith("climatology_")]
        for src in clim_sources:
            try:
                clim_val = float(df_summary.xs(src, level="source")[metric].iloc[0])
            except (KeyError, IndexError):
                continue
            ax.axhline(clim_val, **_style_for_source(src), label=src)
        ax.legend(loc="best", fontsize=9)
    return ax
