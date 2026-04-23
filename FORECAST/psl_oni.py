"""Fetch NOAA PSL Niño 3.4 monthly anomalies (ERSST V5).

Source: https://psl.noaa.gov/data/correlation/nina34.anom.data
        https://psl.noaa.gov/data/timeseries/month/Nino34_CPC/

Why PSL ERSST V5 monthly anomalies (instead of CPC's 3-mo ONI):

  - We need MONTHLY anomalies, not pre-averaged 3-month means, so we can
    combine observed monthly values with SEAS5 monthly anomalies and apply
    a single centered 3-month average ourselves. Mixing CPC's already-
    averaged ONI with SEAS5 monthly values leads to inconsistent smoothing.

  - PSL's reference period is fixed at 1981-2010, NOT the sliding base
    that CPC's operational ONI uses. A fixed baseline is what we want here:
    combining with SEAS5 1993-2016 hindcast-anchored anomalies works
    consistently year-on-year.

PSL standard format
-------------------
    1950 2025                          <- header: first_year last_year
    1950   janval febval ... decval    <- one line per year, 13 numbers
    1951   janval febval ... decval
    ...
    2025   janval febval ... -99.99    <- missing months padded to MISSING
    -99.99                             <- trailer: the missing value
    [free-text metadata may follow]

Missing values: anywhere in the grid, NOT just the trailing months. We mask
on equality with the trailer value (typically -99.99 or -9.9, but read it
from the file rather than hard-coding).
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    import FORECAST.SEAS5.config as config
except ImportError:  # pragma: no cover
    import config

logger = logging.getLogger(__name__)

PSL_NINA34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"
DEFAULT_CACHE = config.DATA_DIR / "psl_nina34.txt"


# =============================================================================
# Fetch + cache
# =============================================================================
def fetch_psl_text(
    url: str = PSL_NINA34_URL,
    cache_path: Path = DEFAULT_CACHE,
    max_age_hours: float = 24.0,
    force_refresh: bool = False,
) -> str:
    """Return PSL text, fetched from network if cache is stale."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600.0
        if age_hours < max_age_hours:
            logger.info(
                "Using cached PSL Nino 3.4 (age=%.1fh): %s", age_hours, cache_path
            )
            return cache_path.read_text()

    logger.info("Fetching PSL Nino 3.4 from %s", url)
    try:
        # Some NOAA PSL endpoints reject default urllib User-Agent.
        req = Request(
            url, headers={"User-Agent": "Mozilla/5.0 (compatible; SIENA-IH-STORM)"}
        )
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as ex:
        if cache_path.exists():
            logger.warning("PSL fetch failed (%s); using stale cache", ex)
            return cache_path.read_text()
        raise RuntimeError(
            f"Cannot fetch PSL Nino 3.4 and no cache available at {cache_path}"
        ) from ex

    cache_path.write_text(text)
    logger.info("Cached PSL Nino 3.4 to %s", cache_path)
    return text


# =============================================================================
# Parse
# =============================================================================
def parse_psl_text(text: str) -> pd.DataFrame:
    """Parse PSL standard-format text into long-form (year, month, value).

    Handles:
        - the (year_start, year_end) header line
        - 13-column data rows (year + 12 monthly values)
        - the trailing scalar that defines the missing-value sentinel
        - any free-text metadata after the data block
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("PSL file is empty")

    # First line: "1950 2025"
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError(f"Cannot parse PSL header line: {lines[0]!r}")
    try:
        year_start, year_end = int(header[0]), int(header[1])
    except ValueError as ex:
        raise ValueError(f"Bad PSL header: {lines[0]!r}") from ex

    # Walk subsequent lines until we have year_end - year_start + 1 data rows
    expected_rows = year_end - year_start + 1
    data_rows = []
    trailer_value = None
    for ln in lines[1:]:
        toks = ln.split()
        if len(toks) == 13:
            try:
                year = int(toks[0])
            except ValueError:
                continue
            try:
                vals = [float(t) for t in toks[1:]]
            except ValueError:
                continue
            data_rows.append((year, vals))
        elif len(toks) == 1:
            # Likely the trailing missing-value sentinel
            try:
                trailer_value = float(toks[0])
            except ValueError:
                pass
            break  # data block is done
        # Other lines (free-text metadata) are simply ignored.

    if len(data_rows) < expected_rows:
        logger.warning(
            "Expected %d data rows from PSL header, got %d. "
            "Continuing with what was parsed.",
            expected_rows,
            len(data_rows),
        )

    # Build long-form DataFrame
    records = []
    for year, vals in data_rows:
        for m, v in enumerate(vals, start=1):
            if trailer_value is not None and np.isclose(v, trailer_value):
                continue
            records.append({"year": year, "month": m, "value": v})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("Parsed PSL file contained no usable values")
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df


# =============================================================================
# Public API
# =============================================================================
def load_monthly_anomaly(
    cache_path: Path = DEFAULT_CACHE,
    max_age_hours: float = 24.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Return the full PSL Nino 3.4 monthly anomaly archive as a DataFrame.

    Columns: ['year', 'month', 'value']  (value is K, baseline 1981-2010).
    """
    text = fetch_psl_text(
        cache_path=cache_path, max_age_hours=max_age_hours, force_refresh=force_refresh
    )
    return parse_psl_text(text)


def load_observed_monthly_dict(
    target_year: int,
    init_month: int,
    cache_path: Path = DEFAULT_CACHE,
    max_age_hours: float = 24.0,
    force_refresh: bool = False,
) -> Dict[int, float]:
    """Observed monthly Niño 3.4 anomalies for months strictly < init_month.

    Returns
    -------
    dict : {month_int: anomaly_K}
        Includes only months in `target_year` for which PSL has a value.
        Also includes Dec of `target_year - 1` keyed as month=0 to allow
        wrap-around 3-month windows that need December (e.g. centered
        ONI for January = mean(Dec-1 + Jan + Feb)). Callers that don't
        need it can simply ignore key 0.
    """
    df = load_monthly_anomaly(
        cache_path=cache_path, max_age_hours=max_age_hours, force_refresh=force_refresh
    )
    obs = {}
    cur = df[(df["year"] == target_year) & (df["month"] < init_month)]
    for _, row in cur.iterrows():
        obs[int(row["month"])] = float(row["value"])

    # Pull December of previous year as month=0 to support January-centered ONI
    prev = df[(df["year"] == target_year - 1) & (df["month"] == 12)]
    if len(prev):
        obs[0] = float(prev["value"].iloc[0])

    logger.info(
        "PSL Nino 3.4: %d observed months for %d (init_month=%d): %s",
        len(obs),
        target_year,
        init_month,
        sorted(obs.keys()),
    )
    return obs
