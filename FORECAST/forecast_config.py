"""
Forecast configuration for SIENA-IH-STORM seasonal forecast mode.

The manifest maps each calendar month to:
  - source: "observed" | "forecast" | "historical"
  - phase:  "LN" | "NEU" | "EN"
  - env_year: int (only for observed/forecast; historical resamples from pool)
  - ensemble_member: int (only for forecast source, optional)

Example forecast_config.json:
{
  "mode": "seasonal_forecast",
  "base_year": 2026,
  "ensemble_member": null,
  "months": {
    "1":  {"source": "observed",   "phase": "LN",  "env_year": 2026},
    "2":  {"source": "observed",   "phase": "LN",  "env_year": 2026},
    "3":  {"source": "observed",   "phase": "LN",  "env_year": 2026},
    "4":  {"source": "observed",   "phase": "NEU", "env_year": 2026},
    "5":  {"source": "forecast",   "phase": "NEU", "env_year": 9999},
    "6":  {"source": "forecast",   "phase": "NEU", "env_year": 9999},
    "7":  {"source": "forecast",   "phase": "EN",  "env_year": 9999},
    "8":  {"source": "forecast",   "phase": "EN",  "env_year": 9999},
    "9":  {"source": "forecast",   "phase": "EN",  "env_year": 9999},
    "10": {"source": "forecast",   "phase": "EN",  "env_year": 9999},
    "11": {"source": "historical", "phase": "EN"},
    "12": {"source": "historical", "phase": "EN"}
  }
}
"""

import json
import os
from CODE.siena_utils import normalize_phase, sample_env_year


def load_forecast_config(path):
    """Load and validate a forecast config JSON."""
    with open(path) as f:
        cfg = json.load(f)

    assert cfg.get("mode") == "seasonal_forecast", "Not a forecast config"
    assert "months" in cfg, "Missing 'months' key"

    # Validate each month entry
    for m_str, entry in cfg["months"].items():
        assert int(m_str) in range(1, 13), f"Invalid month: {m_str}"
        assert entry["source"] in ("observed", "forecast", "historical")
        assert normalize_phase(entry["phase"]) is not None
        if entry["source"] in ("observed", "forecast"):
            assert "env_year" in entry, (
                f"Month {m_str}: source={entry['source']} requires env_year"
            )
    return cfg


def build_forecast_env_years(cfg, env_pool, active_months):
    """
    Build the env_years dict {month: year} for a single simulated year,
    respecting the forecast config.

    - observed/forecast months → deterministic env_year from config
    - historical months → resample from env_pool (phase-consistent)

    Parameters
    ----------
    cfg : dict, loaded forecast config
    env_pool : dict, from load_env_pool()
    active_months : list of int, e.g. [6,7,8,9,10,11]

    Returns
    -------
    dict : {month: int} mapping each active month to a year label
    """
    env_years = {}
    for m in active_months:
        m_str = str(m)
        if m_str not in cfg["months"]:
            # Month not in config — fall back to historical resampling
            # Use NEU as default if no phase specified
            env_years[m] = sample_env_year(env_pool, "NEU", m)
            continue

        entry = cfg["months"][m_str]
        if entry["source"] in ("observed", "forecast"):
            env_years[m] = int(entry["env_year"])
        else:
            # historical: resample from pool, respecting the month's phase
            env_years[m] = sample_env_year(env_pool, entry["phase"], m)
    return env_years


def get_month_phases(cfg, active_months):
    """
    Extract {month: phase_str} for active months from the forecast config.

    Returns
    -------
    dict : {month_int: "LN"|"NEU"|"EN"}
    """
    phases = {}
    for m in active_months:
        m_str = str(m)
        if m_str in cfg["months"]:
            phases[m] = normalize_phase(cfg["months"][m_str]["phase"])
        else:
            phases[m] = "NEU"  # default fallback
    return phases
