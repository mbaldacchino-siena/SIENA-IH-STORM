# SEAS5 Bias-Correction + ENSO Integration

This folder houses everything related to SEAS5 seasonal forecast ingestion
into SIENA-IH-STORM:

  1. Delta (anomaly-based) bias correction of SEAS5 onto the ERA5
     1993-2016 climatological scale.
  2. Clean ENSO phase-schedule construction combining observed ONI
     (from `climate_index.csv`) with SEAS5-derived 3-month centered
     Niño 3.4 ONI, with persistence for gap months.

Both are consumed by `MASTER/MASTER_forecast_fields.py` with no behavioural
changes to downstream TC-generation code: members are still opened one at a
time, regridded to 0.25°, and saved to `env_yearly/` — but the values are
now bias-corrected before PI/GPI/VWS computation.

## Module layout

| File | Purpose |
|---|---|
| `config.py` | Reference periods, domain, variables, paths. Single source of truth. |
| `cds_download.py` | CDS API wrappers for ERA5 monthly means and SEAS5 anomalies. |
| `climatology.py` | Build ERA5 monthly climatology over 1993-2016. |
| `regrid.py` | Bilinear regridding ERA5 → SEAS5 grid, handles lon conventions. |
| `bias_correction.py` | Delta correction: `X_corrected = ERA5_clim + SEAS5_anomaly`. |
| `pipeline.py` | End-to-end orchestrator (steps 1-4), idempotent per step. |
| `loader.py` | Presents corrected per-variable files as `ds_pl` + `ds_sfc` for MASTER. |
| `enso.py` | Observed + SEAS5 + persistence ONI schedule for phase labelling. |
| `requirements.txt` | `cdsapi`, `xarray`, `netCDF4`, `numpy`, `scipy`. |

## How MASTER_forecast_fields uses it

The old `download_seas5()` function is replaced by `prepare_seas5_bias_corrected()`,
which delegates to `loader.prepare_corrected_for_init()`:

```python
# In MASTER_forecast_fields.py __main__
prepare_seas5_bias_corrected(
    init_date=args.init_date,
    lead_months=args.lead_months,
    overwrite=args.force_recorrect,
)
# -> runs steps 1+2 only if climatology files are missing
# -> runs steps 3+4 for this init (cached on disk; idempotent)
```

Per-member processing opens the merged bias-corrected datasets via
`loader.prepare_corrected_for_init(...)` (idempotent read when files exist)
and extracts u, v, t, q, sst, msl just as before. The variable names and
shapes are identical to the previous raw-download output, so
`process_member()` needs no changes.

ENSO phase scheduling (inside `--generate-config`) now calls
`enso.phase_schedule_from_corrected(...)` which:

  - Uses observed ONI from `climate_index.csv` directly (it IS already a
    CPC 3-month centered mean — no further averaging).
  - Derives monthly Niño 3.4 anomalies from the bias-corrected SST using
    the same ERA5 SST climatology as the correction baseline (consistent).
  - Averages those monthly anomalies over centered 3-month windows ONLY
    where the full (m-1, m, m+1) window falls inside the SEAS5 horizon
    (for April init: June-September).
  - Persists the last known phase across gap months
    (April, May, October, November, December for an April init).

## The scientific reasoning summary

See the paper draft and the prior chat for the full discussion. The short
version, for documenting in the whitepaper:

  - Delta correction preserves SEAS5's predicted anomaly pattern while
    re-anchoring to ERA5's climatological absolute scale. This is what an
    ERA5-trained downstream model (your regressions) expects as input.
  - The 1993-2016 reference period is dictated by the CDS
    `seasonal-postprocessed-*` products (fixed by ECMWF product design).
    The warming trend between 1993-2016 and the target forecast year is
    carried legitimately by SEAS5's anomaly (which you verified is
    nonzero in expectation over 2017-present).
  - What the correction does NOT address: variance/tail biases, and
    out-of-training-range extrapolation under continued warming. Document
    in the limitations section.

## Typical operational run (April 2026 forecast)

```bash
python -m MASTER.MASTER_forecast_fields \
    --init-date 2026-04-01 \
    --lead-months 6 \
    --generate-config \
    --active-months 6 7 8 9 10 11 \
    --workers 8
```

First run: downloads ERA5 (once, slow), builds climatology (once), downloads
SEAS5 April-2026 anomaly (fast), applies correction, processes 51 members.

Subsequent runs with the same init: only member processing runs (everything
else is cached on disk).

To force re-download of the SEAS5 anomaly (e.g., if you suspect the file is
corrupted), pass `--force-recorrect`.

## File layout after a run

```
FORECAST/SEAS5/data/
├── era5_raw/                    # raw ERA5 monthly means 1993-2016 (built once)
│   ├── era5_sst_1993-2016.nc
│   ├── era5_msl_1993-2016.nc
│   ├── era5_t_pl_1993-2016.nc
│   ├── era5_u_pl_1993-2016.nc
│   ├── era5_v_pl_1993-2016.nc
│   └── era5_q_pl_1993-2016.nc
├── era5_climatology/            # ERA5 monthly climatology (built once, reused)
│   ├── era5_clim_sst_1993-2016.nc
│   └── ... (one per variable)
├── seas5_anomaly/               # SEAS5 anomalies per init (cached)
│   ├── seas5anom_sst_2026-2026.nc
│   └── ...
└── seas5_corrected/             # X_corrected per variable (consumed by MASTER)
    ├── seas5corrected_sst_2026-2026.nc
    └── ...
```

## Caveats worth citing in the paper

1. **Reference period mismatch with model training.** The bias correction
   uses ERA5 1993-2016; your model is trained on ERA5 1980-2021 (or
   1980-2025 in the final version). The trend between periods is small
   relative to SEAS5 anomaly amplitudes and is carried faithfully by
   the anomaly term. Document this briefly.

2. **Univariate / marginal correction.** Each variable is corrected
   independently. Joint distributional consistency is preserved only to
   the extent that the additive per-variable shift is the same for every
   member — which it is, by construction — so inter-variable covariance
   on anomaly deviations is undistorted. Tail shape and variance are NOT
   corrected.

3. **SEAS5 skill ceiling.** No bias correction improves skill. CRPS and
   reliability diagrams at peak season should be reported against
   climatology and persistence baselines, as in Section 4 of the
   whitepaper draft.

4. **ENSO persistence across gap months.** April/May and Oct-Dec phases
   for an April init are persisted from the nearest computed month.
   This is the operational default; for more aggressive scheduling you
   could run a March-init SEAS5 to fill April, or ingest observed
   monthly Niño 3.4 anomalies separately from the CPC 3-mo ONI.
