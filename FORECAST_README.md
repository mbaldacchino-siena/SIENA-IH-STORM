# Forecast-mode runbook

How to run SIENA-IH-STORM in seasonal-forecast mode end-to-end. Covers what
must exist before each stage, what each stage writes, and the commands to
run in order.

## Stage 0 — Prerequisites (one-time per machine)

These must be in place **before anything else**. They are independent of
which forecast you're running.

### 0.1 Training climatology and preprocessing

Generates the ERA5-based fields and coefficient files the generator reads
at runtime. This is the historical-data foundation that the forecast
pipeline extends, not replaces.

```bash
python -m MASTER.MASTER_climatology      # builds ERA5 climatology fields
python -m MASTER.MASTER_preprocessing    # fits coefficients, pools, env_pool
```

Output locations:
- `env_yearly/*_{year}_{month}.npy` — per-year-month environmental fields
  (VWS, RH600, MSLP, SST, PI, VMAX_PI, VORT850) for 1993–2021 or similar
- `env_yearly/env_pool.json` — which years belong to which ENSO phase
- `COEFFICIENTS_*.txt`, `genesis_*.npy`, etc. — fitted regression coefficients

These files are reused by every forecast run and should not be regenerated
unless the training-side data or model configuration changes.

### 0.2 CDS API credentials

Required for downloading ERA5 and SEAS5. Create `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <YOUR-KEY>
```

Accept the licence terms for each dataset (once, via the CDS web UI):
- ERA5 monthly means (single + pressure levels)
- C3S seasonal forecasts postprocessed (single + pressure levels)

### 0.3 Python environment

```bash
pip install -r FORECAST/SEAS5/requirements.txt
pip install tcpyPI                        # highly recommended, enables full PI
```

Without tcpyPI, `process_member` falls back to the simplified Bister-Emanuel
approximation. Results are still usable but miss the vertical-profile
information that matters for interannual PI variation.

---

## Stage 1 — Forecast fields (per SEAS5 init)

Runs the bias-correction pipeline, extracts bias-corrected fields per
ensemble member, and writes forecast-mode `env_yearly` files and
`forecast_configs/*.json`.

```bash
python -m MASTER.MASTER_forecast_fields \
    --init-date 2026-04-01 \
    --lead-months 6 \
    --member 51 \
    --generate-config \
    --active-months 6 7 8 9 10 11 \
    --workers 8
```

### Required arguments

- `--init-date YYYY-MM-DD` — first-of-month init date. Must match an actual
  SEAS5 init available on CDS (SEAS5 runs on the 1st of every month, released
  around the 5th). For an April 2026 forecast, use `2026-04-01` as soon as
  CDS has published the release (typically April 5, 12:00 UTC).

### Optional arguments

- `--lead-months 6` — how many lead months to retrieve (max 7). For NA
  hurricane season from April init, 6 covers Apr–Sep.
- `--member 51` — process all 51 operational members. Pass a smaller
  integer to process only the first N members (e.g. `--member 6` for a
  quick test or diagnostic).
- `--generate-config` — write `forecast_configs/config_m{N}.json` per
  member. **Required if you want to run Stage 2 in forecast mode.**
- `--active-months 6 7 8 9 10 11` — which months count as active TC
  season. Default is NA basin (Jun–Nov). Change for other basins:
  WP `5 6 7 8 9 10 11`, SP/SI `11 12 1 2 3 4`, NI `4 5 6 9 10 11`.
- `--oni-threshold 0.5` — ±K threshold for EN/LN vs NEU classification.
  Standard is 0.5 (CPC convention).
- `--workers 8` — parallel ensemble-member processing. Set to ~number
  of physical cores. `--workers 1` is sequential (useful for debugging).
- `--skip-download` — skip all CDS access and bias correction. Use only
  when the corrected NetCDFs already exist on disk and you just want to
  re-run the per-member extraction (e.g. after fixing a bug in `process_member`).
- `--force-recorrect` — redo the bias correction even if output NetCDFs
  already exist. Use after changing bias-correction logic.

### What happens during Stage 1

1. ERA5 monthly means (1993–2016) download — **only on first run per machine**
2. ERA5 climatology build — **only on first run**
3. SEAS5 anomaly download for this init — **cached per init**, ~1–2 GB
4. Bias correction (`X_corrected = ERA5_clim + SEAS5_anomaly`) — fast, cached
5. Per-member loop:
   - Regrid to 0.25°
   - Log-p interpolate q and T to 600 hPa (correct RH600 computation)
   - Compute VWS, RH600, vorticity
   - Compute PI + VMAX_PI via tcpyPI on native-resolution q/T profiles
   - Save to `env_yearly/{STEM}_{10000 + member}_{month}.npy`
6. If `--generate-config`: compute ENSO phase schedule per member (PSL
   observed + SEAS5 merged, centered 3-mo ONI) and write
   `forecast_configs/config_m{N}.json`

### Output files

```
env_yearly/
├── VWS_10000_4.npy      # member 0, April (lead 1 for April init)
├── VWS_10000_5.npy      # member 0, May  (lead 2)
├── ...
├── RH600_10050_9.npy    # member 50, September (lead 6)
├── PI_10050_9.npy
├── VMAX_PI_10050_9.npy
├── VORT850_10050_9.npy
├── MSLP_10050_9.npy
├── SST_10050_9.npy
forecast_configs/
├── config_m0.json       # ENSO schedule + active months for member 0
├── ...
└── config_m50.json
```

Naming convention: `env_year = 10000 + member_index`. This puts forecast
fields in a dedicated numeric range that can't collide with training-year
files (1993–2021).

### Wall time expectations (rough)

First run on a new machine (including ERA5 download): ~4–8 hours depending
on bandwidth.

Subsequent runs for a new init (ERA5 already cached): ~1–2 hours total,
dominated by the 51-member processing loop.

Re-running the same init after a code change (with `--skip-download`):
~15–45 minutes for the member loop.

---

## Stage 2 — Storm generation (per init × per basin)

Uses the forecast configs from Stage 1 to sample synthetic storms under
each ensemble member's ENSO schedule.

```bash
python -m MASTER.MASTER_storm_parallel \
    --forecast forecast_configs/config_m0.json \
               forecast_configs/config_m1.json \
               forecast_configs/config_m2.json \
               ... \
               forecast_configs/config_m50.json \
    --basins NA \
    --years 100 \
    --loop 1 \
    --workers 8
```

In bash you can shortcut the config list:

```bash
CONFIGS=$(printf "forecast_configs/config_m%d.json " {0..50})

python -m MASTER.MASTER_storm_parallel \
    --forecast $CONFIGS \
    --basins NA \
    --years 100 \
    --loop 1 \
    --workers 8
```

Or in Python:
```python
forecast_members = [f"forecast_configs/config_m{m}.json" for m in range(51)]
```

### Forecast-mode specifics

- `--forecast <config-list>` — paths to the per-member JSON configs from
  Stage 1. When set, `--phase` is ignored (phases come from each config's
  month-by-month schedule).
- `--basins NA` — one or more basins. Must have corresponding training
  coefficients available. NA is the most validated.
- `--years 100` — **years per member**, not total catalog size. With 51
  members × 100 years = 5 100 synthetic member-years. Increase if the
  return-period tail matters more than ensemble spread at short periods.
- `--loop 1` — number of loops per member. For forecast mode, 1 is
  standard (each member is already a scenario). Raise only if you need
  more statistical power per member with identical environmental inputs.
- `--workers 8` — parallel workers. Jobs are `(basin × member × loop)`
  combinations.

### Prerequisite check before Stage 2

Before launching, confirm:

- All forecast_configs files exist for the members you're passing.
- `env_yearly/` contains the forecast fields for every `env_year` referenced
  in those configs. Each config names its `env_year` in the JSON; grep for it:
  ```bash
  grep env_year forecast_configs/config_m0.json
  # Check that env_yearly/VWS_{that_env_year}_{6..11}.npy all exist
  ```
- Training coefficients exist (`COEFFICIENTS_*.txt` for each basin).

### Output files

```
storm_catalogs/
├── Storm_NA_FCST_10000_0.txt     # basin, pseudo-phase, env_year, loop_idx
├── Storm_NA_FCST_10001_0.txt
├── ...
└── Storm_NA_FCST_10050_0.txt
```

Each file has columns: year, month, storm_id, timestep, lat, lon, pressure,
wind, category, landfall, (and more depending on your version).

---

## What to run **before** both stages, in order

Per machine, one time:

1. **Stage 0.1** — `MASTER_climatology` + `MASTER_preprocessing`. Required
   by Stage 2 (coefficient files) and implicitly by Stage 1 (it reuses the
   `env_yearly` reference fields for grid alignment).
2. **Stage 0.2** — CDS credentials. Required by Stage 1.
3. **Stage 0.3** — Install dependencies including `tcpyPI`.

Per forecast init, in order:

4. **Stage 1** — `MASTER_forecast_fields`. Writes `env_yearly/*_1xxxx_*.npy`
   and `forecast_configs/config_m*.json`.
5. **Stage 2** — `MASTER_storm_parallel --forecast`. Reads the outputs of
   Stage 1. Writes synthetic storm catalogs.

Stage 1 must complete fully before Stage 2 starts. There is no parallel
streaming between them.

---

## Re-running after a code change

If you change `process_member` or any bias-correction logic:

```bash
# Wipe stale forecast fields (env_year >= 10000)
rm env_yearly/*_1[0-9][0-9][0-9][0-9]_*.npy

# Re-run Stage 1 without re-downloading CDS data
python -m MASTER.MASTER_forecast_fields \
    --init-date 2026-04-01 --member 51 --generate-config \
    --active-months 6 7 8 9 10 11 --workers 8 \
    --skip-download

# Re-run Stage 2 with the same command as before
```

If you change a training-side module (`genesis_matrix.py`, coefficients,
etc.): re-run Stage 0.1, then Stage 1, then Stage 2.

If you change bias-correction code itself: re-run Stage 1 with
`--force-recorrect` instead of `--skip-download`. This re-applies the
correction to the cached SEAS5 anomalies without re-downloading them.

---

## Diagnostic run (RH600 bias check, ~15 min)

To quantify the magnitude of the old RH600 bug or to sanity-check the fix,
run a small hindcast:

```bash
python -m MASTER.MASTER_forecast_fields \
    --init-date 2015-05-01 \
    --lead-months 6 \
    --member 6 \
    --workers 6
```

August 2015 (lead 4 from May init) during strong El Niño is the worst case
for the bug. Compare `env_yearly/RH600_10000_8.npy` (forecast member 0)
against `env_yearly/RH600_2015_8.npy` (ERA5 truth). A structured positive
bias in the MDR is the fingerprint of the old bug; with the fix applied,
the difference should be small and spatially unstructured.

---

## Common mistakes

- **Skipping Stage 0.1 on first setup.** Stage 2 reads coefficient files
  produced by `MASTER_preprocessing`; if they're missing, storm generation
  fails with cryptic file-not-found errors partway through.
- **Running Stage 1 before the SEAS5 init is released.** SEAS5 runs for
  a given month are released around the 5th of that month. If you run
  `--init-date 2026-04-01` on April 3rd, the CDS request will succeed but
  return older cached hindcast data, not the operational forecast.
- **Forgetting `--generate-config`.** Stage 2 in forecast mode requires
  the per-member JSON configs. Without them the `--forecast` flag has
  nothing to read and Stage 2 errors on argument parsing.
- **Mixing `--member 51` (count) with index semantics.** `--member 51` means
  "members 0 through 50" (51 total). `--member 50` means "0 through 49"
  and silently drops the last member of the ensemble. Always use 51 for
  operational SEAS5.
- **Running two inits into the same `env_yearly` directory without renaming.**
  `env_year = 10000 + member_idx` is the same across inits, so a second
  April-init run would overwrite the first. If you want to keep multiple
  inits side-by-side, either run each in its own working directory or
  namespace the env_year differently (requires code change).
