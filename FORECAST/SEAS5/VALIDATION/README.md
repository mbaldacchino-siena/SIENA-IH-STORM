# SEAS5 Bias Correction Validation

Three diagnostic figures for Section 2.X of the whitepaper.

## Data inputs needed

Each figure loads three aligned datasets:

1. **Corrected SEAS5** — produced by `pipeline.step4_apply_correction`. Already on disk.
2. **Raw SEAS5** (uncorrected absolute) — download separately.
3. **ERA5 validation** (reanalysis monthly means over 2017-2025, i.e. the validation window, *not* the 1993-2016 climatology).

### Expected file layout (defaults, overridable via kwargs)

```
data/
├── seas5_corrected/seas5corrected_{var}[_pl]_2017-2025.nc   # from pipeline
├── seas5_raw/seas5raw_{var}[_pl]_2017-2025.nc               # you download
└── era5_validation/era5_{var}[_pl]_2017-2025.nc             # you download
```

### Downloading the raw SEAS5 + ERA5 validation

The pipeline's `cds_download` helpers work for this too. For the raw SEAS5
you'd call the `seasonal-monthly-*` (not `seasonal-postprocessed-*`) datasets
to get absolute values instead of anomalies. A short helper script:

```python
import cdsapi
c = cdsapi.Client()

years = [str(y) for y in range(2017, 2026)]
init_months = ["04", "05", "06"]

# Raw SST absolute
c.retrieve(
    "seasonal-monthly-single-levels",
    {
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": "sea_surface_temperature",
        "product_type": "monthly_mean",
        "year": years,
        "month": init_months,
        "leadtime_month": ["1", "2", "3", "4", "5", "6"],
        "format": "netcdf",
    },
    "data/seas5_raw/seas5raw_sst_2017-2025.nc",
)

# ERA5 validation monthly mean
c.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": "monthly_averaged_reanalysis",
        "variable": "sea_surface_temperature",
        "year": years,
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "format": "netcdf",
    },
    "data/era5_validation/era5_sst_2017-2025.nc",
)
```

## Figures

### Figure 1 — Raw bias map (motivation)

`figure1_bias_map.plot_bias_map(...)`. Shows `mean(X_raw - X_ERA5)` across
matching valid months and members. `lead=3` for a single lead, `lead="all"`
to average across all leads.

### Figure 2 — RMSE vs lead (effect of correction)

`figure2_rmse_vs_lead.plot_rmse_vs_lead(...)`. Two lines (raw, corrected)
on the same axis. Spatial RMSE averaged over inits and members, as a
function of lead time. Cosine-latitude weighted to be honest about polar
over-representation.

### Figure 3 — Anomaly preservation (the defensive one)

`figure3_anomaly_preservation.plot_anomaly_scatter(...)`. Hexbin of raw
anomaly vs corrected anomaly at every (init, lead, member, cell) point.
Should sit on the 1:1 line with correlation ~1.0 and slope ~1.0 —
demonstrating that correction doesn't distort the forecast signal,
only the climatological mean.

Anomalies are computed per calendar month of the valid date, using each
dataset's own mean over the validation window. This is deliberate: both
series are centered by *their own* climatology, so they should match
exactly under delta correction.

## Anticipated results for the paper

- Fig 1: spatially structured bias (cooler SEAS5 in the central Atlantic,
  warmer in the Gulf of Mexico is typical; MSLP bias usually a dipole),
  motivating per-cell correction rather than a global offset.
- Fig 2: raw RMSE grows ~monotonically with lead; corrected is ~flat.
- Fig 3: corr ≈ 1.0000, slope ≈ 1.0000 — defense against "does your
  correction distort the forecast signal?" reviewer questions.

## What is intentionally NOT here

- **Corrected minus raw absolute bias maps.** Delta correction trivially
  reduces mean bias at every cell by construction. This map is a
  tautology. See our earlier discussion.
- **Skill scores (CRPS, Brier).** Correction does not improve skill.
  Those belong in the backtest section (whitepaper Section 4), evaluated
  on the downstream TC counts / ACE, not on the input fields.
