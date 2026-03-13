# SIENA-IH-STORM implementation notes

Implemented in this packaged repo:

- pooled preprocessing with ENSO phase labels carried on each track/pressure sample
- optional pooled VWS and RH co-location during preprocessing
- pooled movement regression with ENSO dummy offsets and shrinkage-style ridge penalty on phase terms
- pooled pressure regression with ENSO dummy offsets and VWS/RH terms
- phase-aware storm-count, genesis-month, genesis-location, track, and pressure generation using shared pooled coefficients and phase-specific environmental files when available
- `MASTER_storm.py --phase LN|NEU|EN` entry point
- fixed `input.dat` parsing and added `generation_phase`

Important limitations:

- this is a structural implementation pass, not a scientifically validated release
- ERA5 VWS/RH downloads require valid CDS credentials and were implemented but not exercised here
- `MASTER_return_period.py` already had a pre-existing indentation error and was left untouched
- no full end-to-end run was possible here because the required large climate/IBTrACS assets are not bundled in the zip
- residual modeling is still simplified relative to a full hierarchical Bayesian implementation

Recommended next steps:

1. run `MASTER_climatology.py`
2. run `MASTER_preprocessing.py`
3. generate three catalogs with `MASTER_storm.py --phase LN`, `--phase NEU`, `--phase EN`
4. build the validation/ablation script before trusting the Miami phase ordering
