# Forecasting results — audited consolidation (2026-11-23)

This directory contains **original observations and their deterministic aggregation**, not newly trained models or newly executed Test evaluations.

## Scientific scope and provenance

- **V1:** SIRIUS 1 — 810 original Validation runs from `validation_all_V1.csv`.
- **V2:** Ubuntu RTX 5080 — 810 original Validation runs from `PGAF_validation_V2_Ubuntu.zip`, individually checked against archived `run_spec.json`, `validation_metrics.json` and `validation_done.json` (including the embedded SHA-256 of the metrics/specifications).
- **V3:** SIRIUS 2 — 810 original Validation runs from `validation_all_V3.csv`.
- Architectures: `lstm_sergiot` (called **LSTM** in the manuscript), `itransformer`, `patchtst_mv`. The legacy `lstm_original` has deliberately been excluded.
- Horizons: 4, 8, 16, 24, 48, 96, 144, 192, 288. Lookbacks screened: 48, 96, 192, 288, 384, 672. Seeds: 8, 13, 22, 27, 31.

## Files

- `validation/validation_all.csv`: all **2,430** Validation model/seed/lookback observations; global `nMAE-IQR` and Validation window count.
- `validation/selected_lookbacks.csv`: **81** selections (3 variants × 3 architectures × 9 horizons), by **lowest mean global Validation nMAE-IQR across five seeds**, then lowest sample standard deviation and smaller lookback on ties. **Test scores were never used for model selection.**
- `test/test_by_version_and_horizon.csv`: **81** original five-seed Test aggregates (9 horizons × 3 variants × 3 architectures); includes global and TBS/HR scores and inference latency.
- `test/test_per_seed_9variables.csv`: **3,645** original per-variable Test metric records (= 405 model/seed Test evaluations × 9 sensors). `model` is standardized to the internal architecture identifier.
- `test/persistence_by_version_and_horizon.csv`: **27** original Persistence baseline aggregates, retained separately from the neural architectures.

## Audit checks performed

1. Validation data: 810 runs for each variant, 270 per architecture and variant, five distinct seeds per candidate, six candidates per variant/model/horizon. No missing or duplicate run keys.
2. Archived V2: **810/810** metric/metadata/specification triplets agree; file hashes for metric and run-spec files match the archived completion records.
3. Frozen selection: **81/81** selected lookbacks and all **486/486** candidate averages/standard deviations agree with the authorized `selection_frozen.json` (excluding `lstm_original`). Its SHA-256 matches the `test_authorization.json` authorization.
4. Test aggregates: **81/81** tested model/lookback combinations agree with Validation-only selection; original Test manifest contains all **405** selected model/seed combinations.
5. Per-variable Test: **3,645/3,645** unique results, with five seeds per variant/model/horizon/variable. TBS and RH Test MAE and nMAE-IQR, and global nMAE-IQR, reproduce the 81 aggregate rows within numeric formatting precision.
6. The original source had 540 Test manifest records, including 135 legacy `lstm_original` records. Only the 405 runs associated with the three definitive architectures are included in the published result tables.
7. Resolved V2 PatchTST selection: **H16 → L384** and **H24 → L288**, as shown by Ubuntu Validation and already evaluated in the authorized Test campaign; **no re-training and no new Test evaluation** performed.

## Original-source SHA-256

- `validation_all_V1.csv`: `300579a315843a49a11f465de89366b07e2b4db75e3b8fa05abcc6f5c6e66c59`
- `validation_all_V3.csv`: `de6d8446fdc89b99bcd5687d7ce4634299797aad6a80d50a1724e791dc237da1`
- `PGAF_validation_V2_Ubuntu.zip`: `c1d522a634cd04013d49222d0335aa0c91cbfeb8fa6f4fe62477fde90be94434`
- `PGAF_forecasting_test_original.zip`: `af402ad9342c2fb32d37cc5af6099aa5e569ad0e009e123d98d36439ede36daa`

The raw Test archives, complete checkpoints, per-origin Validation error arrays, and stepwise Persistence prediction arrays are not included in this compact results package. Preserve those source artifacts separately when archiving the full experimental evidence. They are not required to regenerate the tabular comparisons provided here.
