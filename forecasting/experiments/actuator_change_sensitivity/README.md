# Actuator-Change Sensitivity Analysis (H16)

This directory contains the original **OBS / MASK / SUST** evidence and a repository-adapted reproduction runner for the H16 (4-hour) actuator-change analysis in the PGAF manuscript.

## Protocol (as actually executed)

- **Models**: LSTM (`lstm_sergiot`), iTransformer and PatchTST-MV. All **nine** H16 configurations were selected by Validation only; five seeds `(13, 27, 22, 8, 31)` per configuration, **45** checkpoints total.
- **Inputs**: nine sensor measurements, six past actuator measurements and four cyclic time features (**19 channels**). Train-only mean/standard deviation normalization. Actual values for **TBS** and **RH** are evaluated in physical units, with MAE also normalized by each sensor's Train IQR.
- **Candidate episodes**: a single actuator changes at forecast origin `t` by at least the H16 threshold; all six actuator channels are present and their ranges remain *strictly below* their thresholds from `t` through `t+16`. Lookback completeness is checked using maximum lookback 672, so every model within a version uses the same origins.
- **OBS**: keep actual input history.
- **MASK**: replace only the changed actuator value at `t` with `a[t−1]`.
- **SUST**: replace the changed actuator's last 16 historical values, `a[t−15] … a[t]`, with `a[t−1]`.
- No future values are injected into the network. **Future actuator measurements are used to select stable episodes retrospectively**; this is a conditional post-hoc sensitivity test, **not** a real-time intervention or causal experiment.

**Methodological qualification:** the original implementation itself labels the event thresholds as a *candidate definition requiring justification*. The thresholds were applied in producing these results but are not independently validated as universal or physically optimal change thresholds. The runner preserves them instead of silently redesigning the experiment. They must be explained accurately in the manuscript.

## Included evidence

- `results/protocol.json`: original protocol and frozen source-file hashes.
- `results/audit_event_counts.csv`: **75 / 37 / 60** candidate episodes for V1 / V2 / V3 (172 total).
- `results/metrics_by_seed.csv`: **270** rows, all 45 checkpoints × OBS/MASK/SUST × TBS/RH.
- `results/metrics_mean_5_seeds.csv`: **54** rows.
- `results/paired_predictions/`: **45 compressed NPZ** with the exact matched origin rows, actuator IDs, truths and condition-specific predictions. They provide an independent audit trail for recalculating MAEs and shifts.
- `results/reporting/`: paired breakdowns by actuator, seed, version and episode.
- `results/figures/`: plots and tabular figure summaries from the historical analysis.

The original results were generated with dataset SHA-256 `3d74ad9113a9b35a03059786eeeb007119bd7107de6b74eb346625a65673dc15`, matching `preprocessing/data/processed/ETIFA_clean_imputed_1h.csv`. The **frozen original** `experiment.json` and `split_manifest.json` have different byte hashes from the simplified public `forecasting/config.json` and `forecasting/splits.json`; the relevant H16 thresholds, temporal splits, seeds and lookback choices are preserved. `protocol.json` retains the *original* frozen hashes for traceability.

## Reported equal-weighted results

Relative change in mean TBS/RH normalized MAE versus OBS, averaging V1, V2, V3 equally:

| Model | MASK | SUST |
|---|---:|---:|
| LSTM | +1.43% | +3.05% |
| iTransformer | −0.67% | +0.07% |
| PatchTST-MV | −0.34% | +0.20% |

An **increase** in error after MASK or SUST is compatible with useful actuator-change information in the predictor. Unchanged or lower errors **do not** support that claim. Prediction shifts alone are not evidence of better prediction. These are descriptive experimental contrasts, not proof of causal identification.

## Audit the existing evidence

Run from the repository root in an environment with NumPy and pandas:

```bash
python forecasting/experiments/actuator_change_sensitivity/verify_results.py
```

Expected: `EVIDENCIA D VERIFICADA ...`.

## Reproduce episode selection (no inference)

```bash
python forecasting/experiments/actuator_change_sensitivity/run.py
```

Requires `preprocessing/data/processed/ETIFA_clean_imputed_1h.csv`, `forecasting/config.json`, `forecasting/splits.json` and `forecasting/results/validation/selected_lookbacks.csv`. The runner checks the dataset SHA-256 and partition boundaries and should find **75 / 37 / 60** episodes. No files are written in audit mode.

## Verify one historical inference (CUDA smoke test)

With PyTorch installed and the published checkpoints present:

```bash
python forecasting/experiments/actuator_change_sensitivity/run.py --smoke --device cuda
```

This runs one checkpoint (V1/H16/L48/LSTM/seed8) against all V1 event origins, and compares OBS/MASK/SUST predictions with the original saved NPZ, without writing results. A maximum discrepancy of 0.001 in physical units is allowed. **Verified on Ubuntu with CUDA: OBS, MASK and SUST matched the archived predictions within approximately 3.81e-6 physical units for the selected LSTM checkpoint. This does not establish numerical equivalence for all 45 checkpoints.**

## Re-execute inference (optional, *not* needed to publish existing results)

Requires PyTorch, NumPy, pandas and the 45 selected H16 checkpoints in `forecasting/checkpoints/`:

```bash
python forecasting/experiments/actuator_change_sensitivity/run.py \
  --evaluate --confirm-event-definition --device cuda \
  --output-root /tmp/pgaf_actuator_sensitivity_reproduction
```

The runner uses `forecasting.models`, whose three published architectures preserve the parameter layout of the historical selected models. Unlike the original campaign script, it uses current public repository paths; **numerical equivalence of a fresh full inference run must still be verified against the stored NPZ** before claiming exact reproduction of the 45 original runs. The runner refuses to overwrite a nonempty output directory.

No data or checkpoints should be renamed or overwritten for this experiment. The GA's historical Keras `.h5` model is **not** used here.
