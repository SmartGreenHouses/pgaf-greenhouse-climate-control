# PGAF — Genetic Algorithm optimization

Research code for the **single** selected GA technique, `GA_ENGINEERING_AWARE`, used in the *Ecological Informatics* manuscript. Other exploratory evolutionary variants are deliberately omitted. The algorithm simulates control of six greenhouse actuators using an LSTM surrogate and a composite engineering-aware fitness.

## Critical provenance: labels versus actual data and model

The author requested the following names for the historical files:

| Required local file | SHA-256 | Original artifact |
| --- | --- | --- |
| `optimization/models/LSTM_L48_H1.h5` | `49869b812ef659e8cecacd9e0e2789ed17e265642cae9dbbdc7bc182858ada68` | `model_lstm_lb96.h5` from historical GA training |
| `optimization/datasets/ETIFA_clean_imputed_1h.csv` | `6ab084f5d45a4613eaba81d46a165788ea2765f577526b530a816ceccbf08b9b` | `DIRTY-DS-15-SUMMER.csv` used by the historical GA |

**These names are aliases, not proof of equivalence.** The existing `.h5` checkpoint **actually takes 96 historical steps × 17 features** (`(None, 96, 17)`) and predicts 9 environmental variables for the next step (`(None, 9)`). Its renamed filename `LSTM_L48_H1.h5` **does not mean its real lookback is 48**. The GA subsequently optimizes 12 successive 15-minute control decisions (3 hours).

The renamed historical GA CSV is **not verified to be the same dataset** as the ETIFA preprocessing/V1–V3 forecasting dataset. Its SHA-256 differs from the previously supplied ETIFA imputed CSV. Do **not** claim they are the same data or that the GA used the PyTorch `lstm_sergiot` checkpoint. The GA uses its historical **TensorFlow/Keras** checkpoint, not the PyTorch forecasting model. These differences must be made explicit in the manuscript before publication.

The GitHub repository includes the historical model and dataset at the paths above. Their SHA-256 hashes establish artifact identity, while their filenames are aliases that must not be interpreted as architectural or dataset equivalence. A single `.gitignore` is maintained at the repository root.

## Files

```text
optimization/
  README.md
  requirements.txt
  constants.py
  genetic_algorithm_class.py
  individual_class.py
  ga_method_registry.py
  utils_v.py
  run_genetic_algorithm_experiments.py
  run_selected_ga.py
  run_tuning_grid.py
  aggregate_tuning_grid_results.py
  selected_ga_config.json
  tuning_grid_config.json
  results/tuning_results.csv
  tests/test_optimization.py
  models/LSTM_L48_H1.h5                 # original artifact included
  datasets/ETIFA_clean_imputed_1h.csv   # original artifact included
```

The `out/` directory contains generated outputs and is not tracked. The script/config labels containing `lb96_h12` in original grid metadata preserve traceability to archived experiment identifiers; `lb96` describes the **real** model history and `h12` the **GA control duration**.

## Install and test

Python 3.10 in a separate environment is recommended:

```bash
conda create -n pgaf-ga python=3.10 pip -y
conda activate pgaf-ga
python -m pip install -r optimization/requirements.txt
```

Run from the repository root:

```bash
PYTHONPATH=optimization python -m unittest discover -s optimization/tests -p 'test_optimization.py' -v
sha256sum optimization/models/LSTM_L48_H1.h5 optimization/datasets/ETIFA_clean_imputed_1h.csv
```

The eight tests cover the selected GA operators/settings, the 36-config grid, published tuning summary, and hashes of the two original assets. If model and CSV are absent, the asset test is skipped. Unit fitness tests use a simulated predictor; they **do not** test actual TensorFlow inference. To load the actual `.h5`:

```bash
PYTHONPATH=optimization python -c 'from utils_v import get_model; m=get_model(); print(m.input_shape, m.output_shape); assert m.input_shape==(None,96,17) and m.output_shape==(None,9)'
```

## Selected GA configuration

The manuscript's selection criterion is the **minimum mean best fitness** across all 12 control steps of each tuning case, averaged over all five seeds and three episodes. The selected settings are:

- Method: `GA_ENGINEERING_AWARE`
- `POPULATION_NUM=200`, `GENERATIONS_NUM=50`, `K_TOURNAMENT=2`
- Conservative mutation: no mutation `0.80`, mutation 1 `0.15`, mutation 2 `0.04`, mutation 3 `0.01`
- `GAUSSIAN_SIGMA_FRACTION=0.04`, early stopping disabled
- Three 24-hour input episodes × five seeds = **15 cases**, each containing **12 subsequent 15-minute GA control decisions**
- **10,000** LSTM–GA evaluations per control decision; **120,000** per 3-hour simulated episode

```bash
python optimization/run_selected_ga.py --case-count
python optimization/run_selected_ga.py --case-index 0 --output-root optimization/out/reproduction_check
```

The already-performed local Mac smoke run completed case 0 in 12 steps / 120,000 evaluations, with MAE(TBS) = 0.8885 °C and MAE(HR) = 3.2283 percentage points. These are **diagnostic local-run values**, not the manuscript aggregate. Its trajectory does **not** numerically match the archived original case, despite successful execution; full numeric/statistical reproducibility has **not** been established.

## Historical tuning grid and results

- Three budgets: `200×50`, `100×100`, `50×200` (fixed 10,000 evaluations/step)
- Two tournament sizes: `k=2`, `k=5`
- Two mutation profiles: conservative, aggressive
- Three mutation scales: `0.04`, `0.08`, `0.12`

Total: **36 configurations × 3 episodes × 5 seeds = 540 historical cases**, each with 12 control steps. The included `results/tuning_results.csv` contains **all 36 original aggregate rows**, with 15 cases and 180 step records per configuration. Strict-minimum winner: `pop200_gen50_k2_mutconservative_sigma0p04`, mean step-best fitness **0.09259039864241379**.

```bash
python optimization/run_tuning_grid.py --config-count   # 36
python optimization/run_tuning_grid.py --case-count     # 540
# Individual case only, if required:
# python optimization/run_tuning_grid.py --case-index 0
```

`aggregate_tuning_grid_results.py` now uses the **same minimum-mean-fitness rule as the manuscript** when the original per-case output files are supplied. An earlier exploratory script used a **one-standard-error then fastest** rule and selected a different configuration; that alternative was **not** the article's selection criterion. The historical aggregate CSV is provided directly: no full rerun is necessary for reading or checking the reported results.

## Scope limits

The actual checkpoint and dataset are **required to reproduce inference** and must be included separately alongside this code. Full SIRIUS per-case logs and held-out evaluation records are **not included** in the ZIP. Do not claim those are present. Maintain a clear separation between hyperparameter tuning data and independent later evaluation.
