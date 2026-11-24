# Genetic-algorithm optimization

This component contains the **`GA_ENGINEERING_AWARE`** optimizer, configuration files, historical surrogate model and input dataset, and the aggregate tuning results of the PGAF greenhouse control study. Only the GA variant selected for the reported study is provided here; other exploratory variants are outside the released implementation.

## Experimental design

The optimizer searches over **six greenhouse actuators** using a composite objective that includes climate-target performance and engineering-aware penalties, including an actuator/energy-effort proxy. It is a simulation-based optimization experiment, **not a deployed autonomous greenhouse controller**.

| Element | Configuration |
| --- | --- |
| Prediction surrogate | Historical **TensorFlow/TF-Keras LSTM** checkpoint |
| Actual surrogate input | **96** previous 15-minute steps × **17** features |
| Surrogate output | **Nine** environmental variables, **one step ahead** |
| Control horizon | **12 sequential 15-minute decisions (3 hours)** |
| Objective/algorithm | `GA_ENGINEERING_AWARE` |
| Tuning grid | **36** configurations × **3** episodes × **5** seeds = **540** cases |
| Budget per control decision | **10,000** candidate evaluations |
| Selected configuration | Population **200**, generations **50**, tournament **2**, conservative mutation profile, Gaussian sigma fraction **0.04**, early stopping disabled |

The selection criterion was **minimum mean step-best fitness**, averaged across the 12 optimization steps, episodes and seeds. The archived [`results/tuning_results.csv`](results/tuning_results.csv) contains the 36 aggregate configurations. The selected configuration is `pop200_gen50_k2_mutconservative_sigma0p04` (reported aggregate mean fitness approximately **0.0925904**).

**Evaluation scope:** the aggregate selected-configuration results concern the tuning episodes. They should **not** be described as independent held-out controller validation. Full original per-case trajectory logs are not included, and exact numerical reproduction of all historical GA trajectories has not been established.

## Source files

```text
optimization/
├── core/
│   ├── constants.py
│   ├── ga_method_registry.py
│   ├── genetic_algorithm_class.py
│   ├── individual_class.py
│   ├── run_genetic_algorithm_experiments.py
│   └── utils.py
├── scripts/
│   ├── run_selected_ga.py
│   ├── run_tuning_grid.py
│   └── aggregate_tuning_grid_results.py
├── datasets/ETIFA_clean_imputed_1h.csv
├── models/LSTM_L48_H1.h5
├── results/tuning_results.csv
├── selected_ga_config.json
├── tuning_grid_config.json
└── requirements.txt
```

The scripts are launched as **Python modules from the repository root** (`python -m optimization.scripts...`). Generated runs go under `optimization/out/`, excluded from version control. The reusable GA implementation is under `core/`; `utils.py` is the renamed historical utility module, without the old `_v` suffix. Tests live in [`../tests/test_optimization.py`](../tests/test_optimization.py).

## Historical artifacts and naming

The two included artifacts preserve established repository filenames, but those names should **not** be used to infer scientific equivalence with the forecasting component:

| Repository artifact | Historical source name | SHA-256 |
| --- | --- | --- |
| [`models/LSTM_L48_H1.h5`](models/LSTM_L48_H1.h5) | `model_lstm_lb96.h5` | `49869b812ef659e8cecacd9e0e2789ed17e265642cae9dbbdc7bc182858ada68` |
| [`datasets/ETIFA_clean_imputed_1h.csv`](datasets/ETIFA_clean_imputed_1h.csv) | `DIRTY-DS-15-SUMMER.csv` | `6ab084f5d45a4613eaba81d46a165788ea2765f577526b530a816ceccbf08b9b` |

In particular, the checkpoint filename includes `L48`, but the verified Keras input shape is **`(None, 96, 17)`**, not a lookback of 48. The GA dataset has a **different hash and historical provenance** from [`../preprocessing/data/processed/ETIFA_clean_imputed_1h.csv`](../preprocessing/data/processed/ETIFA_clean_imputed_1h.csv). The GA's TensorFlow/Keras model and the forecasting study's PyTorch checkpoint files are **distinct serialized artifacts with different input configurations**. Do not substitute one dataset or checkpoint for the other.

## Environment and checks

The GA inference runtime needs a **separate Python 3.10 environment**, because its [`requirements.txt`](requirements.txt) pins TensorFlow/TF-Keras and NumPy 1.x. From the repository root:

```bash
conda create -n pgaf-ga python=3.10 pip -y
conda activate pgaf-ga
python -m pip install -r optimization/requirements.txt
```

Inspect the provided experiment configurations without running expensive optimization:

```bash
python -m optimization.scripts.run_selected_ga --case-count   # 15
python -m optimization.scripts.run_tuning_grid --config-count # 36
python -m optimization.scripts.run_tuning_grid --case-count   # 540
python -m unittest discover -s tests -p 'test_optimization.py' -v
```

The eight unit checks validate configuration, operators, ranking consistency and artifact hashes. **They use a surrogate test stub for fitness checks, not the real TensorFlow model.** To load the historical `.h5` for a separate inference-environment check:

```bash
python -c 'from optimization.core.utils import get_model; m=get_model(); print(m.input_shape, m.output_shape)'
```

For an individual GA reproduction run, first inspect `--help` and the [`selected_ga_config.json`](selected_ga_config.json) settings; this performs substantially more computation than the checks above:

```bash
python -m optimization.scripts.run_selected_ga --case-index 0 \
  --output-root optimization/out/reproduction_check
```

Do not interpret a successful software smoke run as proof of bit-for-bit reproduction of the original GA trajectories. For forecasting results and the distinct PyTorch LSTM checkpoints, consult [Forecasting](../forecasting/README.md).
