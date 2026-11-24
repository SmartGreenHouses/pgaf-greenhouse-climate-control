# Predictive Analytics and Evolutionary Optimization for Greenhouse Climate Control

Research repository accompanying **“A Data-Driven Predictive Analytics Framework for Sustainable Greenhouse Climate Control Using Deep Learning and Evolutionary Optimization”**, submitted to *Ecological Informatics* (manuscript **ECOINF-D-26-02633**, revised submission R1).

The repository contains greenhouse observations, preprocessing code and audits, multivariate forecasting models and evaluation records, a retrospective actuator-change sensitivity study, and a genetic-algorithm (GA) optimization experiment.

**Versioning:** [`ecoinf-r1-2026-11-24`](https://github.com/SmartGreenHouses/pgaf-greenhouse-climate-control/tree/ecoinf-r1-2026-11-24) preserves the repository snapshot associated with the R1 submission. The `main` branch contains subsequent organizational improvements; it is not an identical file-tree snapshot of the submitted version.

## Research workflow

| Component | Start here | Contents |
| --- | --- | --- |
| **Data preparation** | [Preprocessing](preprocessing/README.md) | Raw and processed observations, cleaning/imputation code, temporal coverage and reconstruction audits |
| **Climate forecasting** | [Forecasting](forecasting/README.md) | LSTM, iTransformer, PatchTST-MV, Persistence baseline, validation/test results and selected checkpoints |
| **Actuator sensitivity (H16)** | [Actuator-change experiment](forecasting/experiments/actuator_change_sensitivity/README.md) | Retrospective OBS/MASK/SUST comparisons, matched predictions and result verification |
| **Evolutionary optimization** | [Optimization](optimization/README.md) | Engineering-aware GA, historical LSTM surrogate, tuning grid and aggregate results |

The preprocessing and forecasting study uses **15-minute observations** and three temporal partitions (**V1/V2/V3**). Forecasting models use **19 input channels** (nine environmental measurements, six historical actuator measurements, four cyclic time features) to predict nine environmental variables. The GA experiment uses **its own historical dataset and TensorFlow/Keras LSTM checkpoint**, with a different serialized artifact and input configuration from the published PyTorch forecasting checkpoints. The two datasets must not be treated as interchangeable.

## Repository layout

```text
.
├── preprocessing/
│   ├── README.md
│   ├── data/{raw,processed}/
│   ├── scripts/
│   └── results/
├── forecasting/
│   ├── README.md
│   ├── core/                 # architectures, models, data and metrics
│   ├── scripts/              # forecasting pipeline
│   ├── experiments/          # H16 actuator-change sensitivity
│   ├── checkpoints/          # selected PyTorch model weights and SHA-256 manifest
│   └── results/              # Validation, Test and Persistence tables
├── optimization/
│   ├── README.md
│   ├── core/                 # GA implementation and utilities
│   ├── scripts/              # existing experiment commands
│   ├── datasets/             # historical GA input
│   ├── models/               # historical TensorFlow/Keras checkpoint
│   └── results/              # original aggregate tuning table
├── tests/                    # unit, protocol and integrity checks
└── README.md
```

Each component retains its own configuration and dependency files. The experiment-specific documentation identifies its inputs, outputs and verification commands.

## Published evidence at a glance

| Study | Available evidence |
| --- | --- |
| Forecasting | **2,430** Validation runs; **81** Validation-selected model/horizon/version configurations; **405** selected PyTorch checkpoints and corresponding Test evaluations |
| Persistence | **27** aggregated baseline comparisons |
| Actuator-change sensitivity | **45** model/seed evaluations across OBS, MASK and SUST; **172** candidate episodes; **45** paired-prediction NPZ files |
| GA tuning | **36** parameter configurations × three episodes × five seeds = **540** historical cases; aggregate table of 36 configurations |

For exact file definitions, selection rules and experiment limitations, see the component READMEs rather than relying on this overview.

## Verify the supplied evidence

Run from the **repository root**, using an environment with NumPy, pandas and PyTorch:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python forecasting/experiments/actuator_change_sensitivity/verify_results.py
python -m optimization.scripts.run_selected_ga --case-count    # 15
python -m optimization.scripts.run_tuning_grid --config-count  # 36
```

These commands inspect configurations and saved evidence. **They do not retrain the forecasting models or rerun the GA tuning grid.** The historical GA surrogate requires a separate TensorFlow/TF-Keras environment for actual model inference; see [Optimization](optimization/README.md).

## Interpretation and limitations

- Forecasting lookbacks were selected from **Validation**, before evaluating **Test**. Results are observational; neither forecasts nor actuator-input sensitivity comparisons establish causal control effects.
- The H16 sensitivity study uses future actuator observations **only for retrospective episode selection**, not as model inputs. Its event thresholds reproduce the historical definition and are not independently established as physically optimal.
- The historical GA model/data artifacts are distinct from the canonical forecasting artifacts. GA tuning summaries do **not** amount to independent held-out controller validation; exact numerical reproduction of all original GA trajectories has not been established.
- Checkpoint manifests and experiment-specific hashes support artifact identification, but do not replace the underlying methodology or establish unrestricted rights to reuse third-party data. No general reuse license is supplied in this repository.

**Citation:** *A Data-Driven Predictive Analytics Framework for Sustainable Greenhouse Climate Control Using Deep Learning and Evolutionary Optimization*, *Ecological Informatics*, manuscript ECOINF-D-26-02633. Journal publication details and a DOI should be added only after they are available.
