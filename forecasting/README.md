# Multivariate greenhouse climate forecasting

This component implements and documents the predictive part of PGAF: **LSTM** (`lstm_sergiot`), **iTransformer** (`itransformer`), **PatchTST-MV** (`patchtst_mv`) and a **Persistence** baseline. The LSTM is presented simply as **LSTM** in the manuscript. The legacy `lstm_original` is not a fourth reported neural model.

## Forecasting protocol

| Element | Published configuration |
| --- | --- |
| Sampling | 15-minute intervals |
| Input | Nine historical environmental sensors + six historical actuators + four cyclic time features = **19 channels** |
| Output | Nine environmental variables |
| Horizons (steps) | **4, 8, 16, 24, 48, 96, 144, 192, 288** |
| Candidate lookbacks (steps) | **48, 96, 192, 288, 384, 672** |
| Dataset versions | **V1, V2, V3**, defined in [`splits.json`](splits.json) |
| Training seeds | **8, 13, 22, 27, 31** |
| Model selection | Lowest mean global Validation nMAE-IQR over five seeds, then standard deviation and smaller lookback as tie-breakers |

Training uses complete observed windows. Validation and Test use the documented future-actuator stability restrictions **to select retrospective evaluation windows**; future actuator observations are not fed to the predictor. Normalization statistics are fitted on Train. For wind direction, the loss and error evaluation account for circular angular differences.

**No Test scores enter training, checkpoint selection or lookback selection.** The evaluation pipeline requires explicit authorization before running Test.

## Where to find things

| Location | Purpose |
| --- | --- |
| [`core/architectures.py`](core/architectures.py) | Model architecture definitions |
| [`core/models.py`](core/models.py) | Published model constructors and output heads |
| [`core/data.py`](core/data.py) | Temporal partitions, scaling and evaluation windows |
| [`core/metrics.py`](core/metrics.py) | Physical and normalized error metrics |
| [`scripts/pipeline.py`](scripts/pipeline.py) | Train, Validation, selection, authorized Test and reporting commands |
| [`config.json`](config.json) / [`splits.json`](splits.json) | Experimental settings and frozen temporal partitions |
| [`results/`](results/) | Audited Validation, Test and Persistence tables |
| [`checkpoints/`](checkpoints/) | Selected PyTorch model weights and [`SHA-256 manifest`](checkpoints/manifest.csv) |
| [`experiments/actuator_change_sensitivity/`](experiments/actuator_change_sensitivity/) | H16 actuator-change study |
| [`../tests/test_forecasting.py`](../tests/test_forecasting.py) | Protocol, architecture and data-integrity checks |

## Included evaluation evidence

- **2,430** original Validation runs (three versions × three architectures × nine horizons × six lookbacks × five seeds).
- **81** Validation-selected combinations (three versions × three architectures × nine horizons).
- **405** selected Test model/seed evaluations and **405** corresponding PyTorch checkpoints.
- **27** aggregated Persistence baseline comparisons.

The source-campaign audit and exact CSV descriptions are in [`results/README.md`](results/README.md). Checkpoint paths follow `checkpoints/<version>/H<horizon>/L<lookback>/<model>/seed<seed>/best.pt`. The 81 selected lookbacks are in [`results/validation/selected_lookbacks.csv`](results/validation/selected_lookbacks.csv).

## Verification and execution

Use a suitable PyTorch environment installed from [`requirements.txt`](requirements.txt). From the repository root:

```bash
python -m unittest discover -s tests -p 'test_forecasting.py' -v
python -m forecasting.scripts.pipeline --help
python forecasting/experiments/actuator_change_sensitivity/verify_results.py
```

The pipeline also supports `preflight`, `jobs`, `train`, `select`, `authorize-test`, `test`, `persistence` and `report` commands. For example, to validate the dataset and model setup without starting training:

```bash
python -m forecasting.scripts.pipeline preflight
```

**Do not rerun selection against Test or launch the training campaign simply to inspect published results.** Full training and inference require the appropriate environment and computational resources. The retained checkpoints and table files document the existing campaign.

## Actuator-change sensitivity

The separate retrospective H16 analysis compares **OBS**, **MASK** and **SUST** input histories for **TBS** and **RH** using 45 selected model/seed checkpoints and 172 candidate episodes. It evaluates sensitivity to actuator-change information, **not a causal greenhouse intervention**. Full definitions, saved paired predictions and audit commands are in the [experiment README](experiments/actuator_change_sensitivity/README.md).

## Scope of the optimization study

The GA uses a **historical TensorFlow/Keras LSTM checkpoint and its own dataset**, not these PyTorch checkpoint files. See [Optimization](../optimization/README.md) for that experiment and the restrictions on comparability and reproducibility.
