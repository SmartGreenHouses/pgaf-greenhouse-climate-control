# Greenhouse data preprocessing

This component contains the original observations, the cleaning and short-gap reconstruction implementation, quality-control audits, and the figures supporting the temporal partitioning of the PGAF forecasting study. **All timestamps and partition boundaries below are in UTC.**

## Datasets and provenance

| File | Description |
| --- | --- |
| [`data/raw/ETIFA_raw_data.csv`](data/raw/ETIFA_raw_data.csv) | Original greenhouse observations (114,048 rows) |
| [`data/processed/ETIFA_clean.csv`](data/processed/ETIFA_clean.csv) | Trimmed and quality-controlled observations, before imputation |
| [`data/processed/ETIFA_clean_imputed_1h.csv`](data/processed/ETIFA_clean_imputed_1h.csv) | Curated 15-minute series after bounded reconstruction; canonical input to forecasting |
| [`data/processed/ETIFA_cleaning_audit.csv`](data/processed/ETIFA_cleaning_audit.csv) | Cleaning-rule invalidations |
| [`data/processed/ETIFA_imputation_summary.csv`](data/processed/ETIFA_imputation_summary.csv) | Reconstruction counts by variable |

Trimming starts at **2019-04-29 07:30 UTC**, after removing 13,182 initial rows that precede availability of the final actuator channel. The curated series contains **100,866 rows** and ends at **2022-03-14 23:45 UTC**. The `1h` suffix indicates the **maximum eligible short-gap duration**, *not* one-hour sampling; observations remain at **15-minute resolution**.

Quality control marks recorded outliers and physical inconsistencies as missing. Reconstruction uses limited linear interpolation for appropriate scalar channels, circular interpolation for wind direction, a derived humidity-deficit estimate where its prerequisite measurements permit, and actuator-specific handling of missing values. It **does not** fill every missing observation.

According to [`results/data_quality/data_quality_manifest.json`](results/data_quality/data_quality_manifest.json), 351,327 missing cells were reconstructed and 25,558 remained missing in the curated dataset. These counts refer to the trimmed and audited data; they must not be misread as missingness over the full raw-data period.

## Temporal coverage and partitions

The curated CSV preserves the data of the excluded 2021 interval. The period **2021-06-27 17:45 UTC through 2021-10-31 23:45 UTC** is excluded from the forecasting partitions, **not deleted from the CSV**. Each V1/V2/V3 allocation contains 88,745 unique timestamps divided into Train, Validation and Test. Exact ranges and row counts are frozen in [`../forecasting/splits.json`](../forecasting/splits.json).

A figure of the arrangements is available at [`results/partitions/ETIFA_V1_V2_V3_temporal_partitions.pdf`](results/partitions/ETIFA_V1_V2_V3_temporal_partitions.pdf).

## Code and evidence

- [`scripts/etifa_preprocess.py`](scripts/etifa_preprocess.py): cleaning and imputation, without dataset splitting.
- [`scripts/audit_missing_data.py`](scripts/audit_missing_data.py): variable-level missingness, temporal coverage and manifest.
- [`scripts/audit_temporal_exclusion.py`](scripts/audit_temporal_exclusion.py): evidence about the historical 2021 exclusion.
- [`scripts/interpolation_sensitivity.py`](scripts/interpolation_sensitivity.py): retrospective artificial-gap reconstruction sensitivity across five periods.
- `scripts/make_*_figure*.py` and `scripts/figure_style.py`: publication figures and shared figure styling.
- [`results/`](results/): archived audits, interpolation summaries and figures.

The interpolation sensitivity study artificially masks selected observed periods; **it is not a comparison of downstream forecasting scores after retraining on alternative imputed datasets**.

## Inspect and reproduce safely

Install the component's dependencies from [`requirements_preprocessing.txt`](requirements_preprocessing.txt). Run from the repository root:

```bash
python -m unittest discover -s tests -p 'test_preprocessing.py' -v

# Recreate the processed CSVs in a NEW directory; do not overwrite published data.
python preprocessing/scripts/etifa_preprocess.py \
  preprocessing/data/raw/ETIFA_raw_data.csv \
  --output-dir /tmp/pgaf-preprocessing-reproduction

# Recalculate quality-control tables in another NEW directory.
python preprocessing/scripts/audit_missing_data.py \
  --output-dir /tmp/pgaf-quality-audit-reproduction
```

Choose empty output directories. The archived datasets and results should not be overwritten when checking reproducibility. For model training and evaluation, continue to [Forecasting](../forecasting/README.md).
