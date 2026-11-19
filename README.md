# PGAF — Research code (initial release)

Source code and experiment configuration for the forecasting component of
**A Data-Driven Predictive Analytics Framework for Sustainable Greenhouse Climate Control Using Deep Learning and Evolutionary Optimization**
(manuscript ECOINF-D-26-02633R1).

## Scope and status

This is an **initial, incomplete code release** extracted from the researchers' ETIFA
backup, not a complete reproducibility package. It contains the main forecasting
model implementations, training/evaluation utilities and the v10.1 forecasting
campaign's experiment configuration and split manifest.

The following are **not included in this initial release** and are pending
verification/publication as permitted: the GA implementation and tuning scripts,
the full data-curation and sensitivity-analysis pipelines, validated numeric
outputs, environment lockfile and detailed reproduction instructions.

## Data availability

The raw/curated greenhouse measurements were supplied by NUTRICONTROL.
Redistribution of those measurements is subject to the provider's permission;
no greenhouse measurements are included here. The split manifest describes
partition intervals and window feasibility, **not the underlying observations**.
For data access inquiries, contact the corresponding authors subject to the
provider's permissions. The SHA-256 recorded in the experiment metadata is
provided solely for provenance, not as a substitute for data access.

## Main source paths

- `src/etifa_direct/models.py` and `original_models.py`: forecasting architectures.
- `src/etifa_direct/data.py`, `metrics.py`, `train.py`: supporting pipeline.
- `campaigns/ETIFA_CAMPANA_6L_CORREGIDA_v10_1_R2_VIENTO_2026-11-14/`:
  versioned forecasting experiment driver and configuration.

The campaign driver explicitly sets the number of prediction targets to nine.
The lower-level reusable model modules also preserve legacy defaults and
alternative architecture names. The final published LSTM is implemented under
`lstm_sergiot`. Avoid running standalone helpers as if they were the final
campaign without first examining the versioned configuration.

## Running and dependencies

Requires a compatible Python/PyTorch environment and access to the authorized,
curated input dataset. See `requirements.txt` for the major imported libraries;
exact versions have not yet been verified. The campaign CLI supports a
`--help` command. No executed results are claimed from this archive.

## Intellectual property and license

No open-source license is asserted in this initial release. Redistribution and
reuse rights for the code should be confirmed by the rights holders before
adding an explicit license. Public access to a repository alone is not a grant
of unrestricted reuse rights.
