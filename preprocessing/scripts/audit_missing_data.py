#!/usr/bin/env python3
"""Read-only, reproducible ETIFA missing-data and temporal-coverage audit.

Requires Python >=3.10, pandas and numpy. Does not modify input files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'

SENSORS = (
    "dry_bulb_temperature", "wet_bulb_temperature", "relative_humidity",
    "humidity_deficit", "outdoor_temperature", "outdoor_relative_humidity",
    "wind_speed", "wind_direction", "radiation",
)
ACTUATORS = (
    "pipe_temperature", "humidification", "window_1_position",
    "window_2_position", "screen_1_position", "screen_2_position",
)
VARIABLES = SENSORS + ACTUATORS
RAW_RENAME = {
    "Fecha": "timestamp", "CajaMedida1.TBS": "dry_bulb_temperature",
    "CajaMedida1.TBH": "wet_bulb_temperature", "CajaMedida1.HR": "relative_humidity",
    "CajaMedida1.DH": "humidity_deficit", "TemperaturaExterior": "outdoor_temperature",
    "HumedadExterior": "outdoor_relative_humidity", "VelocidadViento": "wind_speed",
    "DireccionViento": "wind_direction", "Radiacion": "radiation",
    "TemperaturaTuberia": "pipe_temperature", "Humidificacion": "humidification",
    "PosicionVentana1": "window_1_position", "PosicionVentana2": "window_2_position",
    "PosicionPantalla1": "screen_1_position", "PosicionPantalla2": "screen_2_position",
}
EXCLUDED_START = pd.Timestamp("2021-06-27 17:45:00+00:00")
EXCLUDED_END = pd.Timestamp("2021-10-31 23:45:00+00:00")
SPLITS = {
    "V1": {
        "train": ("2019-04-29 07:30:00+00:00", "2021-02-08 23:45:00+00:00"),
        "validation": ("2021-02-09 00:00:00+00:00", "2021-06-27 17:30:00+00:00"),
        "test": ("2021-11-01 00:00:00+00:00", "2022-03-14 23:45:00+00:00"),
    },
    "V2": {
        "train": ("2019-04-29 07:30:00+00:00", "2021-02-08 23:45:00+00:00"),
        "validation": ("2021-11-01 00:00:00+00:00", "2022-03-14 23:45:00+00:00"),
        "test": ("2021-02-09 00:00:00+00:00", "2021-06-27 17:30:00+00:00"),
    },
    "V3": {
        "train": ("2019-09-10 07:30:00+00:00", "2021-06-27 17:30:00+00:00"),
        "validation": ("2021-11-01 00:00:00+00:00", "2022-03-14 23:45:00+00:00"),
        "test": ("2019-04-29 07:30:00+00:00", "2019-09-10 07:15:00+00:00"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, original: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if original:
        df = df.rename(columns=RAW_RENAME)
    expected = {"timestamp", *VARIABLES}
    if set(df.columns) != expected or len(df.columns) != len(expected):
        raise ValueError(f"Unexpected columns in {path.name}. Missing: {expected - set(df.columns)}; extra: {set(df.columns) - expected}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    if df["timestamp"].isna().any():
        raise ValueError(f"Missing timestamps in {path.name}")
    if not df["timestamp"].is_unique or not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Timestamps are not unique and ascending in {path.name}")
    if not df["timestamp"].diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError(f"15-minute grid is not continuous in {path.name}")
    for col in VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col]).any():
            raise ValueError(f"Infinite values in {path.name}/{col}")
    return df


def missing_run_rows(mask: np.ndarray, timestamps: pd.Series, variable: str, stage: str):
    """Return runs for *every* consecutive missing stretch, even single-point runs."""
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [{
        "stage": stage, "variable": variable,
        "start_utc": timestamps.iloc[int(start)].isoformat(),
        "end_utc": timestamps.iloc[int(end - 1)].isoformat(),
        "missing_samples": int(end - start),
        "duration_hours": float((end - start) / 4),
    } for start, end in zip(starts, ends)]


def coverage(df: pd.DataFrame, label: str) -> dict:
    missing = df[list(VARIABLES)].isna()
    sensors_missing = missing[list(SENSORS)]
    return {
        "period": label,
        "first_utc": df["timestamp"].iloc[0].isoformat() if len(df) else "",
        "last_utc": df["timestamp"].iloc[-1].isoformat() if len(df) else "",
        "rows": int(len(df)),
        "rows_any_missing": int(missing.any(axis=1).sum()),
        "rows_any_missing_pct": float(100 * missing.any(axis=1).mean()) if len(df) else None,
        "rows_all_nine_sensors_missing": int(sensors_missing.all(axis=1).sum()),
        "missing_cells": int(missing.to_numpy().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW / "ETIFA_raw_data.csv", help="Original ETIFA_raw_data CSV (before initial trim)")
    parser.add_argument("--clean", type=Path, default=PROCESSED / "ETIFA_clean.csv", help="ETIFA_clean CSV (after trim and cleaning, before imputation)")
    parser.add_argument("--imputed", type=Path, default=PROCESSED / "ETIFA_clean_imputed_1h.csv", help="ETIFA_clean_imputed_1h CSV")
    parser.add_argument("--cleaning-audit", type=Path, default=PROCESSED / "ETIFA_cleaning_audit.csv", help="Optional ETIFA_cleaning_audit.csv")
    parser.add_argument("--imputation-summary", type=Path, default=PROCESSED / "ETIFA_imputation_summary.csv", help="Optional ETIFA_imputation_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "data_quality")
    args = parser.parse_args()
    outputs = ["missingness_by_variable.csv", "period_coverage.csv", "split_coverage.csv",
               "monthly_coverage.csv", "missing_runs.csv", "cleaning_rules.csv",
               "data_quality_manifest.json", "data_quality_report.md"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any((args.output_dir / name).exists() for name in outputs):
        parser.error("Output files already exist. Choose an empty --output-dir to avoid overwriting results.")

    raw, clean, imputed = load(args.raw, original=True), load(args.clean), load(args.imputed)
    if not clean["timestamp"].equals(imputed["timestamp"]):
        raise ValueError("Clean and imputed timestamps differ")
    if not clean["timestamp"].isin(raw["timestamp"]).all():
        raise ValueError("Clean timestamps are not a subset of the raw source")
    if not clean["timestamp"].equals(raw["timestamp"].iloc[-len(clean):].reset_index(drop=True)):
        raise ValueError("Initial temporal trim is not a contiguous prefix removal")
    trimmed_rows = len(raw) - len(clean)
    raw_retained = raw.iloc[trimmed_rows:].reset_index(drop=True)
    numeric_tolerance = dict(atol=1e-10, rtol=1e-12, equal_nan=True)

    rows = []
    for col in VARIABLES:
        a, b, c = (frame[col].to_numpy(dtype=float) for frame in (raw_retained, clean, imputed))
        originally_present = np.isfinite(a)
        cleaned_present = np.isfinite(b)
        final_present = np.isfinite(c)
        invalidated = originally_present & ~cleaned_present
        filled = ~cleaned_present & final_present
        if np.any(~originally_present & cleaned_present):
            raise ValueError(f"Cleaning inserted missing values in {col}, should only invalidate")
        if np.any(cleaned_present & ~final_present):
            raise ValueError(f"Imputation removed existing values in {col}")
        if not np.allclose(a[originally_present & cleaned_present], b[originally_present & cleaned_present], **numeric_tolerance):
            raise ValueError(f"Cleaning modified a value without invalidation in {col}")
        if not np.allclose(b[cleaned_present], c[cleaned_present], **numeric_tolerance):
            raise ValueError(f"Imputation modified a nonmissing value in {col}")
        rows.append({
            "variable": col,
            "type": "sensor" if col in SENSORS else "actuator_related",
            "original_missing": int((~originally_present).sum()),
            "original_missing_pct": round(100 * (~originally_present).mean(), 6),
            "invalidated": int(invalidated.sum()),
            "missing_after_cleaning": int((~cleaned_present).sum()),
            "reconstructed": int(filled.sum()),
            "final_missing": int((~final_present).sum()),
            "final_missing_pct": round(100 * (~final_present).mean(), 6),
        })
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "missingness_by_variable.csv", index=False)

    audit_rules = pd.DataFrame(columns=["rule", "invalidated_cells"])
    if args.cleaning_audit:
        events = pd.read_csv(args.cleaning_audit)
        if not {"timestamp", "variable", "rule"}.issubset(events.columns):
            raise ValueError("Cleaning audit lacks required columns")
        audit_rules = (events.groupby("rule", as_index=False).size()
                       .rename(columns={"size": "invalidated_cells"})
                       .sort_values("invalidated_cells", ascending=False))
        if int(audit_rules["invalidated_cells"].sum()) != int(table["invalidated"].sum()):
            raise ValueError("Cleaning audit total differs from raw vs clean invalidations")
    audit_rules.to_csv(args.output_dir / "cleaning_rules.csv", index=False)

    if args.imputation_summary:
        recorded = pd.read_csv(args.imputation_summary).set_index("variable")
        for row in rows:
            key = row["variable"]
            if key not in recorded.index:
                raise ValueError(f"Variable {key} missing in imputation summary")
            expected = [row["missing_after_cleaning"], row["reconstructed"], row["final_missing"]]
            actual = recorded.loc[key, ["missing_before", "filled", "missing_after"]].astype(int).tolist()
            if expected != actual:
                raise ValueError(f"Imputation summary mismatch for {key}: expected {expected}, got {actual}")

    mask_excluded = imputed["timestamp"].between(EXCLUDED_START, EXCLUDED_END)
    if mask_excluded.sum() == 0:
        raise ValueError("Expected excluded 2021 interval missing; check source")
    period_table = pd.DataFrame([
        coverage(imputed.loc[~mask_excluded], "retained_experimental_periods"),
        coverage(imputed.loc[mask_excluded], "excluded_2021_gap_period"),
        coverage(imputed, "all_curated_data"),
    ])
    period_table.to_csv(args.output_dir / "period_coverage.csv", index=False)

    splits = []
    for version, parts in SPLITS.items():
        seen = np.zeros(len(imputed), dtype=bool)
        for label, (start, end) in parts.items():
            selected = imputed["timestamp"].between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
            if np.any(seen & selected):
                raise ValueError(f"Overlapping {version} partition: {label}")
            seen |= selected
            splits.append({"version": version, "split": label,
                           **coverage(imputed.loc[selected], label)})
        if seen.sum() != (~mask_excluded).sum() or np.any(seen & mask_excluded):
            raise ValueError(f"{version} splits do not match the expected retained coverage")
    pd.DataFrame(splits).to_csv(args.output_dir / "split_coverage.csv", index=False)

    monthly = []
    for month, group in imputed.groupby(imputed["timestamp"].dt.strftime("%Y-%m"), sort=True):
        record = coverage(group, str(month))
        record["excluded_rows"] = int(group["timestamp"].between(EXCLUDED_START, EXCLUDED_END).sum())
        monthly.append(record)
    pd.DataFrame(monthly).to_csv(args.output_dir / "monthly_coverage.csv", index=False)

    missing_runs = []
    for col in VARIABLES:
        missing_runs += missing_run_rows(clean[col].isna().to_numpy(), clean["timestamp"], col, "before_imputation")
        missing_runs += missing_run_rows(imputed[col].isna().to_numpy(), imputed["timestamp"], col, "after_imputation")
    sensor_mask = imputed[list(SENSORS)].isna()
    missing_runs += missing_run_rows(sensor_mask.all(axis=1).to_numpy(), imputed["timestamp"], "all_9_sensors", "after_imputation")
    missing_runs += missing_run_rows(sensor_mask.any(axis=1).to_numpy(), imputed["timestamp"], "any_sensor", "after_imputation")
    pd.DataFrame(missing_runs).sort_values(["stage", "variable", "start_utc"]).to_csv(
        args.output_dir / "missing_runs.csv", index=False)

    total = table[["original_missing", "invalidated", "missing_after_cleaning", "reconstructed", "final_missing"]].sum().to_dict()
    summary = {
        "audit_script_sha256": sha256(Path(__file__)),
        "raw_rows": int(len(raw)), "initially_trimmed_rows": int(trimmed_rows),
        "curated_rows": int(len(clean)), "variables": int(len(VARIABLES)),
        "cell_counts": {key: int(value) for key, value in total.items()},
        "final_rows_with_any_missing": int(imputed[list(VARIABLES)].isna().any(axis=1).sum()),
        "final_fully_observed_rows": int(imputed[list(VARIABLES)].notna().all(axis=1).sum()),
        "retained_rows": int((~mask_excluded).sum()),
        "excluded_rows": int(mask_excluded.sum()),
        "all_inputs_sha256": {name: sha256(path) for name, path in (("raw", args.raw), ("clean", args.clean), ("imputed", args.imputed))},
        "optional_inputs_sha256": {key: sha256(path) for key, path in (
            ("cleaning_audit", args.cleaning_audit), ("imputation_summary", args.imputation_summary)) if path},
        "scope": "Dataset completeness and documented partitions only; not model-window validity.",
        "caveats": [
            "Original missingness is counted AFTER initial temporal trimming, not across all raw rows.",
            "Initial actuator empty entries reflect change-based logging; they need not be device failures.",
            "The excluded 2021 period is a partitioning decision, not a deletion of those rows from the curated CSV.",
            "No causal diagnosis of sensor/device failures or repair dates is inferred.",
            "Counts do not demonstrate that imputed target values were excluded by the training code.",
        ],
    }
    (args.output_dir / "data_quality_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    retained = period_table.loc[period_table["period"] == "retained_experimental_periods"].iloc[0]
    excluded = period_table.loc[period_table["period"] == "excluded_2021_gap_period"].iloc[0]
    lines = [
        "# ETIFA data quality — reproducible audit", "",
        f"- Input rows: **{len(raw):,}**", f"- Initial rows trimmed: **{trimmed_rows:,}**",
        f"- Curated rows: **{len(clean):,}**", f"- Invalidated cells: **{total['invalidated']:,}**",
        f"- Reconstructed cells: **{total['reconstructed']:,}**",
        f"- Final missing cells: **{total['final_missing']:,}**",
        f"- Final rows with missing values: **{summary['final_rows_with_any_missing']:,}**", "",
        "## Experimental-period coverage (after imputation)", "",
        f"- Retained: {int(retained['rows']):,} rows, {int(retained['rows_any_missing']):,} incomplete ({retained['rows_any_missing_pct']:.2f}%).",
        f"- Excluded: {int(excluded['rows']):,} rows, {int(excluded['rows_any_missing']):,} incomplete ({excluded['rows_any_missing_pct']:.2f}%).",
        f"- Missing cells within excluded block: {int(excluded['missing_cells']):,}.", "",
        "See `missingness_by_variable.csv` for Appendix Table A.1, `missing_runs.csv` for exact gap intervals,",
        "`monthly_coverage.csv` for temporal coverage, and `split_coverage.csv` for V1/V2/V3.", "",
        "**Limitations:** This does not establish the cause of the outages, actual repair dates,",
        "valid model-window counts, or the absence of prediction-time leakage.", "",
    ]
    (args.output_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:10]))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
