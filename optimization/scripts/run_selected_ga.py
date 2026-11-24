from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the simplified selected GA experiment: GA_ENGINEERING_AWARE + LSTM lb96 + horizon 12."
    )
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "selected_ga_config.json"))
    p.add_argument("--case-count", action="store_true", help="Print number of date/seed cases and exit.")
    p.add_argument("--case-index", type=int, default=None, help="Run only one date/seed case, zero based.")
    p.add_argument("--output-root", default=None, help="Root output folder. Defaults to out/<RUN_LABEL>.")
    p.add_argument("--model-path", default=None, help="Override GA_MODEL_PATH for the LSTM .h5 model.")
    p.add_argument("--dry-run", action="store_true", help="Print effective configuration and exit.")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_cases(config: dict) -> list[tuple[dict, int]]:
    dates = [dict(item) for item in config["SELECTED_DATES"]]
    seeds = [int(seed) for seed in config["SEED_VALUES"]]
    return [(date, seed) for date in dates for seed in seeds]


def main():
    args = parse_args()
    if args.model_path:
        os.environ["GA_MODEL_PATH"] = str(Path(args.model_path).expanduser().resolve())

    config_path = Path(args.config)
    config = load_json(config_path)
    cases = build_cases(config)

    if args.case_count:
        print(len(cases))
        return 0

    run_label = str(config.get("RUN_LABEL", "selected_ga_engineering_aware_lb96_h12"))
    if args.output_root:
        output_root = Path(args.output_root)
    else:
        output_root = Path(__file__).resolve().parents[1] / "out" / run_label

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise SystemExit(f"Invalid --case-index {args.case_index}; valid range: 0..{len(cases)-1}")
        selected_date, seed = cases[args.case_index]
        config["SELECTED_DATES"] = [selected_date]
        config["SEED_VALUES"] = [seed]
        config["OUTPUT_FOLDER"] = str(output_root / f"case_{args.case_index:04d}")
        config["EXPERIMENT_NAME"] = f"{run_label}_case_{args.case_index:04d}"
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        config["OUTPUT_FOLDER"] = str(output_root / f"single_process_{stamp}")
        config["EXPERIMENT_NAME"] = run_label

    # Fixed protocol metadata expected by run_genetic_algorithm_experiments.py
    config.setdefault("RUN_FULL_DATE_SET", False)
    config.setdefault("RUN_FULL_SEED_SET", False)
    config.setdefault("FITNESS_PENALTY_SCHEME", "target_first_soft_v1")
    config.setdefault("COMPARISON_GROUP", "selected_ga_only")
    config.setdefault("SCENARIO_GROUP", "lstm_dirty_lb96_h12")
    config.setdefault("OPTIMIZER_CONFIG_GROUP", "manuscript_min_mean_fitness_hyperparameters")
    config.setdefault("OBJECTIVE_GROUP", "engineering_aware_composite_objective")

    if config.get("GA_VARIANT") != "GA_ENGINEERING_AWARE":
        raise SystemExit("This simplified project only supports GA_ENGINEERING_AWARE.")
    if int(config.get("GENERATIONS_NUM")) != 50 or int(config.get("POPULATION_NUM")) != 200:
        print(
            "WARNING: this config does not use the manuscript default 50 generations and population 200.",
            file=sys.stderr,
        )

    Path(config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)
    effective_config_path = Path(config["OUTPUT_FOLDER"]) / "effective_config.json"
    effective_config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print(f"Cases in source config: {len(cases)}")
        print(f"Effective config saved to: {effective_config_path}")
        return 0

    # Import after GA_MODEL_PATH is set so constants.MODEL_PATH sees the override.
    from ..core.constants import HISTORY_STEPS, STEPS_TO_RUN, MODEL_PATH, DATA_CSV_PATH
    from ..core.run_genetic_algorithm_experiments import run_experiments

    print("=" * 80)
    print("Selected GA experiment")
    print("GA variant: GA_ENGINEERING_AWARE")
    print(f"Historical GA dataset: {Path(DATA_CSV_PATH).name}; real model lookback={HISTORY_STEPS}; GA control steps={STEPS_TO_RUN}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output folder: {config['OUTPUT_FOLDER']}")
    print("=" * 80)

    summary = run_experiments(config)
    summary_path = Path(config["OUTPUT_FOLDER"]) / "run_summary.json"
    safe_summary = deepcopy(summary)
    safe_summary.pop("results", None)
    summary_path.write_text(json.dumps(safe_summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
