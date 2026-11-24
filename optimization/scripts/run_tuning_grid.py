from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run selected-GA tuning grid: GA_ENGINEERING_AWARE + LSTM lb96 + horizon 12."
    )
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "tuning_grid_config.json"))
    p.add_argument("--config-count", action="store_true", help="Print number of grid configurations and exit.")
    p.add_argument("--case-count", action="store_true", help="Print number of grid/date/seed cases and exit.")
    p.add_argument("--case-index", type=int, default=None, help="Run one grid/date/seed case, zero based.")
    p.add_argument("--output-root", default=None, help="Root output folder. Defaults to out/<RUN_LABEL>.")
    p.add_argument("--model-path", default=None, help="Override GA_MODEL_PATH for the LSTM .h5 model.")
    p.add_argument("--dry-run", action="store_true", help="Print grid summary and exit.")
    return p.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assert_mutation_profile(profile: dict, name: str) -> None:
    expected_keys = {"no_mutation", "mutation_1", "mutation_2", "mutation_3"}
    if set(profile) != expected_keys:
        raise ValueError(f"Mutation profile {name!r} must have keys {sorted(expected_keys)}")
    total = sum(float(v) for v in profile.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Mutation profile {name!r} sums to {total}, expected 1.0")


def build_grid_configs(config: dict) -> list[dict]:
    grid = config["GRID"]
    mutation_profiles = grid["MUTATION_PROFILES"]
    for name, profile in mutation_profiles.items():
        assert_mutation_profile(profile, name)

    configs = []
    for budget, k, mutation_name, sigma in product(
        grid["BUDGET_SPLITS"],
        grid["K_TOURNAMENT"],
        mutation_profiles.keys(),
        grid["GAUSSIAN_SIGMA_FRACTION"],
    ):
        cfg_id = (
            f"{budget['label']}_k{k}_mut{mutation_name}_sigma{str(sigma).replace('.', 'p')}"
        )
        generated = deepcopy(config["BASE_CONFIG"])
        generated.update({
            "TUNING_CONFIG_ID": cfg_id,
            "EXPERIMENT_NAME": cfg_id,
            "GENERATIONS_NUM": int(budget["generations"]),
            "POPULATION_NUM": int(budget["population"]),
            "K_TOURNAMENT": int(k),
            "MUTATION_PROFILE_NAME": str(mutation_name),
            "MUTATION_PROBABILITIES": deepcopy(mutation_profiles[mutation_name]),
            "GAUSSIAN_SIGMA_FRACTION": float(sigma),
            "FITNESS_PENALTY_SCHEME": "target_first_soft_v1",
            "COMPARISON_GROUP": "selected_ga_tuning_grid",
            "SCENARIO_GROUP": "lstm_dirty_lb96_h12",
            "OPTIMIZER_CONFIG_GROUP": cfg_id,
            "OBJECTIVE_GROUP": "engineering_aware_composite_objective",
            "EXPERIMENT_PHASE": config.get("EXPERIMENT_PHASE"),
            "DATE_SPLIT": config.get("DATE_SPLIT"),
            "CLAIM_SCOPE": config.get("CLAIM_SCOPE"),
        })
        if generated["GA_VARIANT"] != "GA_ENGINEERING_AWARE":
            raise ValueError("This tuning project only supports GA_ENGINEERING_AWARE")
        if generated["GENERATIONS_NUM"] * generated["POPULATION_NUM"] != 10000:
            raise ValueError(f"Budget for {cfg_id} is not 10,000 evaluations")
        configs.append(generated)
    return configs


def build_cases(config: dict) -> list[tuple[int, dict, int, int, int]]:
    grid_configs = build_grid_configs(config)
    dates = [dict(item) for item in config["SELECTED_DATES"]]
    seeds = [int(seed) for seed in config["SEED_VALUES"]]
    cases = []
    for cfg_index, grid_config in enumerate(grid_configs):
        for date_index, selected_date in enumerate(dates):
            for seed_index, seed in enumerate(seeds):
                cases.append((cfg_index, grid_config, date_index, selected_date, seed_index, seed))
    return cases


def main() -> int:
    args = parse_args()
    if args.model_path:
        os.environ["GA_MODEL_PATH"] = str(Path(args.model_path).expanduser().resolve())

    config_path = Path(args.config)
    config = load_json(config_path)
    grid_configs = build_grid_configs(config)
    cases = build_cases(config)

    if args.config_count:
        print(len(grid_configs))
        return 0
    if args.case_count:
        print(len(cases))
        return 0

    run_label = str(config.get("RUN_LABEL", "selected_ga_engineering_aware_lb96_h12_tuning_grid"))
    output_root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parents[1] / "out" / run_label

    if args.dry_run:
        print("Selected GA tuning grid")
        print(f"Config file: {config_path}")
        print(f"Grid configurations: {len(grid_configs)}")
        print(f"Dates: {len(config['SELECTED_DATES'])}")
        print(f"Seeds: {len(config['SEED_VALUES'])}")
        print(f"Total cases: {len(cases)}")
        print("First configurations:")
        for item in grid_configs[:5]:
            print(" -", item["TUNING_CONFIG_ID"])
        return 0

    if args.case_index is None:
        raise SystemExit("Use --case-index for array execution, or --dry-run/--case-count for inspection.")
    if args.case_index < 0 or args.case_index >= len(cases):
        raise SystemExit(f"Invalid --case-index {args.case_index}; valid range: 0..{len(cases)-1}")

    cfg_index, effective_config, date_index, selected_date, seed_index, seed = cases[args.case_index]
    cfg_id = effective_config["TUNING_CONFIG_ID"]
    case_label = f"case_{args.case_index:04d}_date{date_index:02d}_seed{seed}"

    effective_config = deepcopy(effective_config)
    effective_config["SELECTED_DATES"] = [selected_date]
    effective_config["SEED_VALUES"] = [seed]
    effective_config["EXPERIMENT_NAME"] = f"{run_label}_{cfg_id}_{case_label}"
    effective_config["OUTPUT_FOLDER"] = str(output_root / cfg_id / case_label)
    effective_config["TUNING_CASE_INDEX"] = int(args.case_index)
    effective_config["TUNING_CONFIG_INDEX"] = int(cfg_index)
    effective_config["TUNING_DATE_INDEX"] = int(date_index)
    effective_config["TUNING_SEED_INDEX"] = int(seed_index)

    Path(effective_config["OUTPUT_FOLDER"]).mkdir(parents=True, exist_ok=True)
    effective_config_path = Path(effective_config["OUTPUT_FOLDER"]) / "effective_config.json"
    effective_config_path.write_text(json.dumps(effective_config, indent=2, ensure_ascii=False), encoding="utf-8")

    from ..core.constants import HISTORY_STEPS, STEPS_TO_RUN, MODEL_PATH, DATA_CSV_PATH
    from ..core.run_genetic_algorithm_experiments import run_experiments

    print("=" * 80)
    print("Selected GA tuning case")
    print(f"Case index: {args.case_index}/{len(cases)-1}")
    print(f"Tuning config: {cfg_id}")
    print(f"GA variant: {effective_config['GA_VARIANT']}")
    print(f"Population x generations: {effective_config['POPULATION_NUM']} x {effective_config['GENERATIONS_NUM']}")
    print(f"Tournament k: {effective_config['K_TOURNAMENT']}")
    print(f"Mutation profile: {effective_config['MUTATION_PROFILE_NAME']}")
    print(f"Gaussian sigma fraction: {effective_config['GAUSSIAN_SIGMA_FRACTION']}")
    print(f"Historical GA dataset: {Path(DATA_CSV_PATH).name}; real model lookback={HISTORY_STEPS}; GA control steps={STEPS_TO_RUN}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output folder: {effective_config['OUTPUT_FOLDER']}")
    print("=" * 80)

    summary = run_experiments(effective_config)
    safe_summary = deepcopy(summary)
    safe_summary.pop("results", None)
    summary_path = Path(effective_config["OUTPUT_FOLDER"]) / "run_summary.json"
    summary_path.write_text(json.dumps(safe_summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved summary to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
