from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate and rank selected-GA tuning grid outputs.")
    p.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "out" / "selected_ga_engineering_aware_lb96_h12_tuning_grid"))
    return p.parse_args()


def concat_csv(paths):
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df.insert(0, "source_file", str(path))
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def rank_by_min_mean_fitness(summary: pd.DataFrame, metric: str = "mean_step_best_fitness") -> pd.DataFrame:
    """Published criterion: lowest mean best fitness across all GA control steps.

    Historical analysis also examined a one-standard-error/time rule, but that
    alternative rule did NOT determine the manuscript's selected configuration.
    """
    ranked = summary.sort_values(
        [metric, "mean_total_ga_time_s", "config_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "rank_by_mean_fitness", np.arange(1, len(ranked) + 1))
    ranked["selected_for_manuscript"] = False
    if not ranked.empty:
        ranked.loc[0, "selected_for_manuscript"] = True
    return ranked


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    case_files = sorted(root.glob("*/case_*/case_summary.csv"))
    step_files = sorted(root.glob("*/case_*/steps_flat.csv"))
    config_files = sorted(root.glob("*/case_*/effective_config.json"))
    if not case_files:
        raise FileNotFoundError(f"No case_summary.csv files found under {root}/*/case_*")

    cases = concat_csv(case_files)
    steps = concat_csv(step_files)
    cases_path = root / "case_summary_all.csv"
    steps_path = root / "steps_flat_all.csv"
    cases.to_csv(cases_path, index=False)
    if not steps.empty:
        steps.to_csv(steps_path, index=False)

    config_rows = []
    for cfg_path in config_files:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        config_rows.append({
            "source_config": str(cfg_path),
            "config_id": cfg.get("TUNING_CONFIG_ID"),
            "population_num": cfg.get("POPULATION_NUM"),
            "generations_num": cfg.get("GENERATIONS_NUM"),
            "k_tournament": cfg.get("K_TOURNAMENT"),
            "mutation_profile": cfg.get("MUTATION_PROFILE_NAME"),
            "gaussian_sigma_fraction": cfg.get("GAUSSIAN_SIGMA_FRACTION"),
            "seed": (cfg.get("SEED_VALUES") or [None])[0],
            "start_date": (cfg.get("SELECTED_DATES") or [{}])[0].get("from"),
            "end_date": (cfg.get("SELECTED_DATES") or [{}])[0].get("to"),
        })
    config_df = pd.DataFrame(config_rows)
    config_df.to_csv(root / "effective_configs_all.csv", index=False)

    # Prefer per-step fitness because it reflects all 12 receding-horizon optimisations,
    # not only the final step of each 3 h episode.
    if steps.empty or "best_fitness_error_score" not in steps.columns:
        raise ValueError("steps_flat_all.csv must contain best_fitness_error_score")
    steps["config_id"] = steps["optimizer_config_group"]
    cases["config_id"] = cases["optimizer_config_group"]

    step_group = steps.groupby("config_id", as_index=False).agg(
        n_step_records=("best_fitness_error_score", "count"),
        mean_step_best_fitness=("best_fitness_error_score", "mean"),
        std_step_best_fitness=("best_fitness_error_score", "std"),
        median_step_best_fitness=("best_fitness_error_score", "median"),
        mean_step_engineering_violations=("engineering_rule_violations_step", "mean"),
        mean_step_ga_time_s=("ga_time_s", "mean"),
    )
    step_group["se_step_best_fitness"] = step_group["std_step_best_fitness"] / np.sqrt(step_group["n_step_records"].clip(lower=1))

    case_metrics = {
        "mae_tbs": "mean_mae_tbs",
        "mae_hr": "mean_mae_hr",
        "energy_proxy_total": "mean_energy_proxy_total",
        "total_actuator_change_normalized": "mean_total_actuator_change_normalized",
        "engineering_rule_violations": "mean_engineering_rule_violations",
        "total_ga_time_s": "mean_total_ga_time_s",
        "total_evaluations_count": "mean_total_evaluations_count",
    }
    available_case_metrics = {k: v for k, v in case_metrics.items() if k in cases.columns}
    case_group = cases.groupby("config_id", as_index=False).agg(
        n_cases=("config_id", "count"),
        **{new: (old, "mean") for old, new in available_case_metrics.items()}
    )

    meta_cols = [
        "config_id", "population_num", "generations_num", "k_tournament",
        "mutation_profile", "gaussian_sigma_fraction"
    ]
    meta = config_df[meta_cols].drop_duplicates("config_id")
    summary = meta.merge(step_group, on="config_id", how="left").merge(case_group, on="config_id", how="left")
    ranking = rank_by_min_mean_fitness(summary)

    summary_path = root / "tuning_summary_by_config.csv"
    ranking_path = root / "tuning_ranking.csv"
    summary.to_csv(summary_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    selected = ranking[ranking["selected_for_manuscript"]]
    selected_config_path = root / "selected_ga_tuning_winner.json"
    if not selected.empty:
        winner_id = selected.iloc[0]["config_id"]
        first_cfg = next(path for path in config_files if json.loads(path.read_text(encoding="utf-8")).get("TUNING_CONFIG_ID") == winner_id)
        winner_cfg = json.loads(first_cfg.read_text(encoding="utf-8"))
        # Remove single-case fields; keep only the winning hyperparameters and protocol metadata.
        for key in ["OUTPUT_FOLDER", "TUNING_CASE_INDEX", "TUNING_CONFIG_INDEX", "TUNING_DATE_INDEX", "TUNING_SEED_INDEX", "SELECTED_DATES", "SEED_VALUES"]:
            winner_cfg.pop(key, None)
        selected_config_path.write_text(json.dumps(winner_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Aggregated cases: {len(cases)} -> {cases_path}")
    print(f"Aggregated steps: {len(steps)} -> {steps_path}")
    print(f"Summary by config -> {summary_path}")
    print(f"Ranking -> {ranking_path}")
    if not selected.empty:
        print("Winner by minimum mean step best fitness:")
        cols = ["config_id", "mean_step_best_fitness", "se_step_best_fitness", "mean_total_ga_time_s"]
        print(selected[cols].to_string(index=False))
        print(f"Winner config -> {selected_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
