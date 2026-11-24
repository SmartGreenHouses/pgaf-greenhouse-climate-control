import json
import os
import random
import time
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

from .genetic_algorithm_class import (
    algorithm_family,
    count_engineering_rule_violations,
    count_target_progress_rule_violations,
    objective_variant,
    run_genetic_algorithm,
    uses_actuator_effort_penalty,
    uses_hold_last_action,
)
from .utils import add_month_col, encode_cyclical_value, is_it_day_from_date
from .constants import (
    ALL_SEED_VALUES,
    ALL_SELECTED_DATES,
    DATA_CSV_PATH,
    DESIRED_HR_DAY,
    DESIRED_HR_NIGHT,
    DESIRED_TBS_DAY,
    DESIRED_TBS_NIGHT,
    FITNESS_PENALTY_SCHEME,
    LATITUDE,
    LOCAL_TIMEZONE,
    LONGITUDE,
    MONTH_COS_COL,
    MONTH_SIN_COL,
    MSG_MODE,
    MULTIVARIATE_VARIABLES,
    OUTPUT_FOLDER,
    PIPE_ENERGY_EXCESS_WEIGHT,
    PIPE_ENERGY_LEVEL_WEIGHT,
    POSSIBLE_GENES,
    TIMESTAMP_COL,
    STEPS_TO_RUN,
    HISTORY_STEPS
)



REQUIRED_CONFIG_KEYS = {
    "RUN_FULL_DATE_SET", "RUN_FULL_SEED_SET", "EXPERIMENT_NAME",
    "GA_VARIANT", "FITNESS_PERCENTAGE_GOAL", "GENERATIONS_NUM",
    "POPULATION_NUM", "K_TOURNAMENT", "PATIENCE", "STOP_GA_THR",
    "STOP_GA_FROM", "EARLY_STOPPING_ENABLED", "MUTATION_PROBABILITIES",
    "GAUSSIAN_SIGMA_FRACTION", "ENERGY_PROXY_WEIGHTS",
    "ALLOWED_DEVIATION_TBS", "ALLOWED_DEVIATION_HR",
}


def read_csv(dataset_csv_path=None, output_folder=None, fields=None):
    dataset_csv_path = DATA_CSV_PATH if dataset_csv_path is None else dataset_csv_path
    output_folder = OUTPUT_FOLDER if output_folder is None else output_folder
    fields = MULTIVARIATE_VARIABLES if fields is None else list(fields)

    if not os.path.isfile(dataset_csv_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_csv_path}")

    data = pd.read_csv(
        dataset_csv_path,
        sep=",",
        index_col=TIMESTAMP_COL,
        parse_dates=[TIMESTAMP_COL],
        usecols=fields + [TIMESTAMP_COL],
    )
    data = data.sort_index()

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Dataset index must be a pandas.DatetimeIndex")
    if data.index.has_duplicates:
        duplicates = data.index[data.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"Dataset contains duplicate timestamps, e.g. {duplicates}")
    if data.empty:
        raise ValueError("Dataset is empty")
    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Dataset contains missing values in columns: {missing}")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains infinite values")

    if MSG_MODE:
        print(f"Dataset read: {dataset_csv_path}\n")
        print(data.head(2))

    os.makedirs(output_folder, exist_ok=True)
    return data


def sanitize_label(value):
    return "".join(character if character.isalnum() else "_" for character in str(value))


def get_selected_dates(run_full_date_set):
    return ALL_SELECTED_DATES if run_full_date_set else [ALL_SELECTED_DATES[0]]


def get_seed_values(run_full_seed_set):
    return ALL_SEED_VALUES if run_full_seed_set else [ALL_SEED_VALUES[0]]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # Older TensorFlow builds may not expose this switch; the manifest
            # records the exact version so this remains auditable.
            pass
    except ImportError:
        pass


def get_exact_history_window(data, selected_date, expected_rows=HISTORY_STEPS):
    raw_start = pd.Timestamp(selected_date["from"])
    raw_end = pd.Timestamp(selected_date["to"])
    minimum_span = timedelta(minutes=15 * (expected_rows - 1))
    if raw_end < raw_start:
        raise ValueError(
            f"Selected date range has end before start: {raw_start} -> {raw_end}"
        )
    if raw_end - raw_start < minimum_span:
        raise ValueError(
            f"Selected date range is too short for {expected_rows} history rows: "
            f"{raw_start} -> {raw_end}; required at least {minimum_span}"
        )

    index_timezone = data.index.tz
    if index_timezone is not None:
        start = (
            raw_start.tz_localize(index_timezone)
            if raw_start.tzinfo is None
            else raw_start.tz_convert(index_timezone)
        )
        end = (
            raw_end.tz_localize(index_timezone)
            if raw_end.tzinfo is None
            else raw_end.tz_convert(index_timezone)
        )
    else:
        start = raw_start.tz_localize(None) if raw_start.tzinfo is not None else raw_start
        end = raw_end.tz_localize(None) if raw_end.tzinfo is not None else raw_end

    expected_index = pd.date_range(end=end, periods=expected_rows, freq="15min")
    missing_timestamps = expected_index.difference(data.index)
    if not missing_timestamps.empty:
        preview = [str(value) for value in missing_timestamps[:5]]
        raise ValueError(
            f"Selected history is missing {len(missing_timestamps)} timestamps; first: {preview}"
        )

    case_data = data.loc[expected_index].copy()
    if len(case_data) != expected_rows:
        raise ValueError(
            f"Invalid history length: got {len(case_data)}, expected {expected_rows}"
        )
    return case_data


def get_setpoints(timestamp):
    if is_it_day_from_date(timestamp, lat=LATITUDE, lon=LONGITUDE, timezone=LOCAL_TIMEZONE):
        return DESIRED_TBS_DAY, DESIRED_HR_DAY

    return DESIRED_TBS_NIGHT, DESIRED_HR_NIGHT


def build_next_row(ga, next_timestamp):
    new_row_dict = ga.fittest_individual.solution["all_predictions"].copy()

    for key, value in zip(POSSIBLE_GENES, ga.fittest_individual.genotype):
        new_row_dict[key] = value

    month_sin, month_cos = encode_cyclical_value(next_timestamp.month, 12)
    new_row_dict[MONTH_SIN_COL] = float(month_sin)
    new_row_dict[MONTH_COS_COL] = float(month_cos)

    return pd.DataFrame(new_row_dict, index=[next_timestamp])


def compute_error_metrics(step_records):
    if not step_records:
        return {
            "MAE_TBS": 0.0,
            "MAE_HR": 0.0,
            "RMSE_TBS": 0.0,
            "RMSE_HR": 0.0,
            "P95_abs_error_TBS": 0.0,
            "P95_abs_error_HR": 0.0,
            "IAE_TBS": 0.0,
            "IAE_HR": 0.0,
            "IAE_TBS_per_step": 0.0,
            "IAE_HR_per_step": 0.0,
        }

    error_tbs = np.array([record["error_tbs"] for record in step_records], dtype=float)
    error_hr = np.array([record["error_hr"] for record in step_records], dtype=float)
    abs_error_tbs = np.abs(error_tbs)
    abs_error_hr = np.abs(error_hr)
    steps_count = len(step_records)
    iae_tbs = float(np.sum(abs_error_tbs))
    iae_hr = float(np.sum(abs_error_hr))

    return {
        "MAE_TBS": float(np.mean(abs_error_tbs)),
        "MAE_HR": float(np.mean(abs_error_hr)),
        "RMSE_TBS": float(np.sqrt(np.mean(np.square(error_tbs)))),
        "RMSE_HR": float(np.sqrt(np.mean(np.square(error_hr)))),
        "P95_abs_error_TBS": float(np.percentile(abs_error_tbs, 95)),
        "P95_abs_error_HR": float(np.percentile(abs_error_hr, 95)),
        "IAE_TBS": iae_tbs,
        "IAE_HR": iae_hr,
        "IAE_TBS_per_step": float(iae_tbs / steps_count),
        "IAE_HR_per_step": float(iae_hr / steps_count),
    }


def compute_actuator_change_metrics(step_records):
    if not step_records:
        return {
            "total_actuator_change": 0.0,
            "delta_u_squared": 0.0,
            "total_actuator_change_normalized": 0.0,
            "delta_u_squared_normalized": 0.0,
        }

    return {
        "total_actuator_change": float(
            np.sum([record["actuator_abs_change"] for record in step_records])
        ),
        "delta_u_squared": float(
            np.sum([record["actuator_delta_u_squared"] for record in step_records])
        ),
        "total_actuator_change_normalized": float(
            np.sum([record["actuator_abs_change_normalized"] for record in step_records])
        ),
        "delta_u_squared_normalized": float(
            np.sum([record["actuator_delta_u_squared_normalized"] for record in step_records])
        ),
    }


def compute_energy_metrics(step_records):
    if not step_records:
        return {
            "energy_proxy_total": 0.0,
            "pipe_energy_proxy": 0.0,
            "humidification_usage": 0.0,
        }

    return {
        "energy_proxy_total": float(np.sum([record["energy_proxy_step"] for record in step_records])),
        "pipe_energy_proxy": float(np.sum([record["pipe_energy_proxy_step"] for record in step_records])),
        "humidification_usage": float(np.sum([record["humidification_usage_step"] for record in step_records])),
    }


def compute_climate_violation_metrics(step_records):
    if not step_records:
        return {
            "climate_violation_tbs": 0.0,
            "climate_violation_hr": 0.0,
        }

    return {
        "climate_violation_tbs": float(np.sum([record["climate_violation_tbs_step"] for record in step_records])),
        "climate_violation_hr": float(np.sum([record["climate_violation_hr_step"] for record in step_records])),
    }


def compute_engineering_rule_metrics(step_records):
    if not step_records:
        return {
            "engineering_rule_violations": 0,
            "target_progress_rule_violations": 0,
        }

    return {
        "engineering_rule_violations": int(
            np.sum([record["engineering_rule_violations_step"] for record in step_records])
        ),
        "target_progress_rule_violations": int(
            np.sum([record["target_progress_rule_violations_step"] for record in step_records])
        ),
    }


def compute_ga_summary(step_records):
    if not step_records:
        return {
            "ga_executions": 0,
            "stop_reason_counts": {},
            "mean_completed_generations": 0.0,
            "final_completed_generations": None,
            "final_stop_reason": None,
            "final_best_fitness_history": [],
        }

    completed_generations = np.array(
        [record["completed_generations"] for record in step_records],
        dtype=float,
    )
    stop_reason_counts = Counter(record["stop_reason"] for record in step_records)
    last_record = step_records[-1]

    return {
        "ga_executions": len(step_records),
        "stop_reason_counts": dict(stop_reason_counts),
        "mean_completed_generations": float(np.mean(completed_generations)),
        "final_completed_generations": int(last_record["completed_generations"]),
        "final_stop_reason": last_record["stop_reason"],
        "final_best_fitness_history": last_record["best_fitness_history"],
    }


def relative_convergence_generation(best_fitness_history, fraction=0.99):
    """Return the first generation reaching ``fraction`` of the observed gain.

    This post-hoc definition remains meaningful for fixed-budget runs and does
    not compare absolute fitness thresholds across variants with differently
    scaled penalty terms.
    """
    history = np.asarray(best_fitness_history, dtype=float)
    if history.size == 0:
        return None
    if not np.isfinite(history).all():
        raise ValueError("Best-fitness history contains non-finite values")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("Convergence fraction must be in (0, 1]")

    initial = float(history[0])
    final = float(history[-1])
    observed_gain = initial - final
    if observed_gain <= 0.0:
        return 1

    threshold = initial - float(fraction) * observed_gain
    reached = np.flatnonzero(history <= threshold)
    return int(reached[0] + 1) if reached.size else int(history.size)


def compute_convergence_diagnostics(ga, stop_ga_from, fraction=0.99):
    """Return convergence diagnostics without inventing them for fixed policies.

    Optimisers are compared by both iteration and cumulative fitness evaluations.
    HOLD_LAST_ACTION is a non-iterative policy baseline, so convergence fields are
    deliberately ``None`` rather than a misleading value of one.
    """
    if uses_hold_last_action(ga.variant_name):
        return {
            "target_reached_generation": None,
            "target_reached_evaluation": None,
            "convergence_fraction": None,
            "generations_to_converge": None,
            "evaluations_to_converge": None,
        }

    history = list(ga.best_fitness_history)
    evaluation_history = list(ga.best_fitness_evaluations)
    if len(history) != len(evaluation_history):
        raise ValueError(
            "Fitness history and cumulative-evaluation history must have equal length"
        )

    target_index = None
    if stop_ga_from is not None:
        target_index = next(
            (index for index, value in enumerate(history) if value <= float(stop_ga_from)),
            None,
        )

    convergence_generation = relative_convergence_generation(history, fraction=fraction)
    convergence_evaluation = (
        evaluation_history[convergence_generation - 1]
        if convergence_generation is not None
        else None
    )
    return {
        "target_reached_generation": target_index + 1 if target_index is not None else None,
        "target_reached_evaluation": (
            evaluation_history[target_index] if target_index is not None else None
        ),
        "convergence_fraction": float(fraction),
        "generations_to_converge": convergence_generation,
        "evaluations_to_converge": convergence_evaluation,
    }


def compute_ga_runtime_metrics(step_records):
    if not step_records:
        return {
            "total_ga_time_s": 0.0,
            "mean_ga_time_s": 0.0,
            "p95_ga_time_s": 0.0,
            "max_ga_time_s": 0.0,
            "real_time_step_budget_s": 15 * 60,
            "p95_ga_time_within_step_budget": True,
            "max_ga_time_within_step_budget": True,
            "total_evaluations_count": 0,
            "mean_time_per_evaluation": None,
            "mean_generations_to_converge": None,
            "final_generations_to_converge": None,
            "mean_evaluations_to_converge": None,
            "final_evaluations_to_converge": None,
        }

    ga_times = np.array([record.get("ga_time_s", 0.0) for record in step_records], dtype=float)
    total_ga_time_s = float(np.sum(ga_times))
    mean_ga_time_s = float(np.mean(ga_times))
    p95_ga_time_s = float(np.percentile(ga_times, 95))
    max_ga_time_s = float(np.max(ga_times))
    real_time_step_budget_s = 15 * 60
    total_evaluations_count = int(
        np.sum([record.get("evaluations_count", 0) for record in step_records])
    )
    mean_time_per_evaluation = (
        float(total_ga_time_s / total_evaluations_count)
        if total_evaluations_count > 0
        else None
    )

    generations = [
        record["generations_to_converge"]
        for record in step_records
        if record.get("generations_to_converge") is not None
    ]
    evaluations = [
        record["evaluations_to_converge"]
        for record in step_records
        if record.get("evaluations_to_converge") is not None
    ]

    return {
        "total_ga_time_s": total_ga_time_s,
        "mean_ga_time_s": mean_ga_time_s,
        "p95_ga_time_s": p95_ga_time_s,
        "max_ga_time_s": max_ga_time_s,
        "real_time_step_budget_s": real_time_step_budget_s,
        "p95_ga_time_within_step_budget": p95_ga_time_s <= real_time_step_budget_s,
        "max_ga_time_within_step_budget": max_ga_time_s <= real_time_step_budget_s,
        "total_evaluations_count": total_evaluations_count,
        "mean_time_per_evaluation": mean_time_per_evaluation,
        "mean_generations_to_converge": float(np.mean(generations)) if generations else None,
        "final_generations_to_converge": step_records[-1].get("generations_to_converge"),
        "mean_evaluations_to_converge": float(np.mean(evaluations)) if evaluations else None,
        "final_evaluations_to_converge": step_records[-1].get("evaluations_to_converge"),
    }


def run_step(
    date_range,
    step_time,
    ga_variant,
    generations_num,
    population_num,
    k_tournament,
    case_data,
    fitness_percentage_goal,
    patience,
    stop_ga_thr,
    stop_ga_from,
    mutation_probabilities,
    current_timestamp,
    energy_proxy_weights,
    allowed_deviation_tbs,
    allowed_deviation_hr,
    gaussian_sigma_fraction,
    early_stopping_enabled,
):
    step_timestamp = pd.to_datetime(date_range["to"]) + timedelta(minutes=15)
    step_date = step_timestamp.isoformat()
    print(f"            - Step {step_time}: {step_date}")

    last_row = case_data.tail(1).to_dict(orient="records")[0]
    desired_tbs_output, desired_hr_output = get_setpoints(step_timestamp)
    previous_genes = [float(last_row[gene_key]) for gene_key in POSSIBLE_GENES]

    start_ga = time.time()
    ga, _, _ = run_genetic_algorithm(
        generations_num,
        population_num,
        k_tournament,
        last_row,
        case_data,
        last_row["CajaMedida1.TBS"],
        last_row["CajaMedida1.HR"],
        desired_tbs_output,
        desired_hr_output,
        fitness_percentage_goal,
        patience,
        stop_ga_thr,
        stop_ga_from,
        mutation_probabilities,
        variant_name=ga_variant,
        energy_proxy_weights=energy_proxy_weights,
        gaussian_sigma_fraction=gaussian_sigma_fraction,
        early_stopping_enabled=early_stopping_enabled,
    )
    end_ga = time.time()
    if MSG_MODE:
        print(f"                Tiempo de ejecucion del GA: {end_ga - start_ga:.2f} segundos")
    ga_time_s = float(end_ga - start_ga)

    next_timestamp = current_timestamp + timedelta(minutes=15)
    if pd.Timestamp(next_timestamp) != pd.Timestamp(step_timestamp):
        raise ValueError(
            f"Timestamp drift detected: history expects {next_timestamp}, date range expects {step_timestamp}"
        )
    new_row = build_next_row(ga, next_timestamp)
    case_data = pd.concat([case_data, new_row]).iloc[1:]

    next_from = pd.to_datetime(date_range["from"]) + timedelta(minutes=15)
    next_to = pd.to_datetime(date_range["to"]) + timedelta(minutes=15)
    next_date_range = {"from": next_from, "to": next_to}

    predicted_tbs = float(ga.fittest_individual.solution["tbs"])
    predicted_hr = float(ga.fittest_individual.solution["hr"])
    error_tbs = predicted_tbs - float(desired_tbs_output)
    error_hr = predicted_hr - float(desired_hr_output)
    abs_error_tbs = abs(error_tbs)
    abs_error_hr = abs(error_hr)
    current_genes = [float(gene_value) for gene_value in ga.fittest_individual.genotype]
    gene_deltas = np.array(current_genes, dtype=float) - np.array(previous_genes, dtype=float)
    actuator_abs_change = float(np.sum(np.abs(gene_deltas)))
    actuator_delta_u_squared = float(np.sum(np.square(gene_deltas)))
    gene_ranges = np.array([
        max(
            float(gene_config["max"]) - float(gene_config["min"]),
            1e-9,
        )
        for gene_config in POSSIBLE_GENES.values()
    ])
    normalized_gene_deltas = gene_deltas / gene_ranges
    actuator_abs_change_normalized = float(np.sum(np.abs(normalized_gene_deltas)))
    actuator_delta_u_squared_normalized = float(np.sum(np.square(normalized_gene_deltas)))

    current_genes_map = {
        gene_key: float(gene_value)
        for gene_key, gene_value in zip(POSSIBLE_GENES, ga.fittest_individual.genotype)
    }
    best_individual_trace = {
        f"gene_{gene_key}": gene_value
        for gene_key, gene_value in current_genes_map.items()
    }

    pipe_value = current_genes_map["TemperaturaTuberia"]
    pipe_min = float(POSSIBLE_GENES["TemperaturaTuberia"]["min"])
    pipe_max = float(POSSIBLE_GENES["TemperaturaTuberia"]["max"])
    pipe_range = max(pipe_max - pipe_min, 1e-9)
    pipe_norm = np.clip((pipe_value - pipe_min) / pipe_range, 0.0, 1.0)

    tbs_reference = float(last_row["CajaMedida1.TBS"])
    pipe_excess = max(0.0, pipe_value - tbs_reference)
    pipe_excess_norm = np.clip(pipe_excess / pipe_range, 0.0, 1.0)

    # Proxy robusto: combina nivel absoluto de tuberia y exceso sobre TBS.
    pipe_energy_proxy_step = float(
        PIPE_ENERGY_LEVEL_WEIGHT * pipe_norm
        + PIPE_ENERGY_EXCESS_WEIGHT * pipe_excess_norm
    )

    humidification_usage_step = float(current_genes_map["Humidificacion"])

    openings_norm_values = [
        float(current_genes_map["PosicionVentana1"]) / 100.0,
        float(current_genes_map["PosicionVentana2"]) / 100.0,
        float(current_genes_map["PosicionPantalla1"]) / 100.0,
        float(current_genes_map["PosicionPantalla2"]) / 100.0,
    ]
    openings_proxy_step = float(np.mean(openings_norm_values))

    energy_proxy_step = float(
        energy_proxy_weights["pipe"] * pipe_energy_proxy_step
        + energy_proxy_weights["humidification"] * humidification_usage_step
        + energy_proxy_weights["openings"] * openings_proxy_step
    )

    climate_violation_tbs_step = float(max(0.0, abs(error_tbs) - allowed_deviation_tbs))
    climate_violation_hr_step = float(max(0.0, abs(error_hr) - allowed_deviation_hr))

    engineering_rule_violations_step = count_engineering_rule_violations(
        ga.fittest_individual.genotype,
        ga.fittest_individual.solution,
        last_row,
        desired_tbs_output=desired_tbs_output,
        desired_hr_output=desired_hr_output,
    )
    target_progress_rule_violations_step = count_target_progress_rule_violations(
        ga.fittest_individual.solution,
        last_row,
        desired_tbs_output,
        desired_hr_output,
    )

    convergence = compute_convergence_diagnostics(
        ga,
        stop_ga_from=stop_ga_from,
        fraction=0.99,
    )
    planned_evaluation_budget = int(generations_num) * int(population_num)

    step_metrics = {
        "step": step_time,
        "timestamp": step_date,
        "ga_variant": ga_variant,
        "method_name": ga_variant,
        "algorithm_family": algorithm_family(ga_variant),
        "objective_variant": objective_variant(ga_variant),
        "actuator_effort_penalty": uses_actuator_effort_penalty(ga_variant),
        "planned_evaluation_budget": planned_evaluation_budget,
        "predicted_tbs": predicted_tbs,
        "desired_tbs": float(desired_tbs_output),
        "error_tbs": error_tbs,
        "abs_error_tbs": abs_error_tbs,
        "predicted_hr": predicted_hr,
        "desired_hr": float(desired_hr_output),
        "error_hr": error_hr,
        "abs_error_hr": abs_error_hr,
        "actuator_abs_change": actuator_abs_change,
        "actuator_delta_u_squared": actuator_delta_u_squared,
        "actuator_abs_change_normalized": actuator_abs_change_normalized,
        "actuator_delta_u_squared_normalized": actuator_delta_u_squared_normalized,
        "gaussian_sigma_fraction": gaussian_sigma_fraction,
        "early_stopping_enabled": early_stopping_enabled,
        "ga_time_s": ga_time_s,
        "pipe_energy_proxy_step": pipe_energy_proxy_step,
        "humidification_usage_step": humidification_usage_step,
        "openings_proxy_step": openings_proxy_step,
        "energy_proxy_step": energy_proxy_step,
        "climate_violation_tbs_step": climate_violation_tbs_step,
        "climate_violation_hr_step": climate_violation_hr_step,
        "engineering_rule_violations_step": engineering_rule_violations_step,
        "target_progress_rule_violations_step": target_progress_rule_violations_step,
        "ga_engineering_rule_violations_total": ga.engineering_rule_violations_total,
        "completed_generations": ga.completed_generations,
        "stop_reason": ga.stop_reason,
        "best_fitness_error_score": float(ga.fittest_individual.fitness_error_score),
        "best_fitness_history": [float(value) for value in ga.best_fitness_history],
        "best_fitness_evaluations": [int(value) for value in ga.best_fitness_evaluations],
        **best_individual_trace,
        "evaluations_count": ga.evaluations_count,
        "time_per_evaluation": float(ga_time_s / ga.evaluations_count) if ga.evaluations_count > 0 else None,
        **convergence,
    }

    return case_data, next_date_range, next_timestamp, ga, step_metrics


def run_case(data, 
             selected_date, 
             seed,
             experiment_name,
             ga_variant,
             fitness_percentage_goal,
             generations_num,
             population_num,
             k_tournament,
             patience,
             stop_ga_thr,
             stop_ga_from,
             mutation_probabilities,
             energy_proxy_weights,
             allowed_deviation_tbs,
             allowed_deviation_hr,
             gaussian_sigma_fraction,
             early_stopping_enabled,
            ):
    
    case_id = f"{sanitize_label(selected_date['from'])}-{sanitize_label(selected_date['to'])}"
    case_prefix = f"seed_{seed}_{case_id}"

    print(f"\n        Rango de fechas: {selected_date['from']} - {selected_date['to']}")

    case_data = get_exact_history_window(data, selected_date)
    case_data = add_month_col(case_data)
    current_timestamp = case_data.index[-1]
    date_range = {
        "from": current_timestamp - timedelta(days=7),
        "to": current_timestamp,
    }

    start_selected_date = time.time()
    last_ga = None
    case_step_records = []

    for step_time in range(STEPS_TO_RUN):
        # Re-seed each control step so algorithms that consume different
        # numbers of random draws cannot perturb the starting stream of later
        # paired steps. The same case seed/step gives every algorithm the same
        # initial random population.
        optimizer_seed = int(seed) * 10_000 + step_time
        seed_everything(optimizer_seed)
        case_data, date_range, current_timestamp, last_ga, step_metrics = run_step(
            date_range=date_range,
            step_time=step_time,
            ga_variant=ga_variant,
            generations_num=generations_num,
            population_num=population_num,
            k_tournament=k_tournament,
            case_data=case_data,
            fitness_percentage_goal=fitness_percentage_goal,
            patience=patience,
            stop_ga_thr=stop_ga_thr,
            stop_ga_from=stop_ga_from,
            mutation_probabilities=mutation_probabilities,
            current_timestamp=current_timestamp,
            energy_proxy_weights=energy_proxy_weights,
            allowed_deviation_tbs=allowed_deviation_tbs,
            allowed_deviation_hr=allowed_deviation_hr,
            gaussian_sigma_fraction=gaussian_sigma_fraction,
            early_stopping_enabled=early_stopping_enabled,
        )
        step_metrics["optimizer_seed"] = optimizer_seed
        case_step_records.append(step_metrics)

    end_selected_date = time.time()
    elapsed = end_selected_date - start_selected_date
    print(f"          Tiempo de ejecucion {selected_date['from']} - {selected_date['to']}: {elapsed:.2f} segundos")
    error_metrics = compute_error_metrics(case_step_records)
    actuator_change_metrics = compute_actuator_change_metrics(case_step_records)
    energy_metrics = compute_energy_metrics(case_step_records)
    climate_violation_metrics = compute_climate_violation_metrics(case_step_records)
    engineering_rule_metrics = compute_engineering_rule_metrics(case_step_records)
    ga_runtime_metrics = compute_ga_runtime_metrics(case_step_records)
    ga_summary = compute_ga_summary(case_step_records)
    last_ga_summary = {
        "ga_variant": last_ga.variant_name,
        "completed_generations": last_ga.completed_generations,
        "stop_reason": last_ga.stop_reason,
        "best_fitness_history": [float(value) for value in last_ga.best_fitness_history],
        "best_fitness_evaluations": [int(value) for value in last_ga.best_fitness_evaluations],
        "evaluations_count": last_ga.evaluations_count,
        "engineering_rule_violations_total": last_ga.engineering_rule_violations_total,
        "penalized_rule_violations_total": last_ga.penalized_rule_violations_total,
        "gaussian_sigma_fraction": last_ga.gaussian_sigma_fraction,
        "best_fitness_error_score": last_ga.fittest_individual.fitness_error_score,
        "best_genotype": last_ga.fittest_individual.genotype.copy(),
    }

    return {
        "experiment_name": experiment_name,
        "case_prefix": case_prefix,
        "start_date": selected_date["from"],
        "end_date": selected_date["to"],
        "seed": seed,
        "ga_variant": ga_variant,
        "method_name": ga_variant,
        "algorithm_family": algorithm_family(ga_variant),
        "objective_variant": objective_variant(ga_variant),
        "planned_evaluation_budget": int(generations_num) * int(population_num),
        "generations_num": generations_num,
        "population_num": population_num,
        "k_tournament": k_tournament,
        "fitness_percentage_goal": fitness_percentage_goal,
        "fitness_penalty_scheme": FITNESS_PENALTY_SCHEME,
        "actuator_effort_penalty": uses_actuator_effort_penalty(ga_variant),
        "patience": patience,
        "stop_ga_thr": stop_ga_thr,
        "stop_ga_from": stop_ga_from,
        "early_stopping_enabled": early_stopping_enabled,
        "mutation_probabilities": mutation_probabilities,
        "gaussian_sigma_fraction": gaussian_sigma_fraction,
        "energy_proxy_weights": energy_proxy_weights,
        "allowed_deviation_tbs": allowed_deviation_tbs,
        "allowed_deviation_hr": allowed_deviation_hr,
        "code_time": elapsed,
        "final_timestamp": current_timestamp,
        "steps": case_step_records,
        **error_metrics,
        **actuator_change_metrics,
        **energy_metrics,
        **climate_violation_metrics,
        **engineering_rule_metrics,
        **ga_runtime_metrics,
        "ga_summary": ga_summary,
        "last_ga": last_ga_summary,
    }


def validate_experiment_config(config):
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"Missing experiment configuration keys: {sorted(missing)}")
    for key in ("RUN_FULL_DATE_SET", "RUN_FULL_SEED_SET", "EARLY_STOPPING_ENABLED"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    configured_scheme = config.get("FITNESS_PENALTY_SCHEME")
    if configured_scheme is not None and configured_scheme != FITNESS_PENALTY_SCHEME:
        raise ValueError(
            f"Configured fitness scheme {configured_scheme!r} does not match "
            f"the active scheme {FITNESS_PENALTY_SCHEME!r}"
        )


def run_experiments(config):
    """Run exactly one grid configuration.

    All experiment choices come from ``config``. This module intentionally has
    no alternative standalone configuration, preventing hidden state between
    consecutive grid experiments.
    """
    validate_experiment_config(config)

    run_full_date_set = config["RUN_FULL_DATE_SET"]
    run_full_seed_set = config["RUN_FULL_SEED_SET"]
    experiment_name = str(config["EXPERIMENT_NAME"])
    ga_variant = str(config["GA_VARIANT"])
    fitness_percentage_goal = float(config["FITNESS_PERCENTAGE_GOAL"])
    generations_num = int(config["GENERATIONS_NUM"])
    population_num = int(config["POPULATION_NUM"])
    k_tournament = int(config["K_TOURNAMENT"])
    patience = config["PATIENCE"]
    stop_ga_thr = config["STOP_GA_THR"]
    stop_ga_from = config["STOP_GA_FROM"]
    early_stopping_enabled = config["EARLY_STOPPING_ENABLED"]
    mutation_probabilities = dict(config["MUTATION_PROBABILITIES"])
    gaussian_sigma_fraction = float(config["GAUSSIAN_SIGMA_FRACTION"])
    energy_proxy_weights = dict(config["ENERGY_PROXY_WEIGHTS"])
    allowed_deviation_tbs = float(config["ALLOWED_DEVIATION_TBS"])
    allowed_deviation_hr = float(config["ALLOWED_DEVIATION_HR"])
    dataset_csv_path = str(config.get("DATA_CSV_PATH", DATA_CSV_PATH))
    output_folder = str(config.get("OUTPUT_FOLDER", OUTPUT_FOLDER))

    start_code = time.time()
    data = read_csv(
        dataset_csv_path=dataset_csv_path,
        output_folder=output_folder,
        fields=MULTIVARIATE_VARIABLES,
    )

    selected_dates = config.get("SELECTED_DATES")
    if selected_dates is None:
        selected_dates = get_selected_dates(run_full_date_set)
    else:
        selected_dates = [dict(date_range) for date_range in selected_dates]

    seed_values = config.get("SEED_VALUES")
    if seed_values is None:
        seed_values = get_seed_values(run_full_seed_set)
    else:
        seed_values = [int(seed) for seed in seed_values]

    if not selected_dates:
        raise ValueError("SELECTED_DATES cannot be empty")
    if not seed_values:
        raise ValueError("SEED_VALUES cannot be empty")
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("SEED_VALUES must not contain duplicates")
    
    results = []

    for seed in seed_values:
        print(f"\n    Semilla: {seed}")

        for selected_date in selected_dates:
            # Reset before every case so results do not depend on date ordering
            # or on whether other dates were included in the same execution.
            seed_everything(seed)
            results.append(run_case(data,
                                    selected_date, 
                                    seed,
                                    experiment_name,
                                    ga_variant,
                                    fitness_percentage_goal,
                                    generations_num,
                                    population_num,
                                    k_tournament,
                                    patience,
                                    stop_ga_thr,
                                    stop_ga_from,
                                    mutation_probabilities,
                                    energy_proxy_weights,
                                    allowed_deviation_tbs,
                                    allowed_deviation_hr,
                                    gaussian_sigma_fraction,
                                    early_stopping_enabled))

    end_code = time.time()

    # Export flattened per-step CSV for traceability and plots. Fail loudly:
    # a completed experiment without trustworthy outputs must not be accepted.
    os.makedirs(output_folder, exist_ok=True)
    step_rows = []
    for result in results:
        for step in result.get("steps", []):
            row = {
                "experiment_name": result.get("experiment_name"),
                "experiment_phase": config.get("EXPERIMENT_PHASE"),
                "date_split": config.get("DATE_SPLIT"),
                "claim_scope": config.get("CLAIM_SCOPE"),
                "comparison_group": config.get("COMPARISON_GROUP"),
                "scenario_group": config.get("SCENARIO_GROUP"),
                "optimizer_config_group": config.get("OPTIMIZER_CONFIG_GROUP"),
                "objective_group": config.get("OBJECTIVE_GROUP"),
                "seed": result.get("seed"),
                "start_date": result.get("start_date"),
                "end_date": result.get("end_date"),
                "case_prefix": result.get("case_prefix"),
            }
            row.update(step)
            step_rows.append(row)

    steps_path = os.path.join(output_folder, "steps_flat.csv")
    if step_rows:
        steps_df = pd.DataFrame(step_rows)
        steps_df.to_csv(steps_path, index=False)

    case_rows = []
    for result in results:
        row = {
            "experiment_name": result.get("experiment_name"),
            "experiment_phase": config.get("EXPERIMENT_PHASE"),
            "date_split": config.get("DATE_SPLIT"),
            "claim_scope": config.get("CLAIM_SCOPE"),
            "comparison_group": config.get("COMPARISON_GROUP"),
            "scenario_group": config.get("SCENARIO_GROUP"),
            "optimizer_config_group": config.get("OPTIMIZER_CONFIG_GROUP"),
            "objective_group": config.get("OBJECTIVE_GROUP"),
            "ga_variant": result.get("ga_variant"),
            "method_name": result.get("method_name"),
            "algorithm_family": result.get("algorithm_family"),
            "objective_variant": result.get("objective_variant"),
            "planned_evaluation_budget": result.get("planned_evaluation_budget"),
            "case_prefix": result.get("case_prefix"),
            "seed": result.get("seed"),
            "start_date": result.get("start_date"),
            "end_date": result.get("end_date"),
            "run_full_date_set": run_full_date_set,
            "run_full_seed_set": run_full_seed_set,
            "generations_num": result.get("generations_num"),
            "population_num": result.get("population_num"),
            "k_tournament": result.get("k_tournament"),
            "fitness_percentage_goal": result.get("fitness_percentage_goal"),
            "fitness_penalty_scheme": result.get("fitness_penalty_scheme"),
            "actuator_effort_penalty": result.get("actuator_effort_penalty"),
            "patience": result.get("patience"),
            "stop_ga_thr": result.get("stop_ga_thr"),
            "stop_ga_from": result.get("stop_ga_from"),
            "early_stopping_enabled": result.get("early_stopping_enabled"),
            "mutation_probabilities": json.dumps(result.get("mutation_probabilities"), sort_keys=True),
            "gaussian_sigma_fraction": result.get("gaussian_sigma_fraction"),
            "energy_proxy_weights": json.dumps(result.get("energy_proxy_weights"), sort_keys=True),
            "allowed_deviation_tbs": result.get("allowed_deviation_tbs"),
            "allowed_deviation_hr": result.get("allowed_deviation_hr"),
            "code_time": result.get("code_time"),
            "final_timestamp": result.get("final_timestamp"),
        }

        primary_metrics = [
            "MAE_TBS", "MAE_HR", "RMSE_TBS", "RMSE_HR",
            "P95_abs_error_TBS", "P95_abs_error_HR",
            "IAE_TBS", "IAE_HR", "IAE_TBS_per_step", "IAE_HR_per_step",
            "total_actuator_change", "delta_u_squared",
            "total_actuator_change_normalized", "delta_u_squared_normalized",
            "energy_proxy_total", "pipe_energy_proxy", "humidification_usage",
            "climate_violation_tbs", "climate_violation_hr",
            "engineering_rule_violations", "target_progress_rule_violations",
            "total_ga_time_s", "mean_ga_time_s", "p95_ga_time_s", "max_ga_time_s",
            "real_time_step_budget_s", "p95_ga_time_within_step_budget",
            "max_ga_time_within_step_budget", "total_evaluations_count",
            "mean_time_per_evaluation", "mean_generations_to_converge",
            "final_generations_to_converge", "mean_evaluations_to_converge",
            "final_evaluations_to_converge",
        ]
        for metric in primary_metrics:
            row[metric.lower()] = result.get(metric)

        ga_summary = result.get("ga_summary", {})
        row["ga_executions"] = ga_summary.get("ga_executions")
        row["stop_reason_counts"] = json.dumps(ga_summary.get("stop_reason_counts", {}), sort_keys=True)
        row["mean_completed_generations"] = ga_summary.get("mean_completed_generations")
        row["final_completed_generations"] = ga_summary.get("final_completed_generations")
        row["final_stop_reason"] = ga_summary.get("final_stop_reason")

        last_ga = result.get("last_ga", {})
        row["last_ga_variant"] = last_ga.get("ga_variant")
        row["last_best_fitness_error_score"] = last_ga.get("best_fitness_error_score")
        row["last_evaluations_count"] = last_ga.get("evaluations_count")
        row["last_engineering_rule_violations_total"] = last_ga.get("engineering_rule_violations_total")
        row["last_penalized_rule_violations_total"] = last_ga.get("penalized_rule_violations_total")
        row["last_gaussian_sigma_fraction"] = last_ga.get("gaussian_sigma_fraction")
        case_rows.append(row)

    case_path = os.path.join(output_folder, "case_summary.csv")
    if case_rows:
        pd.DataFrame(case_rows).to_csv(case_path, index=False)

    return {
        "seed_values": seed_values,
        "code_time": end_code - start_code,
        "output_folder": output_folder,
        "results": results,
    }
