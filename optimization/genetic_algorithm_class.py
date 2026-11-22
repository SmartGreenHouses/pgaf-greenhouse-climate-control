from copy import deepcopy
import math
import time
import random
import numpy as np
from individual_class import Individual
from constants import POSSIBLE_GENES, MIN_TBS_VALUE, MAX_TBS_VALUE, MIN_HR_VALUE, MAX_HR_VALUE, MSG_MODE, STEPS, PIPE_TEMP_DECREASE_THRESHOLD, DESIRED_TBS_DAY, DESIRED_TBS_NIGHT, DESIRED_HR_DAY, DESIRED_HR_NIGHT, TBS_HEATING_NEED_TOLERANCE, TBS_OVERHEAT_TOLERANCE, HR_HUMIDIFICATION_TOLERANCE, HR_DEHUMIDIFICATION_TOLERANCE, ABS_HUMIDITY_MARGIN, PIPE_OVER_TBS_MARGIN, COLD_EXTERIOR_MAX_WINDOW_POSITION, DEHUMIDIFICATION_MIN_WINDOW_POSITION, HUMIDITY_RISK_MAX_WINDOW_POSITION, SCREEN_RADIATION_THRESHOLD, HIGH_RADIATION_MAX_SCREEN_POSITION, SOLAR_GAIN_MIN_SCREEN_POSITION, BLX_CROSSOVER_ALPHA, PIPE_ENERGY_LEVEL_WEIGHT, PIPE_ENERGY_EXCESS_WEIGHT, ACTUATOR_AWARE_PENALTY_WEIGHT, RICH_ENERGY_PENALTY_WEIGHT, RICH_ACTUATOR_PENALTY_WEIGHT, ENGINEERING_RULE_PENALTY_WEIGHT, TARGET_PROGRESS_RULE_PENALTY_WEIGHT, TARGET_BAND_VIOLATION_WEIGHT, PENALTY_TARGET_TBS_TOLERANCE, PENALTY_TARGET_HR_TOLERANCE, PENALTY_GATE_FLOOR, PENALTY_GATE_POWER
from utils_v import prepare_forecast_data, forecast_from_model_history, forecast_batch_from_model_history
from numpy import mean, abs
POSSIBLE_GENE_KEYS = list(POSSIBLE_GENES)
from ga_method_registry import BASELINE_METHODS, GA_VARIANTS, METHODS, REFERENCE_OBJECTIVE_VARIANT, VARIANTS, algorithm_family, objective_variant, uses_actuator_effort_penalty, uses_enhanced_operators, uses_hold_last_action, uses_random_search, validate_energy_proxy_weights, validate_method

def calculate_solution(individual, forecast_context, steps):
    last_row_updates = {key: individual.genotype[i] for i, key in enumerate(POSSIBLE_GENES)}
    model_history, prediction_columns = forecast_context
    denormalized_model_predictions, prediction_columns = forecast_from_model_history(model_history, prediction_columns, steps=steps, last_row_updates=last_row_updates)
    all_predictions = {}
    for index, column in enumerate(prediction_columns):
        all_predictions[column] = denormalized_model_predictions[-1, index]
    solution = {'tbs': all_predictions['CajaMedida1.TBS'], 'hr': all_predictions['CajaMedida1.HR'], 'all_predictions': all_predictions}
    return solution

def min_max_norm(v, min_v, max_v):
    denom = max_v - min_v
    if denom == 0:
        return 0.0
    return (v - min_v) / denom

def fitness_tbs_and_hr_percent_function(solution, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal):
    partial_desired_tbs_output = last_tbs - (last_tbs - desired_tbs_output) * fitness_percentage_goal
    partial_desired_hr_output = last_hr - (last_hr - desired_hr_output) * fitness_percentage_goal
    fitness_tbs = abs(min_max_norm(solution['tbs'], MIN_TBS_VALUE, MAX_TBS_VALUE) - min_max_norm(partial_desired_tbs_output, MIN_TBS_VALUE, MAX_TBS_VALUE))
    fitness_hr = abs(min_max_norm(solution['hr'], MIN_HR_VALUE, MAX_HR_VALUE) - min_max_norm(partial_desired_hr_output, MIN_HR_VALUE, MAX_HR_VALUE))
    fitness = mean([fitness_tbs, fitness_hr])
    return fitness
fitness_base = fitness_tbs_and_hr_percent_function
RICH_FITNESS_WEIGHTS = {'climate': 1.0, 'energy': RICH_ENERGY_PENALTY_WEIGHT, 'actuator_effort': RICH_ACTUATOR_PENALTY_WEIGHT}
ENERGY_PROXY_WEIGHTS = {'pipe': 5.0, 'humidification': 2.0, 'openings': 0.3}

def compute_energy_proxy(genotype_map, last_tbs, energy_proxy_weights=None):
    if energy_proxy_weights is None:
        energy_proxy_weights = ENERGY_PROXY_WEIGHTS
    pipe_value = genotype_map['TemperaturaTuberia']
    pipe_min = float(POSSIBLE_GENES['TemperaturaTuberia']['min'])
    pipe_max = float(POSSIBLE_GENES['TemperaturaTuberia']['max'])
    pipe_range = max(pipe_max - pipe_min, 1e-09)
    pipe_norm = np.clip((pipe_value - pipe_min) / pipe_range, 0.0, 1.0)
    pipe_excess = max(0.0, pipe_value - last_tbs)
    pipe_excess_norm = np.clip(pipe_excess / pipe_range, 0.0, 1.0)
    pipe_proxy = PIPE_ENERGY_LEVEL_WEIGHT * pipe_norm + PIPE_ENERGY_EXCESS_WEIGHT * pipe_excess_norm
    hum_proxy = float(genotype_map['Humidificacion'])
    openings_proxy = float(np.mean([genotype_map['PosicionVentana1'] / 100.0, genotype_map['PosicionVentana2'] / 100.0, genotype_map['PosicionPantalla1'] / 100.0, genotype_map['PosicionPantalla2'] / 100.0]))
    return float(energy_proxy_weights['pipe'] * pipe_proxy + energy_proxy_weights['humidification'] * hum_proxy + energy_proxy_weights['openings'] * openings_proxy)

def target_distance_normalized(solution, desired_tbs_output, desired_hr_output):
    tbs_scale = max(float(PENALTY_TARGET_TBS_TOLERANCE), 1e-09)
    hr_scale = max(float(PENALTY_TARGET_HR_TOLERANCE), 1e-09)
    tbs_distance = (float(solution['tbs']) - float(desired_tbs_output)) / tbs_scale
    hr_distance = (float(solution['hr']) - float(desired_hr_output)) / hr_scale
    return float(np.hypot(tbs_distance, hr_distance))

def target_aware_penalty_gate(solution, desired_tbs_output, desired_hr_output):
    distance = target_distance_normalized(solution, desired_tbs_output, desired_hr_output)
    close_to_target = 1.0 / (1.0 + distance ** PENALTY_GATE_POWER)
    return float(PENALTY_GATE_FLOOR + (1.0 - PENALTY_GATE_FLOOR) * close_to_target)

def target_band_violation(solution, desired_tbs_output, desired_hr_output):
    tbs_scale = max(float(PENALTY_TARGET_TBS_TOLERANCE), 1e-09)
    hr_scale = max(float(PENALTY_TARGET_HR_TOLERANCE), 1e-09)
    tbs_excess = max(0.0, abs(float(solution['tbs']) - float(desired_tbs_output)) / tbs_scale - 1.0)
    hr_excess = max(0.0, abs(float(solution['hr']) - float(desired_hr_output)) / hr_scale - 1.0)
    return float(np.hypot(tbs_excess, hr_excess))

def target_band_violation_penalty(solution, desired_tbs_output, desired_hr_output):
    return TARGET_BAND_VIOLATION_WEIGHT * target_band_violation(solution, desired_tbs_output, desired_hr_output)

def fitness_rich(solution, genotype, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row=None, energy_proxy_weights=None):
    if energy_proxy_weights is None:
        energy_proxy_weights = ENERGY_PROXY_WEIGHTS
    climate_error = fitness_tbs_and_hr_percent_function(solution, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal)
    genotype_map = {key: genotype[i] for i, key in enumerate(POSSIBLE_GENE_KEYS)}
    energy = compute_energy_proxy(genotype_map, last_tbs, energy_proxy_weights)
    energy_max = energy_proxy_weights['pipe'] + energy_proxy_weights['humidification'] + energy_proxy_weights['openings']
    energy_norm = np.clip(energy / max(energy_max, 1e-09), 0.0, 1.0)
    if last_row is None:
        raise ValueError('last_row is required for rich fitness')
    actuator_effort = compute_normalized_actuator_penalty(genotype, last_row)
    penalty_gate = target_aware_penalty_gate(solution, desired_tbs_output, desired_hr_output)
    weights = RICH_FITNESS_WEIGHTS
    fitness = weights['climate'] * climate_error + target_band_violation_penalty(solution, desired_tbs_output, desired_hr_output) + penalty_gate * weights['energy'] * energy_norm + penalty_gate * weights['actuator_effort'] * actuator_effort
    return float(fitness)

def compute_normalized_actuator_penalty(genotype, last_row):
    weighted_effort = 0.0
    max_weighted_effort = 0.0
    for i, gene in enumerate(genotype):
        gene_key = POSSIBLE_GENE_KEYS[i]
        gene_config = POSSIBLE_GENES[gene_key]
        last_value = float(last_row[gene_key])
        delta = float(gene) - last_value
        relative_change = max(float(gene_config['relative_15m']), 1e-09)
        delta_norm = min(abs(delta) / relative_change, 1.0)
        if delta > 0:
            direction_weight = float(gene_config['penalty_up'])
        elif delta < 0:
            direction_weight = float(gene_config['penalty_down'])
        else:
            direction_weight = 0.0
        weighted_effort += direction_weight * delta_norm
        max_weighted_effort += max(float(gene_config['penalty_up']), float(gene_config['penalty_down']))
    return weighted_effort / max(max_weighted_effort, 1e-09)
MAX_ENGINEERING_RULE_VIOLATIONS = 15

def absolute_humidity_g_m3(temperature_c, relative_humidity):
    relative_humidity = np.clip(float(relative_humidity), 0.0, 100.0)
    temperature_c = float(temperature_c)
    saturation_vapor_pressure = 6.112 * np.exp(17.67 * temperature_c / (temperature_c + 243.5))
    vapor_pressure = relative_humidity / 100.0 * saturation_vapor_pressure
    return 216.7 * vapor_pressure / (temperature_c + 273.15)

def count_engineering_rule_violations(genotype, solution, last_row, desired_tbs_output=None, desired_hr_output=None):
    genotype_map = {key: genotype[i] for i, key in enumerate(POSSIBLE_GENE_KEYS)}
    last_tbs = float(last_row.get('CajaMedida1.TBS', 20.0))
    predicted_tbs = float(solution.get('tbs', last_tbs))
    predicted_hr = float(solution.get('hr', 0.0))
    last_pipe = float(last_row.get('TemperaturaTuberia', 20.0))
    t_exterior = float(last_row.get('TemperaturaExterior', 15.0))
    exterior_hr = float(last_row.get('HumedadExterior', predicted_hr))
    radiation = float(last_row.get('Radiacion', 0.0))
    desired_tbs_ref = float(desired_tbs_output if desired_tbs_output is not None else min(DESIRED_TBS_DAY, DESIRED_TBS_NIGHT))
    desired_hr_ref = float(desired_hr_output if desired_hr_output is not None else max(DESIRED_HR_DAY, DESIRED_HR_NIGHT))
    violations = 0
    current_pipe = genotype_map['TemperaturaTuberia']
    temperature_too_low = predicted_tbs < desired_tbs_ref - TBS_HEATING_NEED_TOLERANCE
    temperature_too_high = predicted_tbs > desired_tbs_ref + TBS_OVERHEAT_TOLERANCE
    pipe_is_meaningfully_hot = current_pipe > last_tbs + PIPE_OVER_TBS_MARGIN
    large_pipe_drop = last_pipe - current_pipe >= PIPE_TEMP_DECREASE_THRESHOLD
    if temperature_too_low and pipe_is_meaningfully_hot and large_pipe_drop:
        violations += 1
    pipe_is_unnecessarily_hot = current_pipe > desired_tbs_ref + PIPE_OVER_TBS_MARGIN
    if temperature_too_high and pipe_is_unnecessarily_hot:
        violations += 1
    if genotype_map['Humidificacion'] == 1 and predicted_hr >= desired_hr_ref - HR_HUMIDIFICATION_TOLERANCE:
        violations += 1
    cold_outside = t_exterior < desired_tbs_ref - TBS_HEATING_NEED_TOLERANCE
    ventilation_is_thermally_safe = not (cold_outside and temperature_too_low)
    if cold_outside and (not temperature_too_high):
        if genotype_map['PosicionVentana1'] > COLD_EXTERIOR_MAX_WINDOW_POSITION:
            violations += 1
        if genotype_map['PosicionVentana2'] > COLD_EXTERIOR_MAX_WINDOW_POSITION:
            violations += 1
    humidity_too_high = predicted_hr > desired_hr_ref + HR_DEHUMIDIFICATION_TOLERANCE
    humidity_too_low = predicted_hr < desired_hr_ref - HR_HUMIDIFICATION_TOLERANCE
    inside_abs_humidity = absolute_humidity_g_m3(predicted_tbs, predicted_hr)
    outside_abs_humidity = absolute_humidity_g_m3(t_exterior, exterior_hr)
    outside_air_is_drier = outside_abs_humidity < inside_abs_humidity - ABS_HUMIDITY_MARGIN
    outside_air_is_more_humid = outside_abs_humidity > inside_abs_humidity + ABS_HUMIDITY_MARGIN
    if humidity_too_high and outside_air_is_drier and ventilation_is_thermally_safe:
        if genotype_map['PosicionVentana1'] < DEHUMIDIFICATION_MIN_WINDOW_POSITION:
            violations += 1
        if genotype_map['PosicionVentana2'] < DEHUMIDIFICATION_MIN_WINDOW_POSITION:
            violations += 1
    if humidity_too_high and outside_air_is_more_humid and (not temperature_too_high):
        if genotype_map['PosicionVentana1'] > HUMIDITY_RISK_MAX_WINDOW_POSITION:
            violations += 1
        if genotype_map['PosicionVentana2'] > HUMIDITY_RISK_MAX_WINDOW_POSITION:
            violations += 1
    if humidity_too_low and outside_air_is_drier and (not temperature_too_high):
        if genotype_map['PosicionVentana1'] > HUMIDITY_RISK_MAX_WINDOW_POSITION:
            violations += 1
        if genotype_map['PosicionVentana2'] > HUMIDITY_RISK_MAX_WINDOW_POSITION:
            violations += 1
    radiation_is_meaningful = radiation >= SCREEN_RADIATION_THRESHOLD
    if radiation_is_meaningful and temperature_too_high:
        if genotype_map['PosicionPantalla1'] > HIGH_RADIATION_MAX_SCREEN_POSITION:
            violations += 1
        if genotype_map['PosicionPantalla2'] > HIGH_RADIATION_MAX_SCREEN_POSITION:
            violations += 1
    if radiation_is_meaningful and temperature_too_low:
        if genotype_map['PosicionPantalla1'] < SOLAR_GAIN_MIN_SCREEN_POSITION:
            violations += 1
        if genotype_map['PosicionPantalla2'] < SOLAR_GAIN_MIN_SCREEN_POSITION:
            violations += 1
    return violations

def apply_engineering_rules(genotype, solution, last_row, fitness, desired_tbs_output=None, desired_hr_output=None):
    desired_tbs_ref = desired_tbs_output if desired_tbs_output is not None else min(DESIRED_TBS_DAY, DESIRED_TBS_NIGHT)
    desired_hr_ref = desired_hr_output if desired_hr_output is not None else max(DESIRED_HR_DAY, DESIRED_HR_NIGHT)
    violations = count_engineering_rule_violations(genotype, solution, last_row, desired_tbs_output=desired_tbs_ref, desired_hr_output=desired_hr_ref)
    normalized_violations = violations / MAX_ENGINEERING_RULE_VIOLATIONS
    penalty_gate = target_aware_penalty_gate(solution, desired_tbs_ref, desired_hr_ref)
    return (fitness + penalty_gate * ENGINEERING_RULE_PENALTY_WEIGHT * normalized_violations, violations)
MAX_TARGET_PROGRESS_RULE_VIOLATIONS = 2

def count_target_progress_rule_violations(solution, last_row, desired_tbs_output, desired_hr_output):
    """Count climate dimensions that fail to move towards an unmet target.

    The rule is inactive once a dimension is inside its target band. Outside
    the band, a candidate violates the rule when its predicted absolute error
    is not smaller than the current absolute error. Using absolute error also
    rejects overshooting to the opposite side of the setpoint.
    """
    last_tbs_error = abs(float(last_row['CajaMedida1.TBS']) - float(desired_tbs_output))
    predicted_tbs_error = abs(float(solution['tbs']) - float(desired_tbs_output))
    last_hr_error = abs(float(last_row['CajaMedida1.HR']) - float(desired_hr_output))
    predicted_hr_error = abs(float(solution['hr']) - float(desired_hr_output))
    violations = 0
    if last_tbs_error > PENALTY_TARGET_TBS_TOLERANCE and predicted_tbs_error >= last_tbs_error - 1e-09:
        violations += 1
    if last_hr_error > PENALTY_TARGET_HR_TOLERANCE and predicted_hr_error >= last_hr_error - 1e-09:
        violations += 1
    return violations

def score_individual_for_variant(individual, variant_name, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row, energy_proxy_weights=None):
    validate_method(variant_name)
    score = fitness_rich(individual.solution, individual.genotype, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row=last_row, energy_proxy_weights=energy_proxy_weights)
    score, violations = apply_engineering_rules(individual.genotype, individual.solution, last_row, score, desired_tbs_output=desired_tbs_output, desired_hr_output=desired_hr_output)
    return (float(score), violations)

def tournament_selection(population, size):
    if not population:
        raise ValueError('Cannot select from an empty population')
    parents = random.choices(population, k=size)
    return min(parents, key=lambda individual: individual.fitness_error_score)

def crossover_blx_alpha(p_1, p_2, last_row):
    genotype_1 = []
    genotype_2 = []
    for i, gene_key in enumerate(POSSIBLE_GENE_KEYS):
        gene_config = POSSIBLE_GENES[gene_key]
        gene_min = float(gene_config['min'])
        gene_max = float(gene_config['max'])
        local_min = max(float(last_row[gene_key]) - float(gene_config['relative_15m']), gene_min)
        local_max = min(float(last_row[gene_key]) + float(gene_config['relative_15m']), gene_max)
        parent_value_1 = float(p_1.genotype[i])
        parent_value_2 = float(p_2.genotype[i])
        if gene_min == 0 and gene_max == 1:
            value_1 = random.choice([parent_value_1, parent_value_2])
            value_2 = random.choice([parent_value_1, parent_value_2])
            genotype_1.append(round(value_1))
            genotype_2.append(round(value_2))
            continue
        lower_parent = min(parent_value_1, parent_value_2)
        upper_parent = max(parent_value_1, parent_value_2)
        parent_distance = upper_parent - lower_parent
        sample_min = lower_parent - BLX_CROSSOVER_ALPHA * parent_distance
        sample_max = upper_parent + BLX_CROSSOVER_ALPHA * parent_distance
        value_1 = random.uniform(sample_min, sample_max)
        value_2 = random.uniform(sample_min, sample_max)
        value_1 = float(np.clip(value_1, local_min, local_max))
        value_2 = float(np.clip(value_2, local_min, local_max))
        genotype_1.append(value_1)
        genotype_2.append(value_2)
    return (Individual(genotype=genotype_1, last_row=last_row), Individual(genotype=genotype_2, last_row=last_row))

def mutate_gaussian(individual, mutation_probabilities, last_row, gaussian_sigma_fraction):
    gene_mutation_num = random.choices([0, 1, 2, 3], [mutation_probabilities['no_mutation'], mutation_probabilities['mutation_1'], mutation_probabilities['mutation_2'], mutation_probabilities['mutation_3']])[0]
    gene_mutation_indexes = random.sample(range(len(POSSIBLE_GENE_KEYS)), k=min(gene_mutation_num, len(POSSIBLE_GENE_KEYS)))
    for gene_i in gene_mutation_indexes:
        gene_key = POSSIBLE_GENE_KEYS[gene_i]
        gene_config = POSSIBLE_GENES[gene_key]
        gene_min = float(gene_config['min'])
        gene_max = float(gene_config['max'])
        local_min = max(float(last_row[gene_key]) - float(gene_config['relative_15m']), gene_min)
        local_max = min(float(last_row[gene_key]) + float(gene_config['relative_15m']), gene_max)
        if gene_min == 0 and gene_max == 1:
            individual.genotype[gene_i] = random.randint(0, 1)
            continue
        sigma = gaussian_sigma_fraction * (gene_max - gene_min)
        new_value = individual.genotype[gene_i] + np.random.normal(0.0, sigma)
        individual.genotype[gene_i] = float(np.clip(new_value, local_min, local_max))
    return individual

def validate_ga_variant(variant_name):
    """Backwards-compatible validator for the complete method registry."""
    validate_method(variant_name)

def validate_ga_parameters(generations_num, population_num, k_tournament, mutation_probabilities, gaussian_sigma_fraction, fitness_percentage_goal=None, patience=None, stop_ga_thr=None, stop_ga_from=None, energy_proxy_weights=None):
    if not isinstance(generations_num, int) or isinstance(generations_num, bool) or generations_num <= 0:
        raise ValueError('generations_num must be a positive integer')
    if not isinstance(population_num, int) or isinstance(population_num, bool) or population_num <= 0:
        raise ValueError('population_num must be a positive integer')
    if not isinstance(k_tournament, int) or isinstance(k_tournament, bool) or k_tournament <= 0:
        raise ValueError('k_tournament must be a positive integer')
    if k_tournament > population_num:
        raise ValueError('k_tournament cannot be greater than population_num')
    required_probabilities = {'no_mutation', 'mutation_1', 'mutation_2', 'mutation_3'}
    if set(mutation_probabilities) != required_probabilities:
        raise ValueError(f'mutation_probabilities must contain exactly {sorted(required_probabilities)}, got {sorted(mutation_probabilities)}')
    values = [float(mutation_probabilities[key]) for key in required_probabilities]
    if any((not math.isfinite(value) or value < 0 for value in values)):
        raise ValueError('Mutation probabilities must be finite and non-negative')
    total_probability = sum(values)
    if not math.isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=1e-09):
        raise ValueError(f'Mutation probabilities must sum to 1.0, got {total_probability}')
    gaussian_sigma_fraction = float(gaussian_sigma_fraction)
    if not math.isfinite(gaussian_sigma_fraction) or gaussian_sigma_fraction < 0:
        raise ValueError('gaussian_sigma_fraction must be finite and non-negative')
    if fitness_percentage_goal is not None:
        fitness_percentage_goal = float(fitness_percentage_goal)
        if not math.isfinite(fitness_percentage_goal) or not 0.0 <= fitness_percentage_goal <= 1.0:
            raise ValueError('fitness_percentage_goal must be between 0 and 1')
    if patience is not None:
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError('patience must be a positive integer or None')
        if stop_ga_thr is None:
            raise ValueError('stop_ga_thr is required when patience is enabled')
    if stop_ga_thr is not None:
        stop_ga_thr = float(stop_ga_thr)
        if not math.isfinite(stop_ga_thr) or stop_ga_thr < 0:
            raise ValueError('stop_ga_thr must be finite and non-negative, or None')
    if stop_ga_from is not None and (not math.isfinite(float(stop_ga_from))):
        raise ValueError('stop_ga_from must be finite or None')
    validate_energy_proxy_weights(energy_proxy_weights)

class GA:

    def __init__(self, generations_num, population_num, k_tournament, variant_name='GA_ENGINEERING_AWARE', energy_proxy_weights=None, gaussian_sigma_fraction=0.08, early_stopping_enabled=True):
        validate_ga_variant(variant_name)
        self.generations_num = generations_num
        self.population_num = population_num
        self.k_tournament = k_tournament
        self.variant_name = variant_name
        self.energy_proxy_weights = dict(ENERGY_PROXY_WEIGHTS if energy_proxy_weights is None else energy_proxy_weights)
        self.gaussian_sigma_fraction = gaussian_sigma_fraction
        self.early_stopping_enabled = bool(early_stopping_enabled)
        self.population = None
        self.fittest_individual = None
        self.generations = []
        self.completed_generations = 0
        self.stop_reason = None
        self.patience_count = 0
        self.best_fitness_error_score = None
        self.previous_fitness_error_score = None
        self.evaluations_count = 0
        self.first_generation_best_fitness = None
        self.last_generation_best_fitness = None
        self.best_fitness_history = []
        self.best_fitness_evaluations = []
        self.engineering_rule_violations_total = 0
        self.penalized_rule_violations_total = 0

    def do_start_population(self, last_row):
        population = []
        for _ in range(self.population_num):
            population.append(Individual(last_row=last_row))
        self.population = population

    def do_start_generations(self, forecast_context, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row, patience, stop_ga_thr, stop_ga_from, mutation_probabilities):
        for gen in range(self.generations_num):
            if MSG_MODE:
                print('     Generation #' + str(gen))
            self.do_fitness(forecast_context, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row)
            stop_reason = None
            if self.early_stopping_enabled:
                stop_reason = self.do_check_gen_stop(patience, stop_ga_thr, stop_ga_from)
            else:
                self.do_check_gen_stop(None, stop_ga_thr, None)
            if stop_reason is not None:
                self.stop_reason = stop_reason if gen < self.generations_num - 1 else 'max_generations'
                break
            if gen == self.generations_num - 1:
                self.stop_reason = 'max_generations'
                break
            selection = self.do_selection()
            new_population = self.do_mating(selection, mutation_probabilities, last_row)
            self.population = new_population
        if self.stop_reason is None:
            self.stop_reason = 'max_generations'
        self.completed_generations = len(self.generations)

    def do_fitness(self, forecast_context, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row):
        if STEPS == 1:
            model_history, prediction_columns = forecast_context
            last_row_updates_list = [{key: individual.genotype[i] for i, key in enumerate(POSSIBLE_GENES)} for individual in self.population]
            batch_predictions, prediction_columns = forecast_batch_from_model_history(model_history, prediction_columns, steps=STEPS, last_row_updates_list=last_row_updates_list)
            for individual, prediction_row in zip(self.population, batch_predictions):
                all_predictions = dict(zip(prediction_columns, prediction_row))
                solution = {'tbs': all_predictions['CajaMedida1.TBS'], 'hr': all_predictions['CajaMedida1.HR'], 'all_predictions': all_predictions}
                individual.solution = solution
                individual.fitness_error_score, violations = score_individual_for_variant(individual, self.variant_name, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row, energy_proxy_weights=self.energy_proxy_weights)
                if not math.isfinite(float(individual.fitness_error_score)):
                    raise ValueError(f'Non-finite fitness for variant {self.variant_name}: {individual.fitness_error_score}')
                self.engineering_rule_violations_total += violations
                self.penalized_rule_violations_total += violations
            self.evaluations_count += len(self.population)
            return
        for individual in self.population:
            solution = calculate_solution(individual, forecast_context, steps=STEPS)
            individual.solution = solution
            individual.fitness_error_score, violations = score_individual_for_variant(individual, self.variant_name, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row, energy_proxy_weights=self.energy_proxy_weights)
            if not math.isfinite(float(individual.fitness_error_score)):
                raise ValueError(f'Non-finite fitness for variant {self.variant_name}: {individual.fitness_error_score}')
            self.engineering_rule_violations_total += violations
            self.penalized_rule_violations_total += violations
        self.evaluations_count += len(self.population)

    def do_check_gen_stop(self, patience, stop_ga_thr, stop_ga_from):
        current_fittest = min(self.population, key=lambda individual: individual.fitness_error_score)
        current_score = float(current_fittest.fitness_error_score)
        if not math.isfinite(current_score):
            raise ValueError(f'Cannot apply stopping criteria to non-finite fitness: {current_score}')
        previous_best = self.best_fitness_error_score
        if previous_best is None or current_score < previous_best:
            self.best_fitness_error_score = current_score
            self.fittest_individual = deepcopy(current_fittest)
        if self.fittest_individual is None:
            self.fittest_individual = deepcopy(current_fittest)
        if MSG_MODE:
            print('Fittest: ' + str(self.fittest_individual))
        target_reached = stop_ga_from is not None and self.best_fitness_error_score <= float(stop_ga_from)
        if patience is not None:
            if previous_best is None:
                self.patience_count = 0
            else:
                improvement = previous_best - self.best_fitness_error_score
                if improvement <= float(stop_ga_thr):
                    self.patience_count += 1
                else:
                    self.patience_count = 0
        else:
            self.patience_count = 0
        self.previous_fitness_error_score = current_score
        snapshot = {'fitness_error_score': float(self.best_fitness_error_score), 'genotype': list(self.fittest_individual.genotype), 'solution': deepcopy(self.fittest_individual.solution)}
        self.generations.append({'fittest_individual': snapshot})
        self.best_fitness_history.append(float(self.best_fitness_error_score))
        self.best_fitness_evaluations.append(int(self.evaluations_count))
        if self.first_generation_best_fitness is None:
            self.first_generation_best_fitness = float(self.best_fitness_error_score)
        self.last_generation_best_fitness = float(self.best_fitness_error_score)
        if target_reached:
            return 'target_reached'
        if patience is not None and self.patience_count >= patience:
            return 'stagnation'
        return None

    def do_selection(self):
        best = min(self.population, key=lambda individual: individual.fitness_error_score)
        selection = [best]
        selection_size = max(2, self.population_num // 2)
        selection_size = min(selection_size, self.population_num)
        for _ in range(selection_size - 1):
            selection.append(tournament_selection(self.population, self.k_tournament))
        return selection

    def do_mating(self, selection, mutation_probabilities, last_row):
        elite = min(selection, key=lambda individual: individual.fitness_error_score)
        new_population = [Individual(genotype=elite.genotype.copy(), last_row=last_row)]
        while len(new_population) < self.population_num:
            parent_1 = tournament_selection(selection, self.k_tournament)
            parent_2 = tournament_selection(selection, self.k_tournament)
            new_population.extend(crossover_blx_alpha(parent_1, parent_2, last_row=last_row))
        new_population = new_population[:self.population_num]
        for i in range(1, len(new_population)):
            new_population[i] = mutate_gaussian(new_population[i], mutation_probabilities, last_row=last_row, gaussian_sigma_fraction=self.gaussian_sigma_fraction)
        return new_population

def run_genetic_algorithm(generations_num, population_num, k_tournament, last_row, case_data, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, patience, stop_ga_thr, stop_ga_from, mutation_probabilities, variant_name='GA_ENGINEERING_AWARE', energy_proxy_weights=None, gaussian_sigma_fraction=0.08, early_stopping_enabled=True):
    validate_ga_parameters(generations_num, population_num, k_tournament, mutation_probabilities, gaussian_sigma_fraction, fitness_percentage_goal=fitness_percentage_goal, patience=patience, stop_ga_thr=stop_ga_thr, stop_ga_from=stop_ga_from, energy_proxy_weights=energy_proxy_weights)
    validate_ga_variant(variant_name)
    forecast_context = prepare_forecast_data(case_data)
    ga = GA(generations_num, population_num, k_tournament, variant_name=variant_name, energy_proxy_weights=energy_proxy_weights, gaussian_sigma_fraction=gaussian_sigma_fraction, early_stopping_enabled=early_stopping_enabled)
    inicio_crear_poblacion = time.time()
    ga.do_start_population(last_row)
    fin_crear_poblacion = time.time()
    inicio_generar_poblacion = time.time()
    ga.do_start_generations(forecast_context, last_tbs, last_hr, desired_tbs_output, desired_hr_output, fitness_percentage_goal, last_row, patience, stop_ga_thr, stop_ga_from, mutation_probabilities)
    fin_generar_poblacion = time.time()
    tiempo_crear_poblacion = fin_crear_poblacion - inicio_crear_poblacion
    tiempo_generar_poblacion = fin_generar_poblacion - inicio_generar_poblacion
    return (ga, tiempo_crear_poblacion, tiempo_generar_poblacion)

def ga_to_state_dict(ga):
    fittest = ga.fittest_individual
    fittest_state = None
    if fittest is not None:
        fittest_state = {'genotype': list(fittest.genotype), 'fitness_error_score': float(fittest.fitness_error_score), 'solution': deepcopy(fittest.solution)}
    return {'fittest_individual': fittest_state, 'completed_generations': ga.completed_generations, 'stop_reason': ga.stop_reason, 'best_fitness_history': list(ga.best_fitness_history), 'best_fitness_evaluations': list(ga.best_fitness_evaluations), 'evaluations_count': ga.evaluations_count, 'engineering_rule_violations_total': ga.engineering_rule_violations_total, 'penalized_rule_violations_total': ga.penalized_rule_violations_total, 'first_generation_best_fitness': ga.first_generation_best_fitness, 'last_generation_best_fitness': ga.last_generation_best_fitness, 'best_fitness_error_score': ga.best_fitness_error_score, 'variant_name': ga.variant_name, 'objective_variant': objective_variant(ga.variant_name), 'algorithm_family': algorithm_family(ga.variant_name), 'gaussian_sigma_fraction': ga.gaussian_sigma_fraction, 'early_stopping_enabled': ga.early_stopping_enabled}
