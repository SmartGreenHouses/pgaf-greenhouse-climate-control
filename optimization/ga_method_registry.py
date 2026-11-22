"""Minimal registry: this project intentionally exposes only the selected GA."""

import math

GA_VARIANTS = ("GA_ENGINEERING_AWARE",)
BASELINE_METHODS = ()
METHODS = GA_VARIANTS
VARIANTS = list(METHODS)
REFERENCE_OBJECTIVE_VARIANT = "GA_ENGINEERING_AWARE"


def validate_method(method_name):
    if method_name not in METHODS:
        raise ValueError(
            f"Unknown method {method_name!r}. This simplified project only supports "
            "GA_ENGINEERING_AWARE."
        )


def objective_variant(method_name):
    validate_method(method_name)
    return "GA_ENGINEERING_AWARE"


def algorithm_family(method_name):
    validate_method(method_name)
    return "genetic_algorithm"


def uses_random_search(method_name):
    validate_method(method_name)
    return False


def uses_hold_last_action(method_name):
    validate_method(method_name)
    return False


def uses_actuator_effort_penalty(method_name):
    validate_method(method_name)
    return True


def uses_enhanced_operators(method_name):
    validate_method(method_name)
    return True


def validate_energy_proxy_weights(weights):
    required = {"pipe", "humidification", "openings"}
    if weights is None:
        return
    if set(weights) != required:
        raise ValueError(
            f"energy_proxy_weights must contain exactly {sorted(required)}, got {sorted(weights)}"
        )
    total = 0.0
    for key, value in weights.items():
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid energy proxy weight {key}={value}")
        total += value
    if total <= 0:
        raise ValueError("At least one energy proxy weight must be greater than zero")
