import math
import random

from .constants import POSSIBLE_GENES

POSSIBLE_GENE_KEYS = tuple(POSSIBLE_GENES)


def _local_gene_bounds(gene_key, last_row):
    gene = POSSIBLE_GENES[gene_key]
    global_min = float(gene["min"])
    global_max = float(gene["max"])

    if last_row is None:
        return global_min, global_max

    if gene_key not in last_row:
        raise KeyError(f"Missing actuator value in last_row: {gene_key}")
    last_value = float(last_row[gene_key])
    if not math.isfinite(last_value):
        raise ValueError(f"Non-finite actuator value for {gene_key}: {last_value}")

    relative_change = float(gene["relative_15m"])
    if relative_change < 0:
        raise ValueError(f"relative_15m cannot be negative for {gene_key}")

    local_min = max(last_value - relative_change, global_min)
    local_max = min(last_value + relative_change, global_max)
    if local_min > local_max:
        raise ValueError(
            f"Current value for {gene_key} is incompatible with configured bounds: "
            f"last={last_value}, global=[{global_min}, {global_max}], "
            f"relative_15m={relative_change}"
        )
    return local_min, local_max


class Individual:
    def __init__(self, genotype=None, last_row=None):
        if genotype is None:
            if last_row is None:
                raise ValueError("last_row is required when creating a random individual")

            values = []
            for key, gene in POSSIBLE_GENES.items():
                min_v, max_v = _local_gene_bounds(key, last_row)
                if gene["min"] == 0 and gene["max"] == 1:
                    values.append(random.randint(0, 1))
                else:
                    values.append(random.uniform(min_v, max_v))
            self.genotype = values
        else:
            if len(genotype) != len(POSSIBLE_GENES):
                raise ValueError(
                    f"Invalid genotype length: got {len(genotype)}, expected {len(POSSIBLE_GENES)}"
                )

            values = []
            for gene_i, (gene_key, gene) in enumerate(POSSIBLE_GENES.items()):
                value = float(genotype[gene_i])
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite genotype value for {gene_key}: {value}")

                min_v, max_v = _local_gene_bounds(gene_key, last_row)
                if value < min_v - 1e-12 or value > max_v + 1e-12:
                    raise ValueError(
                        f"Genotype value for {gene_key} is outside allowed bounds: "
                        f"{value} not in [{min_v}, {max_v}]"
                    )

                if gene["min"] == 0 and gene["max"] == 1:
                    if value not in (0.0, 1.0):
                        raise ValueError(f"Binary gene {gene_key} must be 0 or 1, got {value}")
                    values.append(int(value))
                else:
                    values.append(value)
            self.genotype = values

        self.fitness_error_score = None
        self.solution = None

    def mutate_gene(self, gene_i, last_row):
        if not 0 <= gene_i < len(POSSIBLE_GENE_KEYS):
            raise IndexError(f"Invalid gene index: {gene_i}")

        gene_key = POSSIBLE_GENE_KEYS[gene_i]
        gene = POSSIBLE_GENES[gene_key]
        min_v, max_v = _local_gene_bounds(gene_key, last_row)

        if gene["min"] == 0 and gene["max"] == 1:
            self.genotype[gene_i] = random.randint(0, 1)
        else:
            self.genotype[gene_i] = random.uniform(min_v, max_v)

    def __str__(self):
        return (
            f"Genotype: {self.genotype}, "
            f"Fitness error score: {self.fitness_error_score}, "
            f"Solution: {self.solution}"
        )
