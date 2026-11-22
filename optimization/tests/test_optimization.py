"""Pure scientific operator checks; no real LSTM or TensorFlow required."""
import importlib
import json
import random
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The production utils_v imports the historical .h5 model. These unit checks
# deliberately replace only that dependency; no assertion of real model fidelity.
if "utils_v" not in sys.modules:
    mocked_utils = types.ModuleType("utils_v")
    mocked_utils.prepare_forecast_data = lambda x: x
    mocked_utils.forecast_from_model_history = lambda *a, **k: None
    def batch_stub(model_history, prediction_columns, steps, last_row_updates_list):
        out = []
        for u in last_row_updates_list:
            out.append([17.0 + 0.15 * float(u["TemperaturaTuberia"]),
                        65.0 + 10.0 * float(u["Humidificacion"])])
        return np.asarray(out), prediction_columns
    mocked_utils.forecast_batch_from_model_history = batch_stub
    sys.modules["utils_v"] = mocked_utils

import genetic_algorithm_class as ga
from individual_class import Individual
from constants import POSSIBLE_GENES
from run_tuning_grid import build_cases, build_grid_configs

class TestSelectedGA(unittest.TestCase):
    def setUp(self):
        self.row = {name: (20.0 if name == "TemperaturaTuberia" else
                           0 if name == "Humidificacion" else 50.0)
                    for name in POSSIBLE_GENES}
        self.row.update({"CajaMedida1.TBS":18., "CajaMedida1.HR":70.,
                         "TemperaturaExterior":16., "HumedadExterior":85.,
                         "Radiacion":150.})

    def test_only_one_variant(self):
        from ga_method_registry import GA_VARIANTS, validate_method
        self.assertEqual(GA_VARIANTS, ("GA_ENGINEERING_AWARE",))
        with self.assertRaises(ValueError):
            validate_method("GA_BASE")

    def test_manuscript_tuning_grid(self):
        config = json.loads((ROOT/"tuning_grid_config.json").read_text())
        self.assertEqual(len(build_grid_configs(config)),36)
        self.assertEqual(len(build_cases(config)),540)
        self.assertEqual({c["GA_VARIANT"] for c in build_grid_configs(config)},
                         {"GA_ENGINEERING_AWARE"})

    def test_selected_manuscript_settings(self):
        c = json.loads((ROOT/"selected_ga_config.json").read_text())
        self.assertEqual((c["POPULATION_NUM"],c["GENERATIONS_NUM"],c["K_TOURNAMENT"]),(200,50,2))
        self.assertEqual(c["GAUSSIAN_SIGMA_FRACTION"],0.04)
        self.assertFalse(c["EARLY_STOPPING_ENABLED"])
        self.assertEqual(sum(c["MUTATION_PROBABILITIES"].values()),1.0)

    def test_fitness_is_finite_and_uses_engineering_rules(self):
        random.seed(23); np.random.seed(23)
        item = Individual(last_row=self.row)
        item.solution={"tbs":20.0,"hr":81.0}
        score, violations = ga.score_individual_for_variant(
            item,"GA_ENGINEERING_AWARE",18.,70.,20.,80.,1.,self.row,
            energy_proxy_weights={"pipe":5.0,"humidification":2.0,"openings":0.3})
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(violations,0)

    def test_reproducible_genetic_operators_and_actuator_ranges(self):
        mutations={"no_mutation":.8,"mutation_1":.15,"mutation_2":.04,"mutation_3":.01}
        out=[]
        for repeat in range(2):
            random.seed(101);np.random.seed(101)
            a=ga.GA(3,20,2,variant_name="GA_ENGINEERING_AWARE",
                    gaussian_sigma_fraction=.04,early_stopping_enabled=False)
            a.do_start_population(self.row)
            for i,item in enumerate(a.population):item.fitness_error_score=float(i)
            population=a.do_mating(a.do_selection(),mutations,self.row)
            genotypes=[list(item.genotype) for item in population]
            for genotype in genotypes:
                for key,value in zip(POSSIBLE_GENES, genotype):
                    spec=POSSIBLE_GENES[key]
                    self.assertTrue(spec["min"]<=value<=spec["max"])
                    self.assertLessEqual(abs(float(value)-float(self.row[key])),spec["relative_15m"]+1e-8)
            out.append(genotypes)
        self.assertEqual(out[0],out[1])

    def test_published_tuning_results_match_selected_configuration(self):
        import pandas as pd
        from aggregate_tuning_grid_results import rank_by_min_mean_fitness
        observed = pd.read_csv(ROOT / "results" / "tuning_results.csv")
        self.assertEqual(len(observed), 36)
        self.assertTrue((observed["n_cases"] == 15).all())
        self.assertTrue((observed["n_step_records"] == 180).all())
        selected = observed.loc[observed["selected_for_manuscript"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["config_id"], "pop200_gen50_k2_mutconservative_sigma0p04")
        self.assertAlmostEqual(float(selected.iloc[0]["mean_step_best_fitness"]), 0.09259039864241379)
        recomputed = rank_by_min_mean_fitness(observed.drop(columns=["rank_by_mean_fitness", "selected_for_manuscript"]))
        self.assertEqual(recomputed.iloc[0]["config_id"], selected.iloc[0]["config_id"])

    def test_model_and_dataset_checksums_if_present(self):
        import hashlib
        artifacts = {
            ROOT / "models" / "LSTM_L48_H1.h5": "49869b812ef659e8cecacd9e0e2789ed17e265642cae9dbbdc7bc182858ada68",
            ROOT / "datasets" / "ETIFA_clean_imputed_1h.csv": "6ab084f5d45a4613eaba81d46a165788ea2765f577526b530a816ceccbf08b9b",
        }
        if not all(path.is_file() for path in artifacts):
            self.skipTest("Place the historical GA dataset and model in optimization/ to verify their SHA-256 hashes")
        for path, expected in artifacts.items():
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
            self.assertEqual(digest.hexdigest(), expected, f"Incorrect asset: {path}")

    def test_three_generations_run_budget(self):
        random.seed(37);np.random.seed(37)
        # Pure synthetic forecast surrogate, not the published .h5 model.
        context=(object(),["CajaMedida1.TBS","CajaMedida1.HR"])
        alg,_,_=ga.run_genetic_algorithm(
            generations_num=3,population_num=20,k_tournament=2,
            last_row=self.row,case_data=context,last_tbs=18.,last_hr=70.,
            desired_tbs_output=20.,desired_hr_output=80.,fitness_percentage_goal=1.,
            patience=10,stop_ga_thr=.001,stop_ga_from=-1.,
            mutation_probabilities={"no_mutation":.8,"mutation_1":.15,
                                    "mutation_2":.04,"mutation_3":.01},
            variant_name="GA_ENGINEERING_AWARE",
            energy_proxy_weights={"pipe":5.0,"humidification":2.0,"openings":.3},
            gaussian_sigma_fraction=.04,early_stopping_enabled=False)
        self.assertEqual(alg.evaluations_count,60)
        self.assertEqual(alg.completed_generations,3)

if __name__ == '__main__': unittest.main()
