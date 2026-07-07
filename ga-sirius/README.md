# GA Sirius — protocolo final y campañas confirmatorias

Esta carpeta contiene la implementación consolidada del sistema de control predictivo, el protocolo reproducible por fases, los scripts SLURM y los resultados compactos de las campañas ejecutadas en Sirius.

> Esta es la etapa final del estudio. La evolución previa, los prototipos y las tandas exploratorias se conservan en [`../ga-experimental/`](../ga-experimental/).

## Contenido

- Nueve variantes del algoritmo genético.
- `RANDOM_SEARCH_BUDGET_MATCHED`, con el mismo presupuesto de candidatos.
- `HOLD_LAST_ACTION`, como política de referencia.
- Separación entre screening, validación y confirmación.
- Configuraciones de campañas reproducibles.
- Pruebas sintéticas y de integración.
- Scripts para job arrays de SLURM.
- Resultados de screening, tuning, validación y comparación final.
- Manifiestos de estudio y archivos de estado.

El dataset y el modelo se almacenan en las carpetas compartidas `../datasets/` y `../models/` del repositorio raíz.

## Estructura

```text
.
├── README.md
├── CHANGELOG.md
├── .gitignore
├── requirements.txt
├── requirements-lock.txt
├── analyze_results.py
├── configure_final_confirmatory.py
├── configure_remaining_campaign.py
├── configure_tuning_campaign.py
├── configure_variant_screening.py
├── constants.py
├── ga_method_registry.py
├── genetic_algorithm_class.py
├── individual_class.py
├── integration_test_pipeline.py
├── run_experiment_grid.py
├── run_genetic_algorithm_experiments.py
├── smoke_test_ga_variants.py
├── utils_v.py
├── docs/
│   ├── FINAL_MODEL_SELECTION.txt
│   ├── GRID_CONFIGURATIONS.md
│   ├── README_PROTOCOL.md
│   └── SIRIUS_LAUNCH_GUIDE.md
├── scripts/
│   ├── aggregate_ga_results.sbatch
│   ├── check_sirius_gpu.sbatch
│   ├── run_ga_array.sbatch
│   ├── setup_sirius_env.sh
│   ├── submit_ga_campaign.sh
│   └── submit_preflight.sh
└── out/
    ├── screening/
    ├── validation/
    └── confirmatory/
```

## Métodos evaluados

### Variantes GA

- `GA_BASE`
- `GA_BASE_OPERATOR_ENHANCED`
- `GA_BASE_ENGINEERING`
- `GA_BASE_ENGINEERING_ENHANCED`
- `GA_BASE_ENGINEERING_PROGRESS`
- `GA_ACTUATOR_AWARE`
- `GA_ENERGY_AWARE`
- `GA_OPERATOR_ENHANCED`
- `GA_ENGINEERING_AWARE`

### Baselines

- `RANDOM_SEARCH_BUDGET_MATCHED`
- `HOLD_LAST_ACTION`

Los baselines se puntúan con el objetivo de referencia registrado para facilitar comparaciones controladas. Las comparaciones entre funciones objetivo distintas deben basarse en métricas externas y no exclusivamente en el fitness interno.

## Entorno

El entorno verificado se encuentra bloqueado en `requirements-lock.txt`. Se recomienda Python 3.10.

### Instalación local

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

### Instalación en Sirius

```bash
GA_PYTHON_VERSION=3.10 bash scripts/setup_sirius_env.sh
```

El script crea o reutiliza el entorno Conda, instala las dependencias y genera un inventario local en `requirements-sirius-resolved.txt`.

## Validación previa

Antes de lanzar una campaña:

```bash
python smoke_test_ga_variants.py
python integration_test_pipeline.py
python run_experiment_grid.py --dry-run
python run_experiment_grid.py --config-count
```

- `smoke_test_ga_variants.py` valida el registro y el comportamiento esencial de los once métodos.
- `integration_test_pipeline.py` recorre el pipeline sintético de doce pasos y valida las salidas CSV.
- `--dry-run` comprueba el protocolo y calcula su coste.
- `--config-count` muestra el número efectivo de configuraciones.

Estas pruebas no sustituyen una integración real con el CSV y el modelo `.h5`.

## Protocolo por fases

1. **Piloto técnico:** comprueba carga del modelo, recursos y escritura de resultados.
2. **Variant screening:** compara los once métodos con una configuración común.
3. **Tuning:** ajusta los finalistas manteniendo el mismo presupuesto de 10,000 evaluaciones.
4. **Validation:** selecciona la configuración final en un episodio temporal distinto.
5. **Final confirmatory:** compara los once métodos con hiperparámetros, métricas, fechas y semillas congelados.

Consulta:

- [Protocolo reproducible](docs/README_PROTOCOL.md)
- [Configuraciones definitivas](docs/GRID_CONFIGURATIONS.md)
- [Selección final](docs/FINAL_MODEL_SELECTION.txt)

## Ejecución local

El grid completo puede ejecutarse desde esta carpeta:

```bash
python run_experiment_grid.py --dry-run
python run_experiment_grid.py
```

Opciones útiles:

```bash
python run_experiment_grid.py --show-config 0
python run_experiment_grid.py --config-index 0
python run_experiment_grid.py --aggregate-only
```

No utilices `--confirm-large-run` sin haber inspeccionado previamente el coste mostrado por `--dry-run`.

## Ejecución en Sirius

La guía completa se encuentra en [docs/SIRIUS_LAUNCH_GUIDE.md](docs/SIRIUS_LAUNCH_GUIDE.md).

Flujo resumido:

```bash
bash scripts/submit_preflight.sh
MAX_PARALLEL=2 bash scripts/submit_ga_campaign.sh
```

El envío crea un job array con una configuración por índice y un trabajo dependiente que valida y agrega las salidas.

Para reanudar índices concretos:

```bash
ARRAY_SPEC="3,7,12%2" bash scripts/submit_ga_campaign.sh
```

## Resultados

La carpeta `out/` está organizada por fase y campaña:

```text
out/
├── screening/
│   ├── variant_screening/
│   ├── tuning_200x50/
│   ├── tuning_100x100/
│   └── tuning_50x200/
├── validation/
│   └── model_validation/
└── confirmatory/
    └── final_confirmatory/
```

Los archivos compactos relevantes incluyen:

- `config.json`;
- `status.json`;
- `case_summary.csv`;
- `case_summary_all.csv`;
- `study_manifest.json`;
- `aggregation_status.json`.

Los archivos masivos `steps_flat.csv` y `steps_flat_all.csv`, los logs y los metadatos de SLURM no se versionan en el repositorio público.

## Análisis confirmatorio

Después de completar la fase confirmatoria:

```bash
python analyze_results.py \
  --input out/confirmatory/final_confirmatory/case_summary_all.csv
```

El analizador comprueba que los datos pertenezcan a la fase confirmatoria y que la configuración esté congelada. Después aplica análisis pareados, tamaños de efecto, intervalos bootstrap y corrección de Holm según el protocolo implementado.

## Reproducibilidad

Cada campaña genera un `study_manifest.json` con hashes, entorno, métodos y configuración. Conserva también:

- el commit exacto de Git;
- `requirements-lock.txt`;
- las semillas y ventanas temporales;
- el estado de cada configuración;
- el resumen agregado;
- los logs y metadatos de SLURM en el archivo privado de ejecución.

## Cambios respecto a la etapa experimental

Consulta [`CHANGELOG.md`](CHANGELOG.md) para ver las correcciones de comparabilidad, presupuestos, convergencia, trazabilidad, pruebas y análisis introducidas en la versión orientada a publicación.
