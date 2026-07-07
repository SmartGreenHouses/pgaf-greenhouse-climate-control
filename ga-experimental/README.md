# GA experimental — desarrollo y análisis exploratorio

Esta carpeta documenta la etapa **exploratoria y de desarrollo metodológico** del algoritmo genético para control climático predictivo de invernadero.

Contiene la evolución del método, dos tandas experimentales, documentación técnica, resultados agregados y una selección de snapshots históricos que explican cómo se llegó al protocolo final de Sirius.

> Esta carpeta no representa la evidencia confirmatoria final. Sus resultados deben interpretarse como screening, diagnóstico y desarrollo de la metodología.

## Contenido

- Código experimental más reciente en la raíz.
- Nueve variantes del GA.
- Dos baselines añadidos en la refactorización posterior:
  - `RANDOM_SEARCH_BUDGET_MATCHED`;
  - `HOLD_LAST_ACTION`.
- Dos tandas de resultados exploratorios en [`results/`](results/).
- Documentación metodológica en [`docs/`](docs/).
- Cuatro hitos históricos seleccionados en [`legacy/`](legacy/).
- Documentación de transición al protocolo final en [`docs/transicion-sirius/`](docs/transicion-sirius/).

El dataset y el modelo se almacenan una sola vez en las carpetas compartidas `../datasets/` y `../models/` del repositorio raíz.

## Estructura

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── analyze_results.py
├── constants.py
├── genetic_algorithm_class.py
├── individual_class.py
├── run_experiment_grid.py
├── run_genetic_algorithm_experiments.py
├── smoke_test_ga_variants.py
├── utils_v.py
├── docs/
│   ├── cronologia.md
│   ├── evolucion-penalizaciones.md
│   ├── experimento-base-engineering-enhanced.md
│   ├── metodologia-analisis.md
│   ├── protocolo-experimental.md
│   ├── resultados-experimentales.md
│   ├── variantes.md
│   └── transicion-sirius/
├── results/
│   ├── phase-1/
│   └── phase-2/
└── legacy/
    ├── original-prototype/
    ├── modular-variants-2026-05-19/
    ├── phase-1-code-snapshot/
    └── phase-2-code-snapshot/
```

## Entorno

Se recomienda Python 3.10 o 3.11. Crea un entorno independiente del utilizado en `ga-sirius/`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Comprobación

```bash
python smoke_test_ga_variants.py
python run_experiment_grid.py --dry-run
```

El modo `--dry-run` valida el protocolo y muestra el coste computacional estimado sin ejecutar la campaña.

## Documentación

- [Protocolo experimental](docs/protocolo-experimental.md)
- [Variantes del algoritmo](docs/variantes.md)
- [Resultados de las dos tandas](docs/resultados-experimentales.md)
- [Metodología de análisis](docs/metodologia-analisis.md)
- [Evolución de las penalizaciones](docs/evolucion-penalizaciones.md)
- [Experimento Base Engineering Enhanced](docs/experimento-base-engineering-enhanced.md)
- [Cronología](docs/cronologia.md)
- [Transición al protocolo de Sirius](docs/transicion-sirius/README.md)

## Resultados exploratorios

[`results/`](results/) contiene una selección compacta y auditable:

- agregados a nivel de caso;
- informes de análisis;
- tablas estadísticas;
- perfiles de convergencia;
- frentes de Pareto;
- figuras finales.

No se incluyen los datos masivos a nivel de paso ni todas las carpetas individuales de ejecución.

### Fase 1

La primera tanda incluye 2,520 casos agregados y cinco variantes. Permitió detectar que las penalizaciones secundarias podían dominar el error climático y favorecer soluciones con una actuación insuficiente.

### Fase 2

La segunda tanda aplica el enfoque *target-first*, pesos revisados y nuevas variantes. Su agregado contiene 4,680 casos y nueve variantes. Parte del material original contenía ejecuciones repetidas creadas durante el trabajo local, por lo que esta fase se conserva como evidencia histórica y no confirmatoria.

Los dos baselines de la versión final no aparecen en estas tandas históricas porque se incorporaron durante la refactorización posterior.

## Histórico seleccionado

[`legacy/`](legacy/) no pretende almacenar todas las copias realizadas durante el proyecto. Solo conserva cuatro hitos con valor metodológico. Para cualquier cambio futuro debe utilizarse el historial de Git en lugar de crear nuevas copias manuales fechadas.

## Relación con la fase final

La documentación de [`docs/transicion-sirius/`](docs/transicion-sirius/) explica la separación entre screening, validación y confirmación, la comparación por presupuesto de evaluaciones y las guardas de reproducibilidad que se incorporaron antes de las campañas finales.

La implementación consolidada y los resultados confirmatorios se encuentran en [`../ga-sirius/`](../ga-sirius/).
