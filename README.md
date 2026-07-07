# Control predictivo de invernadero mediante algoritmos genéticos

Repositorio de investigación para el desarrollo, evaluación y validación de un sistema de control predictivo de invernadero basado en un **algoritmo genético (GA)** y un modelo predictivo **LSTM**.

El repositorio reúne dos etapas del estudio sin duplicar los recursos comunes:

- [`ga-experimental/`](ga-experimental/): desarrollo metodológico, evolución del algoritmo y resultados exploratorios.
- [`ga-sirius/`](ga-sirius/): protocolo consolidado, validación y campañas finales ejecutadas en Sirius.
- [`datasets/`](datasets/): dataset compartido por ambas etapas.
- [`models/`](models/): modelo LSTM compartido por ambas etapas.

> Los resultados de `ga-experimental/` son exploratorios. Las conclusiones finales deben basarse en el protocolo y los resultados de `ga-sirius/`.

## Estructura

```text
.
├── README.md
├── .gitignore
├── LICENSE                         # pendiente de seleccionar
├── datasets/
│   ├── README.md
│   └── DIRTY-DS-15-SUMMER.csv
├── models/
│   ├── README.md
│   └── LSTM_CLEAN-DS-15-SUMMER_best_TBS.h5
├── ga-experimental/
│   ├── README.md
│   ├── requirements.txt
│   ├── docs/
│   ├── legacy/
│   ├── results/
│   └── *.py
└── ga-sirius/
    ├── README.md
    ├── CHANGELOG.md
    ├── requirements.txt
    ├── requirements-lock.txt
    ├── docs/
    ├── scripts/
    ├── out/
    └── *.py
```

## Descripción del sistema

En cada instante de control, el sistema evalúa posibles acciones sobre los actuadores del invernadero mediante el siguiente proceso:

1. Se construye una ventana histórica de **672 observaciones de 15 minutos**, equivalente a siete días.
2. Cada individuo del GA representa una posible configuración de los actuadores en el instante actual.
3. La acción candidata sustituye los valores de actuación del último instante de la ventana.
4. La LSTM predice el estado del invernadero en el instante siguiente.
5. La función de fitness evalúa el error climático y, según la variante, el esfuerzo de actuación, el consumo energético aproximado, las reglas agronómicas o el progreso hacia la consigna.
6. El individuo con mejor fitness se utiliza como acción recomendada.

Las consignas diurnas y nocturnas, los límites de normalización, los límites físicos de los actuadores y sus variaciones máximas por intervalo fueron definidos con apoyo de expertos agrónomos de **NUTRICONTROL**.

## Etapas del estudio

```text
Prototipo inicial
      ↓
Diseño de variantes y operadores
      ↓
Experimentos exploratorios
      ↓
Revisión de fitness y penalizaciones
      ↓
Screening y ajuste de hiperparámetros
      ↓
Congelación del protocolo
      ↓
Validación del pipeline
      ↓
Campañas finales en Sirius
      ↓
Análisis confirmatorio
```

### Etapa experimental

[`ga-experimental/`](ga-experimental/) documenta cómo evolucionó el método. Incluye:

- el código experimental más reciente;
- dos tandas de resultados exploratorios;
- documentación metodológica;
- figuras y tablas agregadas;
- cuatro snapshots históricos seleccionados;
- la transición hacia el protocolo final.

### Etapa final en Sirius

[`ga-sirius/`](ga-sirius/) contiene:

- las once estrategias evaluadas;
- el protocolo por fases;
- las configuraciones finales;
- pruebas sintéticas e integración;
- scripts SLURM para Sirius;
- screening, tuning, validación y comparación confirmatoria;
- manifiestos y resultados agregados.

## Recursos compartidos

| Recurso | Archivo | Tamaño | SHA-256 |
|---|---|---:|---|
| Dataset | `datasets/DIRTY-DS-15-SUMMER.csv` | 24,423,700 bytes | `6ab084f5d45a4613eaba81d46a165788ea2765f577526b530a816ceccbf08b9b` |
| Modelo | `models/LSTM_CLEAN-DS-15-SUMMER_best_TBS.h5` | 333,456 bytes | `4f06c1b9b7dad4bb6d974f1ef67f6e356ad4bdaec795b91d1a96e58e2c9fa42b` |

Consulta [`datasets/README.md`](datasets/README.md) y [`models/README.md`](models/README.md) para ver su descripción técnica.

## Entornos de ejecución

Las dos etapas usan dependencias diferentes. Se recomienda crear **un entorno independiente para cada subproyecto**.

### Entorno experimental

```bash
cd ga-experimental
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

En PowerShell:

```powershell
cd ga-experimental
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Comprobación:

```bash
python smoke_test_ga_variants.py
python run_experiment_grid.py --dry-run
```

### Entorno final de Sirius

```bash
cd ga-sirius
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Comprobación:

```bash
python smoke_test_ga_variants.py
python integration_test_pipeline.py
python run_experiment_grid.py --dry-run
```

En Sirius, utiliza el instalador y los scripts SLURM descritos en [`ga-sirius/docs/SIRIUS_LAUNCH_GUIDE.md`](ga-sirius/docs/SIRIUS_LAUNCH_GUIDE.md).

## Resultados incluidos

El repositorio conserva únicamente información útil para interpretar y auditar el estudio:

- agregados a nivel de caso;
- tablas estadísticas;
- figuras finales;
- configuraciones y estados compactos;
- manifiestos de reproducibilidad;
- documentación metodológica.

Se excluyen deliberadamente:

- archivos masivos a nivel de paso, como `steps_flat.csv` y `steps_flat_all.csv`;
- logs de SLURM;
- metadatos temporales de ejecución;
- cachés de Python;
- entornos virtuales;
- copias manuales redundantes;
- archivos generados por el sistema operativo.

Los resultados brutos completos pueden publicarse por separado mediante una release o un repositorio científico con DOI.

## Reproducibilidad

Cada campaña final debe asociarse, como mínimo, con:

- el commit exacto de Git;
- el manifiesto de estudio;
- la configuración experimental;
- las semillas aleatorias;
- los episodios temporales;
- el checksum del dataset y del modelo;
- el entorno bloqueado;
- el presupuesto de evaluaciones;
- los criterios de parada;
- las métricas primarias y secundarias;
- los recursos de cómputo utilizados.

No deben modificarse los hiperparámetros, las fechas, las semillas ni las métricas principales después de inspeccionar los resultados confirmatorios.

## Documentación

- [Etapa experimental](ga-experimental/README.md)
- [Resultados exploratorios](ga-experimental/results/README.md)
- [Transición al protocolo Sirius](ga-experimental/docs/transicion-sirius/README.md)
- [Etapa final en Sirius](ga-sirius/README.md)
- [Protocolo reproducible](ga-sirius/docs/README_PROTOCOL.md)
- [Configuraciones de campañas](ga-sirius/docs/GRID_CONFIGURATIONS.md)
- [Guía de ejecución en Sirius](ga-sirius/docs/SIRIUS_LAUNCH_GUIDE.md)

## Citación

La referencia del artículo y del dataset debe añadirse cuando dispongan de información bibliográfica definitiva.

```bibtex
@article{pendiente,
  title   = {Pendiente},
  author  = {Pendiente},
  journal = {Pendiente},
  year    = {Pendiente},
  doi     = {Pendiente}
}
```

## Licencia

Antes de hacer público el repositorio, añade un archivo `LICENSE` y verifica por separado las condiciones de distribución del código, el dataset y el modelo. Mientras no exista una licencia explícita, este repositorio no concede permisos específicos de reutilización o redistribución.

## Reconocimientos

Este trabajo emplea conocimiento experto de NUTRICONTROL para definir consignas, normalización y restricciones de actuación. Las campañas finales se ejecutaron en la infraestructura de computación **Sirius**.
