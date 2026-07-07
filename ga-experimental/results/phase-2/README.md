# Fase experimental 2

Segunda tanda exploratoria ejecutada después de revisar el fitness y las penalizaciones.

## Resumen

- **Casos agregados:** 4,680.
- **Variantes presentes:** 9.
- **Cambio principal:** esquema *target-first* con pesos revisados.
- **Carácter:** exploratorio.

Esta fase amplió las variantes evaluadas y priorizó el progreso climático antes de aplicar las penalizaciones secundarias.

## Contenido

- `aggregates/case_summary_all.csv`: agregado principal a nivel de caso.
- `analysis_report.md`: informe completo de análisis.
- `plots/`: figuras de distribución, convergencia, perfiles, trayectoria y Pareto.
- `stats/`: contrastes y análisis de sensibilidad.
- `tables/`: resúmenes, rankings, perfiles y trayectoria de la mejor configuración.

## Código asociado

El snapshot contemporáneo se conserva en:

```text
../../legacy/phase-2-code-snapshot/
```

## Advertencia sobre el histórico

El material original de esta tanda contenía ejecuciones repetidas con sufijos creados durante la gestión local de archivos. Por este motivo, el agregado se conserva como evidencia del proceso de desarrollo y no como campaña confirmatoria final.

Los baselines `RANDOM_SEARCH_BUDGET_MATCHED` y `HOLD_LAST_ACTION` se incorporaron después y no forman parte de esta tanda histórica.
