# Resultados de las campañas de Sirius

Esta carpeta contiene los resultados compactos de las campañas finales.

## Estructura

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

## Archivos conservados

- `config.json`: configuración efectiva de una ejecución.
- `status.json`: estado y metadatos compactos.
- `case_summary.csv`: resumen por caso de una configuración.
- `case_summary_all.csv`: agregado validado de una campaña.
- `study_manifest.json`: código, dependencias, protocolo y hashes.
- `aggregation_status.json`: estado de la agregación y posibles índices ausentes.

## Archivos no versionados

- `steps_flat.csv`;
- `steps_flat_all.csv`;
- logs `.out` y `.err`;
- `slurm_metadata/`;
- cachés y archivos temporales.

Los resultados confirmatorios no deben regenerarse con cambios de hiperparámetros, fechas, semillas o métricas bajo la misma etiqueta de campaña. Cualquier modificación del protocolo requiere una nueva carpeta, un nuevo manifiesto y una nueva identificación de versión.
