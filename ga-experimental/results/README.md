# Resultados experimentales conservados

Esta carpeta contiene una selección auditable de las dos tandas exploratorias anteriores a la campaña final de Sirius.

## Organización

```text
results/
├── README.md
├── phase-1/
│   ├── README.md
│   ├── aggregates/
│   ├── analysis_report.md
│   ├── plots/
│   ├── stats/
│   └── tables/
└── phase-2/
    ├── README.md
    ├── aggregates/
    ├── analysis_report.md
    ├── plots/
    ├── stats/
    └── tables/
```

## Material incluido

- `case_summary_all.csv` de cada tanda;
- informe principal generado por el analizador;
- tablas de variantes, convergencia, objetivos y sensibilidad;
- contrastes estadísticos compactos;
- figuras de distribución, convergencia, perfiles temporales y Pareto.

## Material excluido

- `steps_flat.csv` y `steps_flat_all.csv`;
- datos completos a nivel de paso;
- carpetas individuales de cada configuración;
- análisis duplicados de prueba;
- comparaciones auxiliares masivas;
- archivos temporales y cachés.

## Interpretación científica

Estos resultados son **exploratorios**. Se utilizaron para:

- detectar problemas en la escala del fitness;
- estudiar el efecto de penalizaciones y operadores;
- formular hipótesis;
- seleccionar candidatos para el protocolo final.

No deben mezclarse con la evidencia confirmatoria de [`../../ga-sirius/`](../../ga-sirius/).
