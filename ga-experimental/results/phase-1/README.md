# Fase experimental 1

Primera tanda de comparación de variantes del algoritmo genético.

## Resumen

- **Casos agregados:** 2,520.
- **Variantes presentes:** 5.
- **Finalidad:** comparación inicial de estrategias y diagnóstico del fitness.
- **Carácter:** exploratorio.

Esta fase permitió detectar que las penalizaciones secundarias podían dominar el error climático. En determinadas configuraciones, el algoritmo podía obtener una puntuación favorable reduciendo la actuación, aunque el acercamiento a las consignas no fuese el deseado.

## Contenido

- `aggregates/case_summary_all.csv`: agregado principal a nivel de caso.
- `analysis_report.md`: informe completo de análisis.
- `plots/`: figuras de distribución, convergencia, perfiles y compromisos multiobjetivo.
- `stats/`: contrastes y análisis de sensibilidad.
- `tables/`: resúmenes compactos, rankings y frentes de Pareto.

## Código asociado

El snapshot contemporáneo a esta tanda se conserva en:

```text
../../legacy/phase-1-code-snapshot/
```

## Limitación

Los resultados corresponden a una fase de desarrollo. No deben utilizarse como evidencia confirmatoria ni combinarse directamente con los resultados finales de Sirius sin considerar las diferencias de fitness, variantes y protocolo.
