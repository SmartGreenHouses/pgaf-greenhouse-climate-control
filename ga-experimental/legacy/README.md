# Hitos históricos seleccionados

Esta carpeta conserva únicamente snapshots con valor técnico para explicar la evolución del proyecto. No es una copia completa de todos los directorios históricos creados durante el desarrollo.

## Contenido

| Carpeta | Descripción |
|---|---|
| `original-prototype/` | Implementación monolítica inicial del algoritmo. |
| `modular-variants-2026-05-19/` | Primera separación de fitness, operadores y ejecución de variantes. |
| `phase-1-code-snapshot/` | Código contemporáneo a la primera tanda experimental. |
| `phase-2-code-snapshot/` | Código contemporáneo a la segunda tanda experimental y anterior a la consolidación final. |

El código activo de la raíz de `ga-experimental/` representa el estado experimental más reciente e incorpora los baselines, el presupuesto de evaluaciones y las guardas del protocolo por fases.

## Política de mantenimiento

- No añadas nuevas copias completas con fechas o sufijos.
- Usa commits y etiquetas de Git para conservar la evolución futura.
- No ejecutes directamente los snapshots sin revisar sus dependencias y rutas históricas.
- Las carpetas `__pycache__/` y los archivos `.pyc` no forman parte del histórico y deben eliminarse o ignorarse.
