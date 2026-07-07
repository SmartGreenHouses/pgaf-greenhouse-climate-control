# Transición de la etapa experimental al protocolo de Sirius

Esta carpeta documenta la última revisión metodológica realizada antes de congelar las campañas finales.

Su objetivo es explicar cómo se pasó de una comparación exploratoria de variantes a un protocolo por fases con presupuestos comparables, conjuntos temporales separados, semillas nuevas y análisis confirmatorio.

## Documentos

1. [Enfoque final](enfoque-final.md): visión general del protocolo consolidado y del flujo de evaluación.
2. [Plan de experimentos](plan-experimentos.md): piloto, screening, tuning, validación y comparación confirmatoria.
3. [Evaluación y análisis](evaluacion.md): fitness, presupuestos, métricas externas y estadística pareada.
4. [Estado preliminar](estado-preliminar.md): evidencia disponible en la fase experimental y límites de interpretación.
5. [Auditoría del código](auditoria-codigo.md): responsabilidades de los módulos, verificaciones y cambios detectados.
6. [`resultados-preliminares-variantes.csv`](resultados-preliminares-variantes.csv): resumen numérico reutilizable de la campaña exploratoria anterior.

## Principios introducidos

- Separación temporal entre screening, validación y confirmación.
- Semillas confirmatorias no utilizadas durante el desarrollo.
- Hiperparámetros congelados antes de inspeccionar el conjunto confirmatorio.
- Comparación de métodos bajo presupuestos explícitos de evaluaciones.
- Baseline de búsqueda aleatoria con presupuesto equivalente.
- Baseline de mantener la última acción.
- Distinción entre fitness interno y métricas externas comparables.
- Manifiestos, hashes y validación estructural de las salidas.

## Alcance

Los resultados numéricos de esta carpeta proceden de la etapa exploratoria y sirven para justificar decisiones metodológicas. La implementación consolidada, las configuraciones congeladas y la evidencia final se encuentran en [`../../../ga-sirius/`](../../../ga-sirius/).
