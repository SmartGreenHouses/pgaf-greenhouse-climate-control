# Modelo LSTM del invernadero

Esta carpeta contiene el modelo predictivo compartido por las etapas experimental y final.

## Archivo

```text
LSTM_CLEAN-DS-15-SUMMER_best_TBS.h5
```

| Propiedad | Valor |
|---|---|
| Formato | Keras HDF5 (`.h5`) |
| Tamaño | 333,456 bytes |
| Backend guardado | TensorFlow |
| Versión Keras registrada | 2.4.0 |
| SHA-256 | `4f06c1b9b7dad4bb6d974f1ef67f6e356ad4bdaec795b91d1a96e58e2c9fa42b` |

## Arquitectura almacenada

```text
Input:  (672, 17)
LSTM:   70 unidades, activación tanh
Dense:  9 salidas, activación lineal
Loss:   mean squared error (MSE)
Optimizador guardado: Adam, learning rate 1e-4
```

Las 672 observaciones corresponden a siete días con una frecuencia de 15 minutos.

## Entradas

El modelo recibe 17 características:

- 15 variables climáticas, meteorológicas y de actuación;
- `month_sin`;
- `month_cos`.

Las 15 variables originales son:

```text
CajaMedida1.TBS
CajaMedida1.TBH
CajaMedida1.HR
CajaMedida1.DH
TemperaturaExterior
HumedadExterior
VelocidadViento
DireccionViento
Radiacion
TemperaturaTuberia
Humidificacion
PosicionVentana1
PosicionVentana2
PosicionPantalla1
PosicionPantalla2
```

## Salidas

El pipeline interpreta nueve predicciones continuas:

```text
CajaMedida1.TBS
CajaMedida1.TBH
CajaMedida1.HR
CajaMedida1.DH
TemperaturaExterior
HumedadExterior
VelocidadViento
DireccionViento
Radiacion
```

Para el control, las variables de actuación del último instante se sustituyen por cada individuo candidato y se realiza una predicción a un paso, correspondiente a `t+1`.

## Compatibilidad

El archivo procede de una versión antigua de Keras y utiliza el formato HDF5 legado. La fase experimental emplea TensorFlow/Keras 2.13, mientras que la fase Sirius carga el modelo mediante `tf_keras` dentro del entorno bloqueado.

No sobrescribas el archivo después de cargarlo o convertirlo. Si necesitas migrarlo a otro formato, conserva este original y registra por separado:

- herramienta y versión usadas;
- nuevo formato;
- checksum del archivo generado;
- comprobación de equivalencia de las predicciones.

## Integridad

Verificación en Linux o macOS:

```bash
shasum -a 256 LSTM_CLEAN-DS-15-SUMMER_best_TBS.h5
```

Verificación en PowerShell:

```powershell
Get-FileHash LSTM_CLEAN-DS-15-SUMMER_best_TBS.h5 -Algorithm SHA256
```

Resultado esperado:

```text
4f06c1b9b7dad4bb6d974f1ef67f6e356ad4bdaec795b91d1a96e58e2c9fa42b
```

## Procedencia, entrenamiento y licencia

Completa antes de publicar:

- **Autores o entidad responsable:** pendiente.
- **Datos exactos de entrenamiento:** pendiente.
- **División de entrenamiento, validación y prueba:** pendiente.
- **Métricas de evaluación:** pendiente.
- **Licencia del modelo:** pendiente.
- **Referencia o DOI:** pendiente.

La inclusión del modelo en el repositorio no sustituye una licencia explícita para su reutilización o redistribución.
