# Dataset del invernadero

Esta carpeta contiene el dataset compartido por las etapas experimental y final del proyecto.

## Archivo

```text
DIRTY-DS-15-SUMMER.csv
```

| Propiedad | Valor |
|---|---|
| Filas | 86,540 |
| Columnas | 26 |
| Tamaño | 24,423,700 bytes |
| Periodo observado | 2018-12-18 13:00 UTC – 2021-06-06 23:45 UTC |
| Cadencia nominal | 15 minutos |
| Columna temporal | `Fecha` |
| SHA-256 | `6ab084f5d45a4613eaba81d46a165788ea2765f577526b530a816ceccbf08b9b` |

> El nombre `DIRTY-DS-15-SUMMER.csv` se conserva porque es el identificador utilizado por el código y por los experimentos históricos. No implica que el archivo deba editarse directamente.

## Variables

### Clima interior

- `CajaMedida1.TBS`
- `CajaMedida1.TBH`
- `CajaMedida1.HR`
- `CajaMedida1.DH`
- `LecturaCO2`

### Meteorología exterior

- `TemperaturaExterior`
- `HumedadExterior`
- `VelocidadViento`
- `DireccionViento`
- `Radiacion`
- `RadiacionAcumulada`
- `AlarmaLluvia`
- `AlarmaTormenta`

### Consignas y calefacción

- `TemperaturaTuberia`
- `TemperaturaVentilacion`
- `TemperaturaCalefaccion`
- `TemperaturaCalculadaTuberia`

### Actuadores

- `Humidificacion`
- `Destratificacion`
- `PosicionVentana1`
- `PosicionVentana2`
- `PosicionPantalla1`
- `PosicionPantalla2`
- `HileraIluminacion1`
- `HileraIluminacion2`

Para ventanas y pantallas, `0 %` representa cierre completo y `100 %` apertura completa.

## Variables empleadas por el modelo

El pipeline utiliza 15 variables originales:

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

A estas variables se añaden dos codificaciones cíclicas del mes (`month_sin` y `month_cos`), generando las 17 entradas utilizadas por la LSTM.

## Integridad

Verificación en Linux o macOS:

```bash
shasum -a 256 DIRTY-DS-15-SUMMER.csv
```

Verificación en PowerShell:

```powershell
Get-FileHash DIRTY-DS-15-SUMMER.csv -Algorithm SHA256
```

El resultado esperado es:

```text
6ab084f5d45a4613eaba81d46a165788ea2765f577526b530a816ceccbf08b9b
```

## Uso y conservación

- Mantén este archivo como copia inmutable de referencia.
- Realiza cualquier limpieza o transformación sobre una copia nueva.
- Documenta toda transformación que afecte a la selección temporal, normalización o variables.
- No mezcles versiones distintas del dataset bajo el mismo nombre.
- Registra el checksum utilizado en cada campaña experimental.

## Procedencia, licencia y citación

Completa antes de publicar:

- **Procedencia:** pendiente.
- **Autores o entidad responsable:** pendiente.
- **Licencia de los datos:** pendiente.
- **DOI o URL persistente:** pendiente.
- **Referencia bibliográfica:** pendiente.

La presencia del archivo en este repositorio no sustituye la declaración explícita de sus derechos de uso y redistribución.
