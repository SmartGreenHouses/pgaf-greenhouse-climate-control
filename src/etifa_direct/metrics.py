"""Métricas de predicción directa multisalida ETIFA.

- Métricas físicas en unidades originales.
- Dirección del viento evaluada con distancia angular mínima.
- nMAE global: media de los 10 nMAE, usando IQR de Train.
- Métricas agregadas y por paso futuro.
"""

import numpy as np

from .data import SENSORS


WIND_DIRECTION = "DireccionViento"
N_SENSORS = 10


def _angular_error(prediction, target):
    """Error angular firmado en [-180, 180) grados."""
    return (prediction - target + 180.0) % 360.0 - 180.0


def _basic_metrics(target, prediction, error, iqr):
    """Calcula métricas utilizando el error indicado."""
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = (
        float(1.0 - np.sum(error ** 2) / denominator)
        if denominator > 0.0
        else None
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nmae_iqr": float(mae / iqr),
        "bias": float(np.mean(error)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
    }


def _score_physical(yt, yp, iqrs):
    """Evalúa trayectorias en unidades físicas [N, H, 10]."""
    by_sensor = {}
    nmaes = []

    for j, name in enumerate(SENSORS):
        target = yt[:, :, j].reshape(-1)
        prediction = yp[:, :, j].reshape(-1)

        linear_error = prediction - target

        if name == WIND_DIRECTION:
            angular_error = _angular_error(prediction, target)

            metrics = _basic_metrics(
                target, prediction, angular_error, iqrs[j]
            )

            # El R² convencional sigue siendo lineal.
            linear_metrics = _basic_metrics(
                target, prediction, linear_error, iqrs[j]
            )

            metrics["r2"] = linear_metrics["r2"]
            metrics["error_type"] = "angular"
            metrics["r2_type"] = "linear"
            metrics["linear_metrics"] = linear_metrics

        else:
            metrics = _basic_metrics(
                target, prediction, linear_error, iqrs[j]
            )
            metrics["error_type"] = "linear"

        by_sensor[name] = metrics
        nmaes.append(metrics["nmae_iqr"])

    return {
        "nmae_global": float(np.mean(nmaes)),
        "nmae_sensor_count": len(nmaes),
        "by_sensor": by_sensor,
    }


def score(y_true_z, y_pred_z, scaler):
    """Evalúa predicciones normalizadas [muestras, H, 10]."""

    yt_z = np.asarray(y_true_z, dtype=np.float64)
    yp_z = np.asarray(y_pred_z, dtype=np.float64)

    if (
        yt_z.shape != yp_z.shape
        or yt_z.ndim != 3
        or yt_z.shape[-1] != N_SENSORS
        or yt_z.shape[0] == 0
        or yt_z.shape[1] == 0
    ):
        raise ValueError(
            "Se requieren trayectorias no vacías "
            "[muestras, H, 10] de igual forma"
        )

    if not np.isfinite(yt_z).all():
        raise ValueError("Los objetivos contienen NaN o infinitos")

    if not np.isfinite(yp_z).all():
        raise ValueError("Las predicciones contienen NaN o infinitos")

    means = np.asarray(scaler["means"][:N_SENSORS], dtype=np.float64)
    stds = np.asarray(scaler["stds"][:N_SENSORS], dtype=np.float64)
    iqrs = np.asarray(scaler["iqrs"][:N_SENSORS], dtype=np.float64)

    if (
        means.shape != (N_SENSORS,)
        or stds.shape != (N_SENSORS,)
        or iqrs.shape != (N_SENSORS,)
        or not np.isfinite(means).all()
        or not np.isfinite(stds).all()
        or not np.isfinite(iqrs).all()
        or np.any(stds <= 0)
        or np.any(iqrs <= 0)
    ):
        raise ValueError("Estadísticas de Train inválidas")

    yt = yt_z * stds + means
    yp = yp_z * stds + means

    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Desnormalización no finita")

    result = _score_physical(yt, yp, iqrs)

    result["n_samples"] = int(yt.shape[0])
    result["horizon_steps"] = int(yt.shape[1])
    result["metric_version"] = "etifa_angular_v1"

    by_step = []

    for step in range(yt.shape[1]):
        step_result = _score_physical(
            yt[:, step:step + 1, :],
            yp[:, step:step + 1, :],
            iqrs,
        )

        step_result["step"] = step + 1
        step_result["minutes_ahead"] = (step + 1) * 15
        by_step.append(step_result)

    result["by_step"] = by_step

    return result
