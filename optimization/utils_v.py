import ephem
import numpy as np
import pandas as pd
from numpy import pi, sin, cos
from pathlib import Path
from constants import (
    LOCAL_TIMEZONE,
    MODEL_PATH,
    MONTH_SIN_COL,
    MONTH_COS_COL,
    MULTIVARIATE_VARIABLES,
    NORMALIZATION_PARAMETERS,
    HISTORY_STEPS,
)

from tf_keras.models import load_model

MODEL = None
TF_RUNTIME_CONFIGURED = False
LOOK_BACK_15 = HISTORY_STEPS
MODEL_INPUT_COLUMNS = MULTIVARIATE_VARIABLES + [MONTH_SIN_COL, MONTH_COS_COL]
MODEL_INPUT_COLUMN_INDEX = {column: index for index, column in enumerate(MODEL_INPUT_COLUMNS)}
MODEL_PREDICT_PARAMETERS = {
    'batch_size': LOOK_BACK_15,
    'verbose': 0
}
VARIABLES_TO_REMOVE_FROM_OUTPUT = [
    'month_sin',
    'month_cos',
    'Humidificacion',
    'PosicionVentana1',
    'PosicionVentana2',
    'PosicionPantalla1',
    'PosicionPantalla2',
    'TemperaturaTuberia',
]
VARIABLES_TO_REMOVE_FROM_OUTPUT_SET = set(VARIABLES_TO_REMOVE_FROM_OUTPUT)

# Methods

# Encode cyclical features (like month) using sine and cosine transformations.
def encode_cyclical_value(values, max_val):
    if max_val <= 0:
        raise ValueError("max_val must be greater than 0")

    angle = 2 * pi * values / max_val
    return sin(angle), cos(angle)

# Add month column to dataset and encode it as cyclical features.
def add_month_col(data):
    data = data.copy()

    month_one_based = data.index.month
    data[MONTH_SIN_COL], data[MONTH_COS_COL] = encode_cyclical_value(
        month_one_based,
        12
    )

    return data

# Determine if a given date is during the day or night based on the position of the sun.
def is_it_day_from_date(date, lat, lon, timezone=LOCAL_TIMEZONE):
    """Return whether a local civil timestamp is between sunrise and sunset.

    PyEphem interprets naive datetimes as UTC. Dataset timestamps are local to
    Murcia, so they must be localized first and converted to UTC explicitly.
    """
    timestamp = pd.Timestamp(date)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)

    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = ephem.Date(timestamp.tz_convert("UTC").to_pydatetime())
    sun = ephem.Sun(observer)

    return observer.previous_rising(sun) > observer.previous_setting(sun)

####################################################################################################

##################################################
#                     SCALER                     #
##################################################

def min_max_scaler(array, *, min_limit=None, max_limit=None, min_range=0, max_range=1):
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        raise ValueError("array must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("array contains NaN or infinite values")

    min_limit = float(min_limit) if min_limit is not None else float(np.min(array))
    max_limit = float(max_limit) if max_limit is not None else float(np.max(array))
    min_range = float(min_range)
    max_range = float(max_range)
    if max_limit < min_limit:
        raise ValueError("max_limit must be greater than or equal to min_limit")
    if max_range == min_range:
        raise ValueError("max_range and min_range must be different")
    if max_limit == min_limit:
        return np.full_like(array, min_range, dtype=float)

    return ((array - min_limit) / (max_limit - min_limit)) * (max_range - min_range) + min_range

def normalize_data(data):
    data = data.copy()
    missing_params = [
        column
        for column in data.columns
        if column not in (MONTH_SIN_COL, MONTH_COS_COL) and column not in NORMALIZATION_PARAMETERS
    ]
    if missing_params:
        raise KeyError(f"Missing normalization parameters for columns: {missing_params}")

    for column in data.columns:
        if column != MONTH_SIN_COL and column != MONTH_COS_COL:
            params = NORMALIZATION_PARAMETERS[column]
            data[column] = min_max_scaler(
                data[column].values,
                min_limit=params.get('min_limit'),
                max_limit=params.get('max_limit'),
                min_range=params.get('min_range', 0),
                max_range=params.get('max_range', 1))
    return data

def min_max_scale_value(value, column):
    params = NORMALIZATION_PARAMETERS[column]
    min_limit = float(params.get('min_limit'))
    max_limit = float(params.get('max_limit'))
    min_range = float(params.get('min_range', 0))
    max_range = float(params.get('max_range', 1))
    value = float(value)

    if not np.isfinite(value):
        raise ValueError(f"Non-finite value for column {column}: {value}")
    if max_range == min_range:
        raise ValueError(f"Invalid normalization range for column {column}")
    if max_limit == min_limit:
        return min_range

    return ((value - min_limit) / (max_limit - min_limit)) * (max_range - min_range) + min_range


def min_max_scale_values(values, column):
    """Vectorized min-max scaling for a 1D array of values."""
    params = NORMALIZATION_PARAMETERS[column]
    min_limit = float(params.get('min_limit'))
    max_limit = float(params.get('max_limit'))
    min_range = float(params.get('min_range', 0))
    max_range = float(params.get('max_range', 1))

    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values for column {column}")
    if max_range == min_range:
        raise ValueError(f"Invalid normalization range for column {column}")
    denom = (max_limit - min_limit)
    if denom == 0:
        return np.full_like(values, min_range, dtype=float)

    return ((values - min_limit) / denom) * (max_range - min_range) + min_range

def min_max_inverse_scaler(array, *, min_limit=None, max_limit=None, min_range=0, max_range=1):
    array = np.asarray(array, dtype=float)
    if array.size == 0:
        raise ValueError("array must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("array contains NaN or infinite values")

    min_limit = float(min_limit) if min_limit is not None else float(np.min(array))
    max_limit = float(max_limit) if max_limit is not None else float(np.max(array))
    min_range = float(min_range)
    max_range = float(max_range)
    if max_limit < min_limit:
        raise ValueError("max_limit must be greater than or equal to min_limit")
    if max_range == min_range:
        raise ValueError("max_range and min_range must be different")
    if max_limit == min_limit:
        return np.full_like(array, min_limit, dtype=float)

    scale = (max_range - min_range) / (max_limit - min_limit)
    return (array - (min_range - min_limit * scale)) / scale
    
def denormalize_predictions(predictions, output_columns):
    for index, column in enumerate(output_columns):
        params = NORMALIZATION_PARAMETERS[column]
        predictions[:, index] = min_max_inverse_scaler(
            predictions[:, index],
            min_limit=params.get('min_limit'),
            max_limit=params.get('max_limit'),
            min_range=params.get('min_range', 0),
            max_range=params.get('max_range', 1))
        
    return predictions



##################################################
#                    FORECAST                    #
##################################################

def configure_tensorflow_runtime():
    """Configure TensorFlow once before the model allocates GPU memory."""
    global TF_RUNTIME_CONFIGURED

    if TF_RUNTIME_CONFIGURED:
        return

    try:
        import tensorflow as tf

        for device in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                # TensorFlow has already initialized the device. The SLURM job
                # still owns the GPU, so continuing is safe and auditable.
                pass
    finally:
        TF_RUNTIME_CONFIGURED = True


def get_model():
    global MODEL

    if MODEL is None:
        model_path = Path(MODEL_PATH)
        if not model_path.is_file():
            raise FileNotFoundError(f"Forecast model not found: {model_path}")
        configure_tensorflow_runtime()
        MODEL = load_model(str(model_path), compile=False)

    return MODEL

# Prediction method 
def prepare_forecast_data(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Forecast data must use a pandas.DatetimeIndex")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Forecast data index must be sorted in ascending order")
    if data.index.has_duplicates:
        raise ValueError("Forecast data index contains duplicate timestamps")

    if MONTH_SIN_COL not in data.columns or MONTH_COS_COL not in data.columns:
        data = add_month_col(data.copy())

    missing_columns = [col for col in MODEL_INPUT_COLUMNS if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required model input columns: {missing_columns}")

    # Keep only the exact history window expected by the model.
    data = data.loc[:, MODEL_INPUT_COLUMNS].tail(LOOK_BACK_15).copy()

    if len(data) < LOOK_BACK_15:
        raise ValueError(
            f"Not enough history rows for forecast: got {len(data)}, expected {LOOK_BACK_15}"
        )

    if data.isna().any().any():
        missing = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Forecast data contains missing values in columns: {missing}")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Forecast data contains infinite values")

    # Dataset normalization
    normalized_dataset = normalize_data(data)
    output_columns = normalized_dataset.columns.difference(
        VARIABLES_TO_REMOVE_FROM_OUTPUT,
        sort=False
    )

    return normalized_dataset.to_numpy().reshape(1, LOOK_BACK_15, len(MODEL_INPUT_COLUMNS)), output_columns

def forecast_from_model_history(model_history, output_columns, steps, last_row_updates=None):
    model = get_model()
    # Copy only when we will mutate the history array.
    if steps != 1 or last_row_updates:
        model_history = model_history.copy()
    # Keras/TensorFlow typically runs in float32; avoid repeated internal casts.
    model_history = np.asarray(model_history, dtype=np.float32)
    n_features = model_history.shape[2]
    n_features_output = len(output_columns)

    if last_row_updates:
        for column, value in last_row_updates.items():
            column_index = MODEL_INPUT_COLUMN_INDEX[column]
            model_history[0, -1, column_index] = min_max_scale_value(value, column)

    # plot_model(model, to_file='estructura_modelo.png', show_shapes=True, show_layer_names=True)

    # Common GA path: single-step forecast. Avoid rolling-horizon updates.
    if steps == 1:
        _model_prediction = np.asarray(model.predict(
            model_history,
            **MODEL_PREDICT_PARAMETERS,
        ))

        if _model_prediction.shape[0] != 1 or _model_prediction.shape[-1] != n_features_output:
            raise ValueError(
                "Model output dimension does not match output_columns: "
                f"got shape {_model_prediction.shape}, expected (1, {n_features_output})"
            )
        if not np.all(np.isfinite(_model_prediction)):
            raise ValueError("Model returned NaN or infinite predictions")

        model_predictions = np.asarray(_model_prediction, dtype=float).reshape(1, n_features_output)
        denormalized_model_predictions = denormalize_predictions(model_predictions.copy(), output_columns)
        return denormalized_model_predictions, output_columns

    # Forecast with the model
    model_predictions = np.empty((steps, n_features_output), dtype=float)

    for step in range(steps):
        _model_prediction = np.asarray(model.predict(
            model_history,
            **MODEL_PREDICT_PARAMETERS))

        if _model_prediction.shape[0] != 1 or _model_prediction.shape[-1] != n_features_output:
            raise ValueError(
                "Model output dimension does not match output_columns: "
                f"got shape {_model_prediction.shape}, expected (1, {n_features_output})"
            )
        if not np.all(np.isfinite(_model_prediction)):
            raise ValueError("Model returned NaN or infinite predictions")

        # Snapshot the last step before updates so each removed feature
        # carries its own previous value (instead of reusing one column).
        last_step_values = model_history[0, -1, :].copy()
        next_step_values = np.empty(n_features, dtype=float)
        prediction_index_counter = 0
        for index, column_name in enumerate(MODEL_INPUT_COLUMNS):
            if column_name in VARIABLES_TO_REMOVE_FROM_OUTPUT_SET:
                # Keep removed variables constant for next step horizon.
                next_step_values[index] = last_step_values[index]
            else:
                next_step_values[index] = _model_prediction[0][prediction_index_counter]
                prediction_index_counter += 1

        model_history[:, :-1, :] = model_history[:, 1:, :]
        model_history[:, -1, :] = next_step_values
        model_predictions[step, :] = _model_prediction[0]

    # Denormalize forecast
    denormalized_model_predictions = denormalize_predictions(model_predictions.copy(), output_columns)

    return denormalized_model_predictions, output_columns


def forecast_batch_from_model_history(model_history, output_columns, steps, last_row_updates_list):
    """Batch forecast for multiple scenarios sharing the same history.

    Optimized for the GA use case where steps==1:
    - Builds a batch by repeating the shared history across population.
    - Applies last-row actuator updates per individual.
    - Runs a single model.predict over the entire batch.
    """
    if steps != 1:
        raise ValueError("forecast_batch_from_model_history currently supports steps==1 only")

    if not last_row_updates_list:
        raise ValueError("last_row_updates_list must be a non-empty list")

    model = get_model()

    base_history = np.asarray(model_history, dtype=np.float32)
    if base_history.ndim != 3 or base_history.shape[0] != 1:
        raise ValueError(
            f"model_history must have shape (1, T, F); got {base_history.shape}"
        )

    population_size = len(last_row_updates_list)
    n_features_output = len(output_columns)

    batch_history = np.repeat(base_history, repeats=population_size, axis=0)

    # Apply last-row updates in a vectorized way per column.
    update_columns = set()
    for updates in last_row_updates_list:
        update_columns.update(updates.keys())

    for column in update_columns:
        column_index = MODEL_INPUT_COLUMN_INDEX[column]
        column_values = [updates.get(column) for updates in last_row_updates_list]
        # Filter None (column absent in some dict) by keeping previous values.
        mask = np.array([v is not None for v in column_values], dtype=bool)
        if not np.any(mask):
            continue

        present_values = [float(v) for v in column_values if v is not None]
        scaled = min_max_scale_values(present_values, column)
        batch_history[mask, -1, column_index] = scaled

    batch_predictions = np.asarray(model.predict(
        batch_history,
        **MODEL_PREDICT_PARAMETERS,
    ))

    if batch_predictions.shape[0] != population_size or batch_predictions.shape[-1] != n_features_output:
        raise ValueError(
            "Model output dimension does not match output_columns: "
            f"got shape {batch_predictions.shape}, expected ({population_size}, {n_features_output})"
        )
    if not np.all(np.isfinite(batch_predictions)):
        raise ValueError("Model returned NaN or infinite predictions")

    batch_predictions = np.asarray(batch_predictions, dtype=float).reshape(population_size, n_features_output)
    denormalized = denormalize_predictions(batch_predictions.copy(), output_columns)
    return denormalized, output_columns

def forecast(data, steps, last_row_updates=None):
    model_history, output_columns = prepare_forecast_data(data)
    return forecast_from_model_history(
        model_history,
        output_columns,
        steps,
        last_row_updates=last_row_updates
    )
