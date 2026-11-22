from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent

# Set to True to print messages about the execution of the code.
MSG_MODE = False 

# Forecast horizon optimized by the GA: 12 future steps = 3 hours at 15 min.
STEPS_TO_RUN = 4 * 3

# One-step LSTM prediction inside each GA evaluation.
STEPS = 1

# Input history window expected by the selected LSTM model:
# 96 steps = 24 hours at 15 min resolution.
HISTORY_STEPS = 96
LOOK_BACK_STEPS = HISTORY_STEPS
INPUT_WINDOW_STEPS = HISTORY_STEPS
WINDOW_SIZE_STEPS = HISTORY_STEPS

# List of variables that we will use for the multivariate model. 
MULTIVARIATE_VARIABLES = [
    # Sondas
    "CajaMedida1.TBS",
    "CajaMedida1.TBH",
    "CajaMedida1.HR",
    "CajaMedida1.DH",
    "TemperaturaExterior",
    "HumedadExterior",
    "VelocidadViento",
    "DireccionViento",
    "Radiacion",
    "TemperaturaTuberia",
    # Actuadores
    "Humidificacion",
    "PosicionVentana1",
    "PosicionVentana2",
    "PosicionPantalla1",
    "PosicionPantalla2"
]

# Path to the CSV file that contains the data. 
DATA_CSV_PATH = str(PROJECT_ROOT / "datasets" / "ETIFA_clean_imputed_1h.csv")
# The historical checkpoint has input shape (None, 96, 17), output (None, 9).
# Its requested filename LSTM_L48_H1.h5 is an alias, NOT an architectural
# specification: renaming it does not change its true 96-step input history.
# The original checkpoint was trained with the historical GA dataset.
# GA_MODEL_PATH may override its location.
MODEL_PATH = os.environ.get(
    "GA_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "LSTM_L48_H1.h5"),
)

# Path to save the output of the genetic algorithm. 
OUTPUT_FOLDER = str(PROJECT_ROOT / "out")

# Name of the column that contains the timestamps in the dataset.
TIMESTAMP_COL = "Fecha" 

# Name of the column that we use to store the month information in the dataset.
MONTH_COL = "month"
MONTH_SIN_COL = f"{MONTH_COL}_sin"
MONTH_COS_COL = f"{MONTH_COL}_cos"

# Latitude and longitude of MURCIA (ETIFA).
LATITUDE = 37.98704
LONGITUDE = -1.13004
LOCAL_TIMEZONE = "Europe/Madrid"


##########################################
#               CONFIGURAR               #
##########################################

# Full set of dates that we want to run the algorithm on. 

# Each selected episode is identified by its end timestamp ("to").
# The GA extracts the last HISTORY_STEPS=96 rows ending at that timestamp.
# The "from" value is kept for protocol metadata and should span approximately 24 hours.
ALL_SELECTED_DATES = [{
        "from": "2019-08-31 06:00",
        "to":  "2019-09-01 06:00"
    }, {
        "from": "2019-05-21 06:00",
        "to":  "2019-05-22 06:00"
    }, {
        "from": "2020-01-04 06:00",
        "to":  "2020-01-05 06:00"
    }, {
        "from": "2020-10-31 06:00",
        "to":  "2020-11-01 06:00"
    }, {
        "from": "2019-08-31 12:00",
        "to":  "2019-09-01 12:00"
    }, {
        "from": "2019-05-21 12:00",
        "to":  "2019-05-22 12:00"
    }, {
        "from": "2020-01-04 12:00",
        "to":  "2020-01-05 12:00"
    }, {
        "from": "2020-10-31 12:00",
        "to":  "2020-11-01 12:00"
    }, {
        "from": "2019-08-31 18:00",
        "to":  "2019-09-01 18:00"
    }, {
        "from": "2019-05-21 18:00",
        "to":  "2019-05-22 18:00"
    }, {
        "from": "2020-01-04 18:00",
        "to":  "2020-01-05 18:00"
    }, {
        "from": "2020-08-31 18:00",
        "to":  "2020-09-01 18:00"
    }]


# Full set of seed values that we want to run the algorithm on.

ALL_SEED_VALUES = [11,13,16,20,24]

# Desired setpoints for day and night. 
DESIRED_TBS_DAY = 20
DESIRED_HR_DAY = 80

DESIRED_TBS_NIGHT = 15
DESIRED_HR_NIGHT = 70

# Limits for the TBS and HR values. NOT CHANGED IT MUST BE THE SAME AS THE ONES USED IN THE NORMALIZATION PARAMETERS (LSTM)
MIN_TBS_VALUE = -20
MAX_TBS_VALUE = 50
MIN_HR_VALUE = 0
MAX_HR_VALUE = 100

# List of possible genes that we are going to use for the genetic algorithm.
# penalty_up/down are relative actuator-effort weights for increasing or
# decreasing each actuator, not direct unnormalized fitness increments.
POSSIBLE_GENES = {
    "TemperaturaTuberia": {
        "min": 15,
        "max": 55,
        "relative_15m": 15,
        "penalty_up": 1,
        "penalty_down": 0
    },
    "Humidificacion": {  # boolean
        "min": 0,
        "max": 1,
        "relative_15m": 1,
        "penalty_up": 0.5,
        "penalty_down": 0
    },
    "PosicionVentana1": {
        "min": 0,
        "max": 100,
        "relative_15m": 50,
        "penalty_up": 0.15,
        "penalty_down": 0.1
    },
    "PosicionVentana2": {
        "min": 0,
        "max": 100,
        "relative_15m": 50,
        "penalty_up": 0.15,
        "penalty_down": 0.1
    },
    "PosicionPantalla1": {
        "min": 0,
        "max": 100,
        "relative_15m": 50,
        "penalty_up": 0.15,
        "penalty_down": 0.1
    },
    "PosicionPantalla2": {
        "min": 0,
        "max": 100,
        "relative_15m": 50,
        "penalty_up": 0.15,
        "penalty_down": 0.1
    },
}

# Soft target-aware penalty scheme. These weights are intentionally small
# because the climate fitness is normalized and useful differences are often
# in the 0.01-0.05 range.
FITNESS_PENALTY_SCHEME = "target_first_soft_v1"
ACTUATOR_AWARE_PENALTY_WEIGHT = 0.03
RICH_ENERGY_PENALTY_WEIGHT = 0.02
RICH_ACTUATOR_PENALTY_WEIGHT = 0.02
ENGINEERING_RULE_PENALTY_WEIGHT = 0.05
TARGET_PROGRESS_RULE_PENALTY_WEIGHT = 0.05
TARGET_BAND_VIOLATION_WEIGHT = 0.08
PENALTY_TARGET_TBS_TOLERANCE = 1.0
PENALTY_TARGET_HR_TOLERANCE = 5.0
PENALTY_GATE_FLOOR = 0.05
PENALTY_GATE_POWER = 2.0

# Backwards-compatible alias for older imports.
BASE_ACTUATOR_PENALTY_WEIGHT = ACTUATOR_AWARE_PENALTY_WEIGHT
PIPE_TEMP_DECREASE_THRESHOLD = 5
TBS_HEATING_NEED_TOLERANCE = 0.5
TBS_OVERHEAT_TOLERANCE = 0.5
HR_HUMIDIFICATION_TOLERANCE = 1.0
HR_DEHUMIDIFICATION_TOLERANCE = 2.0
ABS_HUMIDITY_MARGIN = 0.5
PIPE_OVER_TBS_MARGIN = 2.0
COLD_EXTERIOR_MAX_WINDOW_POSITION = 50.0
DEHUMIDIFICATION_MIN_WINDOW_POSITION = 20.0
HUMIDITY_RISK_MAX_WINDOW_POSITION = 50.0
SCREEN_RADIATION_THRESHOLD = 100.0
HIGH_RADIATION_MAX_SCREEN_POSITION = 80.0
SOLAR_GAIN_MIN_SCREEN_POSITION = 20.0
BLX_CROSSOVER_ALPHA = 0.3
PIPE_ENERGY_LEVEL_WEIGHT = 0.5
PIPE_ENERGY_EXCESS_WEIGHT = 0.5

NORMALIZATION_PARAMETERS = {
    'CajaMedida1.TBS': {
        'min_limit': -20,
        'max_limit': 50,
        'min_range': 0,
        'max_range': 1
    },
    'CajaMedida1.TBH': {
        'min_limit': -20,
        'max_limit': 50,
        'min_range': 0,
        'max_range': 1
    },
    'CajaMedida1.HR': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    },
    'CajaMedida1.DH': {
        'min_limit': 0,
        'max_limit': 30,
        'min_range': 0,
        'max_range': 1
    },
    'TemperaturaExterior': {
        'min_limit': -20,
        'max_limit': 50,
        'min_range': 0,
        'max_range': 1
    },
    'HumedadExterior': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    },
    'VelocidadViento': {
        'min_limit': 0,
        'max_limit': 150,
        'min_range': 0,
        'max_range': 1
    },
    'DireccionViento': {
        'min_limit': 0,
        'max_limit': 360,
        'min_range': 0,
        'max_range': 1
    },
    'Radiacion': {
        'min_limit': 0,
        'max_limit': 1500,
        'min_range': 0,
        'max_range': 1
    },
    'TemperaturaTuberia': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    },
    'Humidificacion': {
        'min_limit': 0,
        'max_limit': 1,
        'min_range': 0,
        'max_range': 1
    },
    'PosicionVentana1': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    },
    'PosicionVentana2': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    },
    'PosicionPantalla1': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    },
    'PosicionPantalla2': {
        'min_limit': 0,
        'max_limit': 100,
        'min_range': 0,
        'max_range': 1
    }
}
