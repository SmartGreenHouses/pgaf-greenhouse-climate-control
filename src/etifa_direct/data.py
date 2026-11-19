"""Ventanas multisalida sin materialización y escalado ajustado solo con Train."""
from pathlib import Path
import os
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SENSORS = [
    'CajaMedida1.TBS', 'CajaMedida1.TBH', 'CajaMedida1.HR',
    'CajaMedida1.DH', 'TemperaturaExterior', 'HumedadExterior',
    'VelocidadViento', 'DireccionViento', 'Radiacion', 'TemperaturaTuberia',
]
ACTUATORS = [
    'Humidificacion', 'PosicionVentana1', 'PosicionVentana2',
    'PosicionPantalla1', 'PosicionPantalla2',
]
TEMPORAL = ['hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos']
FEATURES = SENSORS + ACTUATORS + TEMPORAL
PHYSICAL = SENSORS + ACTUATORS
SEEDS = (13, 27, 22, 8, 31)
WINDOWS = (48, 96, 288, 672)
HORIZONS = (4, 8, 16)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_frame(root, split):
    data_dir = os.environ.get('ETIFA_DATA_DIR', 'data_temporal')
    path = Path(root) / data_dir / f'ETIFA_{split}.csv'
    if not path.exists():
        raise FileNotFoundError(f'Falta {path}. Primero genera data_temporal.')
    df = pd.read_csv(path)
    expected = ['Fecha'] + FEATURES
    if list(df.columns) != expected:
        raise ValueError(f'Columnas inesperadas en {path}: {list(df.columns)}')
    dt = pd.to_datetime(df['Fecha'], utc=True, errors='raise')
    if dt.isna().any() or dt.duplicated().any() or not dt.is_monotonic_increasing:
        raise ValueError(f'Fechas inválidas en {path}')
    if len(dt) > 1 and not np.all((dt.diff().iloc[1:] == pd.Timedelta(minutes=15)).to_numpy()):
        raise ValueError(f'Serie temporal no continua a 15 minutos: {path}')
    return df, path


def get_scaler(root):
    """Estadísticas del Train; rechaza escaladores obsoletos."""
    root = Path(root)
    train, path = read_frame(root, 'train')
    digest = file_sha256(path)
    data_dir = os.environ.get('ETIFA_DATA_DIR', 'data_temporal')
    if data_dir == 'data_temporal_v3_early_test':
        scaler_name = 'scaler_train_v3_early_test.json'
    else:
        scaler_name = 'scaler_train.json'
    out = root / 'results' / 'direct_multioutput' / scaler_name
    if out.exists():
        cfg = json.loads(out.read_text(encoding='utf-8'))
        if cfg.get('sha256_train') != digest or cfg.get('features') != FEATURES:
            raise ValueError('El Train cambió: revisar el escalador antes de continuar.')
        return cfg
    means, stds, iqrs = [], [], []
    for c in PHYSICAL:
        a = train[c].to_numpy(dtype=np.float64)
        a = a[np.isfinite(a)]
        if not len(a):
            raise ValueError(f'Sin datos válidos para {c}')
        means.append(float(np.mean(a)))
        stds.append(float(np.std(a)) if np.std(a) > 1e-12 else 1.0)
        q25, q75 = np.percentile(a, [25, 75])
        iqrs.append(float(q75 - q25))
    cfg = {
        'method': 'z-score', 'fitted_on': str(path), 'sha256_train': digest,
        'features': FEATURES, 'sensors': SENSORS,
        'means': means, 'stds': stds, 'iqrs': iqrs,
        'notes': 'Variables cíclicas sin escalar. IQR solo para nMAE.',
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
    return cfg


def normalize(df, scaler):
    a = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    a[:, :15] = (a[:, :15] - np.array(scaler['means'], dtype=np.float32)) / np.array(scaler['stds'], dtype=np.float32)
    return a


class StableDataset(Dataset):
    """Selecciona solo ventanas sin cambios de actuadores en el horizonte.

    X: [L,19] completo; Y: [H,10] completo.
    Origen: último instante del histórico. No se leen sensores futuros en X.
    """
    def __init__(self, root, split, lookback, horizon, scaler):
        if lookback not in WINDOWS or horizon not in HORIZONS:
            raise ValueError('L/H fuera de la matriz experimental acordada')
        self.lookback, self.horizon = lookback, horizon
        self.split = split
        df, self.path = read_frame(root, split)
        self.dates = pd.to_datetime(df['Fecha'], utc=True)
        self.values = normalize(df, scaler)
        n = len(df)
        if n < lookback + horizon:
            self.starts = np.empty(0, dtype=np.int64)
            return
        valid = np.isfinite(self.values)
        bad_x = (~valid).any(axis=1).astype(np.int64)
        bad_y = (~valid[:, :10]).any(axis=1).astype(np.int64)
        bx = np.concatenate(([0], np.cumsum(bad_x)))
        by = np.concatenate(([0], np.cumsum(bad_y)))
        actuators = df[ACTUATORS].to_numpy(dtype=np.float64)
        changes = np.zeros(n, dtype=np.int64)
        changes[1:] = np.any(actuators[1:] != actuators[:-1], axis=1)
        cc = np.concatenate(([0], np.cumsum(changes)))
        starts = np.arange(n - lookback - horizon + 1, dtype=np.int64)
        ends = starts + lookback
        future_ends = ends + horizon
        good = (
            (bx[ends] == bx[starts]) &
            (by[future_ends] == by[ends]) &
            (cc[future_ends] == cc[ends])
        )
        self.starts = starts[good]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        e = s + self.lookback
        h = self.horizon
        x = self.values[s:e]
        y = self.values[e:e+h, :10]
        return torch.from_numpy(x), torch.from_numpy(y)

    def origin_and_end(self, idx):
        s = int(self.starts[idx]); e = s + self.lookback
        return str(self.dates.iloc[e-1]), str(self.dates.iloc[e+self.horizon-1])
