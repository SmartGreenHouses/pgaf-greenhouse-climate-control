#!/usr/bin/env python3
"""ETIFA: auditoría y evaluación emparejada OBS/MASK/SUST, H=16.

Por defecto SOLO AUDITA las poblaciones candidatas, sin realizar inferencias.
Usa los archivos definitivos del repositorio PGAF (V1,V2,V3; 9+6+4 canales).
Los modelos publicados preservan la arquitectura y pesos de la campaña ETIFA.
Se conserva la definición original de evento para reproducir las salidas históricas.

Criterio candidato de evento (A REVISAR antes de publicar):
  exactamente un actuador presenta |a[t]-a[t-1]| >= umbral H16;
  los seis actuadores están disponibles en [t,t+16];
  los seis actuadores mantienen (max-min)<umbral H16 en [t,t+16].
Se toman los umbrales del protocolo original de estabilidad H16 SOLO como
propuesta de detección de eventos, no como una decisión metodológica validada.
No se ejecuta inferencia salvo al pasar --evaluate --confirm-event-definition.

OBS : histórico real, incluido a[t].
MASK: sustituye solamente a[t] por a[t-1] para el actuador cambiado.
SUST: sustituye las últimas 16 muestras a[t-15:t] por a[t-1].

Nunca modifica datos, checkpoints o resultados existentes. Salida independiente.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SENSORS = (
    'dry_bulb_temperature', 'wet_bulb_temperature', 'relative_humidity',
    'humidity_deficit', 'outdoor_temperature', 'outdoor_relative_humidity',
    'wind_speed', 'wind_direction', 'radiation',
)
ACTUATORS = (
    'pipe_temperature', 'humidification', 'window_1_position',
    'window_2_position', 'screen_1_position', 'screen_2_position',
)
COLS = ('timestamp',) + SENSORS + ACTUATORS
MODELS = ('lstm_sergiot', 'itransformer', 'patchtst_mv')
VERSIONS = ('V1', 'V2', 'V3')
SEEDS = (13, 27, 22, 8, 31)
H = 16
L_MAX = 672
OUTPUT = Path('forecasting/experiments/actuator_change_sensitivity/out/reproduction')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonfile(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def find_csv(root, cfg, explicit):
    expected = cfg['data_sha256']
    if explicit is not None:
        p = explicit.resolve()
        require(p.is_file(), f'No existe el CSV: {p}')
        require(sha256(p) == expected, f'El CSV no coincide con SHA256 del experimento: {p}')
        return p

    candidates = []
    for field in ('csv', 'csv_path', 'data_csv', 'dataset_csv', 'data_path'):
        if isinstance(cfg.get(field), str):
            path = Path(cfg[field]); candidates.append(path if path.is_absolute() else root / path)
    candidates.append(root / 'preprocessing/data/processed/ETIFA_clean_imputed_1h.csv')
    for subdir in (root, root / 'data', root / 'datasets', root / 'data_processed'):
        if not subdir.is_dir():
            continue
        candidates.extend(subdir.rglob('*.csv') if subdir != root else subdir.glob('*.csv'))
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            with path.open('r', encoding='utf-8-sig') as f:
                first = f.readline().strip()
            if next(csv.reader([first])) != list(COLS):
                continue
            if sha256(path) == expected:
                return path.resolve()
        except (OSError, UnicodeError):
            continue
    raise FileNotFoundError(
        'No pude localizar el CSV definitivo por su SHA256. Vuelve a ejecutar '
        'añadiendo --csv /ruta/al/dataset.csv (debe coincidir exactamente con la campaña).'
    )


def load_partitions(csv_path, manifest, version):
    df = pd.read_csv(csv_path)
    require(tuple(df.columns) == COLS, 'Columnas del CSV distintas de la campaña original')
    require(len(df) == int(manifest['n_rows']), 'Número de filas distinto del manifiesto')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='raise')
    require(df.timestamp.is_monotonic_increasing and not df.timestamp.duplicated().any(),
            'Timestamp desordenado o repetido')
    result = {}
    for part in ('train', 'test'):
        d = manifest['splits'][version][part]
        lo, hi = pd.Timestamp(d['start_utc']), pd.Timestamp(d['end_utc'])
        selected = df.loc[df.timestamp.between(lo, hi, inclusive='both')].copy().reset_index(drop=True)
        require(len(selected) == int(d['rows']) and not selected.empty,
                f'{version}/{part}: número de filas no coincide')
        require(selected.timestamp.iloc[0] == lo and selected.timestamp.iloc[-1] == hi,
                f'{version}/{part}: fechas de partición no coinciden')
        require(selected.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all(),
                f'{version}/{part}: discontinuidad temporal')
        result[part] = selected
    return result


def scale_and_values(frames):
    train = frames['train'][list(SENSORS + ACTUATORS)].to_numpy(dtype=np.float64)
    mean, sd = np.nanmean(train, axis=0), np.nanstd(train, axis=0)
    iqr = np.nanpercentile(train, 75, axis=0) - np.nanpercentile(train, 25, axis=0)
    require(np.isfinite(mean).all() and np.isfinite(sd).all() and
            np.isfinite(iqr[:9]).all() and (iqr[:9] > 0).all(), 'Estadísticas Train inválidas')
    sd[sd <= 1e-12] = 1.0
    scaled = {}
    for part, frame in frames.items():
        physical = frame[list(SENSORS + ACTUATORS)].to_numpy(dtype=np.float64)
        values = (physical - mean) / sd
        t = frame.timestamp.dt
        minutes = (t.hour * 60 + t.minute).to_numpy(dtype=np.float64)
        day = t.day_of_year.to_numpy(dtype=np.float64)
        cyc = np.column_stack((np.sin(2*np.pi*minutes/1440), np.cos(2*np.pi*minutes/1440),
                               np.sin(2*np.pi*day/365.25), np.cos(2*np.pi*day/365.25)))
        scaled[part] = np.column_stack((values, cyc)).astype(np.float32)
    return scaled, {'mean': mean, 'sd': sd, 'iqr': iqr}


def valid_common_origins(values):
    """Los mismos orígenes completos para cualquier lookback hasta 672."""
    n = len(values)
    if n < L_MAX + H:
        return np.empty(0, dtype=np.int64)
    good_x = np.isfinite(values).all(axis=1)
    good_y = np.isfinite(values[:, :9]).all(axis=1)
    bx = np.r_[0, np.cumsum(~good_x, dtype=np.int64)]
    by = np.r_[0, np.cumsum(~good_y, dtype=np.int64)]
    starts = np.arange(n - L_MAX - H + 1, dtype=np.int64)
    e = starts + L_MAX
    f = e + H
    keep = (bx[e] - bx[starts] == 0) & (by[f] - by[e] == 0)
    return starts[keep] + L_MAX - 1  # origin t, no futuro usado como entrada


def event_candidates(frame, common_t, thresholds):
    """Un único salto material en t; 6 actuadores estables desde t hasta t+H."""
    a = frame[list(ACTUATORS)].to_numpy(dtype=np.float64)
    ends, idxs = [], []
    thresholds = np.asarray(thresholds, dtype=np.float64)
    require(thresholds.shape == (6,) and np.isfinite(thresholds).all() and
            (thresholds > 0).all(), 'Se necesitan seis umbrales positivos en H16')
    for t in common_t:
        t = int(t)
        if t < 1 or t + H >= len(a):
            continue
        before, segment = a[t-1], a[t:t+H+1]
        if not (np.isfinite(before).all() and np.isfinite(segment).all()):
            continue
        jumps = np.abs(segment[0] - before) >= thresholds
        if jumps.sum() != 1:
            continue
        if np.any(np.ptp(segment, axis=0) >= thresholds):
            continue
        # El episodio no puede contener otro salto material en [t+1,t+H].
        if np.any(np.abs(np.diff(segment, axis=0)) >= thresholds):
            continue
        ends.append(t+1)
        idxs.append(int(np.flatnonzero(jumps)[0]))
    return np.asarray(ends, dtype=np.int64), np.asarray(idxs, dtype=np.int64)


def equivalence_stats(values, ends, idxs, lookbacks):
    """Cuántos episodios tienen MASK==SUST de forma exacta en su actuador."""
    out = {}
    for L in lookbacks:
        count = 0
        for end, j in zip(ends, idxs):
            t = end-1
            if np.all(values[t-15:t, 9+j] == values[t-1, 9+j]):
                count += 1
        out[str(L)] = {'identical_mask_sust': count, 'total': len(ends)}
    return out


def selected_lookbacks(base):
    p = base / 'forecasting/results/validation/selected_lookbacks.csv'
    require(p.is_file(), f'Falta selección validada: {p}')
    d = pd.read_csv(p)
    require(set(('version', 'horizon', 'model', 'lookback')).issubset(d.columns),
            'Esquema inválido de selección')
    out = {}
    for version in VERSIONS:
        for model in MODELS:
            subset = d.loc[(d.version == version) & (d.horizon == H) & (d.model == model)]
            require(len(subset) == 1, f'Selección ambigua: {version}/H{H}/{model}')
            L = int(subset.iloc[0].lookback)
            require(L in (48, 96, 192, 288, 384, 672), f'Lookback no permitido: {L}')
            out[(version, model)] = L
    return out


def load_state(checkpoint, device):
    """Carga restringida de checkpoints propios que contienen arrays NumPy.

    La campaña ETIFA guardó metadatos NumPy junto al state_dict. En PyTorch
    reciente, weights_only=True necesita autorizar sus constructores y dtypes.
    NO utiliza weights_only=False ni permite clases arbitrarias de terceros.
    Incluye torch.torch_version.TorchVersion usada por metadatos PyTorch.
    """
    import torch
    from numpy._core.multiarray import _reconstruct, scalar
    from torch.torch_version import TorchVersion

    # Los checkpoints ETIFA contienen metadatos con np.ndarray/dtype y
    # la clase de version de PyTorch. Autorizamos solo estas clases conocidas.
    allowed = [_reconstruct, scalar, np.ndarray, np.dtype, TorchVersion]
    # np.dtypes.*DType son clases distintas de np.dtype en NumPy 2.x.
    for type_name in (
        'Float16DType', 'Float32DType', 'Float64DType',
        'Int8DType', 'Int16DType', 'Int32DType', 'Int64DType',
        'UInt8DType', 'UInt16DType', 'UInt32DType', 'UInt64DType',
        'BoolDType',
    ):
        cls = getattr(np.dtypes, type_name, None)
        if cls is not None:
            allowed.append(cls)
    # Preflight: no aceptar ninguna clase inesperada en el pickle del checkpoint.
    # La comprobacion esttica no sustituye a la carga restringida de PyTorch.
    if hasattr(torch.serialization, 'get_unsafe_globals_in_checkpoint'):
        unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)) - {
            f'{obj.__module__}.{obj.__qualname__}' for obj in allowed
        }
        if unknown:
            raise RuntimeError(
                f'Checkpoint contiene clases no autorizadas: {checkpoint} -> {sorted(unknown)}. '
                'No se cargaran automaticamente.'
            )
    try:
        with torch.serialization.safe_globals(allowed):
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f'No se pudo cargar el checkpoint restringidamente: {checkpoint}. '
            'No se realizará una deserialización insegura automática. '
            f'Detalle: {exc}'
        ) from exc
    if isinstance(saved, dict):
        for key in ('state_dict', 'model_state_dict', 'model_state', 'model', 'network_state_dict'):
            if key in saved and isinstance(saved[key], dict):
                return saved[key]
        if saved and all(isinstance(k, str) and hasattr(v, 'shape') for k, v in saved.items()):
            return saved
    raise RuntimeError(
        f'Formato desconocido de checkpoint {checkpoint}; '
        f'claves={list(saved)[:25] if isinstance(saved, dict) else type(saved)}'
    )


def infer_three(model, values, ends, indices, L, device, batch_size):
    import torch
    n = len(ends)
    actual = np.empty((n, H, 9), dtype=np.float32)
    predictions = {condition: np.empty((n, H, 9), dtype=np.float32)
                   for condition in ('OBS', 'MASK', 'SUST')}
    for offset in range(0, n, batch_size):
        sl = slice(offset, min(offset+batch_size, n))
        eb = ends[sl]
        js = indices[sl]
        x = np.stack([values[e-L:e] for e in eb]).astype(np.float32)
        y = np.stack([values[e:e+H, :9] for e in eb]).astype(np.float32)
        actual[sl] = y
        xs = {'OBS': x, 'MASK': x.copy(), 'SUST': x.copy()}
        for i, (e, j) in enumerate(zip(eb, js)):
            col = 9+int(j)
            prev = values[e-2, col]  # a[t-1], pues t = end-1
            xs['MASK'][i, -1, col] = prev
            xs['SUST'][i, -16:, col] = prev
        with torch.inference_mode():
            for condition in ('OBS', 'MASK', 'SUST'):
                pred = model(torch.from_numpy(xs[condition]).to(device)).detach().cpu().numpy()
                require(pred.shape == y.shape,
                        f'Dimensiones de inferencia incorrectas: {condition} {pred.shape}/{y.shape}')
                predictions[condition][sl] = pred
    return actual, predictions


def evaluate(root, versions, data, samples, lbs, device, batch_size, out):
    import torch
    # Arquitecturas PGAF con los mismos módulos y nombres de parámetros
    # que los tres modelos definitivos del código histórico ETIFA.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from forecasting.core import models
    require(models.N_SENSORS == 9, 'La salida debe incluir nueve sensores')
    records = []
    # El intento anterior pudo crear una carpeta vacía antes de cargar el
    # primer checkpoint. Se permite SOLO reusar una carpeta completamente vacía.
    out.mkdir(parents=True, exist_ok=True)
    for version in versions:
        values, scaler = data[version]
        ends, idxs = samples[version]
        if len(ends) == 0:
            print(f'AVISO: {version} sin episodios, no evaluada')
            continue
        for model_name in MODELS:
            L = lbs[(version, model_name)]
            for seed in SEEDS:
                key = f'{version}/H{H}/L{L}/{model_name}/seed{seed}'
                checkpoint = root / 'forecasting/checkpoints' / key / 'best.pt'
                require(checkpoint.is_file(), f'Falta checkpoint seleccionado: {checkpoint}')
                model = models.build_model(model_name, L, H, n_inputs=19).to(device)
                model.load_state_dict(load_state(checkpoint, device), strict=True)
                model.eval()
                truth_z, yp_z = infer_three(model, values, ends, idxs, L, device, batch_size)
                sd, mean, iqr = scaler['sd'][:9], scaler['mean'][:9], scaler['iqr'][:9]
                truth = truth_z.astype(np.float64)*sd[None,None,:] + mean[None,None,:]
                preds = {k: val.astype(np.float64)*sd[None,None,:]+mean[None,None,:]
                         for k, val in yp_z.items()}
                errors = {k: np.abs(pred-truth) for k, pred in preds.items()}
                for cond in ('OBS', 'MASK', 'SUST'):
                    for col, metric in ((0, 'TBS'), (2, 'RH')):
                        records.append({
                            'version': version, 'horizon': H, 'model': model_name,
                            'lookback': L, 'seed': seed, 'condition': cond, 'sensor': metric,
                            'n_episodes': len(ends),
                            'MAE': float(errors[cond][:,:,col].mean()),
                            'nMAE_IQR': float(errors[cond][:,:,col].mean()/iqr[col]),
                            'delta_MAE_vs_OBS': float((errors[cond][:,:,col]-errors['OBS'][:,:,col]).mean()),
                            'mean_prediction_shift_vs_OBS': float(np.abs(preds[cond][:,:,col]-preds['OBS'][:,:,col]).mean()),
                        })
                dirname = out / 'paired_predictions' / version / f'L{L}' / model_name
                dirname.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dirname / f'seed{seed}.npz',
                    origin_t=ends-1, actuator_index=idxs, truth_TBS_RH=truth[:,:,[0,2]].astype('float32'),
                    obs_TBS_RH=preds['OBS'][:,:,[0,2]].astype('float32'),
                    mask_TBS_RH=preds['MASK'][:,:,[0,2]].astype('float32'),
                    sust_TBS_RH=preds['SUST'][:,:,[0,2]].astype('float32'))
                print(f'OK {key}: {len(ends)} episodios, OBS/MASK/SUST')
                del model
                if device.type == 'cuda': torch.cuda.empty_cache()
    result = pd.DataFrame(records)
    result.to_csv(out / 'metrics_by_seed.csv', index=False)
    if len(result):
        grouped = result.groupby(['version','model','condition','sensor'],as_index=False)[
            ['MAE','nMAE_IQR','delta_MAE_vs_OBS','mean_prediction_shift_vs_OBS']].mean()
        grouped.to_csv(out / 'metrics_mean_5_seeds.csv', index=False)
    print('Salidas:', out.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    ap.add_argument('--csv', type=Path, default=None, help='CSV original (SHA256 verificado)')
    ap.add_argument('--config', type=Path, default=None)
    ap.add_argument('--manifest', type=Path, default=None)
    ap.add_argument('--evaluate', action='store_true', help='Ejecutar los 45 checkpoints, no solo auditar')
    ap.add_argument('--smoke', action='store_true',
                    help='Verificar una inferencia V1/L48/LSTM/seed8 contra el NPZ original')
    ap.add_argument('--confirm-event-definition', action='store_true',
                    help='Confirmación explícita de usar umbrales H16 como umbrales de salto')
    ap.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--output-root', type=Path, default=None,
                    help='Destino NUEVO para reproducción (no sobrescribe resultados publicados)')
    args = ap.parse_args()
    root = args.root.resolve()
    cfg_path = args.config or (root / 'forecasting/config.json')
    man_path = args.manifest or (root / 'forecasting/splits.json')
    require(cfg_path.is_file() and man_path.is_file(),
            'No se encuentran forecasting/config.json y forecasting/splits.json')
    cfg, man = jsonfile(cfg_path), jsonfile(man_path)
    # La configuración pública elimina metadatos de campaña, pero conserva
    # la semántica de H16. El hash auténtico está en splits.json.
    cfg['data_sha256'] = man['sha256']
    published = jsonfile(root/'forecasting/experiments/actuator_change_sensitivity/results/protocol.json')
    require(cfg['data_sha256'] == published['csv_sha256'], 'Dataset distinto del estudio')
    require(tuple(cfg['lookbacks']) == (48,96,192,288,384,672), 'Lookbacks distintos')
    require(H in cfg['horizons'] and len(ACTUATORS)==6 and len(SENSORS)==9, 'Configuración distinta')
    require(tuple(cfg['seeds']) == SEEDS, 'Semillas distintas de la campaña aprobada')
    require(int(man['n_rows'])==100866, 'Número de filas del dataset distinto')
    eps_dict = cfg['thresholds_by_h'][str(H)]
    # No interpretar None como tolerancia 0, ni inventar tolerancias.
    eps = [eps_dict.get(a) for a in ACTUATORS]
    require(all(v is not None and float(v) > 0 for v in eps),
            f'H16 no define umbrales de los seis actuadores: {dict(zip(ACTUATORS,eps))}')
    eps = np.array(eps, dtype=float)
    require(dict(zip(ACTUATORS,eps.tolist())) == published['event_min_deltas'],
            'Los umbrales H16 cambiaron respecto a los resultados publicados')
    print('=== AUDITORÍA METODOLÓGICA (NO RESULTADOS) ===')
    print('Criterio CANDIDATO: salto >= umbral H16; estabilidad futura < umbral H16')
    print('Umbrales propuestos:', dict(zip(ACTUATORS, eps)))
    csv_path = find_csv(root, cfg, args.csv)
    print('CSV original validado SHA256:', csv_path)
    lbs = selected_lookbacks(root)
    data, samples, audit = {}, {}, []
    for v in VERSIONS:
        frames = load_partitions(csv_path, man, v)
        values, scaler = scale_and_values(frames)
        origins = valid_common_origins(values['test'])
        ends, idxs = event_candidates(frames['test'], origins, eps)
        data[v] = (values['test'], scaler)
        samples[v] = (ends, idxs)
        counts = Counter(ACTUATORS[int(j)] for j in idxs)
        row = {'version':v, 'common_origins_complete':len(origins), 'candidate_episodes':len(ends),
               **{f'event_{a}':counts.get(a,0) for a in ACTUATORS}}
        audit.append(row)
        print(f'{v}: orígenes completos={len(origins)}, episodios candidatos={len(ends)}')
        print('   Por actuador:',dict(counts))
        print('   MASK==SUST por lookback:', equivalence_stats(values['test'],ends,idxs,
                                                   set(lbs[(v,m)] for m in MODELS)))
    out = (args.output_root.resolve() if args.output_root is not None
           else root / OUTPUT)
    if args.smoke:
        import torch
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from forecasting.core import models
        require(args.device != 'cuda' or torch.cuda.is_available(),
                'CUDA solicitado pero no disponible')
        device = torch.device(
            'cuda' if args.device == 'cuda' or
            (args.device == 'auto' and torch.cuda.is_available())
            else 'cpu'
        )
        print('Dispositivo del smoke:', device)
        version, model_name, seed = 'V1', 'lstm_sergiot', 8
        L = lbs[(version,model_name)]
        require(L == 48, 'El ejemplo histórico del smoke requiere L48')
        checkpoint = root / 'forecasting/checkpoints' / version / f'H{H}' / f'L{L}' / model_name / f'seed{seed}' / 'best.pt'
        require(checkpoint.is_file(), f'Falta el checkpoint de smoke: {checkpoint}')
        model = models.build_model(model_name,L,H,n_inputs=19).to(device)
        model.load_state_dict(load_state(checkpoint, device), strict=True)
        model.eval()
        actual, predicted = infer_three(model, data[version][0], samples[version][0],
                                        samples[version][1], L, device, args.batch_size)
        norm = data[version][1]
        sd = norm['sd'][[0,2]]
        mu = norm['mean'][[0,2]]
        physical_truth = actual[:,:,[0,2]].astype(np.float64)*sd + mu
        source = root/'forecasting/experiments/actuator_change_sensitivity/results/paired_predictions/V1/L48/lstm_sergiot/seed8.npz'
        with np.load(source,allow_pickle=False) as z:
            require(np.array_equal(z['origin_t'],samples[version][0]-1),
                    'Los orígenes del smoke difieren de los originales')
            require(np.array_equal(z['actuator_index'],samples[version][1]),
                    'Los actuadores del smoke difieren de los originales')
            delta_truth = float(np.abs(physical_truth-z['truth_TBS_RH']).max())
            print('Diferencia máxima ground truth (unidades físicas):',delta_truth)
            require(delta_truth < 1e-4, 'La verdad física no coincide')
            for name in ('OBS','MASK','SUST'):
                physical = predicted[name][:,:,[0,2]].astype(np.float64)*sd+mu
                delta = float(np.abs(physical-z[f'{name.lower()}_TBS_RH']).max())
                print(f'Diferencia máxima {name} (unidades físicas): {delta:.8f}')
                require(delta < 1e-3, f'La reproducción {name} difiere del archivo original')
        print('SMOKE INFERENCIA D OK: origen, actuador, TBS y HR, OBS/MASK/SUST')
        return
    if not args.evaluate:
        print('\nAUDITORÍA terminada: NO se han ejecutado predicciones ni escrito resultados.')
        print('Revisar que los umbrales de evento representan cambios físicos relevantes.')
        return
    require(args.confirm_event_definition,
            'Para inferencia, aprobar explícitamente este criterio de evento con '
            '--confirm-event-definition, tras revisar el informe de auditoría.')
    require(all(len(samples[v][0]) > 0 for v in VERSIONS),
            'Alguna variante no tiene episodios elegibles; revisar definición de evento')
    require(not out.exists() or (out.is_dir() and not any(out.iterdir())),
            f'No se sobrescribe directorio de salida que contiene archivos: {out}')
    import torch
    if args.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA solicitado pero no disponible')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    evaluate(root,VERSIONS,data,samples,lbs,device,args.batch_size,out)
    with (out/'audit_event_counts.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    (out/'protocol.json').write_text(json.dumps({
        'csv_sha256':cfg['data_sha256'],
        'cfg_sha256':published['cfg_sha256'],
        'manifest_sha256':published['manifest_sha256'],
        'repository_cfg_sha256':sha256(cfg_path),
        'repository_manifest_sha256':sha256(man_path),
        'H':H,'event_min_deltas':dict(zip(ACTUATORS,eps.tolist())),
        'event_policy':'exactly_one_large_jump_at_t_and_stable_six_actuators_through_t_plus_H',
        'control_conditions':{'OBS':'observed','MASK':'a[t]=a[t-1]',
                             'SUST':'a[t-15:t]=a[t-1]'},
        'selection':'validation_only','seed_unit':'5 seeds per V,H,model',
        'data_leakage':'none; Train scaler; future used only for episode selection and ground truth'
    },indent=2,ensure_ascii=False),encoding='utf-8')


if __name__ == '__main__':
    main()
