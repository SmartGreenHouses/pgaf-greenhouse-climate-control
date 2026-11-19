"""Entrenamiento y comprobación de modelos ETIFA directos multisalida."""
import argparse
import json
import os
import random
import time
from pathlib import Path

# Configurar cuBLAS antes de inicializar CUDA.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StableDataset, get_scaler, SEEDS, WINDOWS, HORIZONS
from .models import build_model
from .metrics import score

MODELS = ('lstm_original', 'lstm_sergiot', 'itransformer', 'patchtst_mv', 'persistence')


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def evaluate(model, loader, device, scaler):
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            p = model(x.to(device)).cpu().numpy()
            targets.append(y.numpy()); preds.append(p)
    return score(np.concatenate(targets), np.concatenate(preds), scaler)


def latency(model, dataset, device, repeats=30):
    model.eval()
    x = dataset[0][0].unsqueeze(0).to(device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(5):
            model(x)
        sync()
        samples = []
        for _ in range(repeats):
            sync(); t0 = time.perf_counter()
            model(x); sync()
            samples.append((time.perf_counter() - t0) * 1000)
    return {'median_ms': float(np.median(samples)), 'p95_ms': float(np.percentile(samples, 95)),
            'batch_size': 1, 'warmup': 5, 'repetitions': repeats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--window', type=int, choices=WINDOWS, required=True)
    p.add_argument('--horizon', type=int, choices=HORIZONS, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, default=13)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument(
        '--campaign',
        choices=('pilot_gpu_checks', 'pilot_reproducibility', 'pilot_itransformer_norm', 'pilot_runtime_heavy', 'campaign_v1_stable_angular', 'campaign_v2_swapped_seasons', 'campaign_v3_early_test', 'campaign_v3_early_test_race'),
        default='campaign_v1_stable_angular',
        help='Directorio de resultados del experimento',
    )
    p.add_argument('--smoke-test', action='store_true', help='Solo validar ventanas y formas, sin entrenar')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA solicitado, pero no disponible')
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else
                          'cpu' if args.device == 'auto' else args.device)
    set_seed(args.seed)
    scaler = get_scaler(args.root)
    train = StableDataset(args.root, 'train', args.window, args.horizon, scaler)
    val = StableDataset(args.root, 'val', args.window, args.horizon, scaler)
    if not len(train) or not len(val):
        raise RuntimeError(f'Sin ventanas suficientes: Train={len(train)}, Val={len(val)}')
    model = build_model(args.model, args.window, args.horizon).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f'Modelo={args.model} L={args.window} H={args.horizon} seed={args.seed} device={device}')
    print(f'Ventanas estables: Train={len(train):,}, Val={len(val):,}; parámetros={nparams:,}')
    with torch.no_grad():
        x, y = train[0]
        yp = model(x.unsqueeze(0).to(device)).cpu()
        assert x.shape == (args.window, 19), x.shape
        assert y.shape == (args.horizon, 10), y.shape
        assert yp.shape == (1, args.horizon, 10), yp.shape
        assert torch.isfinite(yp).all(), 'Predicción no finita'
        print(f'Formas verificadas: X={tuple(x.shape)}, Y={tuple(y.shape)}, pred={tuple(yp.shape)}')
        print('Primer origen y fin:', train.origin_and_end(0))
    if args.smoke_test:
        print('SMOKE TEST CORRECTO. No se ha entrenado ni evaluado Test.')
        return
    if args.model == 'persistence':
        print('Persistencia: evaluación independiente pendiente.')
        return
    lr = args.lr if args.lr is not None else 1e-4
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )
    criterion = torch.nn.MSELoss()
    output_root = (
        args.root / 'results' / 'direct_multioutput'
        / args.campaign
    )
    checkpoint_dir = output_root / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    name = f'{args.model}_L{args.window}_H{args.horizon}_seed{args.seed}'
    checkpoint = checkpoint_dir / (name + '.pt')
    report_path = output_root / 'runs' / (name + '.json')
    if checkpoint.exists():
        raise FileExistsError(f'Checkpoint ya existe: {checkpoint}. No se sobrescribe.')
    if report_path.exists():
        raise FileExistsError(f'Informe ya existe: {report_path}. No se sobrescribe.')
    best, patience_left, best_epoch = float('inf'), args.patience, 0
    t0 = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f'Pérdida Train no finita: época {epoch}'
                )
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError(
                        f'Gradiente no finito: época {epoch}'
                    )
            optimizer.step()
            total += float(loss.item()) * len(x); count += len(x)
        model.eval(); vtotal, vcount = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f'Pérdida Validation no finita: época {epoch}'
                    )
                vtotal += float(loss.item()) * len(x); vcount += len(x)
        train_mse, val_mse = total / count, vtotal / vcount
        history.append({
            'epoch': epoch,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })
        scheduler.step(val_mse)
        print(f'Epoch {epoch:3d} | train MSE {train_mse:.6f} | val MSE {val_mse:.6f}', flush=True)
        if val_mse < best - 1e-8:
            best, best_epoch, patience_left = val_mse, epoch, args.patience
            torch.save({'state_dict': model.state_dict(), 'model': args.model,
                        'window': args.window, 'horizon': args.horizon,
                        'seed': args.seed, 'scaler_sha256': scaler['sha256_train']}, checkpoint)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print('Early stopping.'); break
    training_sec = time.perf_counter() - t0
    try:
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        saved = torch.load(checkpoint, map_location=device)
    model.load_state_dict(saved['state_dict'])
    report = {
        'model': args.model, 'window': args.window, 'horizon': args.horizon,
        'seed': args.seed, 'scenario': 'stable_actuators',
        'n_train_windows': len(train), 'n_val_windows': len(val),
        'n_parameters': nparams, 'device': str(device), 'torch_version': torch.__version__,
        'learning_rate': lr, 'batch_size': args.batch_size,
        'best_epoch': best_epoch, 'best_val_mse': best,
        'training_seconds': training_sec, 'history': history,
        'val_metrics': evaluate(model, val_loader, device, scaler),
        'latency': latency(model, val, device), 'checkpoint': str(checkpoint),
    }
    out = output_root / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (name + '.json')
    with dest.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False))
    print('Modelo guardado:', checkpoint)
    print('Informe guardado:', dest)

if __name__ == '__main__':
    main()
