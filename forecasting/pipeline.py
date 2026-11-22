#!/usr/bin/env python3
"""Reproduce Train/Validation, freezes lookbacks and evaluates Test separately.

Examples, run from the repository root:
  python -m forecasting.pipeline preflight
  python -m forecasting.pipeline jobs
  python -m forecasting.pipeline train --id 0
  python -m forecasting.pipeline select                 # all runs finished
  python -m forecasting.pipeline authorize-test --selection-sha SHA256
  python -m forecasting.pipeline test --id 0           # after authorization
  python -m forecasting.pipeline persistence --id 0
  python -m forecasting.pipeline report

No Test observations enter training, checkpoint selection or lookback selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from .data import SENSORS, WIND_INDEX, load_partitions, scaler_and_values, valid_starts, matching_origins, as_loader
from .metrics import metric_tables, mixed_circular_mse, eval_validation_objectives
from .models import build_model

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_DATA = PROJECT_ROOT / 'preprocessing/data/processed/ETIFA_clean_imputed_1h.csv'
DEFAULT_OUTPUT = HERE / 'runs'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def write_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def files_provenance(cfgpath, splits_path, csv):
    # No fixed SHA rejection: rebuilding the same scientific CSV can produce
    # byte-different float serialization across NumPy/Pandas environments.
    source = ('__init__.py', 'architectures.py', 'models.py', 'data.py',
              'metrics.py', 'pipeline.py')
    return {
        'dataset_sha256': digest(csv),
        'config_sha256': digest(cfgpath),
        'splits_sha256': digest(splits_path),
        'source_sha256': {name: digest(HERE / name) for name in source},
    }


def approved_config(cfg):
    if cfg['models'] != ['lstm_sergiot', 'itransformer', 'patchtst_mv']:
        raise ValueError('Only the three reported architectures are allowed')
    if cfg['train_policy'] != 'all_complete_windows':
        raise ValueError('Training must use complete windows, not actuator-stable windows')
    if cfg['validation_test_policy'] != 'range_over_t_to_t_plus_H_strict_less_than_threshold':
        raise ValueError('Unexpected actuator stability policy')
    if cfg['monitoring'] != dict(scheduler='validation_nmae_iqr',
                                 early_stopping='validation_nmae_iqr',
                                 best_checkpoint='validation_nmae_iqr'):
        raise ValueError('Validation nMAE/IQR must monitor all training decisions')
    if cfg['loss']['name'] != 'mixed_circular_mse':
        raise ValueError('Unexpected loss function')
    for v in cfg['versions']:
        if v not in ('V1','V2','V3'):
            raise ValueError('Unexpected temporal split')
    if not cfg['lookbacks'] or max(cfg['lookbacks']) != 672:
        raise ValueError('Expected L672 common-origin history')
    return cfg


def load_context(args):
    configpath = Path(args.config).resolve()
    splitpath = Path(args.splits).resolve()
    csv = Path(args.data).resolve()
    if not csv.is_file():
        raise FileNotFoundError(csv)
    cfg = approved_config(load_json(configpath))
    man = load_json(splitpath)
    if set(cfg['versions']) != set(man['splits']):
        raise ValueError('Mismatch between configuration versions and split definitions')
    prov = files_provenance(configpath, splitpath, csv)
    return cfg, man, csv, Path(args.output).resolve(), prov


def jobs(cfg):
    return [(v, h, L, m, seed)
            for v in cfg['versions'] for h in cfg['horizons']
            for L in cfg['lookbacks'] for m in cfg['models'] for seed in cfg['seeds']]


def jobdir(out, record):
    v, h, L, m, seed = record
    return out / 'train' / v / f'H{h}' / f'L{L}' / m / f'seed{seed}'


def threshold(cfg, H):
    p = cfg['thresholds_by_h'][str(H)]
    from .data import ACTUATORS
    return [p[name] for name in ACTUATORS]


def prepare(cfg, man, csv, v, h, L, include_test=False):
    parts = ('train', 'validation', 'test') if include_test else ('train', 'validation')
    frames = load_partitions(csv, man, v, parts)
    values, scaler = scaler_and_values(frames)
    eps = threshold(cfg, h)
    starts = {'train': valid_starts(frames['train'], values['train'], L, h, None)}
    for part in parts:
        if part != 'train':
            starts[part] = matching_origins(frames[part], values[part], L, h, eps,
                                            max(cfg['lookbacks']))
    for part, indices in starts.items():
        if not len(indices):
            raise RuntimeError(f'No complete windows: {v} H{h} L{L} {part}')
    return frames, values, scaler, starts


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_rng():
    return {'python':random.getstate(), 'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng(states):
    random.setstate(states['python'])
    np.random.set_state(states['numpy'])
    torch.set_rng_state(states['torch'])
    if torch.cuda.is_available() and states['cuda'] is not None:
        torch.cuda.set_rng_state_all(states['cuda'])


def choose_device(arg):
    if arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if arg == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    return torch.device(arg)


def collect(model, values, starts, L, H, batch, device, baseline=False):
    true, pred = [], []
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for x, y in as_loader(values, starts, L, H, batch):
            yp = x[:, -1:, :9].expand(-1,H,-1) if baseline else model(x.to(device)).cpu()
            if yp.shape != y.shape:
                raise RuntimeError(f'Incorrect prediction shape: {yp.shape} vs {y.shape}')
            true.append(y.numpy())
            pred.append(yp.detach().numpy())
    return np.concatenate(true), np.concatenate(pred)


def measure_latency(model, values, starts, L, H, device, baseline=False, repeats=20):
    x = torch.from_numpy(values[int(starts[0]):int(starts[0])+L]).unsqueeze(0).to(device)
    if model is not None:
        model.eval()
    def forward():
        return x[:, -1:, :9].expand(-1,H,-1) if baseline else model(x)
    with torch.inference_mode():
        for _ in range(5):
            forward()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeats):
            t = time.perf_counter()
            forward()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            timings.append(1000*(time.perf_counter()-t))
    return {'device':str(device),'batch_size':1,'repeats':repeats,'warmup':5,
            'median_ms':float(np.median(timings)),
            'p95_ms':float(np.percentile(timings,95))}


def train_one(args, ctx, idx):
    cfg, man, csv, out, prov = ctx
    if (out / 'test_authorization.json').exists() or (out/'selection.json').exists():
        raise RuntimeError('Training is frozen after lookback selection')
    rec = jobs(cfg)[idx]
    v, H, L, m, seed = rec
    dest = jobdir(out, rec)
    finished = dest/'validation.json'
    bestpath = dest/'best.pt'
    if finished.exists():
        result = load_json(finished)
        if result['provenance'] != prov or digest(bestpath) != result['checkpoint_sha256']:
            raise RuntimeError('Existing validation result or checkpoint changed')
        print('ALREADY COMPLETE',rec)
        return
    frames,values,scaler,starts = prepare(cfg,man,csv,v,H,L)
    seed_all(seed)
    device=choose_device(args.device)
    model=build_model(m,L,H).to(device)
    optimcfg=cfg['optimizer'];schedulecfg=cfg['scheduler']
    opt=torch.optim.AdamW(model.parameters(),lr=float(optimcfg['lr']),
                          weight_decay=float(optimcfg['weight_decay']))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode='min',factor=schedulecfg['factor'],patience=schedulecfg['patience'],
        min_lr=schedulecfg['min_lr'],threshold=schedulecfg['threshold'],
        threshold_mode=schedulecfg['threshold_mode'],
        cooldown=schedulecfg['cooldown'],eps=schedulecfg['eps'])
    training=cfg['training']
    batch=int(training['batch_size']);max_epochs=int(training['epochs'])
    patience=int(training['patience']);delta=float(training['min_delta'])
    last=dest/'last.pt';history=[];bad=0;best=math.inf;best_epoch=0
    epoch_start=1;elapsed_before=0.0
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False)
        if saved['provenance'] != prov or tuple(saved['record']) != rec:
            raise RuntimeError('Resume checkpoint source/config/data mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);set_rng(saved['rng'])
        history=saved['history'];bad=saved['bad'];best=saved['best']
        best_epoch=saved['best_epoch'];epoch_start=saved['epoch']+1
        elapsed_before=saved['elapsed']
        if best_epoch and not bestpath.is_file():
            raise RuntimeError('Best checkpoint missing on resumed run')
        print('RESUME',rec,'epoch',epoch_start,flush=True)
    start_clock=time.monotonic()
    for epoch in range(epoch_start,max_epochs+1):
        if bad >= patience: break
        lr_used=float(opt.param_groups[0]['lr'])
        model.train();training_sq=0.0;n=0
        for x,y in as_loader(values['train'],starts['train'],L,H,batch,shuffle=True):
            x=x.to(device);y=y.to(device)
            opt.zero_grad(set_to_none=True)
            predicted=model(x)
            if predicted.shape!=y.shape:
                raise ValueError('Model output shape differs from nine-sensor targets')
            loss=mixed_circular_mse(predicted,y,float(scaler['sd'][WIND_INDEX]))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(training['clip_grad_norm']))
            opt.step();training_sq+=float(loss.item())*y.numel();n+=y.numel()
        val=eval_validation_objectives(model,values['validation'],starts['validation'],
                                       L,H,batch,device,scaler)
        vnmae=float(val['val_nmae_iqr'])
        scheduler.step(vnmae)
        history.append({'epoch':epoch,'train_mixed_loss_z':training_sq/n,
                        'val_mixed_loss_z':val['val_mixed_loss_z'],
                        'val_nmae_iqr':vnmae,'lr_used':lr_used,
                        'lr_next':float(opt.param_groups[0]['lr'])})
        if vnmae < best-delta:
            best=vnmae;bad=0;best_epoch=epoch
            write_torch(bestpath,{'model':model.state_dict(), 'record':rec,
                                  'best_epoch':epoch,'best_val_nmae_iqr':best,
                                  'scaler':scaler,'provenance':prov})
        else:
            bad+=1
        write_torch(last,{'epoch':epoch,'model':model.state_dict(),
                          'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                          'rng':get_rng(),'history':history,'bad':bad,'best':best,
                          'best_epoch':best_epoch,'elapsed':elapsed_before+time.monotonic()-start_clock,
                          'provenance':prov,'record':rec})
        print(f'{rec} epoch={epoch} val_nmae={vnmae:.6f} best={best:.6f}',flush=True)
    if best_epoch<1 or not bestpath.exists():
        raise RuntimeError('Training produced no checkpoint')
    selected=torch.load(bestpath,map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'])
    true,pred=collect(model,values['validation'],starts['validation'],L,H,batch,device)
    global_metrics,by_sensor,by_step,truth_phys,pred_phys=metric_tables(true,pred,scaler,H)
    if not math.isclose(float(global_metrics['nmae_global_iqr']),best,abs_tol=1e-5,rel_tol=1e-6):
        raise RuntimeError('Best Validation checkpoint does not match monitored nMAE')
    # Keep per-origin numeric results. No illustrative Test episode selected here.
    errors = np.abs(pred_phys.astype(np.float64)-truth_phys.astype(np.float64))
    errors[:,:,WIND_INDEX]=np.abs((pred_phys[:,:,WIND_INDEX].astype(np.float64)-
                                  truth_phys[:,:,WIND_INDEX].astype(np.float64)+180)%360-180)
    per_sensor=errors.mean(axis=1)/scaler['iqr'][:9]
    origins=(frames['validation'].timestamp.iloc[starts['validation']+L-1]
             .astype('int64').to_numpy()//10**6)
    write_npz(dest/'validation_origin_errors.npz',origin_epoch_millis=origins,
              nmae_iqr=per_sensor.mean(axis=1).astype(np.float32),
              nmae_iqr_by_sensor=per_sensor.astype(np.float32))
    write_json(finished,{
        'version':v,'horizon':H,'lookback':L,'model':m,'seed':seed,
        'n_train':int(len(starts['train'])),'n_validation':int(len(starts['validation'])),
        'best_epoch':best_epoch,'best_validation_nmae_iqr':best,
        'validation_nmae_iqr':global_metrics['nmae_global_iqr'],
        'validation':{'global':global_metrics,'by_sensor':by_sensor,'by_step':by_step},
        'history':history,'training_seconds':elapsed_before+time.monotonic()-start_clock,
        'n_parameters':int(sum(p.numel() for p in model.parameters())),
        'checkpoint_sha256':digest(bestpath),
        'origin_errors_sha256':digest(dest/'validation_origin_errors.npz'),
        'provenance':prov})
    last.unlink(missing_ok=True)
    print('VALIDATION COMPLETE',rec,flush=True)


def verify_train(result, expected, prov):
    # The caller checks provenance, checkpoint integrity, and run identity.
    if (result['version'],result['horizon'],result['lookback'],
        result['model'],result['seed']) != expected:
        raise RuntimeError(f'Wrong training run identity: {expected}')
    if result['provenance'] != prov:
        raise RuntimeError(f'Provenance changed: {expected}')
    if result['best_epoch']<1 or not math.isfinite(result['validation_nmae_iqr']):
        raise RuntimeError(f'Unusable Validation result: {expected}')


def build_selection(cfg,out,prov):
    done={}
    for rec in jobs(cfg):
        directory=jobdir(out,rec)
        path=directory/'validation.json'
        if not path.is_file():
            raise RuntimeError(f'Validation not complete: {rec}')
        r=load_json(path)
        verify_train(r,rec,prov)
        if digest(directory/'best.pt')!=r['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint changed: {rec}')
        if digest(directory/'validation_origin_errors.npz')!=r['origin_errors_sha256']:
            raise RuntimeError(f'Validation origin scores changed: {rec}')
        done[rec]=r
    selection=[]
    for v in cfg['versions']:
        for H in cfg['horizons']:
            for m in cfg['models']:
                candidate=[]
                for L in cfg['lookbacks']:
                    rows=[done[(v,H,L,m,s)] for s in cfg['seeds']]
                    if len({row['n_validation'] for row in rows})!=1:
                        raise RuntimeError(f'Different Validation counts across seeds: {v,H,L,m}')
                    scores=np.array([row['validation_nmae_iqr'] for row in rows])
                    candidate.append({'lookback':L,'mean_nmae_iqr':float(scores.mean()),
                                      'std_nmae_iqr':float(scores.std(ddof=1)),
                                      'n_validation':rows[0]['n_validation']})
                if len({x['n_validation'] for x in candidate})!=1:
                    raise RuntimeError(f'Validation lookbacks lack common origins: {v,H,m}')
                best=min(candidate,key=lambda x:(x['mean_nmae_iqr'],
                                                 x['std_nmae_iqr'],x['lookback']))
                selection.append({'version':v,'horizon':H,'model':m,
                                  'lookback':best['lookback'],
                                  'mean_validation_nmae_iqr':best['mean_nmae_iqr'],
                                  'std_validation_nmae_iqr':best['std_nmae_iqr'],
                                  'candidates':candidate})
    return {'provenance':prov,'criterion':'mean_5seeds_validation_nmae_iqr_then_std_then_smaller_L',
            'n_training_runs':len(done),'selected':selection},done


def select(args,ctx):
    cfg,man,csv,out,prov=ctx
    if (out/'test_authorization.json').exists():
        raise RuntimeError('Validation selection is frozen after Test authorization')
    frozen,done=build_selection(cfg,out,prov)
    path=out/'selection.json'
    if path.exists():
        if load_json(path)!=frozen:
            raise RuntimeError('Previously frozen selection does not match current Validation')
    else:
        write_json(path,frozen)
    rows=[]
    for rec,item in done.items():
        rows.append({'version':rec[0],'horizon':rec[1],'lookback':rec[2],
                     'model':rec[3],'seed':rec[4],
                     'validation_nmae_iqr':item['validation_nmae_iqr'],
                     'n_validation':item['n_validation'],
                     'n_train':item['n_train'],'best_epoch':item['best_epoch'],
                     'train_seconds':item['training_seconds']})
    pd.DataFrame(rows).to_csv(out/'validation_all.csv',index=False)
    pd.DataFrame([{k:row[k] for k in ('version','horizon','model','lookback',
                 'mean_validation_nmae_iqr','std_validation_nmae_iqr')}
                 for row in frozen['selected']]).to_csv(out/'selected_lookbacks.csv',index=False)
    print('SELECTION FROZEN. REVIEW Validation BEFORE Test:',path)
    print('SELECTION SHA256:',digest(path))
    print('SELECTED GROUPS:',len(frozen['selected']))


def require_authorized(out,prov):
    a=out/'test_authorization.json'; f=out/'selection.json'
    if not a.is_file() or not f.is_file():
        raise RuntimeError('Test blocked: select and explicitly authorize first')
    approval=load_json(a)
    if (approval.get('approved_selection_sha256')!=digest(f) or
        approval.get('provenance')!=prov or
        approval.get('explicitly_authorized') is not True):
        raise RuntimeError('Test blocked: authorization no longer matches immutable inputs')
    selection=load_json(f)
    if selection.get('provenance')!=prov:
        raise RuntimeError('Test blocked: selection or sources changed')
    return selection


def authorize(args,ctx):
    cfg,man,csv,out,prov=ctx
    f=out/'selection.json'
    if not f.exists():
        raise RuntimeError('Must finish and review Validation before authorizing Test')
    frozen,done=build_selection(cfg,out,prov)
    if frozen!=load_json(f):
        raise RuntimeError('Frozen Validation selection no longer matches results')
    actual=digest(f)
    if args.selection_sha!=actual:
        raise RuntimeError(f'Explicit selection SHA does not match; expected {actual}')
    item={'explicitly_authorized':True,'approved_selection_sha256':actual,
          'provenance':prov,'scope':'selected_Test_and_Persistence_only'}
    f=out/'test_authorization.json'
    if f.exists() and load_json(f)!=item:
        raise RuntimeError('Conflicting prior authorization')
    write_json(f,item)
    print('TEST AUTHORIZED (not executed):',actual)


def selected_jobs(cfg,selected):
    return [(row['version'],row['horizon'],row['lookback'],row['model'],seed)
            for row in selected['selected'] for seed in cfg['seeds']]


def eval_one(args,ctx,idx):
    cfg,man,csv,out,prov=ctx
    frozen=require_authorized(out,prov)
    picks=selected_jobs(cfg,frozen)
    rec=picks[idx]
    v,H,L,m,seed=rec
    dest=out/'test'/v/f'H{H}'/m/f'seed{seed}'
    result_path=dest/'test.json'
    ckptdir=jobdir(out,rec)
    val=load_json(ckptdir/'validation.json')
    verify_train(val,rec,prov)
    checkpoint=ckptdir/'best.pt'
    if digest(checkpoint)!=val['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed since Validation')
    if result_path.exists():
        old=load_json(result_path)
        if old['provenance']!=prov or old['checkpoint_sha256']!=digest(checkpoint):
            raise RuntimeError('Existing Test result conflicts with inputs')
        print('ALREADY EVALUATED',rec)
        return
    frames,values,scaler,starts=prepare(cfg,man,csv,v,H,L,include_test=True)
    device=choose_device(args.device)
    model=build_model(m,L,H).to(device)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['provenance']!=prov or tuple(saved['record'])!=rec:
        raise RuntimeError('Checkpoint provenance mismatch')
    model.load_state_dict(saved['model'])
    latency=measure_latency(model,values['test'],starts['test'],L,H,device)
    true,pred=collect(model,values['test'],starts['test'],L,H,
                      int(cfg['training']['batch_size']),device)
    gl,by_sensor,by_step,yp,pp=metric_tables(true,pred,scaler,H)
    errors=pp.astype(np.float64)-yp.astype(np.float64)
    errors[:,:,WIND_INDEX]=(errors[:,:,WIND_INDEX]+180)%360-180
    per_sensor=np.abs(errors).mean(axis=1)/scaler['iqr'][:9]
    origins=(frames['test'].timestamp.iloc[starts['test']+L-1]
             .astype('int64').to_numpy()//10**6)
    write_npz(dest/'test_origin_errors.npz',origin_epoch_millis=origins,
              nmae_iqr=per_sensor.mean(axis=1).astype(np.float32),
              nmae_iqr_by_sensor=per_sensor.astype(np.float32))
    write_json(result_path,{
        'version':v,'horizon':H,'lookback':L,'model':m,'seed':seed,
        'n_test':len(starts['test']), 'test':{'global':gl,'by_sensor':by_sensor,'by_step':by_step},
        'validation_nmae_iqr':val['validation_nmae_iqr'],
        'test_nmae_iqr':gl['nmae_global_iqr'],
        'latency':latency,'n_parameters':val['n_parameters'],
        'train_seconds':val['training_seconds'],
        'checkpoint_sha256':digest(checkpoint),'provenance':prov})
    print('TEST COMPLETE',rec,'nmae_iqr=',gl['nmae_global_iqr'])


def baseline_one(args,ctx,idx):
    cfg,man,csv,out,prov=ctx
    frozen=require_authorized(out,prov)
    groups=sorted({(row['version'],row['horizon']) for row in frozen['selected']},
                  key=lambda g:(cfg['versions'].index(g[0]),cfg['horizons'].index(g[1])))
    v,H=groups[idx]
    # With common Lmax origins, Persistence is identical for every selected L.
    L=max(cfg['lookbacks'])
    dest=out/'persistence'/v/f'H{H}'
    if (dest/'metrics.json').is_file():
        if load_json(dest/'metrics.json')['provenance']!=prov:
            raise RuntimeError('Persistence provenance changed')
        print('ALREADY COMPLETE Persistence',v,H)
        return
    frames,values,scaler,starts=prepare(cfg,man,csv,v,H,L,include_test=True)
    results={}
    for part in ('validation','test'):
        true,pred=collect(None,values[part],starts[part],L,H,
                          int(cfg['training']['batch_size']),torch.device('cpu'),baseline=True)
        gl,by_sensor,by_step,yp,pp=metric_tables(true,pred,scaler,H)
        results[part]={'global':gl,'by_sensor':by_sensor,'by_step':by_step}
    latency=measure_latency(None,values['test'],starts['test'],L,H,
                            torch.device('cpu'),baseline=True)
    write_json(dest/'metrics.json',{'version':v,'horizon':H,'baseline':'persistence',
                                   'lookback_for_common_origins':L,'metrics':results,
                                   'latency':latency,'provenance':prov})
    print('PERSISTENCE COMPLETE',v,H,'test nmae',results['test']['global']['nmae_global_iqr'])


def report(args,ctx):
    cfg,man,csv,out,prov=ctx
    frozen=require_authorized(out,prov)
    records=[]
    for rec in selected_jobs(cfg,frozen):
        v,H,L,m,seed=rec
        path=out/'test'/v/f'H{H}'/m/f'seed{seed}'/'test.json'
        if not path.is_file():
            raise RuntimeError(f'Missing selected Test evaluation {rec}')
        result=load_json(path)
        if result['provenance']!=prov or (v,H,L,m,seed)!=tuple(result[k] for k in
                ('version','horizon','lookback','model','seed')):
            raise RuntimeError(f'Wrong evaluation artifact {rec}')
        records.append({'version':v,'horizon':H,'lookback':L,'model':m,'seed':seed,
                        'validation_nmae_iqr':result['validation_nmae_iqr'],
                        'test_nmae_iqr':result['test_nmae_iqr'],'n_test':result['n_test'],
                        'n_parameters':result['n_parameters'],
                        'train_seconds':result['train_seconds'],
                        'latency_median_ms':result['latency']['median_ms'],
                        'latency_p95_ms':result['latency']['p95_ms'],
                        'latency_device':result['latency']['device']})
    for v in cfg['versions']:
        for H in cfg['horizons']:
            path=out/'persistence'/v/f'H{H}'/'metrics.json'
            if not path.is_file():
                raise RuntimeError(f'Missing Persistence baseline for {v,H}')
            baseline=load_json(path)
            if baseline['provenance']!=prov:
                raise RuntimeError(f'Persistence provenance changed: {v,H}')
            records.append({'version':v,'horizon':H,'lookback':max(cfg['lookbacks']),
                            'model':'persistence','seed':-1,
                            'validation_nmae_iqr':baseline['metrics']['validation']['global']['nmae_global_iqr'],
                            'test_nmae_iqr':baseline['metrics']['test']['global']['nmae_global_iqr'],
                            'n_test':baseline['metrics']['test']['global']['n_windows'],
                            'n_parameters':0,'train_seconds':0,
                            'latency_median_ms':baseline['latency']['median_ms'],
                            'latency_p95_ms':baseline['latency']['p95_ms'],
                            'latency_device':baseline['latency']['device']})
    df=pd.DataFrame(records)
    df.to_csv(out/'test_all.csv',index=False)
    selected=df[df.model!='persistence']
    summary=(selected.groupby(['version','horizon','model','lookback'],as_index=False)
             .agg(validation_mean_nmae=('validation_nmae_iqr','mean'),
                  validation_std_nmae=('validation_nmae_iqr','std'),
                  test_mean_nmae=('test_nmae_iqr','mean'),
                  test_std_nmae=('test_nmae_iqr','std'),
                  n_test=('n_test','first'),seeds=('seed','count'),
                  n_parameters=('n_parameters','first'),
                  train_seconds_mean=('train_seconds','mean'),
                  latency_median_mean_ms=('latency_median_ms','mean'),
                  latency_device=('latency_device','first')))
    baseline_lookup={(r.version,r.horizon):r.test_nmae_iqr for r in
                     df[df.model=='persistence'].itertuples()}
    summary['persistence_test_nmae']=summary.apply(
        lambda r:baseline_lookup[(r.version,r.horizon)],axis=1)
    summary['skill_vs_persistence_pct']=(100*(summary['persistence_test_nmae']-
         summary['test_mean_nmae'])/summary['persistence_test_nmae'])
    summary.to_csv(out/'summary_selected_5_seeds.csv',index=False)
    print('REPORT COMPLETE:',len(selected),'model seeds and',len(baseline_lookup),'baseline groups')


def preflight(args,ctx):
    cfg,man,csv,out,prov=ctx
    print('DATA SHA256:',prov['dataset_sha256'])
    print('HISTORICAL reference dataset SHA256:',man['sha256'])
    if prov['dataset_sha256']!=man['sha256']:
        print('NOTE: byte hashes differ; checking data schema and temporal split counts instead.')
    print('TRAINING JOBS:',len(jobs(cfg)))
    for v in cfg['versions']:
        frames=load_partitions(csv,man,v,('train','validation'))
        values,scaler=scaler_and_values(frames)
        for H in cfg['horizons']:
            starts=matching_origins(frames['validation'],values['validation'],
                                    max(cfg['lookbacks']),H,threshold(cfg,H),max(cfg['lookbacks']))
            if not len(starts):
                raise RuntimeError(f'No Validation windows: {v} H{H}')
            print(v,'H',H,'common Validation origins:',len(starts))
    for m in cfg['models']:
        for L,H in ((48,4),(672,288)):
            model=build_model(m,L,H)
            model.eval()
            with torch.no_grad():
                x=torch.zeros(1,L,19)
                p=model(x)
                if p.shape!=(1,H,9) or not torch.isfinite(p).all():
                    raise RuntimeError(f'Failed model smoke: {m} L{L} H{H}')
            print('MODEL OK:',m,'L',L,'H',H)
    print('PREFLIGHT OK')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('preflight','jobs','train','select',
                                           'authorize-test','test','persistence','report'))
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--splits',type=Path,default=HERE/'splits.json')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--id',type=int,default=None,help='0-based job index for train/test/persistence')
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--selection-sha',help='Reviewed frozen selection SHA-256; required to approve Test')
    args=parser.parse_args()
    ctx=load_context(args)
    cfg,man,csv,out,prov=ctx
    if args.command=='jobs':
        print('TRAIN',len(jobs(cfg)))
        for i,rec in enumerate(jobs(cfg)):
            print(i,*rec)
    elif args.command=='preflight':
        preflight(args,ctx)
    elif args.command=='train':
        if args.id is None or not 0<=args.id<len(jobs(cfg)):
            parser.error(f'--id required in range 0..{len(jobs(cfg))-1}')
        train_one(args,ctx,args.id)
    elif args.command=='select':
        select(args,ctx)
    elif args.command=='authorize-test':
        if not args.selection_sha:
            parser.error('Provide --selection-sha from reviewed Validation selection')
        authorize(args,ctx)
    elif args.command=='test':
        if args.id is None:
            parser.error('--id required; selected Test indices start at zero')
        eval_one(args,ctx,args.id)
    elif args.command=='persistence':
        if args.id is None:
            parser.error('--id required; baseline indices start at zero')
        baseline_one(args,ctx,args.id)
    elif args.command=='report':
        report(args,ctx)


if __name__=='__main__':
    main()
