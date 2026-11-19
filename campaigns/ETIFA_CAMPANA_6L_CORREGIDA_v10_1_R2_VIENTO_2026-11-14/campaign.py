#!/usr/bin/env python3
"""ETIFA new CSV: 9 sensor targets, 6 actuator inputs, 6 lookbacks, 9 horizons.

Design: full-window Train; identical actuator-stability-filtered common-origin Validation
for all lookbacks; frozen Validation-based lookback selection; selected Test afterward.
Original ETIFA architectures are imported without editing src/etifa_direct.
Test and Persistence require explicit post-review SHA-256-bound approval.
For each V/H/model/lookback group, the best illustrative episode is selected
by the five-seed MEAN of nMAE(TBS) and nMAE(HR), independently in Validation
and (after authorization) Test. All seed metrics remain complete; a posteriori
Test episode selection never changes checkpoint, seed, lookback, or test scores.
"""
import argparse
import hashlib
import fcntl
import json
import math
import os
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before CUDA initialization for deterministic cuBLAS operations.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
SENSORS = ('dry_bulb_temperature','wet_bulb_temperature','relative_humidity',
           'humidity_deficit','outdoor_temperature','outdoor_relative_humidity',
           'wind_speed','wind_direction','radiation')
ACTUATORS = ('pipe_temperature','humidification','window_1_position',
             'window_2_position','screen_1_position','screen_2_position')
COLS = ('timestamp',) + SENSORS + ACTUATORS
EXPECTED_HASH = '3d74ad9113a9b35a03059786eeeb007119bd7107de6b74eb346625a65673dc15'
WIND_INDEX = SENSORS.index('wind_direction')
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'REQUEST STOP: signal={signum}; checkpoint at end of current epoch', flush=True)


def hash_file(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp,'w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_torch(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp)
    with open(tmp,'rb') as fd:os.fsync(fd.fileno())
    os.replace(tmp,path)


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+'.tmp.npz')
    np.savez_compressed(tmp,**arrays)
    with open(tmp,'rb') as fd:os.fsync(fd.fileno())
    os.replace(tmp,path)


def artifact_hashes(dest, filenames):
    """Record file content digests, not just path existence."""
    dest=Path(dest)
    return {name:hash_file(dest/name) for name in filenames}


def require_artifacts(dest, expected, context):
    if not isinstance(expected,dict) or not expected:
        raise RuntimeError(f'{context}: missing artifact integrity manifest')
    for name,sha in expected.items():
        if Path(name).name!=name or not isinstance(sha,str) or len(sha)!=64:
            raise RuntimeError(f'{context}: invalid artifact manifest entry {name!r}')
        path=Path(dest)/name
        if not path.is_file() or hash_file(path)!=sha:
            raise RuntimeError(f'{context}: corrupted or missing artifact {path}')


def read_json(path):
    with open(path,encoding='utf-8') as fd:
        return json.load(fd)


def compare_close(a,b,label,atol=1e-12):
    if not (math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=atol)):
        raise RuntimeError(f'{label}: numerical inconsistency ({a} vs {b})')


def verify_validation(dest,prov,verify_checkpoint=False):
    """Validate the checkpoint/result agreement before using Validation scores."""
    dest=Path(dest); done=dest/'validation_done.json'
    d=read_json(done)
    required=['run_spec.json','validation_metrics.json','train_history.json','validation_origin_errors.npz']
    require_artifacts(dest,d.get('artifacts_sha256'),f'Validation {dest}')
    # Shared observed Validation trajectory is immutable and checked per run.
    truth_name=d.get('validation_truth_name')
    truth_sha=d.get('validation_truth_sha256')
    if not isinstance(truth_name,str) or not truth_name.endswith('.npz') or Path(truth_name).name!=truth_name:
        raise RuntimeError(f'Validation truth reference invalid: {dest}')
    truth_path=dest.parents[5]/'validation_truth'/truth_name
    if not truth_path.is_file() or hash_file(truth_path)!=truth_sha:
        raise RuntimeError(f'Validation truth damaged or missing: {truth_path}')
    if set(d['artifacts_sha256'])!=set(required):
        raise RuntimeError(f'Validation manifest has unexpected files: {dest}')
    if d.get('campaign_code_sha256')!=prov['campaign_code_sha256']:
        raise RuntimeError(f'Validation code hash changed: {dest}')
    met=read_json(dest/'validation_metrics.json')
    spec=read_json(dest/'run_spec.json')
    hist=read_json(dest/'train_history.json')
    if not hist or not isinstance(hist,list):
        raise RuntimeError(f'Validation history missing: {dest}')
    if spec.get('run_key')!=d.get('run_key') or d['best_epoch']<1:
        raise RuntimeError(f'Validation run identity mismatch: {dest}')
    if spec.get('loss_config',{}).get('name')!='mixed_circular_mse' or d.get('loss_name')!='mixed_circular_mse':
        raise RuntimeError(f'Validation loss configuration mismatch: {dest}')
    if hist[-1]['epoch']<d['best_epoch']:
        raise RuntimeError(f'Best epoch exceeds training history: {dest}')
    compare_close(d['val_nmae_iqr'],met['global']['nmae_global_iqr'],'Validation nMAE')
    with np.load(dest/'validation_origin_errors.npz',allow_pickle=False) as z:
        if len(z['nmae_iqr'])!=int(d['n_validation']) or z['nmae_iqr_by_sensor'].shape!=(int(d['n_validation']),9):
            raise RuntimeError('Validation per-origin errors are malformed')
        compare_close(np.mean(z['nmae_iqr']),d['val_nmae_iqr'],'Validation per-origin nMAE',atol=1e-6)
    if spec.get('monitoring')!={'scheduler':'validation_nmae_iqr',
                               'early_stopping':'validation_nmae_iqr',
                               'best_checkpoint':'validation_nmae_iqr'}:
        raise RuntimeError(f'Validation monitor configuration mismatch: {dest}')
    if d.get('monitor_metric')!='validation_nmae_iqr':
        raise RuntimeError(f'Validation monitor identity mismatch: {dest}')
    if int(d['n_train'])!=int(spec['n_windows']['train']) or int(d['n_validation'])!=int(spec['n_windows']['validation']):
        raise RuntimeError(f'Validation window count mismatch: {dest}')
    if int(d['n_parameters'])!=int(spec['n_parameters']):
        raise RuntimeError(f'Validation parameter count mismatch: {dest}')
    delta=float(spec.get('min_delta',1e-6))
    expected_best=math.inf;expected_epoch=None
    for z in hist:
        if 'val_nmae_iqr' not in z or not math.isfinite(float(z['val_nmae_iqr'])):
            raise RuntimeError(f'Validation nMAE missing/non-finite in history: {dest}')
        if float(z['val_nmae_iqr'])<expected_best-delta:
            expected_best=float(z['val_nmae_iqr']);expected_epoch=int(z['epoch'])
    if expected_epoch!=int(d['best_epoch']):
        raise RuntimeError(f'Validation selected epoch mismatch: {dest}')
    compare_close(d['best_val_nmae_iqr'],expected_best,'Best epoch nMAE/IQR',atol=1e-9)
    compare_close(d['val_nmae_iqr'],expected_best,'Best checkpoint nMAE/IQR',atol=1e-5)
    # The MSE is always logged, but is NOT used for selecting best/checkpoint/scheduler.
    selected_history=next(z for z in hist if int(z['epoch'])==int(d['best_epoch']))
    compare_close(d['best_epoch_val_mixed_loss_z'],selected_history['val_mixed_loss_z'],
                  'Selected epoch mixed circular loss',atol=1e-8)
    if verify_checkpoint:
        bp=dest/'best.pt'
        if not bp.is_file() or hash_file(bp)!=d['checkpoint_sha256']:
            raise RuntimeError(f'Validation checkpoint damaged: {dest}')
    return d


def verify_eval_done(dest,marker,expected_selection_sha,expected_run=None,expected_checkpoint=None):
    """Check all predictions/metrics for both models and Persistence."""
    dest=Path(dest); d=read_json(dest/marker)
    if marker=='test_done.json':
        required={'test_origin_errors.npz','metrics.json'}
        if d.get('test_episode_policy')!='best_five_seed_mean_TBS_HR_illustration_only':
            raise RuntimeError('Test episode policy must be group-based and illustrative')
    else:
        required={'validation_predictions.npz','test_predictions.npz','metrics.json'}
    if set(d.get('artifacts_sha256',{}))!=required:
        raise RuntimeError(f'Evaluation manifest incomplete: {dest}')
    require_artifacts(dest,d['artifacts_sha256'],f'Evaluation {dest}')
    if d.get('frozen_selection_sha256')!=expected_selection_sha:
        raise RuntimeError(f'Frozen selection changed for {dest}')
    if expected_run is not None and d.get('run_key')!=expected_run:
        raise RuntimeError(f'Wrong evaluation key: {dest}')
    if expected_checkpoint is not None and d.get('checkpoint_sha256')!=expected_checkpoint:
        raise RuntimeError(f'Checkpoint changed after evaluation: {dest}')
    met=read_json(dest/'metrics.json')
    compare_close(d['test_nmae_iqr'],met['test']['global']['nmae_global_iqr'],'Test nMAE')
    if marker=='test_done.json':
        with np.load(dest/'test_origin_errors.npz',allow_pickle=False) as origin:
            by_seed=origin['nmae_iqr']
            if len(by_seed)!=int(d['n_test']) or origin['nmae_iqr_by_sensor'].shape!=(len(by_seed),len(SENSORS)):
                raise RuntimeError('Test per-origin errors are malformed')
            compare_close(np.mean(by_seed),d['test_nmae_iqr'],
                          'Test nMAE vs per-origin archive',atol=1e-6)
    if 'validation_nmae_iqr' in d:
        compare_close(d['validation_nmae_iqr'],met['validation']['global']['nmae_global_iqr'],'Validation nMAE')
    if int(d['n_test'])!=int(met['test']['global']['n_windows']):
        raise RuntimeError(f'Test count mismatch: {dest}')
    return d


def measure_latency(model,values,starts,L,H,device,warmup=5,repeats=20,baseline=False):
    """Batch=1 warm inference latency; sync CUDA; explicitly label device."""
    if len(starts)==0:raise ValueError('Cannot benchmark latency with no windows')
    st=int(starts[0]);x=torch.from_numpy(values[st:st+L]).unsqueeze(0).to(device)
    if model is not None:model.eval()
    def forward():
        return x[:,-1:,:9].expand(-1,H,-1) if baseline else model(x)
    if device.type=='cuda':torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(warmup):forward()
        if device.type=='cuda':torch.cuda.synchronize(device)
        samples=[]
        for _ in range(repeats):
            t0=time.perf_counter();forward()
            if device.type=='cuda':torch.cuda.synchronize(device)
            samples.append((time.perf_counter()-t0)*1000)
    return {'device':str(device),'batch_size':1,'warmup':warmup,'repeats':repeats,
            'median_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),
            'mean_ms':float(np.mean(samples))}


def read_config():
    cfg=json.loads((HERE/'experiment.json').read_text(encoding='utf-8'))
    man=json.loads((HERE/'split_manifest.json').read_text(encoding='utf-8'))
    constraints={
        'lookbacks':[48,96,192,288,384,672],
        'horizons':[4,8,16,24,48,96,144,192,288],
        'versions':['V1','V2','V3'],
        'models':['lstm_original','lstm_sergiot','itransformer','patchtst_mv'],
        'seeds':[13,27,22,8,31],
        'train_policy':'all_complete_windows'}
    if cfg.get('optimizer')!={'name':'AdamW','lr':0.0001,'weight_decay':0.01}:
        raise ValueError('The approved AdamW configuration has changed')
    expected_scheduler={'name':'ReduceLROnPlateau','factor':0.5,'patience':4,
                        'min_lr':0.000001,'threshold':0.0001,
                        'threshold_mode':'rel','cooldown':0,'eps':1e-08}
    if cfg.get('scheduler')!=expected_scheduler:
        raise ValueError('The approved LR scheduler configuration has changed')
    expected_monitoring={'scheduler':'validation_nmae_iqr','early_stopping':'validation_nmae_iqr',
                         'best_checkpoint':'validation_nmae_iqr'}
    if cfg.get('monitoring')!=expected_monitoring:
        raise ValueError('The approved Validation nMAE monitoring policy has changed')
    expected_loss={'name':'mixed_circular_mse','wind_direction':'shortest_angular_difference',
                   'wind_period_degrees':360,'wind_scale':'train_zscore_std',
                   'sensor_reduction':'equal_weight_mean_over_9'}
    if cfg.get('loss')!=expected_loss:
        raise ValueError('The approved mixed circular loss configuration has changed')
    h48=cfg.get('thresholds_by_h',{}).get('48',{})
    approved_h48={'name':'pantallas_libres_ventanas_40','pipe_temperature':8,
                  'humidification':0.5,'window_1_position':40,'window_2_position':40,
                  'screen_1_position':None,'screen_2_position':None}
    if h48!=approved_h48:
        raise ValueError('H48 thresholds do not match the approved free-screens policy')
    if cfg.get('test_requires_explicit_approval') is not True:
        raise ValueError('Test must require independent manual approval')
    if cfg.get('prediction_archiving_policy')!='best_5seed_TBS_HR_episode_per_phase_keep_all_numeric_results':
        raise ValueError('Unexpected approved v10 episode retention policy')
    if cfg.get('data_sha256')!=EXPECTED_HASH or man.get('sha256')!=EXPECTED_HASH:
        raise ValueError('Unexpected CSV SHA-256 in experiment manifest')
    for key,expected in constraints.items():
        if cfg.get(key)!=expected:raise ValueError(f'Unexpected experiment configuration {key}')
    return cfg,man


def make_optimizer_and_scheduler(model,cfg):
    """One frozen training recipe for all four historical architectures."""
    opt_cfg=cfg['optimizer'];sch_cfg=cfg['scheduler']
    if opt_cfg.get('name')!='AdamW' or sch_cfg.get('name')!='ReduceLROnPlateau':
        raise ValueError('Frozen campaign requires AdamW and ReduceLROnPlateau')
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(opt_cfg['lr']),
                                weight_decay=float(opt_cfg['weight_decay']))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=float(sch_cfg['factor']),
        patience=int(sch_cfg['patience']),threshold=float(sch_cfg['threshold']),
        threshold_mode=sch_cfg['threshold_mode'],cooldown=int(sch_cfg['cooldown']),
        min_lr=float(sch_cfg['min_lr']),eps=float(sch_cfg['eps']))
    return optimizer,scheduler


def gradient_input_audit(gradient,model_name,L,H):
    """Fail closed if ANY of 19 historical inputs is disconnected at initialisation.

    Uses a deterministic gradient of the real preflight batch. Nonzero gradients
    show computational connectivity, NOT predictive usefulness or causal impact.
    """
    if gradient is None or gradient.ndim!=3 or gradient.shape[-1]!=19:
        raise RuntimeError(f'{model_name}/L{L}/H{H}: missing input gradient')
    if not torch.isfinite(gradient).all():
        raise RuntimeError(f'{model_name}/L{L}/H{H}: non-finite input gradients')
    peak=gradient.detach().abs().amax(dim=(0,1)).cpu().tolist()
    missing=[i for i,amount in enumerate(peak) if amount==0.0]
    if missing:
        raise RuntimeError(f'{model_name}/L{L}/H{H}: disconnected input channels {missing}; '
                           'inspect the historical architecture before training')
    return {'sensor_peak_abs':peak[:9],'actuator_peak_abs':peak[9:15],
            'time_peak_abs':peak[15:19],
            'interpretation':'Gradient connectivity only, not feature importance'}


def jobs(cfg):
    return [(v,h,L,m,s) for v in cfg['versions'] for h in cfg['horizons']
            for L in cfg['lookbacks'] for m in cfg['models'] for s in cfg['seeds']]


def run_key(v,h,L,m,seed):
    return f'{v}/H{h}/L{L}/{m}/seed{seed}'


def policy(cfg,h):
    p=cfg['thresholds_by_h'][str(h)]
    return p['name'], [p[k] for k in ACTUATORS]


def load_partitions(csv, man, version, parts=('train','validation')):
    df=pd.read_csv(csv)
    if tuple(df.columns)!=COLS:
        raise ValueError(f'CSV schema mismatch: {list(df.columns)}')
    if len(df)!=man['n_rows']:
        raise ValueError(f'CSV row count mismatch: {len(df)}')
    df['timestamp']=pd.to_datetime(df['timestamp'],utc=True,errors='raise')
    if df['timestamp'].isna().any() or not df['timestamp'].is_monotonic_increasing or df['timestamp'].duplicated().any():
        raise ValueError('Unsorted or duplicate timestamps')
    result={}
    for part in parts:
        d=man['splits'][version][part]
        lo=pd.Timestamp(d['start_utc']);hi=pd.Timestamp(d['end_utc'])
        selected=df.loc[df.timestamp.between(lo,hi,inclusive='both')].copy().reset_index(drop=True)
        if len(selected)!=d['rows'] or selected.empty:
            raise ValueError(f'{version}/{part}: row count mismatch')
        if not selected.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
            raise ValueError(f'{version}/{part}: timestamps not continuous')
        if selected.timestamp.iloc[0]!=lo or selected.timestamp.iloc[-1]!=hi:
            raise ValueError(f'{version}/{part}: unexpected endpoints')
        result[part]=selected
    return result


def scaler_and_values(frames):
    if 'train' not in frames:
        raise ValueError('Train must be supplied to fit scaler')
    train=frames['train'][list(SENSORS+ACTUATORS)].to_numpy(dtype=np.float64)
    mean=np.nanmean(train,axis=0)
    sd=np.nanstd(train,axis=0)
    q25=np.nanpercentile(train,25,axis=0)
    q75=np.nanpercentile(train,75,axis=0)
    iqr=q75-q25
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or not np.isfinite(iqr[:9]).all():
        raise ValueError('Scaler contains invalid stats')
    if (iqr[:9]<=0).any(): raise ValueError('Zero or negative sensor IQR')
    sd[sd<=1e-12]=1.0
    result={}
    for part,frame in frames.items():
        physical=frame[list(SENSORS+ACTUATORS)].to_numpy(dtype=np.float64)
        values=(physical-mean)/sd
        t=frame.timestamp.dt
        minutes=(t.hour*60+t.minute).to_numpy(dtype=np.float64)
        day=t.dayofyear.to_numpy(dtype=np.float64)
        cyclic=np.column_stack((np.sin(2*np.pi*minutes/1440),np.cos(2*np.pi*minutes/1440),
                                np.sin(2*np.pi*day/365.25),np.cos(2*np.pi*day/365.25)))
        result[part]=np.column_stack((values,cyclic)).astype(np.float32)
    return result, dict(mean=mean,sd=sd,iqr=iqr)


def valid_starts(frame,values,L,H,eps=None):
    """Window [start..start+L-1], target [start+L..start+L+H-1].
    Actor stability over origin t at start+L-1 through forecast end inclusive.
    """
    n=len(frame)
    if n<L+H:return np.empty(0,dtype=np.int64)
    okx=np.isfinite(values).all(axis=1)
    oky=np.isfinite(values[:,:9]).all(axis=1)
    bx=np.r_[0,np.cumsum(~okx,dtype=np.int64)]
    by=np.r_[0,np.cumsum(~oky,dtype=np.int64)]
    starts=np.arange(n-L-H+1,dtype=np.int64)
    end=starts+L
    finish=end+H
    keep=(bx[end]-bx[starts]==0)&(by[finish]-by[end]==0)
    if eps is not None:
        if len(eps)!=len(ACTUATORS):
            raise ValueError('Exactly six actuator thresholds required')
        a=frame[list(ACTUATORS)].to_numpy(dtype=np.float64)
        # Availability and stability are independent: even unconstrained screens
        # must be observed at EVERY time from origin t through t+H inclusive.
        bad_a=(~np.isfinite(a).all(axis=1)).astype(np.int64)
        ca=np.r_[0,np.cumsum(bad_a,dtype=np.int64)]
        keep &= (ca[finish]-ca[end-1]==0)
        for j,threshold in enumerate(eps):
            if threshold is None:continue
            if threshold<=0:raise ValueError('Threshold must be positive')
            z=pd.Series(a[:,j])
            mx=z.rolling(H+1,min_periods=H+1).max().to_numpy()
            mn=z.rolling(H+1,min_periods=H+1).min().to_numpy()
            keep &= (mx[finish-1]-mn[finish-1]) < threshold
    return starts[keep]


def matching_origins(frame,values,L,H,eps,maxL):
    """Use the Lmax-valid origins for ALL L to guarantee fair comparison."""
    if L>maxL:raise ValueError('L > maxL')
    starts_max=valid_starts(frame,values,maxL,H,eps)
    origins=starts_max+maxL-1
    starts=origins-L+1
    if len(starts) and not np.isin(starts,valid_starts(frame,values,L,H,eps)).all():
        raise RuntimeError('Common-origin intersection inconsistency')
    return starts


class WindowDataset(Dataset):
    def __init__(self,values,starts,L,H):
        self.values=values
        self.starts=np.asarray(starts,dtype=np.int64)
        self.L=L;self.H=H
    def __len__(self):return len(self.starts)
    def __getitem__(self,idx):
        s=int(self.starts[idx]);t=s+self.L
        return torch.from_numpy(self.values[s:t]),torch.from_numpy(self.values[t:t+self.H,:9].copy())


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(d):
    random.setstate(d['python']);np.random.set_state(d['numpy']);torch.set_rng_state(d['torch'])
    if torch.cuda.is_available() and d['cuda'] is not None:
        torch.cuda.set_rng_state_all(d['cuda'])


def build_model(repo,model_name,L,H):
    src=Path(repo)/'src'
    if not (src/'etifa_direct/models.py').is_file():
        raise FileNotFoundError(src/'etifa_direct/models.py')
    if str(src) not in sys.path:sys.path.insert(0,str(src))
    from etifa_direct import models
    models.N_SENSORS=9
    model=models.build_model(model_name,L,H,n_inputs=19)
    if not isinstance(model,torch.nn.Module):raise TypeError(f'{model_name} not a torch Module')
    return model


def check_device(allow_cpu=False):
    if not allow_cpu and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; refusing to start training on CPU')
    return torch.device('cuda' if torch.cuda.is_available() and not allow_cpu else 'cpu')


def prepare(cfg,man,csv,version,H,L,include_test=False):
    parts=('train','validation','test') if include_test else ('train','validation')
    frames=load_partitions(csv,man,version,parts)
    values,scaler=scaler_and_values(frames)
    pname,eps=policy(cfg,H)
    train_starts=valid_starts(frames['train'],values['train'],L,H,None)
    maxL=max(cfg['lookbacks'])
    origins={part:matching_origins(frames[part],values[part],L,H,eps,maxL)
             for part in parts if part!='train'}
    starts={'train':train_starts,**origins}
    if any(len(a)==0 for a in starts.values()):
        raise RuntimeError(f'{version}/H{H}/L{L}: no samples for partition {[k for k,a in starts.items() if not len(a)]}')
    return frames,values,scaler,starts


def metric_tables(y_true,y_pred,scaler,H):
    """Inverse-transform to original units and evaluate circular wind direction."""
    y_true=np.asarray(y_true,dtype=np.float64);y_pred=np.asarray(y_pred,dtype=np.float64)
    if y_true.shape!=y_pred.shape or y_true.ndim!=3 or y_true.shape[1:]!=(H,9):
        raise ValueError(f'Invalid arrays for metrics {y_true.shape} / {y_pred.shape}')
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError('Non-finite predictions/targets')
    means=scaler['mean'][:9];std=scaler['sd'][:9];iqrs=scaler['iqr'][:9]
    yt=y_true*std+means;yp=y_pred*std+means
    # Circular predictions are physically identical modulo 360 degrees.
    # Publish a canonical angle in [0,360) for human-readable forecasts.
    # Keep observed y_true untouched, including any original 360-degree readings.
    yp[:,:,WIND_INDEX]=np.mod(yp[:,:,WIND_INDEX],360.0)
    err=yp-yt
    err[:,:,WIND_INDEX]=(err[:,:,WIND_INDEX]+180.0)%360.0-180.0
    ae=np.abs(err)
    sq=err**2
    by_sensor=[];by_step=[]
    for j,name in enumerate(SENSORS):
        true=yt[:,:,j]; e=err[:,:,j]
        # Conventional R² is not invariant to circular wrap (359° and 1° are
        # only 2° apart). Do NOT calculate or report it for wind direction.
        # The physically meaningful angular MAE/RMSE/nMAE are retained above.
        if j==WIND_INDEX:
            r2=None
            r2_status='not_applicable_angular'
        else:
            sst=np.sum((true-true.mean())**2)
            r2=float(1-np.sum(e**2)/sst) if sst>0 else None
            r2_status='linear' if sst>0 else 'undefined_constant_truth'
        by_sensor.append({'sensor':name,'error_type':'angular' if j==WIND_INDEX else 'linear',
                          'mae':float(np.mean(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),
                          'mse':float(np.mean(e*e)),'bias':float(np.mean(e)),
                          'p95_abs_error':float(np.percentile(np.abs(e),95)),
                          'nmae_iqr':float(np.mean(np.abs(e))/iqrs[j]),'iqr_train':float(iqrs[j]),
                          'r2_linear':r2,'r2_status':r2_status})
    for step in range(H):
        row={'step':step+1,'minutes_ahead':(step+1)*15,'n':int(len(yt)),
             'global_nmae_iqr':float(np.mean(ae[:,step,:].mean(axis=0)/iqrs))}
        for j,name in enumerate(SENSORS):
            row['nmae_'+name]=float(np.mean(ae[:,step,j])/iqrs[j])
            row['mae_'+name]=float(np.mean(ae[:,step,j]))
        by_step.append(row)
    global_metrics={'n_windows':int(len(yt)),'horizon':H,
                    'nmae_global_iqr':float(np.mean([x['nmae_iqr'] for x in by_sensor])),
                    'sensor_count':9,
                    'note':'Wind direction uses wrapped angular errors; conventional R2 is not applicable to wind direction (null). Other sensor R2 values use linear physical units.'}
    return global_metrics,by_sensor,by_step,yt.astype(np.float32),yp.astype(np.float32)


def as_loader(values,starts,L,H,batch_size,shuffle=False):
    return DataLoader(WindowDataset(values,starts,L,H),batch_size=batch_size,
                      shuffle=shuffle,num_workers=0,pin_memory=False)


def collect_predictions(model,values,starts,L,H,batch_size,device,baseline=False):
    dl=as_loader(values,starts,L,H,batch_size,shuffle=False)
    true=[];pred=[]
    if model is not None:model.eval()
    with torch.inference_mode():
        for x,y in dl:
            if baseline:
                p=x[:,-1:,:9].expand(-1,H,-1)
            else:
                p=model(x.to(device)).cpu()
            if p.shape!=y.shape:raise ValueError('prediction/target shape mismatch')
            true.append(y.numpy());pred.append(p.detach().numpy())
    return np.concatenate(true,axis=0),np.concatenate(pred,axis=0)


def mixed_circular_mse(prediction,target,wind_train_sd):
    """Normalized squared error over nine sensors, with circular wind residual.

    Each ordinary sensor contributes squared z-score residual. The wind channel
    contributes the shortest angular displacement (degrees) divided by its
    Train-only standard deviation, squared. This keeps 1/9 weighting and the
    existing scale convention while respecting 0 deg == 360 deg.

    torch.remainder is piecewise differentiable; at exact antipodal errors
    (+/-180 deg) the shortest-angle derivative has an unavoidable cusp.
    """
    if prediction.shape!=target.shape or prediction.ndim!=3 or prediction.shape[-1]!=len(SENSORS):
        raise ValueError(f'Wrong shapes for circular loss: {prediction.shape} / {target.shape}')
    if not math.isfinite(float(wind_train_sd)) or wind_train_sd<=0:
        raise ValueError('Wind Train standard deviation must be positive and finite')
    delta=prediction-target
    period=360.0/float(wind_train_sd)
    wind_delta=torch.remainder(delta[...,WIND_INDEX]+period/2.0,period)-period/2.0
    # One squared contribution per sensor, per forecast step, per observation.
    return (delta[...,:WIND_INDEX].square().sum()+
            wind_delta.square().sum()+
            delta[...,WIND_INDEX+1:].square().sum())/prediction.numel()


def eval_loss(model,values,starts,L,H,batch_size,device,wind_train_sd):
    model.eval();se=0.0;n=0
    with torch.inference_mode():
        for x,y in as_loader(values,starts,L,H,batch_size):
            yp=model(x.to(device));y=y.to(device)
            batch_loss=mixed_circular_mse(yp,y,wind_train_sd)
            se+=float(batch_loss.item())*y.numel();n+=y.numel()
    return se/n


def eval_validation_objectives(model,values,starts,L,H,batch_size,device,scaler):
    """One Validation forward pass: mixed circular MSE and circular physical nMAE/IQR.

    Streaming sum avoids accumulating all [N,H,9] predictions each epoch.
    The angular difference is wrapped in z-score units using Train-only SD;
    other errors are inverse-scaled to original physical units. IQR divisors
    are fitted on Train only and identical to metric_tables() for model selection.
    """
    sd=np.asarray(scaler['sd'][:9],dtype=np.float64)
    iqr=np.asarray(scaler['iqr'][:9],dtype=np.float64)
    if not (np.isfinite(sd).all() and (sd>0).all() and
            np.isfinite(iqr).all() and (iqr>0).all()):
        raise ValueError('Invalid Train statistics for Validation nMAE/IQR')
    device_sd=torch.as_tensor(sd,dtype=torch.float64,device=device)
    wind_sd=float(sd[WIND_INDEX])
    period=360.0/wind_sd
    squared_sum=0.0;count=0
    absolute_sums=np.zeros(9,dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for x,y in as_loader(values,starts,L,H,batch_size):
            predicted=model(x.to(device)); actual=y.to(device)
            if predicted.shape!=actual.shape:
                raise ValueError(f'Validation shape mismatch {predicted.shape} != {actual.shape}')
            if not torch.isfinite(predicted).all():
                raise RuntimeError('Non-finite Validation model predictions')
            batch_loss=mixed_circular_mse(predicted,actual,wind_sd)
            if not torch.isfinite(batch_loss):
                raise RuntimeError('Non-finite Validation loss')
            squared_sum+=float(batch_loss.item())*actual.numel()
            count+=int(actual.shape[0]*actual.shape[1])
            residual=(predicted-actual).to(torch.float64)
            residual[...,WIND_INDEX]=torch.remainder(
                residual[...,WIND_INDEX]+period/2.0,period)-period/2.0
            absolute_sums+=(residual.abs()*device_sd).sum(dim=(0,1)).cpu().numpy()
    if count==0:raise RuntimeError('Empty Validation: cannot select checkpoint')
    nmae=float(np.mean(absolute_sums/count/iqr))
    mse=float(squared_sum/(count*len(SENSORS)))
    if not math.isfinite(nmae) or not math.isfinite(mse):
        raise RuntimeError('Non-finite Validation monitoring metrics')
    return {'val_mixed_loss_z':mse,'val_nmae_iqr':nmae}


def provenance(repo,csv,cfg,man):
    model_source=Path(repo)/'src/etifa_direct/models.py'
    # A model constructor may import classes from original_models.py or a
    # sibling module. Freeze every direct ETIFA Python module to prevent drift.
    source_dir=model_source.parent
    module_hashes={p.name:hash_file(p) for p in sorted(source_dir.glob('*.py'))}
    return {'etifa_source_files_sha256':module_hashes,'campaign':cfg['campaign'],'dataset_sha256':hash_file(csv),
            'config_sha256':hash_file(HERE/'experiment.json'),
            'split_manifest_sha256':hash_file(HERE/'split_manifest.json'),
            'campaign_code_sha256':hash_file(HERE/'campaign.py'),
            'slurm_code_sha256':hash_file(HERE/'job_sequential.sbatch'),
            'launch_code_sha256':hash_file(HERE/'launch.sh'),
            'etifa_models_sha256':hash_file(model_source),
            'pytorch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}


def init_output(args,cfg,man):
    repo=Path(args.repo).resolve();csv=Path(args.csv).resolve();out=Path(args.output).resolve()
    if not (repo/'src/etifa_direct/models.py').is_file():
        raise FileNotFoundError('Missing original ETIFA models.py in repo/src/etifa_direct/')
    if not csv.is_file():raise FileNotFoundError(csv)
    if hash_file(csv)!=EXPECTED_HASH:raise ValueError('NEW CSV SHA256 DOES NOT MATCH AUDITED DATASET')
    prov=provenance(repo,csv,cfg,man)
    out.mkdir(parents=True,exist_ok=True)
    p=out/'campaign_provenance.json'
    if p.exists():
        prior=json.loads(p.read_text())
        # Operational scripts may need a site-specific partition/QOS adjustment;
        # this must not prevent resume of immutable scientific checkpoints.
        if any(prior.get(k)!=prov[k] for k in ['dataset_sha256','config_sha256','split_manifest_sha256',
                     'campaign_code_sha256','etifa_models_sha256','etifa_source_files_sha256','pytorch','numpy','pandas']):
            raise ValueError('Campaign sources changed after initialization; refusing mixed results')
    else:
        atomic_json(p,prov)
        snapshots=out/'source_snapshot';snapshots.mkdir(exist_ok=True)
        for path in (HERE/'experiment.json',HERE/'split_manifest.json',HERE/'campaign.py',
                     HERE/'job_sequential.sbatch',HERE/'launch.sh'):
            shutil.copy2(path,snapshots/path.name)
        shutil.copy2(repo/'src/etifa_direct/models.py',snapshots/'etifa_models.py')
        src_snap=snapshots/'etifa_direct';src_snap.mkdir(exist_ok=True)
        for name in prov['etifa_source_files_sha256']:
            shutil.copy2(repo/'src'/'etifa_direct'/name,src_snap/name)
    return repo,csv,out,prov


def train_one(args,cfg,man,record):
    v,H,L,m,seed=record
    repo,csv,out,prov=init_output(args,cfg,man)
    if (out/'test_authorization.json').exists():
        raise RuntimeError('TRAIN BLOCKED: Test already authorized; no retraining in a frozen campaign')
    dest=out/'training'/run_key(v,H,L,m,seed)
    done=dest/'validation_done.json'
    bestpt=dest/'best.pt'
    if done.is_file():
        verify_validation(dest,prov,verify_checkpoint=True)
        return False
    dest.mkdir(parents=True,exist_ok=True)
    frames,values,scaler,starts=prepare(cfg,man,csv,v,H,L)
    seed_everything(seed)
    device=check_device(args.allow_cpu)
    model=build_model(repo,m,L,H).to(device)
    opt,scheduler=make_optimizer_and_scheduler(model,cfg)
    max_epochs=int(cfg['training']['epochs']); patience=int(cfg['training']['patience'])
    batch=int(cfg['training']['batch_size']);delta=float(cfg['training']['min_delta'])
    latest=dest/'latest.pt'
    hist=[];start_epoch=1;best=math.inf;bad=0;best_epoch=0;elapsed=0.0
    best_epoch_mixed_loss=None
    best_ref=None;best_ref_sha=None
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['key']!=run_key(v,H,L,m,seed) or saved['code_sha256']!=prov['campaign_code_sha256']:
            raise RuntimeError('Incompatible resume checkpoint')
        best_ref=saved.get('best_file');best_ref_sha=saved.get('best_sha256')
        if best_ref is None or not (dest/best_ref).is_file() or hash_file(dest/best_ref)!=best_ref_sha:
            raise RuntimeError(f'Resume points to missing/corrupted best checkpoint: {dest}')
        if 'scheduler' not in saved:
            raise RuntimeError('Resume checkpoint lacks approved scheduler state')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        # torch.optim.Optimizer.load_state_dict maps moments to parameter devices
        # while preserving Adam's step tensor placement requirements.
        hist=saved['history'];start_epoch=saved['epoch']+1
        if saved.get('monitor_metric')!='validation_nmae_iqr':
            raise RuntimeError('Resume monitoring metric mismatch')
        best=saved['best'];bad=saved['bad'];best_epoch=saved['best_epoch'];elapsed=saved['elapsed']
        best_epoch_mixed_loss=saved['best_epoch_val_mixed_loss_z']
        restore_rng(saved['rng'])
        print(f'RESUME {run_key(v,H,L,m,seed)} epoch {start_epoch}',flush=True)
    else:
        atomic_json(dest/'run_spec.json',{
            'run_key':run_key(v,H,L,m,seed),'train_filter':'complete','eval_filter':policy(cfg,H)[0],
            'selection':'common_L672_origins','n_windows':{k:len(st) for k,st in starts.items()},
            'max_epochs':max_epochs,'patience':patience,'batch_size':batch,'min_delta':delta,
            'optimizer_config':cfg['optimizer'],'scheduler_config':cfg['scheduler'],
            'monitoring':cfg['monitoring'],
            'loss_config':cfg['loss'],'wind_train_sd':float(scaler['sd'][WIND_INDEX]),
            'n_parameters':int(sum(p.numel() for p in model.parameters())),
            'provenance':prov,'seed':seed})
    tstart=time.monotonic()
    if bad>=patience:
        print('RESUME: prior epoch already satisfied early stopping',flush=True)
    for epoch in range(start_epoch,max_epochs+1) if bad<patience else ():
        lr_at_start=float(opt.param_groups[0]['lr'])
        model.train();training_sq=0.0; n=0
        for x,y in as_loader(values['train'],starts['train'],L,H,batch,shuffle=True):
            x=x.to(device);y=y.to(device);opt.zero_grad(set_to_none=True)
            output=model(x)
            if output.shape!=y.shape:raise ValueError(f'Shape mismatch {output.shape} != {y.shape}')
            loss=mixed_circular_mse(output,y,float(scaler['sd'][WIND_INDEX]))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['clip_grad_norm']))
            opt.step();training_sq+=float(loss.item())*y.numel();n+=y.numel()
        val=eval_validation_objectives(model,values['validation'],starts['validation'],
                                      L,H,batch,device,scaler)
        vmse=val['val_mixed_loss_z']
        vnmae=val['val_nmae_iqr']
        # Scheduler, checkpoint, and early stopping all monitor the same held-out
        # circular nMAE/IQR. Mixed MSE stays the differentiable TRAINING objective.
        scheduler.step(vnmae)
        lr_for_next_epoch=float(opt.param_groups[0]['lr'])
        hist.append({'epoch':epoch,'train_mixed_loss_z':training_sq/n,
                     'val_mixed_loss_z':vmse,'val_nmae_iqr':vnmae,
                     'lr_used':lr_at_start,'lr_next':lr_for_next_epoch})
        previous_best_ref=best_ref
        if vnmae<best-delta:
            best=vnmae;bad=0;best_epoch=epoch
            best_epoch_mixed_loss=vmse
            # First write a versioned candidate; latest.pt is the commit pointer.
            # A crash between these writes leaves the previous committed best valid.
            best_ref=f'best_epoch_{epoch:04d}.pt'
            atomic_torch(dest/best_ref,{'model':model.state_dict(),'epoch':epoch,
                         'run_key':run_key(v,H,L,m,seed),'scaler':scaler,'source_hashes':prov,
                         'optimizer_config':cfg['optimizer'],
                         'scheduler_config':cfg['scheduler'], 'loss_config':cfg['loss'],
                         'monitor_metric':'validation_nmae_iqr',
                         'best_val_nmae_iqr':best})
            best_ref_sha=hash_file(dest/best_ref)
        else:bad+=1
        atomic_torch(latest,{'epoch':epoch,'model':model.state_dict(), 'optimizer':opt.state_dict(),
             'scheduler':scheduler.state_dict(),
             'rng':rng_state(),'history':hist,'best':best,'bad':bad,'best_epoch':best_epoch,
             'best_file':best_ref,'best_sha256':best_ref_sha,
             'elapsed':elapsed+time.monotonic()-tstart,'key':run_key(v,H,L,m,seed),
             'monitor_metric':'validation_nmae_iqr',
             'best_epoch_val_mixed_loss_z':best_epoch_mixed_loss,
             'code_sha256':prov['campaign_code_sha256']})
        # Remove only candidates NOT referenced by the committed latest.pt.
        for stale in dest.glob('best_epoch_*.pt'):
            if stale.name!=best_ref:
                stale.unlink()
        print(f'{run_key(v,H,L,m,seed)} epoch={epoch} train_mse={hist[-1]["train_mixed_loss_z"]:.5f} val_mse={vmse:.5f} val_nmae={vnmae:.5f}',flush=True)
        if bad>=patience:break
        if STOP_REQUESTED:
            print('Stopping after safe epoch checkpoint',flush=True)
            return False
    if best_ref is None or not (dest/best_ref).is_file() or hash_file(dest/best_ref)!=best_ref_sha:
        raise RuntimeError('Missing/corrupted committed best checkpoint after training')
    # Materialize the published checkpoint atomically, preserving best_ref until
    # all metadata has been durably committed.
    best_tmp=bestpt.with_suffix('.pt.tmp')
    shutil.copyfile(dest/best_ref,best_tmp)
    os.replace(best_tmp,bestpt)
    checkpoint=torch.load(bestpt,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    y,p=collect_predictions(model,values['validation'],starts['validation'],L,H,batch,device)
    glob,by_sensor,by_step,y_phys,p_phys=metric_tables(y,p,scaler,H)
    atomic_json(dest/'validation_metrics.json',{'global':glob,'by_sensor':by_sensor,'by_step':by_step})
    # Preserve per-origin numerical errors for EVERY seed. Full arrays are staged
    # only until the five-seed best TBS+HR episode has been selected.
    truth_path,truth_sha,origins=save_validation_forecasts(
        out,v,H,L,frames['validation'],starts['validation'],y_phys)
    stage_path=stage_episode_seed(exemplar_dir(out,v,H,L,m),seed,origins,p_phys)
    origin_errors,sensor_errors=per_origin_errors(y_phys,p_phys,scaler)
    atomic_npz(dest/'validation_origin_errors.npz',origin_epoch_millis=origins,
               nmae_iqr=origin_errors.astype(np.float32),
               nmae_iqr_by_sensor=sensor_errors.astype(np.float32),
               score_TBS_HR=((sensor_errors[:,0]+sensor_errors[:,2])/2).astype(np.float32))
    atomic_json(dest/'train_history.json',hist)
    spec=read_json(dest/'run_spec.json')
    compare_close(best,glob['nmae_global_iqr'],'Best checkpoint nMAE/IQR',atol=1e-5)
    atomic_json(done,{'run_key':run_key(v,H,L,m,seed),'best_epoch':best_epoch,
                      'best_val_nmae_iqr':best,'best_epoch_val_mixed_loss_z':best_epoch_mixed_loss,
                      'monitor_metric':'validation_nmae_iqr',
                      'loss_name':cfg['loss']['name'],
                      'val_nmae_iqr':glob['nmae_global_iqr'],'n_train':int(len(starts['train'])),
                      'n_validation':int(len(starts['validation'])),
                      'n_parameters':int(spec['n_parameters']),
                      'train_seconds':elapsed+time.monotonic()-tstart,
                      'checkpoint_sha256':hash_file(bestpt),
                      'validation_truth_name':truth_path.name,
                      'validation_truth_sha256':truth_sha,
                      'episode_staging_sha256':hash_file(stage_path),
                      'artifacts_sha256':artifact_hashes(dest,('run_spec.json','validation_metrics.json','train_history.json','validation_origin_errors.npz')),
                      'campaign_code_sha256':prov['campaign_code_sha256']})
    latest.unlink(missing_ok=True)
    for stale in dest.glob('best_epoch_*.pt'):stale.unlink()
    finalize_episode_group(out,cfg,v,H,L,m,scaler,phase='validation')
    return True


def frozen_from_validation(cfg,out,prov,verify_checkpoints):
    complete={};missing=[]
    for v,H,L,m,seed in jobs(cfg):
        key=run_key(v,H,L,m,seed)
        dest=out/'training'/key
        if not (dest/'validation_done.json').is_file():
            missing.append(key)
            continue
        complete[(v,H,L,m,seed)]=verify_validation(dest,prov,verify_checkpoint=verify_checkpoints)
    if missing:
        raise RuntimeError(f'FREEZE BLOCKED: {len(missing)}/{len(jobs(cfg))} trainings incomplete: {missing[:5]}')
    selected=[]
    for v in cfg['versions']:
        for H in cfg['horizons']:
            for m in cfg['models']:
                candidates=[]
                for L in cfg['lookbacks']:
                    rows=[complete[(v,H,L,m,seed)] for seed in cfg['seeds']]
                    counts={r['n_validation'] for r in rows}
                    if len(counts)!=1:raise ValueError('Different validation origin counts by seed')
                    scores=np.array([r['val_nmae_iqr'] for r in rows])
                    candidates.append({'L':L,'mean':float(scores.mean()),
                                       'std':float(scores.std(ddof=1)) if len(scores)>1 else 0.,
                                       'n_validation':int(rows[0]['n_validation'])})
                if len({c['n_validation'] for c in candidates})!=1:
                    raise ValueError('Lookback comparison not on identical origin count')
                winner=min(candidates,key=lambda r:(r['mean'],r['std'],r['L']))
                selected.append({'version':v,'horizon':H,'model':m,'lookback':winner['L'],
                                 'mean_validation_nmae_iqr':winner['mean'],
                                 'std_validation_nmae_iqr':winner['std'],
                                 'candidates':candidates})
    all_rows=[dict(version=v,horizon=H,lookback=L,model=m,seed=seed,
        val_nmae_iqr=complete[(v,H,L,m,seed)]['val_nmae_iqr'],
        n_validation=complete[(v,H,L,m,seed)]['n_validation'],
        n_train=complete[(v,H,L,m,seed)]['n_train'],
        best_epoch=complete[(v,H,L,m,seed)]['best_epoch'],
        n_parameters=complete[(v,H,L,m,seed)].get('n_parameters'),
        train_seconds=complete[(v,H,L,m,seed)]['train_seconds'])
        for v,H,L,m,seed in jobs(cfg)]
    frozen={'provenance':prov,'criterion':'mean_validation_nMAE_IQR_across_5_seeds_then_std_then_smaller_L',
            'all_training_runs':len(complete),'selected':selected}
    return frozen,all_rows


def atomic_csv(path, rows):
    """Atomic CSV for review artifacts (no partial reports after node interruption)."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    pd.DataFrame(rows).to_csv(tmp,index=False)
    with open(tmp,'rb') as fd:os.fsync(fd.fileno())
    os.replace(tmp,path)


REVIEW_FILES=('validation_all_lookbacks_seeds.csv',
              'validation_candidates_by_lookback.csv',
              'validation_proposed_lookbacks.csv',
              'validation_selected_by_sensor.csv',
              'validation_selected_by_step.csv',
              'validation_selected_training_curves.csv',
              'validation_persistence_only.csv',
              'validation_trajectory_manifest.csv',
              'validation_examples_index.csv',
              'validation_review_checklist.csv',
              'validation_review.md')
APPROVAL_PHRASE='APRUEBO_TEST_TRAS_REVISAR_VALIDATION'
# Check complete Validation artifacts once per Test-driver process; each selected
# run is additionally verified before Test. Individual CLI commands are new processes.
_APPROVAL_CHECK_CACHE=set()


def _write_svg_validation_example(path, rec, sensors=SENSORS):
    """Small deterministic 3x3 physical-unit SVG preview; no plotting dependencies.

    These snapshots are chosen by temporal index (first/middle/last), NOT error.
    They are illustrations only; metrics use ALL Validation origins.
    """
    from html import escape
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    width,height=1260,890
    margin_x,margin_y=42,100
    cellw,cellh=390,238
    x_steps=np.arange(rec['y_true'].shape[0],dtype=float)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:17px;font-weight:bold}.small{font-size:12px}.label{font-size:14px;font-weight:bold}</style>',
           f'<text x="30" y="32" class="title">{escape(rec["title"])}</text>',
           f'<text x="30" y="55" class="small">Origen UTC {escape(rec["origin"])} | horizonte {rec["y_true"].shape[0]*15} min | {escape(rec.get("selection_note","ejemplo ilustrativo"))}</text>',
           '<line x1="32" y1="76" x2="60" y2="76" stroke="#374151" stroke-width="3"/><text x="66" y="81" class="small">Observado</text>',
           '<line x1="176" y1="76" x2="204" y2="76" stroke="#2563eb" stroke-width="3"/><text x="210" y="81" class="small">Media 5 semillas</text>',
           '<line x1="316" y1="76" x2="344" y2="76" stroke="#16a34a" stroke-width="3"/><text x="350" y="81" class="small">Persistence</text>']
    for k,sensor in enumerate(sensors):
        cx=margin_x+(k%3)*410;cy=margin_y+(k//3)*252
        left,right,top,bottom=cx+45,cx+cellw-17,cy+28,cy+cellh-35
        yt=np.asarray(rec['y_true'][:,k],dtype=float);yp=np.asarray(rec['y_pred'][:,k],dtype=float)
        sd=np.asarray(rec.get('y_std',np.zeros_like(rec['y_pred']))[:,k],dtype=float) if 'y_std' in rec else np.zeros_like(yp)
        persist=np.full(yt.shape,rec['last_sensor'][k],dtype=float)
        lo=float(min(np.min(yt),np.min(yp-sd),np.min(persist)));hi=float(max(np.max(yt),np.max(yp+sd),np.max(persist)))
        space=max(1e-6,(hi-lo)*.08);lo-=space;hi+=space
        def pt(seq):
            return ' '.join(f'{left+((i/max(len(seq)-1,1))*(right-left)):.1f},{bottom-(float(y)-lo)/(hi-lo)*(bottom-top):.1f}' for i,y in enumerate(seq))
        def pt_reverse(seq):
            return ' '.join(f'{left+((i/max(len(seq)-1,1))*(right-left)):.1f},{bottom-(float(seq[i])-lo)/(hi-lo)*(bottom-top):.1f}' for i in range(len(seq)-1,-1,-1))
        parts.extend([f'<text x="{cx+4}" y="{cy+14}" class="label">{escape(sensor)}</text>',
          f'<rect x="{left:.1f}" y="{top:.1f}" width="{right-left:.1f}" height="{bottom-top:.1f}" fill="#f8fafc" stroke="#d1d5db"/>',
          f'<text x="{cx}" y="{top+5}" class="small">{hi:.2f}</text>',
          f'<text x="{cx}" y="{bottom}" class="small">{lo:.2f}</text>',
          f'<polyline points="{pt(yt)}" fill="none" stroke="#374151" stroke-width="2"/>',
          f'<polygon points="{pt(yp+sd)} {pt_reverse(yp-sd)}" fill="#93c5fd" opacity="0.32"/>',
          f'<polyline points="{pt(yp)}" fill="none" stroke="#2563eb" stroke-width="2"/>',
          f'<polyline points="{pt(persist)}" fill="none" stroke="#16a34a" stroke-width="1.7" stroke-dasharray="5 4"/>',
          f'<text x="{left}" y="{bottom+17}" class="small">+15 min</text>',
          f'<text x="{right-58}" y="{bottom+17}" class="small">+{len(yt)*15} min</text>'])
    parts.extend(['<text x="35" y="878" class="small">Banda azul = media ± 1 desviación estándar entre semillas (NO intervalo de confianza). Viento: media circular en grados, banda meramente ilustrativa.</text>','</svg>'])
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    with open(tmp,'rb') as fd:os.fsync(fd.fileno())
    os.replace(tmp,path)


def _review_plot_records(cfg,out,proposals,score_by_key):
    """One best TBS+HR episode per selected lookback, averaging FIVE seeds."""
    plots=out/'validation_plots';plots.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sel in proposals:
        v,H,L,m=(sel['version'],int(sel['horizon']),int(sel['proposed_lookback']),sel['model'])
        meta,path=checked_episode(out,v,H,L,m)
        with np.load(path,allow_pickle=False) as z:
            prediction=z['y_pred_mean'].copy();band=z['y_pred_std'].copy()
            true=z['y_true'].copy();last=z['last_sensor'].copy()
        utc=datetime.fromtimestamp(int(meta['origin_epoch_millis'])/1000,tz=timezone.utc).isoformat()
        fname=f'{v}_H{H}_{m}_L{L}_best_5seed_episode.svg'
        fig=plots/fname
        _write_svg_validation_example(fig,{
            'title':f'{v} H{H} {m} L{L} | mejor episodio TBS+HR, 5 semillas',
            'origin':utc,'y_true':true,'y_pred':prediction,'y_std':band,
            'last_sensor':last,'selection_note':'mínimo nMAE TBS+HR medio entre cinco semillas (solo Validation)'})
        rows.append({'version':v,'horizon':H,'model':m,'lookback':L,
                     'seed_count':len(meta['seed_ids']),'selection':'best_5seed_TBS_HR_episode',
                     'origin_position':'best_TBS_HR','origin_utc':utc,
                     'origin_epoch_millis':int(meta['origin_epoch_millis']),
                     'score_mean_TBS_HR':meta['score_mean_TBS_HR'],
                     'plot_path':str(fig.relative_to(out)),'sha256':hash_file(fig)})
    return rows


def _verify_review_plots(out):
    path=out/'validation_examples_index.csv'
    df=pd.read_csv(path)
    if df.empty:raise RuntimeError('Validation review has no visual examples')
    for row in df.itertuples(index=False):
        rel=Path(str(row.plot_path))
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='validation_plots':
            raise RuntimeError('Invalid plot path in review manifest')
        file=out/rel
        if not file.is_file() or hash_file(file)!=row.sha256:
            raise RuntimeError(f'Validation visual artifact changed: {rel}')


def review_validation(args,cfg,man):
    """Generate HUMAN review package using Validation ONLY. Never touches Test.

    It deliberately does NOT create selection_frozen.json or any authorization.
    """
    _,csv,out,prov=init_output(args,cfg,man)
    if (out/'test_authorization.json').exists():
        raise RuntimeError('Validation review already approved: do not rewrite approved reports')
    candidate,rows=frozen_from_validation(cfg,out,prov,verify_checkpoints=True)
    # Finalize any five-seed groups left uncommitted by an interrupted Train process.
    for v in cfg['versions']:
        frames=load_partitions(csv,man,v,parts=('train',))
        _,scaler=scaler_and_values(frames)
        for H in cfg['horizons']:
            for L in cfg['lookbacks']:
                for m in cfg['models']:
                    finalize_episode_group(out,cfg,v,H,L,m,scaler)
    choices=[];proposals=[];sensor_rows=[];step_rows=[];curve_rows=[]
    score_by_key={(r['version'],r['horizon'],r['lookback'],r['model'],r['seed']):r['val_nmae_iqr']
                  for r in rows}
    for sel in candidate['selected']:
        head={k:sel[k] for k in ('version','horizon','model')}
        per_seed_winner={seed:min(cfg['lookbacks'],key=lambda L:(
            score_by_key[(sel['version'],sel['horizon'],L,sel['model'],seed)],L))
            for seed in cfg['seeds']}
        votes=sum(L==sel['lookback'] for L in per_seed_winner.values())
        rank=sorted(sel['candidates'],key=lambda r:(r['mean'],r['std'],r['L']))
        proposals.append({**head,'proposed_lookback':sel['lookback'],
                          'mean_validation_nmae_iqr':sel['mean_validation_nmae_iqr'],
                          'std_validation_nmae_iqr':sel['std_validation_nmae_iqr'],
                          'n_validation':sel['candidates'][0]['n_validation'],
                          'five_seeds':len(cfg['seeds']),
                          'seed_winner_votes':votes,
                          'seed_winners':'; '.join(f'{seed}:L{per_seed_winner[seed]}' for seed in cfg['seeds']),
                          'second_best_gap_nmae':rank[1]['mean']-rank[0]['mean'] if len(rank)>1 else None,
                          'review_flag': 'SEED_DISAGREEMENT' if votes<3 else ''})
        for rec in sel['candidates']:
            choices.append({**head,'lookback':rec['L'],
                            'mean_validation_nmae_iqr':rec['mean'],
                            'std_validation_nmae_iqr':rec['std'],
                            'n_validation':rec['n_validation'],
                            'proposed':rec['L']==sel['lookback']})
        for seed in cfg['seeds']:
            key=run_key(sel['version'],sel['horizon'],sel['lookback'],sel['model'],seed)
            met=read_json(out/'training'/key/'validation_metrics.json')
            for hrow in read_json(out/'training'/key/'train_history.json'):
                curve_rows.append({**head,'lookback':sel['lookback'],
                                   'seed':seed,**hrow})
            for r in met['by_sensor']:
                sensor_rows.append({**head,'lookback':sel['lookback'],
                                    'seed':seed,**r})
            for r in met['by_step']:
                step_rows.append({**head,'lookback':sel['lookback'],
                                  'seed':seed,**r})
    # Persistence is evaluated ONLY on Validation, never Test. It does not
    # participate in selecting the lookback, and introduces no new training.
    persistence_scores={}
    for v in cfg['versions']:
        frames=load_partitions(csv,man,v,('train','validation'))
        values,scaler=scaler_and_values(frames)
        for H in cfg['horizons']:
            _,eps=policy(cfg,H)
            unique_L=sorted({r['proposed_lookback'] for r in proposals
                             if r['version']==v and r['horizon']==H})
            for L in unique_L:
                st=matching_origins(frames['validation'],values['validation'],L,H,eps,max(cfg['lookbacks']))
                target_offsets=st[:,None]+L+np.arange(H)[None,:]
                y=values['validation'][target_offsets,:len(SENSORS)]
                p=np.broadcast_to(values['validation'][st+L-1,None,:len(SENSORS)],y.shape)
                global_metrics,by_sensor,by_step,_,_=metric_tables(y,p,scaler,H)
                persistence_scores[(v,H,L)]=global_metrics['nmae_global_iqr']
    for row in proposals:
        val=persistence_scores[(row['version'],row['horizon'],row['proposed_lookback'])]
        row['persistence_validation_nmae_iqr']=val
        row['skill_vs_persistence_validation_pct']=100*(1-row['mean_validation_nmae_iqr']/val) if val>0 else None
    persistence_rows=[{'version':v,'horizon':H,'lookback':L,
                       'persistence_validation_nmae_iqr':score}
                      for (v,H,L),score in sorted(persistence_scores.items())]
    # Validation example: one group episode containing FIVE seeds, selected by
    # TBS+HR score for illustration. Lookback selection remains 9-sensor nMAE.
    trajectory_rows=[]
    for sel in candidate['selected']:
        v,H,L,m=(sel['version'],sel['horizon'],sel['lookback'],sel['model'])
        meta,archive=checked_episode(out,v,H,L,m)
        done=read_json(out/'training'/run_key(v,H,L,m,cfg['seeds'][0])/'validation_done.json')
        trajectory_rows.append({'version':v,'horizon':H,'lookback':L,'model':m,
            'seed_count':len(meta['seed_ids']),'seed_ids':';'.join(map(str,meta['seed_ids'])),
            'n_validation_origins':done['n_validation'],
            'origin_position':meta['origin_position'],
            'origin_epoch_millis':meta['origin_epoch_millis'],
            'score_mean_TBS_HR':meta['score_mean_TBS_HR'],
            'path':str(archive.relative_to(out)),'sha256':meta['sha256'],
            'validation_truth_path':f"validation_truth/{done['validation_truth_name']}",
            'validation_truth_sha256':done['validation_truth_sha256'],
            'selection_basis':'minimum_5seed_mean_TBS_HR_per_episode_illustrative_only'})
    figures=_review_plot_records(cfg,out,proposals,score_by_key)
    # No manual override of the automatic lookback rule: acknowledge/hold concerns
    # together before authorization, preserving the fixed Validation protocol.
    checklist=[]
    for r in proposals:
        flags=[]
        if r['review_flag']:flags.append(r['review_flag'])
        if r['skill_vs_persistence_validation_pct'] is not None and r['skill_vs_persistence_validation_pct']<=0:
            flags.append('NO_SKILL_VS_PERSISTENCE')
        if r['n_validation']<100:flags.append('LOW_ORIGIN_COUNT_LT100')
        if r['second_best_gap_nmae'] is not None and abs(r['second_best_gap_nmae'])<0.005:
            flags.append('SMALL_LOOKBACK_GAP_LT0.005')
        checklist.append({'version':r['version'],'horizon':r['horizon'],'model':r['model'],
           'proposed_lookback':r['proposed_lookback'],
           'mean_validation_nmae_iqr':r['mean_validation_nmae_iqr'],
           'seed_winner_votes':r['seed_winner_votes'],
           'review_indicators':';'.join(flags),'review_required':'YES' if flags else 'STANDARD',
           'note':'Indicators trigger discussion, NOT an automatic model exclusion or new selection rule'})
    atomic_csv(out/REVIEW_FILES[0],rows)
    atomic_csv(out/REVIEW_FILES[1],choices)
    atomic_csv(out/REVIEW_FILES[2],proposals)
    atomic_csv(out/REVIEW_FILES[3],sensor_rows)
    atomic_csv(out/REVIEW_FILES[4],step_rows)
    atomic_csv(out/REVIEW_FILES[5],curve_rows)
    atomic_csv(out/REVIEW_FILES[6],persistence_rows)
    atomic_csv(out/'validation_trajectory_manifest.csv',trajectory_rows)
    atomic_csv(out/'validation_examples_index.csv',figures)
    atomic_csv(out/'validation_review_checklist.csv',checklist)
    lines=['# Revisión obligatoria de Validation (sin Test)', '',
           f'- Entrenamientos verificados: **{len(rows)}**.',
           f'- Propuestas de selección por versión/horizonte/modelo: **{len(proposals)}**.',
           '- Criterio: mínimo nMAE/IQR medio entre semillas (desempate: std, L menor).',
           '- **Este informe no es autorización.** No se han calculado predicciones ni métricas de Test.',
           '- Revisar resultados completos, sensibilidad por semilla y anomalías antes de aprobar.',
           '- Persistence de Validation se calcula sin acceso a Test y NO decide el lookback.',
           '- Todas las métricas se conservan; cada trayectoria representativa es el episodio TBS+HR mejor entre orígenes, con las cinco semillas.',
           '- Gráfica: media ± desviación entre cinco semillas del episodio con menor score TBS+HR de Validation; NO sustituye las métricas de todos los episodios.',
           '- La lista de indicadores de revisión es descriptiva: no sustituye el criterio predefinido de selección.',
           '', '| Versión | H | Modelo | L propuesto | nMAE Val | std | votos semilla | Persistence Val | Mejora % Val |',
           '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in proposals:
        lines.append('| {version} | {horizon} | {model} | L{proposed_lookback} | {mean_validation_nmae_iqr:.6f} | {std_validation_nmae_iqr:.6f} | {seed_winner_votes}/{five_seeds} | {persistence_validation_nmae_iqr:.6f} | {skill_vs_persistence_validation_pct:.2f} |'.format(**r))
    flagged=[r for r in proposals if r['review_flag']]
    lines+=['',f'## Revisión de consistencia: {len(flagged)} grupos con desacuerdo entre semillas',
            '', 'No cambiar la regla de selección después de ver Test. Revisar cada caso señalado con las curvas y métricas guardadas.']
    for r in flagged:
        lines.append(f"- {r['version']} H{r['horizon']} {r['model']}: L{r['proposed_lookback']} solo gana {r['seed_winner_votes']}/{r['five_seeds']} semillas; ganadores: {r['seed_winners']}")
    lines += ['', '## Ejemplos gráficos de Validation (solo ilustrativos)',
              '', 'Índice completo: `validation_examples_index.csv`. Archivos: `validation_plots/`.',
              'Los SVG comparan las nueve variables físicas y no se usan para seleccionar modelos.', '']
    for example in figures:
        lines.append(f"- {example['version']} H{example['horizon']} {example['model']}: [mejor episodio TBS+HR de Validation]({example['plot_path']})")
    lines += ['', '## Indicadores de anomalías',
              '', 'Consultar `validation_review_checklist.csv` para revisar desacuerdos de semillas, poca cobertura, gaps pequeños o ausencia de mejora frente a Persistence. No constituyen exclusiones automáticas.', '']
    md=out/REVIEW_FILES[-1]
    tmp=md.with_name(md.name+'.tmp')
    tmp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.replace(tmp,md)
    review={'schema_version':1,'status':'AWAITING_HUMAN_REVIEW',
            'provenance':prov,'n_trainings_verified':len(rows),
            'n_proposed_configurations':len(proposals),
            'n_seed_evaluations_selected':len(selected_jobs(cfg,candidate)),
            'proposed_selection':candidate,
            'review_files_sha256':artifact_hashes(out,REVIEW_FILES),
            'validation_prediction_archives_count':len(trajectory_rows),
            'episode_selection':'minimum_5seed_mean_TBS_HR_by_origin_illustrative_only',
            'validation_visual_examples_count':len(figures),
            'no_test_predictions_calculated':True}
    atomic_json(out/'validation_review.json',review)
    review_sha=hash_file(out/'validation_review.json')
    print('VALIDATION REVIEW READY: NO TEST, NO FREEZE, NO APPROVAL',flush=True)
    print('REVIEW_FILE',out/'validation_review.md',flush=True)
    print('REVIEW_SHA256',review_sha,flush=True)
    return review


def _verified_review(out,prov):
    review_file=out/'validation_review.json'
    if not review_file.exists():
        raise RuntimeError('TEST BLOCKED: validation review has not been generated')
    r=read_json(review_file)
    if r.get('status')!='AWAITING_HUMAN_REVIEW':
        raise RuntimeError('TEST BLOCKED: invalid Validation review')
    if r.get('provenance')!=prov:
        raise RuntimeError('TEST BLOCKED: review provenance changed')
    if set(r.get('review_files_sha256',{}))!=set(REVIEW_FILES):
        raise RuntimeError('TEST BLOCKED: incomplete review artifacts')
    require_artifacts(out,r['review_files_sha256'],'Validation human review')
    _verify_review_exemplars(out)
    _verify_review_plots(out)
    return r,hash_file(review_file)


def prune_unselected_validation_exemplars(out,selection):
    """After approval, retain only selected lookback examples for each V/H/model.

    This affects full forecast examples only. All five seeds' Validation metrics,
    learning curves and immutable best checkpoints remain untouched.
    """
    out=Path(out)
    retain={(r['version'],r['horizon'],r['lookback'],r['model'])
            for r in selection['selected']}
    root=out/'validation_exemplars'
    for pointer in root.glob('*/H*/*/L*/best.json'):
        meta=read_json(pointer)
        group=(meta['version'],meta['horizon'],meta['lookback'],meta['model'])
        if group not in retain:
            # Only delete paths generated in our own campaign; never checkpoints.
            shutil.rmtree(pointer.parent)


def approve_test(args,cfg,man):
    """Explicit user action, pinned to a reviewed SHA-256; no Test execution."""
    _,_,out,prov=init_output(args,cfg,man)
    if getattr(args,'approval_phrase',None)!=APPROVAL_PHRASE:
        raise RuntimeError('TEST BLOCKED: missing exact human approval phrase')
    r,sha=_verified_review(out,prov)
    if getattr(args,'review_sha256',None)!=sha:
        raise RuntimeError('TEST BLOCKED: review SHA-256 not explicitly approved')
    candidate,_=frozen_from_validation(cfg,out,prov,verify_checkpoints=True)
    if candidate!=r.get('proposed_selection'):
        raise RuntimeError('TEST BLOCKED: Validation results changed after review')
    frozen_file=out/'selection_frozen.json'
    if frozen_file.exists():
        if read_json(frozen_file)!=candidate:
            raise RuntimeError('TEST BLOCKED: existing frozen selection differs from review')
    else:
        atomic_json(frozen_file,candidate)
    selection_sha=hash_file(frozen_file)
    authorization={'schema_version':1,'explicit_human_approval':True,
                   'review_sha256':sha,'frozen_selection_sha256':selection_sha,
                   'campaign_code_sha256':prov['campaign_code_sha256'],
                   'config_sha256':prov['config_sha256'],
                   'approved_utc':datetime.now(timezone.utc).isoformat(),
                   'approval_scope':'selected_Test_and_Persistence_only'}
    auth_path=out/'test_authorization.json'
    if auth_path.exists():
        prior=read_json(auth_path)
        if any(prior.get(k)!=authorization[k] for k in
               ('review_sha256','frozen_selection_sha256','campaign_code_sha256','config_sha256')):
            raise RuntimeError('TEST BLOCKED: conflicting existing authorization')
        authorization=prior
    else:
        atomic_json(auth_path,authorization)
    # Remove unselected full Validation trajectories only AFTER durable approval.
    # The selected report/plot manifests remain intact and verifiable.
    prune_unselected_validation_exemplars(out,candidate)
    print('TEST AUTHORIZED (NOT EXECUTED). FROZEN',frozen_file,flush=True)
    return candidate


def require_test_approval(args,cfg,man):
    """Fail CLOSED for every Test, Persistence and final-report entry point."""
    _,_,out,prov=init_output(args,cfg,man)
    auth=out/'test_authorization.json'
    if not auth.is_file():
        raise RuntimeError('TEST BLOCKED: no explicit human authorization. Review Validation first.')
    a=read_json(auth)
    if a.get('schema_version')!=1 or a.get('explicit_human_approval') is not True or \
       a.get('approval_scope')!='selected_Test_and_Persistence_only':
        raise RuntimeError('TEST BLOCKED: invalid approval marker')
    r,review_sha=_verified_review(out,prov)
    frozen_file=out/'selection_frozen.json'
    if not frozen_file.is_file():
        raise RuntimeError('TEST BLOCKED: missing frozen selection')
    sel=read_json(frozen_file)
    if sel!=r.get('proposed_selection') or any([
       a.get('review_sha256')!=review_sha,
       a.get('frozen_selection_sha256')!=hash_file(frozen_file),
       a.get('campaign_code_sha256')!=prov['campaign_code_sha256'],
       a.get('config_sha256')!=prov['config_sha256']]):
        raise RuntimeError('TEST BLOCKED: review, approval or frozen selection changed')
    key=(str(out.resolve()),review_sha,hash_file(frozen_file))
    if key not in _APPROVAL_CHECK_CACHE:
        recomputed,_=frozen_from_validation(cfg,out,prov,verify_checkpoints=True)
        if recomputed!=sel:
            raise RuntimeError('TEST BLOCKED: Frozen selection does not match validated Validation scores')
        _APPROVAL_CHECK_CACHE.add(key)
    return sel


def freeze(args,cfg,man):
    """Backwards-compatible read-only accessor. Approval is always required."""
    return require_test_approval(args,cfg,man)


def selected_jobs(cfg,frozen):
    return [(r['version'],r['horizon'],r['lookback'],r['model'],seed)
            for r in frozen['selected'] for seed in cfg['seeds']]


def save_validation_forecasts(out,v,H,L,frame,starts,y_true_phys):
    """Separate immutable observed truth once per V/H from model forecasts.

    Shared temporal origins are equal for all lookbacks and seeds by design.
    This avoids archiving the same large y_true tensor 120 times per V/H.
    """
    out=Path(out)
    idx=np.asarray(starts,dtype=np.int64)+L-1
    origins=frame.timestamp.iloc[idx].astype('int64').to_numpy()//10**6
    last_sensor=frame.iloc[idx][list(SENSORS)].to_numpy(dtype=np.float32)
    truth_path=out/'validation_truth'/f'{v}_H{H}.npz'
    if truth_path.is_file():
        with np.load(truth_path,allow_pickle=False) as z:
            if (not np.array_equal(z['origin_epoch_millis'],origins) or
                not np.array_equal(z['y_true'],y_true_phys) or
                not np.array_equal(z['last_sensor'],last_sensor)):
                raise RuntimeError(f'Validation observed truth mismatch: {truth_path}')
    else:
        atomic_npz(truth_path,origin_epoch_millis=origins,
                   step_minutes=np.arange(1,H+1,dtype=np.int16)*15,
                   y_true=y_true_phys,last_sensor=last_sensor)
    # Callers pass the destination explicitly to avoid mutable dataframe metadata.
    return truth_path,hash_file(truth_path),origins


def exemplar_dir(out,v,H,L,m):
    return Path(out)/'validation_exemplars'/v/f'H{H}'/m/f'L{L}'


def episode_per_origin_score(truth, predictions, iqr):
    """[seeds,origins] score, averaged over H then TBS and HR (equal weights).

    Validation and Test use the exact same formula; IQR comes from Train only.
    """
    truth=np.asarray(truth,dtype=np.float64)
    predictions=np.asarray(predictions,dtype=np.float64)
    if predictions.ndim!=4 or predictions.shape[1:]!=truth.shape:
        raise RuntimeError('Episode predictions must have shape [seeds,origins,H,9]')
    iqr=np.asarray(iqr,dtype=np.float64)
    if not (np.isfinite(iqr[[0,2]]).all() and (iqr[[0,2]]>0).all()):
        raise RuntimeError('Invalid Train IQR for TBS/HR episode ranking')
    score=(np.abs(predictions[:,:,:,0]-truth[None,:,:,0]).mean(axis=2)/iqr[0]
          +np.abs(predictions[:,:,:,2]-truth[None,:,:,2]).mean(axis=2)/iqr[2])/2.0
    if not np.isfinite(score).all():raise RuntimeError('Non-finite episode score')
    return score


def episode_stats(pred_by_seed):
    """Mean and sample SD at each lead time across seeds; wind uses circular mean.

    This band is between-seed dispersion, NOT a confidence interval.
    """
    arr=np.asarray(pred_by_seed,dtype=np.float64)
    mu=arr.mean(axis=0)
    sd=arr.std(axis=0,ddof=1) if len(arr)>1 else np.zeros_like(mu)
    theta=np.deg2rad(arr[:,:,WIND_INDEX])
    angle=np.arctan2(np.sin(theta).mean(axis=0),np.cos(theta).mean(axis=0))
    mu[:,WIND_INDEX]=np.mod(np.rad2deg(angle),360.0)
    residual=(arr[:,:,WIND_INDEX]-mu[None,:,WIND_INDEX]+180)%360-180
    sd[:,WIND_INDEX]=residual.std(axis=0,ddof=1) if len(arr)>1 else 0.
    return mu.astype(np.float32),sd.astype(np.float32)


def stage_episode_seed(directory,seed,origins,predictions):
    """Temporary full trajectories, removed once all five seeds form an episode."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'pending_seed_{seed}.npz'
    orig=np.asarray(origins,dtype=np.int64);pred=np.asarray(predictions,dtype=np.float32)
    if path.exists():
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['origin_epoch_millis'],orig) or not np.array_equal(z['y_pred'],pred):
                raise RuntimeError(f'Conflicting staged episode seed: {path}')
    else:
        atomic_npz(path,origin_epoch_millis=orig,y_pred=pred)
    return path


def finalize_episode_group(out,cfg,v,H,L,m,scaler,phase='validation'):
    """Select the lowest five-seed MEAN TBS+HR score by forecast origin.

    Crash-safe: best.json is written last; staged files only removed after commit.
    If an interruption occurs, the next review/report can finalize the group.
    """
    if phase not in ('validation','test'):raise ValueError('Unknown episode partition')
    out=Path(out);seeds=tuple(cfg['seeds'])
    group=(exemplar_dir(out,v,H,L,m) if phase=='validation' else
           out/'test_episodes'/v/f'H{H}'/m/f'L{L}')
    pointer=group/'best.json'
    if pointer.exists():
        meta,path=checked_episode(out,v,H,L,m,phase)
        # Recover space if interrupted after atomic commit but before cleanup.
        for stale in group.glob('pending_seed_*.npz'):
            stale.unlink(missing_ok=True)
        return meta,path
    if phase=='validation':
        done_files=[out/'training'/run_key(v,H,L,m,seed)/'validation_done.json' for seed in seeds]
        truth_path=out/'validation_truth'/f'{v}_H{H}.npz'
    else:
        done_files=[out/'selected'/run_key(v,H,L,m,seed)/'test_done.json' for seed in seeds]
        truth_path=out/'test_truth'/f'{v}_H{H}.npz'
    if not all(x.is_file() for x in done_files):return None
    files=[group/f'pending_seed_{seed}.npz' for seed in seeds]
    if not all(x.is_file() for x in files):
        raise RuntimeError(f'Episode {phase} group incomplete/corrupt staged predictions: {group}')
    with np.load(truth_path,allow_pickle=False) as z:
        truth=z['y_true'].copy();orig=z['origin_epoch_millis'].copy();last=z['last_sensor'].copy()
    predictions=[]
    for seed,f,done in zip(seeds,files,done_files):
        if read_json(done).get('episode_staging_sha256')!=hash_file(f):
            raise RuntimeError(f'{phase} staged episode forecast corrupted: {f}')
        with np.load(f,allow_pickle=False) as z:
            if not np.array_equal(orig,z['origin_epoch_millis']):
                raise RuntimeError(f'Episode origins differ among seeds: {group}')
            predictions.append(z['y_pred'].copy())
    pred=np.stack(predictions,axis=0)
    score_by_seed=episode_per_origin_score(truth,pred,scaler['iqr'])
    means=score_by_seed.mean(axis=0)
    winner=int(np.argmin(means))  # earliest origin breaks exact ties
    if truth.shape[1:]!=(H,9):raise RuntimeError('Episode H/sensor dimensions differ')
    five=pred[:,winner]
    mu,sd=episode_stats(five)
    archive=group/'best_episode.npz'
    atomic_npz(archive,seed_ids=np.asarray(seeds,dtype=np.int32),
               origin_epoch_millis=np.int64(orig[winner]),origin_position=np.int64(winner),
               step_minutes=np.arange(1,H+1,dtype=np.int32)*15,
               y_true=truth[winner],last_sensor=last[winner],
               y_pred_by_seed=five,y_pred_mean=mu,y_pred_std=sd,
               scores_by_origin_mean=means.astype(np.float32),
               scores_by_seed_at_winner=score_by_seed[:,winner].astype(np.float32))
    source_hashes={str(seed):hash_file(done) for seed,done in zip(seeds,done_files)}
    metadata={'version':v,'horizon':int(H),'lookback':int(L),'model':m,'phase':phase,
              'filename':archive.name,'sha256':hash_file(archive),'truth_sha256':hash_file(truth_path),
              'seed_ids':list(seeds),'origin_epoch_millis':int(orig[winner]),
              'origin_position':winner,'score_mean_TBS_HR':float(means[winner]),
              'selection_rule':'minimum_over_origins_of_five_seed_mean_TBS_HR_nMAE_Train_IQR',
              'illustration_only':phase=='test', 'source_done_sha256':source_hashes}
    atomic_json(pointer,metadata)
    for f in files:f.unlink(missing_ok=True)
    return metadata,archive


def checked_episode(out,v,H,L,m,phase='validation'):
    group=(exemplar_dir(out,v,H,L,m) if phase=='validation' else
           Path(out)/'test_episodes'/v/f'H{H}'/m/f'L{L}')
    pointer=group/'best.json'
    if not pointer.is_file():raise RuntimeError(f'Missing {phase} episode: {pointer}')
    meta=read_json(pointer)
    if tuple(meta.get(k) for k in ('version','horizon','lookback','model','phase'))!=(v,H,L,m,phase):
        raise RuntimeError(f'{phase} episode identity mismatch: {group}')
    f=group/meta['filename']
    if not f.is_file() or hash_file(f)!=meta['sha256']:
        raise RuntimeError(f'{phase} episode altered or missing: {f}')
    truth_path=Path(out)/f'{phase}_truth'/f'{v}_H{H}.npz'
    if not truth_path.exists() or hash_file(truth_path)!=meta['truth_sha256']:
        raise RuntimeError(f'{phase} episode truth altered or missing')
    for seed in meta['seed_ids']:
        d=(Path(out)/('training' if phase=='validation' else 'selected')/
           run_key(v,H,L,m,int(seed))/('validation_done.json' if phase=='validation' else 'test_done.json'))
        if not d.is_file() or hash_file(d)!=meta['source_done_sha256'][str(seed)]:
            raise RuntimeError(f'{phase} episode seed provenance altered')
    with np.load(f,allow_pickle=False) as z:
        if z['seed_ids'].tolist()!=meta['seed_ids'] or int(z['origin_position'])!=meta['origin_position'] or \
           int(z['origin_epoch_millis'])!=meta['origin_epoch_millis']:
            raise RuntimeError(f'{phase} episode archived identity mismatch')
        arr=z['y_pred_by_seed'];true=z['y_true'];mean=z['y_pred_mean'];std=z['y_pred_std']
        if arr.shape!=(len(meta['seed_ids']),H,9) or true.shape!=(H,9) or mean.shape!=(H,9) or std.shape!=(H,9):
            raise RuntimeError(f'{phase} episode archived shape mismatch')
        mu,sd=episode_stats(arr)
        if not np.allclose(mean,mu,atol=2e-5) or not np.allclose(std,sd,atol=2e-5):
            raise RuntimeError(f'{phase} episode mean/std mismatch')
        scores=z['scores_by_origin_mean']
        if int(np.argmin(scores))!=int(z['origin_position']):
            raise RuntimeError(f'{phase} episode winner not minimum among archived origin scores')
        # Independently verify the winning origin from EVERY seed's retained
        # numerical scores; the whole trajectory is only illustrative.
        origin_scores=[]
        for seed in meta['seed_ids']:
            source=Path(out)/('training' if phase=='validation' else 'selected')/run_key(v,H,L,m,int(seed))
            scored=source/('validation_origin_errors.npz' if phase=='validation' else 'test_origin_errors.npz')
            with np.load(scored,allow_pickle=False) as per:
                origin_scores.append(per['score_TBS_HR'].copy())
        expected_scores=np.mean(np.stack(origin_scores,axis=0),axis=0)
        if not np.allclose(scores,expected_scores,atol=3e-6,rtol=1e-5):
            raise RuntimeError(f'{phase} episode score array differs from five seed numeric archives')
        compare_close(float(scores[int(z['origin_position'])]),meta['score_mean_TBS_HR'],
                      f'{phase} episode score mismatch',atol=2e-6)
    return meta,f


def _verify_review_exemplars(out):
    df=pd.read_csv(Path(out)/'validation_trajectory_manifest.csv')
    if df.empty:raise RuntimeError('Review has no saved Validation episodes')
    for r in df.itertuples(index=False):
        rel=Path(str(r.path))
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0]!='validation_exemplars':
            raise RuntimeError('Invalid Validation episode path in review')
        f=Path(out)/rel
        if not f.is_file() or hash_file(f)!=str(r.sha256):
            raise RuntimeError(f'Validation episode altered or missing: {f}')
        meta,_=checked_episode(out,r.version,int(r.horizon),int(r.lookback),r.model)
        if meta['sha256']!=str(r.sha256) or int(meta['origin_epoch_millis'])!=int(r.origin_epoch_millis):
            raise RuntimeError('Validation episode index differs from committed pointer')
        truth=Path(out)/str(r.validation_truth_path)
        if not truth.is_file() or hash_file(truth)!=str(r.validation_truth_sha256):
            raise RuntimeError(f'Validation truth altered: {truth}')
        if meta['truth_sha256']!=str(r.validation_truth_sha256):
            raise RuntimeError('Validation episode truth hash disagrees with manifest')


def save_predictions(path,y_true_phys,y_pred_phys,frame,starts,L):
    origin_index=starts+L-1
    epoch_millis=frame.timestamp.iloc[origin_index].astype('int64').to_numpy()//10**6
    atomic_npz(path,y_true=y_true_phys,y_pred=y_pred_phys,
               origin_index=origin_index,origin_epoch_millis=epoch_millis,
               step_minutes=np.arange(1,y_true_phys.shape[1]+1,dtype=np.int16)*15)


def per_origin_errors(y_true_phys,y_pred_phys,scaler):
    """Error normalized by Train IQR at each origin, retaining all five seeds."""
    delta=np.asarray(y_pred_phys,dtype=np.float64)-np.asarray(y_true_phys,dtype=np.float64)
    delta[:,:,WIND_INDEX]=(delta[:,:,WIND_INDEX]+180.)%360.-180.
    iqr=np.asarray(scaler['iqr'][:len(SENSORS)],dtype=np.float64)
    if np.any(iqr<=0):raise RuntimeError('Invalid Train IQR for per-origin errors')
    by_sensor=np.mean(np.abs(delta),axis=1)/iqr[None,:]
    return by_sensor.mean(axis=1),by_sensor


def save_test_truth(out,v,H,L,frame,starts,y_true_phys):
    """Shared test truth, created ONLY after valid human authorization."""
    out=Path(out);idx=np.asarray(starts,dtype=np.int64)+L-1
    origins=frame.timestamp.iloc[idx].astype('int64').to_numpy()//10**6
    last=frame.iloc[idx][list(SENSORS)].to_numpy(dtype=np.float32)
    f=out/'test_truth'/f'{v}_H{H}.npz'
    if f.exists():
        with np.load(f,allow_pickle=False) as z:
            if not np.array_equal(z['origin_epoch_millis'],origins) or not np.array_equal(z['y_true'],y_true_phys) or not np.array_equal(z['last_sensor'],last):
                raise RuntimeError('Inconsistent shared Test truth / origins')
    else:
        atomic_npz(f,origin_epoch_millis=origins,y_true=y_true_phys,last_sensor=last,
                   step_minutes=np.arange(1,H+1,dtype=np.int16)*15)
    return f,hash_file(f),origins


def test_selected_one(args,cfg,man,record,frozen=None):
    v,H,L,m,seed=record
    repo,csv,out,prov=init_output(args,cfg,man)
    approved=require_test_approval(args,cfg,man)
    if frozen is not None and frozen!=approved:
        raise RuntimeError('Test input does not match approved selection')
    frozen=approved
    allowed=set(selected_jobs(cfg,frozen))
    if record not in allowed:raise ValueError('Run was not selected via Validation')
    dest=out/'selected'/run_key(v,H,L,m,seed)
    done=dest/'test_done.json'
    train_dir=out/'training'/run_key(v,H,L,m,seed)
    spec=verify_validation(train_dir,prov,verify_checkpoint=True)
    ckpt_path=train_dir/'best.pt'
    frozen_sha=hash_file(out/'selection_frozen.json')
    if done.exists():
        d=verify_eval_done(dest,'test_done.json',frozen_sha,run_key(v,H,L,m,seed),spec['checkpoint_sha256'])
        if d.get('test_episode_policy')!='best_five_seed_mean_TBS_HR_illustration_only':
            raise RuntimeError('Test episode policy differs from approved design')
        return False
    frames,values,scaler,starts=prepare(cfg,man,csv,v,H,L,include_test=True)
    device=check_device(args.allow_cpu)
    model=build_model(repo,m,L,H).to(device)
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['model'])
    batch=cfg['training']['batch_size']
    # No individual Test seed is selected. The five seeds jointly rank episodes.
    summaries={'validation':read_json(train_dir/'validation_metrics.json')}
    latency=measure_latency(model,values['validation'],starts['validation'],L,H,device)
    n_parameters=sum(p.numel() for p in model.parameters())
    infer_start=time.perf_counter()
    y,p=collect_predictions(model,values['test'],starts['test'],L,H,batch,device)
    glob,by_sensor,by_step,yp,pp=metric_tables(y,p,scaler,H)
    summaries['test']={'global':glob,'by_sensor':by_sensor,'by_step':by_step}
    errors,per_sensor=per_origin_errors(yp,pp,scaler)
    compare_close(errors.mean(),glob['nmae_global_iqr'],
                  'Per-origin vs Test global nMAE',atol=1e-7)
    epoch_millis=(frames['test'].timestamp.iloc[starts['test']+L-1]
                  .astype('int64').to_numpy()//10**6)
    atomic_npz(dest/'test_origin_errors.npz',origin_epoch_millis=epoch_millis,
               nmae_iqr=errors.astype(np.float32),
               nmae_iqr_by_sensor=per_sensor.astype(np.float32),
               score_TBS_HR=((per_sensor[:,0]+per_sensor[:,2])/2).astype(np.float32))
    truth_path,truth_sha,origins=save_test_truth(out,v,H,L,frames['test'],starts['test'],yp)
    stage_path=stage_episode_seed(out/'test_episodes'/v/f'H{H}'/m/f'L{L}',seed,origins,pp)
    summaries['performance']={'n_parameters':int(n_parameters),'latency':latency,
                              'evaluation_seconds':float(time.perf_counter()-infer_start),
                              'train_seconds':float(spec['train_seconds'])}
    atomic_json(dest/'metrics.json',summaries)
    files=('metrics.json','test_origin_errors.npz')
    atomic_json(done,{'run_key':run_key(v,H,L,m,seed),'checkpoint_sha256':spec['checkpoint_sha256'],
                      'validation_nmae_iqr':summaries['validation']['global']['nmae_global_iqr'],
                      'test_nmae_iqr':summaries['test']['global']['nmae_global_iqr'],
                      'test_episode_policy':'best_five_seed_mean_TBS_HR_illustration_only',
                      'episode_staging_sha256':hash_file(stage_path),
                      'n_parameters':int(n_parameters),'latency':latency,
                      'train_seconds':float(spec['train_seconds']),
                      'n_test':int(len(starts['test'])),'frozen_selection_sha256':frozen_sha,
                      'artifacts_sha256':artifact_hashes(dest,files)})
    finalize_episode_group(out,cfg,v,H,L,m,scaler,phase='test')
    return True


def baseline_jobs(cfg,frozen):
    return sorted({(r['version'],r['horizon'],r['lookback']) for r in frozen['selected']},
                  key=lambda x:(cfg['versions'].index(x[0]),cfg['horizons'].index(x[1]),x[2]))


def persistence_one(args,cfg,man,record,frozen=None):
    v,H,L=record
    _,csv,out,prov=init_output(args,cfg,man)
    approved=require_test_approval(args,cfg,man)
    if frozen is not None and frozen!=approved:
        raise RuntimeError('Test input does not match approved selection')
    frozen=approved
    if record not in baseline_jobs(cfg,frozen):
        raise ValueError('Persistence L not selected by any model')
    frozen_sha=hash_file(out/'selection_frozen.json')
    dest=out/'persistence'/v/f'H{H}'/f'L{L}'
    if (dest/'done.json').exists():
        verify_eval_done(dest,'done.json',frozen_sha,expected_run=f'Persistence/{v}/H{H}/L{L}')
        return False
    frames,values,scaler,starts=prepare(cfg,man,csv,v,H,L,include_test=True)
    summaries={}
    latency=measure_latency(None,values['validation'],starts['validation'],L,H,torch.device('cpu'),baseline=True)
    infer_start=time.perf_counter()
    for part in ('validation','test'):
        y,p=collect_predictions(None,values[part],starts[part],L,H,cfg['training']['batch_size'],torch.device('cpu'),baseline=True)
        glob,by_sensor,by_step,yp,pp=metric_tables(y,p,scaler,H)
        summaries[part]={'global':glob,'by_sensor':by_sensor,'by_step':by_step}
        save_predictions(dest/(part+'_predictions.npz'),yp,pp,frames[part],starts[part],L)
    summaries['performance']={'n_parameters':0,'latency':latency,
                              'evaluation_seconds':float(time.perf_counter()-infer_start),
                              'train_seconds':0.0}
    atomic_json(dest/'metrics.json',summaries)
    atomic_json(dest/'done.json',{'run_key':f'Persistence/{v}/H{H}/L{L}',
                 'version':v,'horizon':H,'lookback':L,'n_parameters':0,
                 'latency':latency,'train_seconds':0.0,
                 'validation_nmae_iqr':summaries['validation']['global']['nmae_global_iqr'],
                 'test_nmae_iqr':summaries['test']['global']['nmae_global_iqr'],
                 'n_test':int(len(starts['test'])),'frozen_selection_sha256':frozen_sha,
                 'artifacts_sha256':artifact_hashes(dest,('validation_predictions.npz','test_predictions.npz','metrics.json'))})
    return True


def make_report(args,cfg,man,frozen=None):
    _,_,out,prov=init_output(args,cfg,man)
    approved=require_test_approval(args,cfg,man)
    if frozen is not None and frozen!=approved:
        raise RuntimeError('Report input does not match approved selection')
    frozen=approved
    frozen_sha=hash_file(out/'selection_frozen.json')
    table=[];unfinished=[]
    for v,H,L,m,seed in selected_jobs(cfg,frozen):
        dest=out/'selected'/run_key(v,H,L,m,seed)
        if not (dest/'test_done.json').is_file():
            unfinished.append(run_key(v,H,L,m,seed));continue
        ckpt=out/'training'/run_key(v,H,L,m,seed)/'best.pt'
        spec=verify_validation(ckpt.parent,prov,verify_checkpoint=True)
        d=verify_eval_done(dest,'test_done.json',frozen_sha,run_key(v,H,L,m,seed),spec['checkpoint_sha256'])
        if d.get('test_episode_policy')!='best_five_seed_mean_TBS_HR_illustration_only':
            raise RuntimeError('Published Test episode policy mismatch')
        table.append(dict(version=v,horizon=H,lookback=L,model=m,seed=seed,
                          val_nmae_iqr=d['validation_nmae_iqr'],test_nmae_iqr=d['test_nmae_iqr'],
                          n_test=d['n_test'],n_parameters=d['n_parameters'],
                          latency_median_ms=d['latency']['median_ms'],latency_p95_ms=d['latency']['p95_ms'],
                          latency_device=d['latency']['device'],train_seconds=d['train_seconds']))
    for v,H,L in baseline_jobs(cfg,frozen):
        dest=out/'persistence'/v/f'H{H}'/f'L{L}'
        if not (dest/'done.json').is_file():
            unfinished.append(f'Persistence {v}/H{H}/L{L}');continue
        d=verify_eval_done(dest,'done.json',frozen_sha,f'Persistence/{v}/H{H}/L{L}')
        table.append(dict(version=v,horizon=H,lookback=L,model='persistence',seed=None,
                          val_nmae_iqr=d['validation_nmae_iqr'],test_nmae_iqr=d['test_nmae_iqr'],
                          n_test=d['n_test'],n_parameters=0,
                          latency_median_ms=d['latency']['median_ms'],latency_p95_ms=d['latency']['p95_ms'],
                          latency_device=d['latency']['device'],train_seconds=0.0))
    if unfinished:raise RuntimeError(f'Report incomplete ({len(unfinished)} tasks): {unfinished[:5]}')
    # Test episodes are selected a posteriori ONLY FOR ILLUSTRATION, never for metrics.
    episode_rows=[]
    scaler_by_version={}
    for v in cfg['versions']:
        train=load_partitions(args.csv,man,v,parts=('train',))
        _,scaler_by_version[v]=scaler_and_values(train)
    for item in frozen['selected']:
        v,H,L,m=(item['version'],item['horizon'],item['lookback'],item['model'])
        result=finalize_episode_group(out,cfg,v,H,L,m,scaler_by_version[v],phase='test')
        if result is None:raise RuntimeError(f'Missing five-seed Test episode for {v} H{H} {m}')
        meta,archive=checked_episode(out,v,H,L,m,phase='test')
        with np.load(archive,allow_pickle=False) as ep:
            target=ep['y_true'].copy(); pred=ep['y_pred_mean'].copy()
            band=ep['y_pred_std'].copy();last=ep['last_sensor'].copy()
        utc=datetime.fromtimestamp(meta['origin_epoch_millis']/1000,tz=timezone.utc).isoformat()
        fig=out/'test_episode_plots'/f'{v}_H{H}_{m}_L{L}_best_five_seed_episode.svg'
        _write_svg_validation_example(fig,{'title':f'{v} H{H} {m} L{L} | mejor episodio de Test (a posteriori)',
            'origin':utc,'y_true':target,'y_pred':pred,'y_std':band,'last_sensor':last,
            'selection_note':'seleccionado tras Test SOLO para ilustrar; no determina métricas ni selección'})
        episode_rows.append({'version':v,'horizon':H,'lookback':L,'model':m,
             'origin_epoch_millis':meta['origin_epoch_millis'],'score_mean_TBS_HR':meta['score_mean_TBS_HR'],
             'seed_count':len(meta['seed_ids']),'path':str(archive.relative_to(out)),
             'sha256':meta['sha256'],'plot_path':str(fig.relative_to(out)),
             'plot_sha256':hash_file(fig),'illustration_selected_post_hoc':True})
    atomic_csv(out/'test_best_episode_manifest.csv',episode_rows)
    pd.DataFrame(table).to_csv(out/'results_final.csv',index=False)
    summary=(pd.DataFrame([r for r in table if r['model']!='persistence'])
             .groupby(['version','horizon','model','lookback'],as_index=False)
             .agg(validation_mean_nmae=('val_nmae_iqr','mean'),
                  validation_std_nmae=('val_nmae_iqr','std'),
                  test_mean_nmae=('test_nmae_iqr','mean'),
                  test_std_nmae=('test_nmae_iqr','std'),
                  n_test=('n_test','first'),seeds=('seed','count'),
                  n_parameters=('n_parameters','first'),
                  train_seconds_mean=('train_seconds','mean'),
                  train_seconds_std=('train_seconds','std'),
                  latency_median_mean_ms=('latency_median_ms','mean'),
                  latency_p95_mean_ms=('latency_p95_ms','mean'),
                  latency_device=('latency_device','first')))
    persistence_lookup={(r['version'],r['horizon'],r['lookback']):r['test_nmae_iqr']
                        for r in table if r['model']=='persistence'}
    summary['persistence_test_nmae']=summary.apply(
        lambda r:persistence_lookup[(r['version'],r['horizon'],r['lookback'])],axis=1)
    summary['skill_vs_persistence_pct']=100.0*(
        summary['persistence_test_nmae']-summary['test_mean_nmae']
    )/summary['persistence_test_nmae']
    summary.to_csv(out/'summary_selected_5_seeds.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ('version','horizon','model','lookback',
         'mean_validation_nmae_iqr','std_validation_nmae_iqr')}
         for r in frozen['selected']]).to_csv(out/'selection_by_validation.csv',index=False)
    print(f'COMPLETE: {len(table)} evaluated records, {len(frozen["selected"])} selected groups',flush=True)


def preflight(args,cfg,man):
    repo,csv,out,prov=init_output(args,cfg,man)
    print('HASHES',json.dumps(prov,ensure_ascii=False,default=str),flush=True)
    print(f'CONFIG: {len(jobs(cfg))} trainings = 3 versions x 9 H x 6 L x 4 models x 5 seeds',flush=True)
    print('PERSISTENCE: no training; same common origins as selected model lookbacks',flush=True)
    summaries=[]
    for v in cfg['versions']:
        frames=load_partitions(csv,man,v,('train','validation'))
        vals,scaler=scaler_and_values(frames)
        for H in cfg['horizons']:
            _,eps=policy(cfg,H)
            val_origins=matching_origins(frames['validation'],vals['validation'],
                                           max(cfg['lookbacks']),H,eps,max(cfg['lookbacks']))
            if len(val_origins)==0:raise ValueError(f'{v}/H{H} has empty Validation')
            row=dict(version=v,horizon=H,policy=policy(cfg,H)[0],
                     common_validation=len(val_origins))
            summaries.append(row)
            print('COVERAGE',row,flush=True)
    pd.DataFrame(summaries).to_csv(out/'coverage_common_origins.csv',index=False)
    if args.skip_model_smoke:
        print('Model architecture smoke deferred to GPU job',flush=True)
        atomic_json(out/'preflight_data_only.json',{'csv_sha256':prov['dataset_sha256'],'coverage':summaries})
        return
    audit_rows=[]
    for m in cfg['models']:
        for L,H in ((48,4),(672,288)):
            seed_everything(13)
            model=build_model(repo,m,L,H)
            device=check_device(args.allow_cpu) if args.gpu_smoke else torch.device('cpu')
            model=model.to(device)
            # The GPU smoke uses REAL training windows and targets, at the real
            # batch size, to detect model/data integration errors and GPU OOM.
            if args.gpu_smoke:
                frames, values, scaler, starts=prepare(cfg,man,csv,cfg['versions'][0],H,L)
                smoke_batch=int(cfg['training']['batch_size'])
                x,y=next(iter(as_loader(values['train'],starts['train'],L,H,smoke_batch)))
                x=x.to(device).detach().requires_grad_(True);y=y.to(device)
            else:
                smoke_batch=2
                x=torch.zeros(smoke_batch,L,19,device=device)
                y=None
            p=model(x)
            if p.shape!=(len(x),H,9) or not torch.isfinite(p).all():
                raise ValueError(f'FAIL smoke {m}/L{L}/H{H}: {p.shape}')
            if args.gpu_smoke:
                loss=mixed_circular_mse(p,y,float(scaler['sd'][WIND_INDEX]))
                if not torch.isfinite(loss):
                    raise ValueError(f'Non-finite real-data smoke loss {m}/L{L}/H{H}')
                loss.backward()
                sensitivity=gradient_input_audit(x.grad,m,L,H)
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['clip_grad_norm']))
                smoke_opt,smoke_scheduler=make_optimizer_and_scheduler(model,cfg)
                smoke_opt.step()
                smoke_scheduler.step(float(loss.detach()))
                audit_rows.append(dict(model=m,lookback=L,horizon=H,
                                       output_shape=list(p.shape),
                                       trainable_parameters=sum(z.numel() for z in model.parameters() if z.requires_grad),
                                       gradient_sensitivity=sensitivity))
            print(f'MODEL SMOKE PASS: {m}/L{L}/H{H}, batch={len(x)}, real_data={args.gpu_smoke}, backward={args.gpu_smoke}',flush=True)
            del model,p,x,y
            if torch.cuda.is_available():torch.cuda.empty_cache()
    if args.gpu_smoke:
        atomic_json(out/'architecture_preflight.json',{
            'models_sha256':prov['etifa_models_sha256'],
            'all_source_files_sha256':prov['etifa_source_files_sha256'],
            'architecture_checks':audit_rows})
    gpu_info=({'device_name':torch.cuda.get_device_name(0),
               'memory_bytes':torch.cuda.get_device_properties(0).total_memory,
               'cuda_runtime':torch.version.cuda} if args.gpu_smoke and torch.cuda.is_available() else None)
    atomic_json(out/'preflight_success.json',{'config_sha256':prov['config_sha256'],
        'csv_sha256':prov['dataset_sha256'],'models_sha256':prov['etifa_models_sha256'],
        'architecture_check_file_sha256':hash_file(out/'architecture_preflight.json') if args.gpu_smoke else None,
        'campaign_code_sha256':prov['campaign_code_sha256'],'gpu_smoke':args.gpu_smoke,
        'actual_training_batch_tested':int(cfg['training']['batch_size']) if args.gpu_smoke else None,
        'gpu':gpu_info,'coverage':summaries})
    print('PREFLIGHT PASSED',flush=True)


def _campaign_driver_lock(out):
    lockfile=open(out/'.driver.lock','a+')
    try:
        fcntl.flock(lockfile.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lockfile.close()
        raise RuntimeError('Another campaign driver is already running on this output') from exc
    return lockfile


def driver(args,cfg,man):
    """Train+Validation ONLY; generate review on completion. NEVER use Test."""
    _,_,out,prov=init_output(args,cfg,man)
    if (out/'test_authorization.json').exists():
        raise RuntimeError('Training driver cannot resume after Test authorization')
    lockfile=_campaign_driver_lock(out)
    try:
        start=time.monotonic();budget=args.budget_seconds
        all_jobs=jobs(cfg)
        for idx,rec in enumerate(all_jobs):
            if STOP_REQUESTED or time.monotonic()-start>=budget:return 75
            print(f'TRAIN {idx+1}/{len(all_jobs)}: {rec}',flush=True)
            train_one(args,cfg,man,rec)
            if STOP_REQUESTED:return 75
        if STOP_REQUESTED or time.monotonic()-start>=budget:return 75
        review_validation(args,cfg,man)
        print('TRAIN/VALIDATION COMPLETE. STOPPING FOR HUMAN REVIEW. TEST NOT RUN.',flush=True)
        return 0
    finally:
        lockfile.close()


def driver_test(args,cfg,man):
    """Separate second-stage driver, impossible without manual approval."""
    frozen=require_test_approval(args,cfg,man)
    _,_,out,prov=init_output(args,cfg,man)
    lockfile=_campaign_driver_lock(out)
    try:
        start=time.monotonic();budget=args.budget_seconds
        def must_stop():return STOP_REQUESTED or time.monotonic()-start>=budget
        selection=selected_jobs(cfg,frozen)
        for idx,rec in enumerate(selection):
            if must_stop():return 75
            print(f'TEST {idx+1}/{len(selection)}: {rec}',flush=True)
            test_selected_one(args,cfg,man,rec,frozen=frozen)
        for idx,rec in enumerate(baseline_jobs(cfg,frozen)):
            if must_stop():return 75
            print(f'PERSISTENCE {idx+1}: {rec}',flush=True)
            persistence_one(args,cfg,man,rec,frozen=frozen)
        if must_stop():return 75
        make_report(args,cfg,man,frozen=frozen)
        return 0
    finally:
        lockfile.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','train','review','approve-test','freeze','test','persistence','report','driver','driver-test','list'])
    p.add_argument('--repo',required=True,help='Root of historical etifa source')
    p.add_argument('--csv',required=True,help='Exact audited ETIFA_clean_imputed.csv')
    p.add_argument('--output',required=True,help='Fresh output directory (old results untouched)')
    p.add_argument('--id',type=int,default=0)
    p.add_argument('--allow-cpu',action='store_true',help='For local small-fixture testing only')
    p.add_argument('--gpu-smoke',action='store_true',help='Forward+backward of extreme model shapes')
    p.add_argument('--skip-model-smoke',action='store_true',help='Fast data/coverage-only check on login node')
    p.add_argument('--budget-seconds',type=int,default=18*3600)
    p.add_argument('--review-sha256',help='Exact SHA-256 of the human-reviewed validation_review.json')
    p.add_argument('--approval-phrase',help='Explicit human confirmation phrase (required only for approve-test)')
    args=p.parse_args()
    cfg,man=read_config()
    signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    try:
        if args.command=='list':
            rows=jobs(cfg)
            print('TOTAL',len(rows))
            for i,row in enumerate(rows):print(i,*row)
        elif args.command=='preflight':preflight(args,cfg,man)
        elif args.command=='train':
            r=jobs(cfg)
            if not 0<=args.id<len(r):raise IndexError('Job index out of range')
            train_one(args,cfg,man,r[args.id])
        elif args.command=='review':review_validation(args,cfg,man)
        elif args.command=='approve-test':approve_test(args,cfg,man)
        elif args.command=='freeze':freeze(args,cfg,man)
        elif args.command=='test':
            selection=selected_jobs(cfg,freeze(args,cfg,man))
            if not 0<=args.id<len(selection):raise IndexError('Test index out of range')
            test_selected_one(args,cfg,man,selection[args.id])
        elif args.command=='persistence':
            recs=baseline_jobs(cfg,freeze(args,cfg,man))
            if not 0<=args.id<len(recs):raise IndexError('Persistence index out of range')
            persistence_one(args,cfg,man,recs[args.id])
        elif args.command=='report':make_report(args,cfg,man)
        elif args.command=='driver':sys.exit(driver(args,cfg,man))
        elif args.command=='driver-test':sys.exit(driver_test(args,cfg,man))
    except Exception as e:
        print(f'FATAL: {type(e).__name__}: {e}',file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
