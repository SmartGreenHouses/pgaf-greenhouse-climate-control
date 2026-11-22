"""Exact nine-sensor input, partition and common-origin protocol used on SIRIUS.

Train: complete windows. Validation/Test: threshold-stable future actuation,
common forecast origins across all six lookback lengths; fit scaler on Train.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SENSORS = ('dry_bulb_temperature','wet_bulb_temperature','relative_humidity',
           'humidity_deficit','outdoor_temperature','outdoor_relative_humidity',
           'wind_speed','wind_direction','radiation')
ACTUATORS = ('pipe_temperature','humidification','window_1_position',
             'window_2_position','screen_1_position','screen_2_position')
COLS = ('timestamp',) + SENSORS + ACTUATORS
WIND_INDEX = SENSORS.index('wind_direction')

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


def as_loader(values, starts, L, H, batch_size, shuffle=False):
    return DataLoader(WindowDataset(values, starts, L, H), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)
