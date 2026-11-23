#!/usr/bin/env python3
"""Verificación offline de la evidencia D; no carga modelos ni modifica archivos."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent / 'results'


def validate() -> None:
    records = pd.read_csv(BASE/'metrics_by_seed.csv')
    averages = pd.read_csv(BASE/'metrics_mean_5_seeds.csv')
    audit = pd.read_csv(BASE/'audit_event_counts.csv')
    selected = pd.read_csv(REPO/'forecasting/results/validation/selected_lookbacks.csv')
    assert len(records) == 270 and len(averages) == 54
    assert set(records.condition) == {'OBS','MASK','SUST'}
    assert set(records.sensor) == {'TBS','RH'}
    assert set(records.version) == {'V1','V2','V3'}
    assert set(records.model) == {'lstm_sergiot','itransformer','patchtst_mv'}
    assert all(len(g) == 6 for _,g in records.groupby(['version','model','seed']))
    assert sorted(records.seed.unique().tolist()) == [8,13,22,27,31]
    assert (records.horizon == 16).all()
    assert audit.candidate_episodes.sum() == 172
    for _,a in audit.iterrows():
        event_columns = [c for c in audit if c.startswith('event_')]
        assert int(a[event_columns].sum()) == int(a.candidate_episodes)
        assert (records.loc[records.version==a.version,'n_episodes'] == a.candidate_episodes).all()
    for (v,model,L),group in records.groupby(['version','model','lookback']):
        row = selected[(selected.version==v)&(selected.model==model)&(selected.horizon==16)]
        assert len(row)==1 and int(row.iloc[0].lookback)==L, (v,model,L)
        assert len(group)==30
    for col in ['MAE','nMAE_IQR','delta_MAE_vs_OBS','mean_prediction_shift_vs_OBS']:
        recomputed = records.groupby(['version','model','condition','sensor'])[col].mean()
        observed = averages.set_index(['version','model','condition','sensor'])[col]
        assert np.allclose(recomputed.loc[observed.index],observed,atol=1e-12)
    count=0
    for (v,model,L,seed),grp in records.groupby(['version','model','lookback','seed']):
        fn=BASE/'paired_predictions'/v/f'L{L}'/model/f'seed{seed}.npz'
        with np.load(fn,allow_pickle=False) as z:
            origin=z['origin_t'];j=z['actuator_index']
            truth=z['truth_TBS_RH']
            assert len(origin)==len(np.unique(origin))==int(grp.iloc[0].n_episodes)
            assert truth.shape==(len(origin),16,2)
            assert (j>=0).all() and (j<6).all()
            obs=z['obs_TBS_RH'];err_obs=np.abs(obs-truth)
            for _,r in grp.iterrows():
                k={'TBS':0,'RH':1}[r.sensor]
                pred=z[r.condition.lower()+'_TBS_RH'];err=np.abs(pred-truth)
                mae=float(err[:,:,k].mean(dtype=np.float64))
                delta=float((err[:,:,k]-err_obs[:,:,k]).mean(dtype=np.float64))
                shift=float(np.abs(pred[:,:,k]-obs[:,:,k]).mean(dtype=np.float64))
                assert abs(mae-r.MAE)<2e-6, (fn,r.condition,r.sensor,'MAE')
                assert abs(delta-r.delta_MAE_vs_OBS)<2e-6, (fn,r.condition,r.sensor,'delta')
                assert abs(shift-r.mean_prediction_shift_vs_OBS)<2e-6, (fn,r.condition,r.sensor,'shift')
        count+=1
    assert count==45
    print('EVIDENCIA D VERIFICADA: 45 NPZ, 270 métricas, 54 medias, 9 lookbacks H16, 172 episodios')

if __name__ == '__main__':
    validate()
