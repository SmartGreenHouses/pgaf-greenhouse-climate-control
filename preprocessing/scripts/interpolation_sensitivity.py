#!/usr/bin/env python3
"""ETIFA — five-period sensitivity to short-gap interpolation (NO forecasting).

Dataset: data/processed/ETIFA_clean.csv, AFTER quality control and BEFORE imputation.
Assesses artificial fully bounded missing-data gaps of 15,30,...,120 minutes.
The REAL preprocessing interpolates only gaps <=60 min for continuous variables
and pipe temperature; 75-120 minutes are a hypothetical stress test ONLY.
The five actuator-state channels use ffill in production, not examined here.

Primary comparison on the SAME randomly selected and non-overlapping windows:
  * 8 continuous scalar variables: two-sided linear vs persistence.
  * Wind direction: shortest-arc circular vs arithmetic linear vs persistence.
  * DH-only missing: Magnus from OBSERVED TBS/RH vs linear DH vs persistence.
  * DH/TBS/RH all missing: Magnus from LINEARLY interpolated TBS/RH vs
    linear DH vs persistence; uses EXACTLY THE SAME windows as DH-only.

All wind errors are shortest circular distances (degrees), regardless of method.
Synthetic tests do not reproduce the final psychrometric coherence veto in
etifa_preprocess.py and cannot establish forecasting sensitivity or real-outage
imputation quality.

Requires: numpy, pandas, matplotlib. Python 3.10+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'

BASE_SEEDS = (20261117, 20261118, 20261119)
FREQ_MIN = 15
MAX_ACTUAL_FILL_MIN = 60
GAP_LENGTHS_MIN = tuple(range(15, 121, 15))
TRAIN_START = pd.Timestamp('2019-04-29T07:30:00Z')
TRAIN_END = pd.Timestamp('2021-02-08T23:45:00Z')
# Five predefined, disjoint historical periods from V1 TRAIN only, across all seasons.
# End timestamps are inclusive, UTC. The 2019 spring period is 31 days; per-period
# window counts are always reported and may differ across variables and durations.
PERIODS = [
    {'id':'P1_spring_2019', 'start':'2019-05-01T00:00:00Z', 'end':'2019-05-31T23:45:00Z'},
    {'id':'P2_summer_2019', 'start':'2019-07-01T00:00:00Z', 'end':'2019-08-31T23:45:00Z'},
    {'id':'P3_autumn_2019', 'start':'2019-10-01T00:00:00Z', 'end':'2019-11-30T23:45:00Z'},
    {'id':'P4_winter_2020', 'start':'2020-01-01T00:00:00Z', 'end':'2020-02-29T23:45:00Z'},
    {'id':'P5_autumn_2020', 'start':'2020-09-01T00:00:00Z', 'end':'2020-10-31T23:45:00Z'},
]
SCALARS = (
    'dry_bulb_temperature', 'wet_bulb_temperature', 'relative_humidity',
    'outdoor_temperature', 'outdoor_relative_humidity', 'wind_speed',
    'radiation', 'pipe_temperature',
)
OTHER = ('wind_direction', 'humidity_deficit')
VARIABLES = SCALARS + OTHER
UNITS = {
    'dry_bulb_temperature':'degC', 'wet_bulb_temperature':'degC',
    'relative_humidity':'percentage_points', 'outdoor_temperature':'degC',
    'outdoor_relative_humidity':'percentage_points', 'wind_speed':'dataset_units',
    'radiation':'dataset_units', 'pipe_temperature':'degC',
    'wind_direction':'degrees', 'humidity_deficit':'g_per_m3',
}


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def magnus_dh(temp: np.ndarray, rh: np.ndarray) -> np.ndarray:
    # Exactly the same formula as used by etifa_preprocess.py.
    es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
    return 216.7 * es / (temp + 273.15) * (1.0 - rh / 100.0)


def angular_error(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    return (estimated - true + 180.0) % 360.0 - 180.0


def choose_nonoverlap(arrays: list[np.ndarray], gap: int,
                      rng: np.random.Generator, maximum: int,
                      exclude_opposite_wind: bool=False) -> np.ndarray:
    """Select COMPLETE (gap+2)-sample blocks, including two true endpoints.

    Each used block is disjoint from every other block within this exact
    (period, variable/scenario, seed, gap) evaluation. This does not imply
    independence across gap lengths, scenarios, or repeated seeds.
    """
    good=np.logical_and.reduce([np.isfinite(a) for a in arrays])
    block=gap+2
    # int32 prevents overflow in convolution counts and works for longer intervals.
    valid=np.convolve(good.astype(np.int32),np.ones(block,np.int32),mode='valid')==block
    candidates=np.flatnonzero(valid)
    if exclude_opposite_wind and candidates.size:
        first=arrays[0][candidates]
        last=arrays[0][candidates+gap+1]
        arcs=(last-first+180.0)%360.0-180.0
        candidates=candidates[~np.isclose(np.abs(arcs),180.0,rtol=0,atol=1e-10)]
    occupied=np.zeros(len(good),dtype=bool)
    chosen=[]
    for s in rng.permutation(candidates):
        s=int(s)
        if not occupied[s:s+block].any():
            chosen.append(s)
            occupied[s:s+block]=True
            if len(chosen)>=maximum:
                break
    return np.asarray(chosen,dtype=np.int64)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                      repetitions: int) -> tuple[float,float]:
    if len(values)<2 or repetitions<1:
        return float('nan'),float('nan')
    # Resampling unit: masked WINDOW, NOT individual correlated time samples.
    bs=np.empty(repetitions,dtype=float)
    for k in range(repetitions):
        bs[k]=np.mean(values[rng.integers(0,len(values),len(values))])
    return tuple(float(x) for x in np.quantile(bs,[0.025,0.975]))


def evaluate_block(*, var: str, arrays: dict[str,np.ndarray], selected: np.ndarray,
                   gap: int, seed: int, period_id: str, timestamps: np.ndarray,
                   repetitions: int, ci_rng: np.random.Generator):
    """Calculate paired metrics per method; return metrics, window audits, deltas."""
    k=np.arange(1,gap+1,dtype=np.int64)
    ix=selected[:,None]+k[None,:]
    frac=k[None,:]/(gap+1)
    if var=='humidity_deficit':
        a=arrays['humidity_deficit']
        truth=a[ix]
        linear=a[selected,None]+(a[selected+gap+1,None]-a[selected,None])*frac
        persistence=np.broadcast_to(a[selected,None],truth.shape)
        # Both hypotheses evaluated on EXACTLY the same windows.
        methods_by_scenario={
            'dh_only': {
                'magnus_observed_aux':magnus_dh(arrays['dry_bulb_temperature'][ix],
                                                 arrays['relative_humidity'][ix]),
                'linear_dh':linear, 'persistence_dh':persistence,
            },
            'dh_joint': {
                'magnus_interpolated_aux':magnus_dh(
                    arrays['dry_bulb_temperature'][selected,None] +
                    (arrays['dry_bulb_temperature'][selected+gap+1,None] -
                     arrays['dry_bulb_temperature'][selected,None])*frac,
                    arrays['relative_humidity'][selected,None] +
                    (arrays['relative_humidity'][selected+gap+1,None] -
                     arrays['relative_humidity'][selected,None])*frac),
                'linear_dh':linear, 'persistence_dh':persistence,
            },
        }
    elif var=='wind_direction':
        a=arrays[var]
        truth=a[ix]
        left=a[selected,None]
        right=a[selected+gap+1,None]
        arc=(right-left+180)%360-180
        circ=(left+arc*frac)%360
        arithmetic=left+(right-left)*frac
        persistence=np.broadcast_to(left,truth.shape)
        methods_by_scenario={'wind_direction':{
            'circular':circ,'arithmetic_linear':arithmetic,'persistence':persistence,
        }}
    else:
        a=arrays[var]
        truth=a[ix]
        left=a[selected,None]
        linear=left+(a[selected+gap+1,None]-left)*frac
        persistence=np.broadcast_to(left,truth.shape)
        methods_by_scenario={'scalar':{'linear':linear,'persistence':persistence}}

    metrics_rows=[]
    window_rows=[]
    paired_rows=[]
    gap_min=gap*FREQ_MIN
    interval_type='production_range' if gap_min<=MAX_ACTUAL_FILL_MIN else 'stress_test_only'
    for scenario, predictions in methods_by_scenario.items():
        errors_by_method={
            method:angular_error(pred,truth) if var=='wind_direction' else pred-truth
            for method,pred in predictions.items()
        }
        wmae={name:np.mean(np.abs(err),axis=1) for name,err in errors_by_method.items()}
        for name,errors in errors_by_method.items():
            ae=np.abs(errors)
            low,high=bootstrap_mean_ci(wmae[name],ci_rng,repetitions)
            metrics_rows.append({
                'period':period_id,'seed':seed,'variable':var,'unit':UNITS[var],
                'scenario':scenario,'method':name,'gap_min':gap_min,
                'range':interval_type,'n_windows':len(selected),
                'n_masked_values':int(truth.size),
                'mae':float(ae.mean()),'rmse':float(np.sqrt(np.mean(errors**2))),
                'p95_ae':float(np.quantile(ae,0.95)),
                'bias':float(np.mean(errors)),
                'ci95_mae_low':low,'ci95_mae_high':high,
                'fraction_ae_over_45deg':float((ae>45).mean()) if var=='wind_direction' else float('nan'),
                'fraction_ae_over_90deg':float((ae>90).mean()) if var=='wind_direction' else float('nan'),
            })
        primary_name=next(iter(predictions))
        other_names=[x for x in predictions if x!=primary_name]
        for other in other_names:
            diff=wmae[primary_name]-wmae[other]
            lo,hi=bootstrap_mean_ci(diff,ci_rng,repetitions)
            paired_rows.append({
                'period':period_id,'seed':seed,'variable':var,
                'scenario':scenario,'gap_min':gap_min,'range':interval_type,
                'primary_method':primary_name,'comparison_method':other,
                'n_windows':len(selected),'mae_primary_minus_comparison':float(np.mean(diff)),
                'paired_ci95_low':lo,'paired_ci95_high':hi,
            })
        # Compact audit: one row per window and scenario, same selected timestamps.
        first_masked=timestamps[selected+1]
        for j in range(len(selected)):
            row={
                'period':period_id,'seed':seed,'variable':var,'scenario':scenario,
                'gap_min':gap_min,'range':interval_type,
                'first_masked_utc':pd.Timestamp(first_masked[j]).isoformat(),
                'last_masked_utc':pd.Timestamp(timestamps[selected[j]+gap]).isoformat(),
                'left_endpoint':float(arrays[var][selected[j]]),
                'right_endpoint':float(arrays[var][selected[j]+gap+1]),
            }
            row.update({f'window_mae_{method}':float(wmae[method][j]) for method in predictions})
            if var=='wind_direction':
                speed=arrays['wind_speed'][selected[j]+1:selected[j]+gap+1]
                row['observed_mean_wind_speed']=float(np.nanmean(speed)) if np.isfinite(speed).any() else float('nan')
            window_rows.append(row)
    return metrics_rows,window_rows,paired_rows


def plot_results(period_summary: pd.DataFrame, pooled: pd.DataFrame, folder: Path):
    """Draw figures in the shared ETIFA publication style (see make_interpolation_figures.py)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_interpolation_figures import draw_all
    draw_all(period_summary, pooled, folder)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clean',type=Path,default=PROCESSED/'ETIFA_clean.csv',help='ETIFA_clean.csv: after cleaning, before imputation')
    parser.add_argument('--output-dir',type=Path,default=RESULTS/'interpolation_sensitivity_5periods')
    parser.add_argument('--max-windows',type=int,default=120,help='Max disjoint windows per variable, period, duration, seed')
    parser.add_argument('--seeds',default=','.join(map(str,BASE_SEEDS)),help='Comma-separated independent RNG seeds')
    parser.add_argument('--bootstrap-reps',type=int,default=500,help='Window-level bootstrap repetitions')
    parser.add_argument('--periods-json',type=Path,help='Optional JSON array of exactly five {id,start,end} UTC periods')
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    if args.max_windows<2 or args.bootstrap_reps<0:
        parser.error('Need --max-windows >=2 and --bootstrap-reps >=0')
    try:
        seeds=[int(x) for x in args.seeds.split(',')]
        assert seeds and len(seeds)==len(set(seeds))
    except (ValueError,AssertionError):
        parser.error('--seeds must contain distinct comma-separated integers')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory not empty; use another path to preserve reproducibility')
    periods=json.loads(args.periods_json.read_text()) if args.periods_json else PERIODS
    if not isinstance(periods,list) or len(periods)!=5:
        parser.error('Exactly five historical periods are required')
    bounds=[]
    for p in periods:
        start=pd.Timestamp(p['start'])
        end=pd.Timestamp(p['end'])
        if start.tzinfo is None or end.tzinfo is None:
            parser.error('Period boundaries must have timezone information (use UTC Z)')
        start,end=start.tz_convert('UTC'),end.tz_convert('UTC')
        if end<=start or start<TRAIN_START or end>TRAIN_END:
            parser.error(f'Period {p["id"]} is invalid or outside V1 training')
        bounds.append((p['id'],start,end))
    if len(set(x[0] for x in bounds))!=5:
        parser.error('Period identifiers must be unique')
    chronological=sorted(bounds,key=lambda x:x[1])
    for a,b in zip(chronological,chronological[1:]):
        if a[2]>=b[1]:
            parser.error(f'Overlapping study periods: {a[0]}, {b[0]}')

    usecols=['timestamp',*VARIABLES]
    frame=pd.read_csv(args.clean,usecols=usecols)
    frame['timestamp']=pd.to_datetime(frame['timestamp'],utc=True,errors='raise')
    if frame.timestamp.isna().any() or frame.timestamp.duplicated().any():
        raise ValueError('Invalid or duplicate timestamps')
    if not frame.timestamp.is_monotonic_increasing:
        raise ValueError('Timestamps must be sorted ascending')
    for c in VARIABLES:
        frame[c]=pd.to_numeric(frame[c],errors='raise')
        if np.isinf(frame[c]).any():
            raise ValueError(f'Infinite values in {c}')
    metrics_rows,window_rows,paired_rows,period_rows=[] ,[],[],[]
    for ordinal,(period_id,start,end) in enumerate(bounds):
        d=frame.loc[frame.timestamp.between(start,end)].reset_index(drop=True)
        if len(d)<200 or not d.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
            raise ValueError(f'Not enough data or discontinuous grid for {period_id}')
        if d.timestamp.iloc[0]!=start or d.timestamp.iloc[-1]!=end:
            raise ValueError(f'Expected full UTC period coverage for {period_id}')
        period_rows.append({'period':period_id,'first_utc':start.isoformat(),'last_utc':end.isoformat(),
                            'n_timestamps':len(d),'fraction_complete_all_10':
                             float(d[list(VARIABLES)].notna().all(axis=1).mean())})
        arr={c:d[c].to_numpy(dtype=float) for c in VARIABLES}
        t=d.timestamp.to_numpy()
        for vi,var in enumerate(VARIABLES):
            necessary=[arr[var]]
            if var=='humidity_deficit':
                necessary += [arr['dry_bulb_temperature'],arr['relative_humidity']]
            for gap_min in GAP_LENGTHS_MIN:
                gap=gap_min//15
                for seed in seeds:
                    # Stable seed derived from numeric values; do not use Python hash().
                    rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal,vi,gap,123456]))
                    starts=choose_nonoverlap(necessary,gap,rng,args.max_windows,
                                exclude_opposite_wind=(var=='wind_direction'))
                    if len(starts)<2:
                        print(f'WARNING: only {len(starts)} valid windows: {period_id} {var} {gap_min}min {seed}',file=sys.stderr)
                        continue
                    ci_rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal,vi,gap,789]))
                    m,w,pr=evaluate_block(var=var,arrays=arr,selected=starts,gap=gap,
                        seed=seed,period_id=period_id,timestamps=t,
                        repetitions=args.bootstrap_reps,ci_rng=ci_rng)
                    metrics_rows.extend(m)
                    window_rows.extend(w)
                    paired_rows.extend(pr)
        print(f'{period_id}: {len(d):,} timestamps, completed',flush=True)
    if not metrics_rows:
        raise RuntimeError('No eligible windows selected')
    mr=pd.DataFrame(metrics_rows)
    wr=pd.DataFrame(window_rows)
    pr=pd.DataFrame(paired_rows)
    # Repeated seeds are NOT independent datasets: average seeds within each period.
    per=(mr.groupby(['period','variable','unit','scenario','method','gap_min','range'],sort=False)
         .agg(mae_mean_seeds=('mae','mean'),mae_min_seed=('mae','min'),
              mae_max_seed=('mae','max'),rmse_mean_seeds=('rmse','mean'),
              p95_ae_mean_seeds=('p95_ae','mean'),
              windows_mean_per_seed=('n_windows','mean'),n_seeds=('seed','nunique'))
         .reset_index())
    pooled=(per.groupby(['variable','unit','scenario','method','gap_min','range'],sort=False)
        .agg(mae_mean_of_periods=('mae_mean_seeds','mean'),
             mae_min_period=('mae_mean_seeds','min'),mae_max_period=('mae_mean_seeds','max'),
             p95_ae_mean_of_periods=('p95_ae_mean_seeds','mean'),
             n_periods=('period','nunique'),mean_windows_per_period_seed=('windows_mean_per_seed','mean'))
        .reset_index())
    diff=(pr.groupby(['period','variable','scenario','gap_min','range','primary_method','comparison_method'],sort=False)
          .agg(mean_paired_delta_over_seeds=('mae_primary_minus_comparison','mean'),
               min_paired_delta_seed=('mae_primary_minus_comparison','min'),
               max_paired_delta_seed=('mae_primary_minus_comparison','max'))
          .reset_index())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    mr.to_csv(args.output_dir/'metrics_by_period_seed.csv',index=False)
    wr.to_csv(args.output_dir/'paired_window_audit.csv',index=False)
    pr.to_csv(args.output_dir/'paired_differences_by_period_seed.csv',index=False)
    per.to_csv(args.output_dir/'summary_by_period.csv',index=False)
    pooled.to_csv(args.output_dir/'summary_across_five_periods.csv',index=False)
    diff.to_csv(args.output_dir/'paired_differences_summary.csv',index=False)
    pd.DataFrame(period_rows).to_csv(args.output_dir/'periods_checked.csv',index=False)
    # Only 60min table: the longest genuinely filled missing-data gap.
    rows_60=per[per.gap_min.eq(60)].copy()
    rows_60.to_csv(args.output_dir/'table_60min_by_period.csv',index=False)
    metadata={
        'study_type':'retrospective synthetic masking / interpolation sensitivity; no forecasting',
        'input':args.clean.name,'input_sha256':sha256_file(args.clean),
        'script_sha256':sha256_file(Path(__file__)),
        'periods':periods,'source_subset':'V1 TRAIN only; no validation/test periods',
        'gaps_minutes':list(GAP_LENGTHS_MIN),
        'max_production_interpolation_minutes':MAX_ACTUAL_FILL_MIN,
        'stress_test_only_minutes':[x for x in GAP_LENGTHS_MIN if x>MAX_ACTUAL_FILL_MIN],
        'seeds':seeds,'max_nonoverlapping_windows_per_var_period_gap_seed':args.max_windows,
        'bootstrap_repetitions':args.bootstrap_reps,
        'windows_disjoint_within_configuration':True,
        'windows_can_overlap_across_seeds_variables_and_durations':True,
        'aggregation':'mean across seeds within each period, then equal-weight mean across five periods; no false independent pooling of seeds',
        'critical_caveats':[
            'Ground truth is QC-retained observed data BEFORE imputation. Synthetic missingness differs from real outages.',
            'Interpolation is retrospective and has observed values on BOTH sides of each hidden gap.',
            'DH-only case leaves TBS and RH observed at missing DH points; coupled case hides all three simultaneously.',
            'Magnus uses ETIFA preprocessing formula; original DH is highly consistent with this derived metric, so the comparison is not independent formula validation.',
            'Actual preprocessing also performs a subsequent psychrometric coherence rejection pass, NOT simulated here.',
            'Circular wind interpolation follows shortest arc and excludes exactly opposite endpoints (180 deg); even correct circular geometry may have high error.',
            'Wind and scalar errors are reported in their respective units; do not pool across variables.',
            'Five logged actuator-state variables are forward-filled in production (not evaluated in this continuous interpolation study).',
            'These tests do NOT assess downstream forecasting or GA efficacy.',
            'Window-level bootstrap CIs are descriptive within each seed/period and cannot turn repeated seeds into independent temporal datasets.',
        ],
    }
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# ETIFA interpolation sensitivity in five historical periods','',
        '**Study:** synthetic masking of cleaned measured observations, not forecasting.',
        '**Interpolation in production:** <=60 min for scalar sensors, wind direction and pipe temperature.',
        '**Sensitivity only:** 75, 90, 105 and 120 minutes; do NOT extend production filling.',
        '**Period choice:** five nonoverlapping periods from V1 training, covering spring, summer, autumn, winter and a second autumn in another year.','',
        '|Period|Start UTC|End UTC|15-min records|Complete all 10 variables|',
        '|---|---|---|---:|---:|']
    for p in period_rows:
        lines.append(f"|{p['period']}|{p['first_utc']}|{p['last_utc']}|{p['n_timestamps']}|{p['fraction_complete_all_10']:.1%}|")
    lines += ['', '## Results at 60 minutes: per-period MAE (averaged across seeds)', '',
              '|Period|Scenario|Method|MAE|Unit|', '|---|---|---|---:|---|']
    for _,row in rows_60[rows_60.scenario.isin(['dh_only','dh_joint','wind_direction'])].iterrows():
        lines.append(f"|{row['period']}|{row['scenario']}|{row['method']}|{row['mae_mean_seeds']:.4f}|{row['unit']}|")
    lines += ['', '## Files','',
        '- metrics_by_period_seed.csv: absolute and angular-error metrics + window-bootstrap CIs.',
        '- paired_window_audit.csv: UTC window starts/ends and per-window errors for exact reproduction.',
        '- paired_differences_by_period_seed.csv: paired method differences and descriptive bootstrap CIs.',
        '- summary_by_period.csv: mean of seed-specific metrics within the same period.',
        '- summary_across_five_periods.csv: equally-weighted mean of five periods.',
        '- paired_differences_summary.csv: period-specific paired differences, averaged over seeds.',
        '- table_60min_by_period.csv: ready-to-review results for the actual maximum interpolation length.',
        '- metadata.json: original file hash, script hash, experimental design and limitations.',
        '- figures/: paired comparisons and period plots (unless --no-plots).','',
        '## Interpretation constraints','',
        'The study compares interpolation/reconstruction errors for artificially hidden, originally valid measurements.',
        'Five periods are independent in time, but different methods use paired windows; repeated seeds may select overlapping windows.',
        'DH-only Magnus is not a fair model of simultaneous sensor failure; dh_joint evaluates that separately.',
        'No claim is made about predictive model sensitivity, real missingness, or a universally optimal 60-min threshold.','',
    ]
    (args.output_dir/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    if not args.no_plots:
        plot_results(per,pooled,args.output_dir/'figures')
    print('\nFinished. Summary at longest PRODUCTION gap (60min):')
    display=pooled[(pooled.gap_min==60)&
                   (pooled.scenario.isin(['dh_only','dh_joint','wind_direction']))]
    print(display[['scenario','method','mae_mean_of_periods','mae_min_period','mae_max_period']].to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print(f'\nOutputs: {args.output_dir.resolve()}')


if __name__=='__main__':
    main()
