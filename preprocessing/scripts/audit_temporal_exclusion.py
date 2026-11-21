#!/usr/bin/env python3
"""Read-only evidence audit for ETIFA's historical 2021 exclusion.

The established cutoffs are inputs, NOT an automatically detected result.
This script measures missingness and sensor outages independently of those dates.
It never modifies input datasets or attributes missingness to equipment failures.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'

SENSORS = ['dry_bulb_temperature','wet_bulb_temperature','relative_humidity',
           'humidity_deficit','outdoor_temperature','outdoor_relative_humidity',
           'wind_speed','wind_direction','radiation']
ACTUATORS = ['pipe_temperature','humidification','window_1_position',
             'window_2_position','screen_1_position','screen_2_position']
VARIABLES = SENSORS + ACTUATORS


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    df=pd.read_csv(path, low_memory=False)
    if set(df.columns) != set(['timestamp']+VARIABLES) or len(df.columns)!=16:
        raise ValueError(f'Incorrect columns: {path}')
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True,errors='raise')
    if not df.timestamp.is_unique or not df.timestamp.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('Expected unique consecutive UTC timestamps at 15-minute resolution')
    for col in VARIABLES:
        df[col]=pd.to_numeric(df[col],errors='raise')
    if np.isinf(df[VARIABLES].to_numpy()).any():
        raise ValueError('Infinite sensor value')
    return df


def make_flags(df):
    s=df[SENSORS].isna()
    allmiss=df[VARIABLES].isna()
    return pd.DataFrame({'timestamp':df.timestamp,
        'any_sensor_missing':s.any(axis=1),
        'all_nine_missing':s.all(axis=1),
        'missing_sensor_cells':s.sum(axis=1),
        'any_variable_missing':allmiss.any(axis=1),
        'missing_total_cells':allmiss.sum(axis=1)})


def runs(f,stage,typ):
    flag=f[typ].to_numpy(dtype=bool)
    a=np.r_[False,flag,False]
    starts=np.flatnonzero(~a[:-1] & a[1:])
    ends=np.flatnonzero(a[:-1] & ~a[1:])
    return [dict(stage=stage,criterion=typ,
        start_utc=f.timestamp.iloc[int(i)].isoformat(),
        end_utc=f.timestamp.iloc[int(j-1)].isoformat(),
        samples=int(j-i),hours=float((j-i)/4)) for i,j in zip(starts,ends)]


def stats(f,stage,label,mask):
    z=f.loc[mask]
    n=len(z)
    return dict(stage=stage,period=label,rows=n,
        missing_rows=int(z.any_variable_missing.sum()),
        missing_rows_pct=100*float(z.any_variable_missing.mean()) if n else np.nan,
        missing_sensor_rows=int(z.any_sensor_missing.sum()),
        all_nine_missing_rows=int(z.all_nine_missing.sum()),
        missing_sensor_cells=int(z.missing_sensor_cells.sum()),
        missing_total_cells=int(z.missing_total_cells.sum()))


def aggregated(f,stage,fmt):
    g=f.groupby(f.timestamp.dt.strftime(fmt),sort=True)
    tab=g.agg(rows=('timestamp','size'),rows_any_sensor_missing=('any_sensor_missing','sum'),
         rows_all_nine_missing=('all_nine_missing','sum'),
         missing_sensor_cells=('missing_sensor_cells','sum')).reset_index(names='period')
    tab.insert(0,'stage',stage)
    tab['missing_sensor_rows_pct']=100*tab.rows_any_sensor_missing/tab.rows
    return tab


SENSOR_LABELS={'dry_bulb_temperature':'TBS','wet_bulb_temperature':'TBH',
    'relative_humidity':'HR','humidity_deficit':'DH','outdoor_temperature':'T ext',
    'outdoor_relative_humidity':'HR ext','wind_speed':'Wind speed',
    'wind_direction':'Wind dir.','radiation':'Radiation'}


def plot(clean,imputed,start,end,output):
    """Daily share of missing values per environmental sensor:
    (a) cleaned data before imputation, (b) after imputation.
    Dashed lines mark the historical V1/V2/V3 excluded period."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates
    fig,ax=plt.subplots(2,1,figsize=(12,8.5),sharex=True,layout='constrained')
    fig.get_layout_engine().set(hspace=.06)
    x0,x1=start,end+pd.Timedelta(minutes=15)
    for a,df,letter in zip(ax,(clean,imputed),'ab'):
        miss=df[SENSORS].isna().groupby(df.timestamp.dt.floor('D')).mean().mul(100)
        days=mdates.date2num(miss.index.to_pydatetime())
        mesh=a.pcolormesh(np.r_[days,days[-1]+1],np.arange(len(SENSORS)+1),miss.T.to_numpy(),
                          cmap='Purples',vmin=0,vmax=100,shading='flat',rasterized=True)
        for x in (x0,x1):
            a.axvline(x,color='0.35',lw=.8,ls='--')
        a.set_yticks(np.arange(len(SENSORS))+.5)
        a.set_yticklabels([SENSOR_LABELS[s] for s in SENSORS],fontsize=17)
        a.invert_yaxis()
        for side in ('left','right','top','bottom'):
            a.spines[side].set_visible(False)
        a.tick_params(axis='y',length=0)
        a.tick_params(axis='x',labelsize=16)
        a.text(-.13,1,f'({letter})',transform=a.transAxes,fontsize=19,va='top',fontweight='bold')
    cb=fig.colorbar(mesh,ax=list(ax),fraction=.02,pad=.015)
    cb.outline.set_visible(False)
    cb.set_label('Missing values (% of the day)',fontsize=17)
    cb.ax.tick_params(labelsize=16,length=0)
    ax[-1].set_xlabel('Date',fontsize=18)
    ax[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m',tz=mdates.UTC))
    plt.setp(ax[-1].get_xticklabels(),rotation=45,ha='right')
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean',type=Path,default=PROCESSED/'ETIFA_clean.csv')
    p.add_argument('--imputed',type=Path,default=PROCESSED/'ETIFA_clean_imputed_1h.csv')
    p.add_argument('--cleaning-audit',type=Path,default=PROCESSED/'ETIFA_cleaning_audit.csv',help='Optional ETIFA_cleaning_audit.csv for confirmed cleaning-rule invalidations')
    p.add_argument('--excluded-start',default='2021-06-27T17:45:00Z')
    p.add_argument('--excluded-end',default='2021-10-31T23:45:00Z')
    p.add_argument('--out',type=Path,default=RESULTS/'exclusion_2021')
    args=p.parse_args()
    start=pd.Timestamp(args.excluded_start)
    end=pd.Timestamp(args.excluded_end)
    if start.tz is None or end.tz is None:
        p.error('Use timestamps with explicit UTC offset / Z')
    start=start.tz_convert('UTC'); end=end.tz_convert('UTC')
    if start>=end: p.error('Excluded start must precede end')
    paths=['period_comparison.csv','monthly_sensor_coverage.csv','daily_sensor_coverage.csv',
           'sensor_outages.csv','boundary_context.csv','temporal_coverage.png',
           'temporal_exclusion_report.md','manifest.json','cleaning_invalidations.csv']
    args.out.mkdir(parents=True,exist_ok=True)
    if any((args.out/x).exists() for x in paths):
        p.error('Outputs already exist; choose an unused --out directory')
    clean=load(args.clean); imputed=load(args.imputed)
    if not clean.timestamp.equals(imputed.timestamp):
        raise ValueError('The cleaned and imputed datasets must have the same grid')
    for c in VARIABLES:
        orig=clean[c].to_numpy(); new=imputed[c].to_numpy()
        mask=np.isfinite(orig)
        if not np.allclose(orig[mask],new[mask],atol=1e-10,rtol=1e-12):
            raise ValueError(f'Imputation changed recorded values: {c}')
    ts=imputed.timestamp
    periods={'before_exclusion':(ts<start),
             'excluded_2021':ts.between(start,end),
             'after_exclusion':(ts>end),
             'retained_combined':(ts<start)|(ts>end),
             'all_curated':pd.Series(True,index=ts.index)}
    if not periods['excluded_2021'].any():
        raise ValueError('Historical excluded dates do not intersect the CSV')
    summary=[]; monthly=[]; daily=[]; outages=[]; obs={}
    for stage,df in [('before_imputation',clean),('after_imputation',imputed)]:
        f=make_flags(df); obs[stage]=f
        summary.extend(stats(f,stage,k,v) for k,v in periods.items())
        monthly.append(aggregated(f,stage,'%Y-%m'))
        daily.append(aggregated(f,stage,'%Y-%m-%d'))
        outages += runs(f,stage,'all_nine_missing')
        outages += runs(f,stage,'any_sensor_missing')
    period_df=pd.DataFrame(summary)
    monthly_df=pd.concat(monthly,ignore_index=True)
    daily_df=pd.concat(daily,ignore_index=True)
    outage_df=pd.DataFrame(outages).sort_values(['stage','criterion','hours'],ascending=[True,True,False])
    period_df.to_csv(args.out/paths[0],index=False)
    monthly_df.to_csv(args.out/paths[1],index=False)
    daily_df.to_csv(args.out/paths[2],index=False)
    outage_df[outage_df.hours>=1].to_csv(args.out/paths[3],index=False)
    f=obs['after_imputation']
    at_boundary=(ts.between(start-pd.Timedelta(hours=2),start+pd.Timedelta(hours=3)) |
                 ts.between(end-pd.Timedelta(hours=2),end+pd.Timedelta(hours=3)))
    f.loc[at_boundary].to_csv(args.out/paths[4],index=False)
    events=None
    if args.cleaning_audit:
        events=pd.read_csv(args.cleaning_audit)
        if not {'timestamp','rule','variable'}.issubset(events.columns):
            raise ValueError('Invalid cleaning-audit schema')
        events['timestamp']=pd.to_datetime(events.timestamp,utc=True,errors='raise')
    plot(clean,imputed,start,end,args.out/paths[5])
    a=period_df[period_df.stage.eq('after_imputation')].set_index('period')
    inside=a.loc['excluded_2021']; kept=a.loc['retained_combined']; total=a.loc['all_curated']
    large=outage_df[(outage_df.stage=='after_imputation') & (outage_df.criterion=='all_nine_missing')].sort_values('hours',ascending=False)
    examples=['| Start UTC | End UTC | Hours |','|---|---|---:|']
    examples += [f'| {r.start_utc} | {r.end_utc} | {r.hours:.2f} |' for r in large.head(10).itertuples()]
    audit_section='## Quality-control invalidations\nNo cleaning audit provided.\n'
    invalidation_rows=[]
    if events is not None:
        excluded_a=events.timestamp.between(start,end)
        for name,mask in [('excluded_2021',excluded_a),('retained_combined',~excluded_a)]:
            z=events.loc[mask]
            for rule,group in z.groupby('rule'):
                invalidation_rows.append({'period':name,'rule':rule,'invalidated_cells':len(group)})
        pd.DataFrame(invalidation_rows,columns=['period','rule','invalidated_cells']).to_csv(args.out/paths[8],index=False)
        n_inside=int(excluded_a.sum()); n_outside=int((~excluded_a).sum())
        audit_section=(f'## Quality-control invalidations\n'
          f'- {n_inside:,}/{len(events):,} invalidated cells ({100*n_inside/len(events):.2f}%) are in the excluded interval.\n'
          f'- {n_outside:,}/{len(events):,} are in the retained interval.\n'
          f'- Invalidations per 1,000 timestamps: {1000*n_inside/inside.rows:.2f} excluded vs '
          f'{1000*n_outside/kept.rows:.2f} retained.\n'
          f'- These represent script-defined anomaly flags, **not externally confirmed hardware failures**.\n')
    else:
        pd.DataFrame(columns=['period','rule','invalidated_cells']).to_csv(args.out/paths[8],index=False)
    report=f'''# ETIFA — Evidence audit of the historical 2021 exclusion

**Methodological warning:** this script compares the historical excluded dates with the sensor measurements. **It does not retrieve the original reasoning or automatically rediscover the exact cutoff dates.** No device failure or restoration date has been confirmed.

## Historical V1/V2/V3 exclusion
- Start: {start.isoformat()} (inclusive)
- End: {end.isoformat()} (inclusive)
- Data are **excluded from experimental partitions**, not deleted from the curated CSV.

## After imputation
| Measurement | Excluded | Retained |
|---|---:|---:|
| Rows | {int(inside.rows):,} | {int(kept.rows):,} |
| Rows with any missing variable | {int(inside.missing_rows):,} | {int(kept.missing_rows):,} |
| Rows with missing value (%) | {inside.missing_rows_pct:.2f} | {kept.missing_rows_pct:.2f} |
| Rows without all nine environmental sensors | {int(inside.all_nine_missing_rows):,} | {int(kept.all_nine_missing_rows):,} |
| Missing cells | {int(inside.missing_total_cells):,} | {int(kept.missing_total_cells):,} |

The historical excluded interval accounts for {100*inside.missing_total_cells/total.missing_total_cells:.2f}% of all missing cells remaining after imputation. Its valid records are **not necessarily erroneous**.

{audit_section}\n## Major simultaneous environmental-sensor outages
{chr(10).join(examples)}

## What the boundaries do (and do not) establish
- The first complete nine-sensor outage on 27 June 2021 starts at 18:00 UTC, one sample after the chosen exclusion begins (17:45 UTC).
- The last major October outage ends on 19 October 2021, **not on 31 October**; 1 November is a conservative calendar-aligned split start, not a verified restoration date.
- Some rows before and after the interval are incomplete, so eligibility of prediction windows must also be checked.
- An elevated missingness rate warrants exclusion for a study requiring complete multivariate windows, but does not establish why equipment stopped reporting.

## Output files
- `period_comparison.csv`: before/inside/after/retained metrics for both cleaning stages.
- `monthly_sensor_coverage.csv` and `daily_sensor_coverage.csv`: coverage calculated independently of the historical dates.
- `sensor_outages.csv`: runs of missing sensor data, including simultaneous failures.
- `boundary_context.csv`: timestamp-by-timestamp evidence around both cutoffs.
- `temporal_coverage.png`: daily missing values per sensor, (a) before and (b) after imputation; dashed lines mark the excluded period.
- `cleaning_invalidations.csv`: quality-control invalidations by split and rule, if supplied.\n- `manifest.json`: provenance and hashes.
'''
    (args.out/paths[6]).write_text(report,encoding='utf8')
    (args.out/paths[7]).write_text(json.dumps({
        'script_sha256':sha256(Path(__file__)),
        'clean_sha256':sha256(args.clean),'imputed_sha256':sha256(args.imputed),
        'cleaning_audit_sha256':sha256(args.cleaning_audit) if args.cleaning_audit else None,
        'historical_exclusion':{'start_utc':start.isoformat(),'end_utc':end.isoformat()},
        'warning':'Cutoffs taken from established splits; not automatically inferred.',
        'device_failure_diagnosis':'not performed'},indent=2),encoding='utf8')
    print(report)

if __name__=='__main__':
    main()
