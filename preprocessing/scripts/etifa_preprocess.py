#!/usr/bin/env python3
"""
ETIFA cleaning and imputation without dataset splitting.
    Cleaning: temperature -5 -> NaN; if TBS or TBH is -5, all indoor
sensor-box variables in that row -> NaN; isolated temperature spikes >=8 degrees C
with neighbors within 2 degrees C; humidity spikes >=15 percentage points
with neighbors within 5 points; negative humidity deficit -> NaN;
frozen sensor (same value >= 8 readings, saturation plateaus allowed) -> NaN;
TBH > TBS + 2 -> TBS NaN; values outside PHYSICAL_RANGES -> NaN;
sensor-box contradiction (measured HR more than 10 points away from the HR
implied psychrometrically by TBS and TBH, saturation excluded) -> HR, TBH
and DH NaN, TBS kept.
Spike detection uses both neighbors at +/-15 minutes, in a single pass.
    Imputation: six actuators; unlimited forward fill for five of them
(humidification, windows, screens); linear interpolation for pipe temperature
(the sixth actuator) and scalar sensors over complete gaps <=4 samples (1 hour);
angular interpolation for wind direction over gaps <=4 samples.
humidity_deficit is not interpolated independently: it is derived from the
imputed dry-bulb temperature and relative humidity (Magnus psychrometrics,
same formula as the reconstruction step).
Missing humidity_deficit values are reconstructed from
dry_bulb_temperature and relative_humidity for complete
gaps of up to 4 samples.

Originally observed DH values are preserved unless
invalidated by an explicit cleaning rule.
A final coherence pass applies cleaning rule 8 to imputed values: an
interpolated TBS, TBH or HR that contradicts its own row psychrometrically
is rejected (stays missing), along with any DH derived from a rejected value.
    Wind direction: shortest angular arc; opposite endpoints remain missing.
    Trimming: leading rows before the first valid screen_2_position value
(TRIM_TO_FIRST_VALID) are dropped before cleaning, so every output starts
where all 15 variables are being logged.
Requires a continuous 15-minute grid.  
DH is named humidity_deficit, not vapor_pressure_deficit (no unit conversion).
    Usage: python3 scripts/etifa_preprocess.py data/raw/ETIFA_raw_data.csv --output-dir data/reproduced
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
RESULTS = ROOT / 'results'

COLUMN_MAPPING = {'Fecha': 'timestamp', 'CajaMedida1.TBS': 'dry_bulb_temperature', 'CajaMedida1.TBH': 'wet_bulb_temperature', 'CajaMedida1.HR': 'relative_humidity', 'CajaMedida1.DH': 'humidity_deficit', 'TemperaturaExterior': 'outdoor_temperature', 'HumedadExterior': 'outdoor_relative_humidity', 'VelocidadViento': 'wind_speed', 'DireccionViento': 'wind_direction', 'Radiacion': 'radiation', 'TemperaturaTuberia': 'pipe_temperature', 'Humidificacion': 'humidification', 'PosicionVentana1': 'window_1_position', 'PosicionVentana2': 'window_2_position', 'PosicionPantalla1': 'screen_1_position', 'PosicionPantalla2': 'screen_2_position'}

TEMPERATURES = ["dry_bulb_temperature", "wet_bulb_temperature", "outdoor_temperature"]
HUMIDITIES = ["relative_humidity", "outdoor_relative_humidity"]

MISSING_MARKERS = ['', 'nan', 'na', 'null', 'none', '<na>']

# The dataset starts months before this actuator was first logged; rows
# without it are useless for training, so everything before its first valid
# value is dropped (2018-12-13 to 2019-04-29 in the current raw).
TRIM_TO_FIRST_VALID = 'screen_2_position'


def validate(df):
    """Check schema and a continuous, ordered 15-minute timestamp grid."""
    required = ['timestamp'] + SENSORS + ACTUATORS
    if not df.columns.is_unique or set(df.columns) != set(required):
        raise ValueError('The 16 expected ETIFA columns are required.')
    t = pd.to_datetime(df['timestamp'], utc=True, errors='raise')
    if t.isna().any() or not t.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
        raise ValueError('Timestamps must be unique, ordered and continuous every 15 minutes.')


def parse_numeric(df):
    """Parse string columns to float. Returns (numeric frame, missing mask frame).
    Only empty fields and explicit missing markers count as missing."""
    numeric = df.copy()
    missing = pd.DataFrame(index=df.index)
    for col in SENSORS + ACTUATORS:
        text = df[col].str.strip()
        miss = text.str.lower().isin(MISSING_MARKERS)
        s = pd.to_numeric(text.mask(miss), errors='raise').astype(float)
        if np.isinf(s).any():
            raise ValueError(f'{col}: infinite values found; review before imputation.')
        numeric[col] = s
        missing[col] = miss
    return numeric, missing


# Variables measured by the indoor sensor box (CajaMedida1).
SENSOR_BOX = ['dry_bulb_temperature', 'wet_bulb_temperature',
              'relative_humidity', 'humidity_deficit']

# Physically plausible ranges (inclusive). Indoor greenhouse air can exceed
# 50 degrees C at midday with closed vents; heating pipes run up to ~62.
PHYSICAL_RANGES = {
    'dry_bulb_temperature': (0, 60),
    'wet_bulb_temperature': (0, 60),
    'relative_humidity': (0, 100),
    'humidity_deficit': (0, 60),
    'outdoor_temperature': (-15, 50),
    'outdoor_relative_humidity': (0, 100),
    'wind_speed': (0, 60),
    'wind_direction': (0, 360),
    'radiation': (0, 1500),
    'pipe_temperature': (0, 90),
}

# Wet bulb can never exceed dry bulb; small excess is sensor noise.
WET_ABOVE_DRY_TOLERANCE = 2.0

# Sensor-box internal contradiction: psychrometrics ties TBS, TBH and HR
# together; a measured HR further than this many points from HR(TBS, TBH)
# means some reading is wrong (sensor noise is 2-3 points; the reconstruction
# step uses 5). P_EFF_HPA is the same effective empirical pressure used there.
HR_CONTRADICTION_TOLERANCE = 10.0
P_EFF_HPA = 965.0


def relative_humidity_from_bulbs(t, w):
    """HR (%) implied by dry- and wet-bulb temperatures (psychrometric Magnus);
    identical to the reconstruction step."""
    es_t = 6.112 * np.exp(17.67 * t / (t + 243.5))
    es_w = 6.112 * np.exp(17.67 * w / (w + 243.5))
    return 100 * (es_w - 0.00066 * (1 + 0.00115 * w) * P_EFF_HPA * (t - w)) / es_t

# Frozen sensor: the exact same value repeated for >= FROZEN_MIN_RUN consecutive
# valid readings (missing samples break the run).
# Low-resolution variables (wind, radiation, pipe) repeat legitimately and are
# not checked. Values in FROZEN_ALLOWED are physical plateaus (saturation).
FROZEN_MIN_RUN = 8  # 2 hours
FROZEN_CHECK = [
    'dry_bulb_temperature',
    'wet_bulb_temperature',
    'outdoor_temperature',
    'relative_humidity',
    'outdoor_relative_humidity'
]
FROZEN_ALLOWED = {
    'relative_humidity': (99, 100),
    'outdoor_relative_humidity': (99, 100),
}


def frozen_runs(s, min_run):
    """Detect consecutive identical valid readings without crossing NaN gaps."""
    valid = s.notna()
    same_as_previous = valid & s.shift().notna() & s.eq(s.shift())

    run_id = (~same_as_previous).cumsum()
    run_length = valid.groupby(run_id).transform('sum')

    return valid & run_length.ge(min_run)


def clean(df):
    """Apply cleaning rules to a validated numeric frame (see validate/parse_numeric).
    Rules run in order; each cell is invalidated (and audited) at most once.
      1. temperature_equals_minus_5: API error code.
      2. sensor_box_error_row: if TBS or TBH is -5, the whole indoor box
         reading at that timestamp is corrupt (TBS, TBH, HR, DH).
      3. isolated_spike: single-sample spikes (-5 codes excluded as neighbors).
      4. negative_humidity_deficit.
      5. frozen_sensor: same exact value for >= FROZEN_MIN_RUN readings.
      6. wet_bulb_above_dry_bulb: TBH > TBS + tolerance -> TBS invalid.
      7. out_of_physical_range.
      8. sensor_box_contradiction: measured HR disagrees with HR(TBS, TBH)
         beyond HR_CONTRADICTION_TOLERANCE (saturation excluded) -> HR, TBH
         and DH invalid, TBS kept.
    The grid is continuous, so first/last rows simply lack a neighbor (NaN)."""
    out = df.copy()
    audit = []
    nan = pd.Series(np.nan, index=df.index)

    def invalidate(col, mask, rule, before=nan, after=nan):
        mask = mask & out[col].notna()  # only cells still valid
        for i in df.index[mask]:
            audit.append({'timestamp': df.at[i, 'timestamp'], 'variable': col,
                          'original_value': df.at[i, col], 'new_value': np.nan,
                          'rule': rule, 'previous': before.at[i], 'next': after.at[i]})
        out.loc[mask, col] = np.nan

    # 1-2. Error codes and corrupt sensor-box rows.
    for col in TEMPERATURES:
        invalidate(col, df[col].eq(-5), 'temperature_equals_minus_5')
    box_error = df['dry_bulb_temperature'].eq(-5) | df['wet_bulb_temperature'].eq(-5)
    for col in SENSOR_BOX:
        invalidate(col, box_error, 'sensor_box_error_row')

    # 3. Isolated spikes. Neighbor evidence excludes only -5 codes: values in
    # sensor_box_error_row rows are often plausible and still useful as reference.
    for col in TEMPERATURES + HUMIDITIES:
        usable = df[col].mask(df[col].eq(-5)) if col in TEMPERATURES else df[col]
        before, after = usable.shift(1), usable.shift(-1)
        jump, similar = (8.0, 2.0) if col in TEMPERATURES else (15.0, 5.0)
        peak = ((usable > before) & (usable > after)) | ((usable < before) & (usable < after))
        spike = (peak & (usable - before).abs().ge(jump) &
                 (usable - after).abs().ge(jump) & (before - after).abs().le(similar))
        invalidate(col, spike, 'isolated_spike', before, after)

    # 4. Negative humidity deficit.
    invalidate('humidity_deficit', df['humidity_deficit'].lt(0), 'negative_humidity_deficit')

    # 5. Frozen sensor (-5 codes excluded so they do not form runs).
    for col in FROZEN_CHECK:
        s = df[col].mask(df[col].eq(-5)) if col in TEMPERATURES else df[col]
        frozen = frozen_runs(s, FROZEN_MIN_RUN)
        if col in FROZEN_ALLOWED:
            low, high = FROZEN_ALLOWED[col]
            frozen &= ~s.between(low, high)
        invalidate(col, frozen, 'frozen_sensor')

    # 6. Wet bulb above dry bulb (on values still valid).
    wet_above = out['wet_bulb_temperature'] > out['dry_bulb_temperature'] + WET_ABOVE_DRY_TOLERANCE
    invalidate('dry_bulb_temperature', wet_above, 'wet_bulb_above_dry_bulb')

    # 7. Physical ranges.
    for col, (low, high) in PHYSICAL_RANGES.items():
        invalidate(col, (df[col] < low) | (df[col] > high), 'out_of_physical_range')

    # 8. Sensor-box internal contradiction (on values still valid). HR >= 99
    # with TBH >= TBS is coherent saturation, not a contradiction. Some
    # reading is wrong but not which one: TBS, the most robust sensor, is
    # kept; HR, TBH and DH (a function of HR) are dropped.
    t_v, w_v = out['dry_bulb_temperature'], out['wet_bulb_temperature']
    h_v = out['relative_humidity']
    contradiction = ((h_v - relative_humidity_from_bulbs(t_v, w_v)).abs()
                     .gt(HR_CONTRADICTION_TOLERANCE)
                     & ~(h_v.ge(99) & w_v.ge(t_v)))
    for col in ['relative_humidity', 'wet_bulb_temperature', 'humidity_deficit']:
        invalidate(col, contradiction, 'sensor_box_contradiction')

    return out, pd.DataFrame(audit, columns=['timestamp', 'variable', 'original_value',
                                             'new_value', 'rule', 'previous', 'next'])


def interpolate_angular(s):
    """Fill bounded gaps <=4 via the shortest arc; 180 degrees is ambiguous."""
    out = s.copy()
    missing = s.isna().to_numpy()
    i = 0
    while i < len(s):
        if not missing[i]:
            i += 1
            continue
        start = i
        while i < len(s) and missing[i]:
            i += 1
        n = i - start
        if n > 4 or start == 0 or i == len(s):
            continue
        a, b = float(s.iloc[start-1]), float(s.iloc[i])
        if not (0 <= a <= 360 and 0 <= b <= 360):
            continue
        delta = (b - a + 180) % 360 - 180
        if np.isclose(abs(delta), 180, rtol=0, atol=1e-10):
            continue
        out.iloc[start:i] = (a + delta * np.arange(1, n+1) / (n+1)) % 360
    return out

ACTUATORS = [
    'humidification',
    'window_1_position',
    'window_2_position',
    'screen_1_position',
    'screen_2_position',
    'pipe_temperature',
]

SENSORS = [
    'dry_bulb_temperature',
    'wet_bulb_temperature',
    'relative_humidity',
    'humidity_deficit',
    'outdoor_temperature',
    'outdoor_relative_humidity',
    'wind_speed',
    'wind_direction',
    'radiation',
]

if set(SENSORS) & set(ACTUATORS):
    raise RuntimeError('A variable cannot be both a sensor and an actuator.')
if len(SENSORS) != 9 or len(ACTUATORS) != 6:
    raise RuntimeError('Expected 9 sensors and 6 actuators.')

def interpolate_short_gaps(s, max_gap=4):
    missing = s.isna()
    groups = missing.ne(missing.shift()).cumsum()
    length = missing.groupby(groups).transform('sum')

    eligible = missing & length.le(max_gap)

    interpolated = s.interpolate(
        method='linear',
        limit_area='inside'
    )

    result = s.copy()
    result.loc[eligible] = interpolated.loc[eligible]

    return result


def humidity_deficit_from_air(t, hr):
    """DH (g/m3) from dry-bulb temperature (degrees C) and relative humidity (%),
    Magnus saturation formula; identical to the reconstruction step."""
    es = 6.112 * np.exp(17.67 * t / (t + 243.5))
    return 216.7 * es / (t + 273.15) * (1 - hr / 100)


def impute(original):
    validate(original)
    numeric, missing_all = parse_numeric(original)
    output = original.copy()
    methods, fill_masks, updated_values = {}, {}, {}
    for col in SENSORS + ACTUATORS:
        s = numeric[col]
        missing = missing_all[col]
        if col == 'pipe_temperature':
            updated, method = interpolate_short_gaps(s, max_gap=4), 'linear<=4'
        elif col in ACTUATORS:
            updated, method = s.ffill(), 'ffill'
        elif col == 'wind_direction':
            updated, method = interpolate_angular(s), 'angular<=4'
        elif col == 'humidity_deficit':
            # Reconstruct missing DH from TBS and HR only
            # for complete gaps of at most 4 samples.
            dh = humidity_deficit_from_air(
                updated_values['dry_bulb_temperature'],
                updated_values['relative_humidity']
            )

            groups = missing.ne(missing.shift()).cumsum()
            gap_length = missing.groupby(groups).transform('sum')

            eligible = missing & gap_length.le(4)

            low, high = PHYSICAL_RANGES['humidity_deficit']
            valid_dh = dh.where(dh.between(low, high))

            updated = s.copy()
            updated.loc[eligible] = valid_dh.loc[eligible]

            method = 'derived_from_tbs_hr_gap_le_4'
        else:
            updated, method = interpolate_short_gaps(s, max_gap=4), 'linear<=4'
        updated_values[col] = updated
        fill_mask = missing & updated.notna()
        if col in ACTUATORS and col != 'pipe_temperature':
            # Preserve original actuator state representation.
            states = original[col].mask(missing).ffill()
            output.loc[fill_mask, col] = states.loc[fill_mask]
        else:
            # Write interpolated values, including pipe temperature.
            output.loc[fill_mask, col] = updated.loc[fill_mask].map(repr)
        if not output.loc[~missing, col].equals(original.loc[~missing, col]):
            raise RuntimeError(f'{col}: imputation modified non-missing values.')
        methods[col], fill_masks[col] = method, fill_mask

    # Final coherence pass (cleaning rule 8, applied to imputed values): an
    # interpolated value that creates a psychrometric contradiction with its
    # own row is rejected. Measured values are never touched.
    t, w, h = (updated_values[c] for c in
               ['dry_bulb_temperature', 'wet_bulb_temperature', 'relative_humidity'])
    psychrometric_bad = (
        (h - relative_humidity_from_bulbs(t, w)).abs()
        .gt(HR_CONTRADICTION_TOLERANCE)
        & ~(h.ge(99) & w.ge(t))
    )

    wet_above_bad = w.gt(t + WET_ABOVE_DRY_TOLERANCE)

    bad = psychrometric_bad | wet_above_bad
    dh_source_rejected = pd.Series(False, index=output.index)
    for col in ['dry_bulb_temperature', 'wet_bulb_temperature', 'relative_humidity']:
        reject = bad & fill_masks[col]
        output.loc[reject, col] = ''
        updated_values[col] = updated_values[col].mask(reject)
        fill_masks[col] &= ~reject
        if col != 'wet_bulb_temperature':  # DH derives from TBS and HR only
            dh_source_rejected |= reject
    reject_dh = fill_masks['humidity_deficit'] & dh_source_rejected
    output.loc[reject_dh, 'humidity_deficit'] = ''
    updated_values['humidity_deficit'] = updated_values['humidity_deficit'].mask(reject_dh)
    fill_masks['humidity_deficit'] &= ~reject_dh

    summary = [{'variable': col, 'method': methods[col],
                'missing_before': int(missing_all[col].sum()),
                'filled': int(fill_masks[col].sum()),
                'missing_after': int(updated_values[col].isna().sum())}
               for col in SENSORS + ACTUATORS]
    return output, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?', default=RAW / 'ETIFA_raw_data.csv')
    parser.add_argument('--output-dir', type=Path, default=PROCESSED)
    args = parser.parse_args()
    names = ['ETIFA_clean.csv', 'ETIFA_cleaning_audit.csv',
             'ETIFA_clean_imputed_1h.csv',
             'ETIFA_imputation_summary.csv']
    paths = [args.output_dir / name for name in names]
    if any(p.exists() or p.resolve() == args.input.resolve() for p in paths):
        parser.error('Outputs exist or match the input. Choose a different output directory.')
    original = pd.read_csv(args.input, dtype=str, keep_default_na=False,
                           encoding='utf-8-sig')
    original = original.rename(columns=COLUMN_MAPPING)
    # Validate schema, numeric values and timestamps before applying rules.
    validate(original)
    numeric, _ = parse_numeric(original)
    # Drop leading rows recorded before TRIM_TO_FIRST_VALID was logged.
    start = numeric[TRIM_TO_FIRST_VALID].first_valid_index()
    if start is None:
        raise ValueError(f'{TRIM_TO_FIRST_VALID} has no valid values.')
    if start > 0:
        print(f'Trimmed {start} leading rows before {original.at[start, "timestamp"]} '
              f'(first valid {TRIM_TO_FIRST_VALID}).')
        original = original.iloc[start:].reset_index(drop=True)
        numeric = numeric.iloc[start:].reset_index(drop=True)
    cleaned_numeric, audit = clean(numeric)
    cleaned = original.copy()
    for col in SENSORS + ACTUATORS:
        invalidated = numeric[col].notna() & cleaned_numeric[col].isna()
        cleaned.loc[invalidated, col] = ''
    filled, summary = impute(cleaned)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for frame, path in zip([cleaned, audit, filled, summary], paths):
        frame.to_csv(path, index=False)
    print('Invalidated cells:', len(audit))
    print(summary.to_string(index=False))
    for path in paths:
        print(path)

if __name__ == '__main__':
    main()
