from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLS = ['unit', 'cycle'] + [f'op{i}' for i in range(1,4)] + [f's{i}' for i in range(1,22)]
SENSORS = [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]

def validate(df):
    if not set(COLS).issubset(df.columns):
        raise ValueError('Required columns: ' + ', '.join(COLS))
    df = df[COLS].copy()
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in COLS):
        raise ValueError('All columns must be numeric.')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('Missing or infinite values are not supported.')
    if ((df[['unit','cycle']] < 1) | (df[['unit','cycle']] % 1 != 0)).any().any():
        raise ValueError('Unit and cycle must be positive integers.')
    if df.duplicated(['unit','cycle']).any():
        raise ValueError('Duplicate unit/cycle observations.')
    return df.sort_values(['unit','cycle']).reset_index(drop=True)

def read(name):
    return validate(pd.read_csv(ROOT / 'data' / name, sep=r'\s+', header=None, names=COLS))

def features(df):
    df = validate(df)
    out = df[['cycle'] + SENSORS].copy()
    for sensor in SENSORS:
        g = df.groupby('unit')[sensor]
        out[sensor+'_mean10'] = g.transform(lambda x: x.rolling(10,min_periods=1).mean())
        out[sensor+'_delta10'] = df[sensor] - g.shift(10).fillna(df[sensor])
    return out

def labels(df, offsets=None):
    end = df.groupby('unit')['cycle'].transform('max')
    if offsets is not None:
        end = end + df.unit.map(offsets)
    return end - df.cycle
