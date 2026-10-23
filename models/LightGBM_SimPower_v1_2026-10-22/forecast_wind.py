"""
forecast_wind.py - Forecast total South African wind production (MW) from wind speed per site.

The only input is hourly wind speed at 100 m (m/s) for each site, with timestamps.
Everything else (model, power curve, site capacities and commissioning dates) is read
from the files in this folder.

From Python:
    import sys
    sys.path.append('/content/drive/MyDrive/.../Wind_Production_Forecast/LightGBM_SimPower_v1_YYYY-MM-DD')
    from forecast_wind import forecast
    result = forecast('my_wind_speeds.csv')     # a CSV path or a DataFrame
    result = forecast(wind_speed_df)            # index = timestamp, one column per site

From the command line:
    python forecast_wind.py my_wind_speeds.csv forecast_output.csv

Input columns: 'timestamp' plus one column per site, named as in farms.csv
("Cookhouse" or "Cookhouse_wind_speed" both work). Extra columns are ignored.
Output: forecast_mw (total wind production), power_curve_mw, capacity_online_mw per hour.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_loaded = {}


def load(model_dir=MODEL_DIR):
    """Load model, settings and sites once (cached)."""
    if model_dir not in _loaded:
        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        farms = pd.read_csv(os.path.join(model_dir, 'farms.csv'), parse_dates=['online_from', 'aligned_online'])
        farms = farms.set_index('name').loc[config['site_order']].reset_index()   # training order
        _loaded[model_dir] = (model, config, farms)
    return _loaded[model_dir]


def read_wind_speed(source):
    """Wind speed table from a CSV path or DataFrame -> timestamp index, columns named after sites."""
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_csv(source, sep=None, engine='python')
    else:
        df = source.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        df = df.set_index(ts_col)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'timestamp'
    df.columns = [str(c).strip().removesuffix('_wind_speed') for c in df.columns]
    return df.sort_index()


def forecast(wind_speed, model_dir=MODEL_DIR, allow_missing_sites=False):
    """
    Forecast total wind production (MW) for every hour in wind_speed.
    allow_missing_sites: if True, sites without wind speed are passed as missing (less accurate);
                         if False (default), missing sites raise an error.
    """
    model, config, farms = load(model_dir)
    pc = config['power_curve']
    s = read_wind_speed(wind_speed)

    missing = [n for n in config['site_order'] if n not in s.columns]
    if missing and not allow_missing_sites:
        raise ValueError(f"Wind speed missing for {len(missing)} sites: {missing}. "
                         "Add them, or call forecast(..., allow_missing_sites=True).")
    for n in missing:
        s[n] = np.nan
    v = s[config['site_order']].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

    # Standard power curve (identical to training)
    e, ci, r = pc['exponent'], pc['cut_in'], pc['rated']
    cf = np.clip((v ** e - ci ** e) / (r ** e - ci ** e), 0, 1)
    if pc.get('cut_out'):
        cf = np.where(v >= pc['cut_out'], 0.0, cf)
    cf = np.where(np.isnan(v), np.nan, cf)

    # Capacity online each hour: 0 before each site's commissioning date
    t = s.index.to_numpy()
    cap = np.zeros_like(cf)
    for k, f in enumerate(farms.itertuples()):
        on = np.ones(len(t), bool) if pd.isna(f.aligned_online) else t >= np.datetime64(f.aligned_online)
        cap[:, k] = on * f.capacity_mw

    X = pd.DataFrame(cf * cap, columns=config['feature_names'], index=s.index)
    out = pd.DataFrame({'forecast_mw': np.clip(model.predict(X), 0, None),
                        'power_curve_mw': np.nansum(cf * cap, axis=1),
                        'capacity_online_mw': cap.sum(axis=1)}, index=s.index)
    if missing:
        out['sites_missing'] = len(missing)
    return out


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = forecast(sys.argv[1])
    if len(sys.argv) > 2:
        result.to_csv(sys.argv[2])
        print(f"Forecast for {len(result):,} hours written to {sys.argv[2]}")
    else:
        print(result.to_string())
