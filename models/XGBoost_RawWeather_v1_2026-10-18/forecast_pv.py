"""
forecast_pv.py - Forecast total South African PV production (MW) from hourly weather per PV plant.

Inputs
  weather      : hourly weather per plant (CSV path or DataFrame), a 'timestamp' column (or datetime index)
                 plus columns <plant>_ghi, <plant>_dni, <plant>_dhi (W/m2), <plant>_temp (degC),
                 <plant>_cloud (%). See input_template.csv. Times are SAST (UTC+2), hour beginning.
  past_output  : only if the model uses the lag (config.json "lag_hours"): Eskom's actual PV output (MW)
                 per hour, covering at least lag_hours before the first forecast hour.
  radiation_convention : "hour_beginning" (default; radiation is the mean over t to t+1, as in training)
                 or "open_meteo" (radiation is the mean over the preceding hour; shifted 1 h earlier here).

Everything else (model, physics settings, plant capacities and commissioning dates) is read from this folder.

From Python:
    import sys
    sys.path.append('/content/drive/MyDrive/.../PV_Production_Forecast/<model folder>')
    from forecast_pv import forecast
    result = forecast('weather.csv')                                   # model without lag
    result = forecast('weather.csv', past_output='eskom_pv_actual.csv')  # model with lag
    result = forecast(open_meteo_df, radiation_convention='open_meteo')

From the command line:
    python forecast_pv.py weather.csv [--past eskom_pv_actual.csv] [--out forecast.csv] [--open-meteo]

Output per hour: forecast_mw (total PV production), physics_mw (physics model before the ML correction,
if used), capacity_online_mw.
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
VARS = ['ghi', 'dni', 'dhi', 'temp', 'cloud']
_loaded = {}


def load(model_dir=MODEL_DIR):
    """Load model, settings and plants once (cached)."""
    if model_dir not in _loaded:
        with open(os.path.join(model_dir, 'config.json')) as f:
            config = json.load(f)
        model = joblib.load(os.path.join(model_dir, 'model.joblib'))
        plants = pd.read_csv(os.path.join(model_dir, 'plants.csv'), parse_dates=['online_from', 'aligned_online'])
        _loaded[model_dir] = (model, config, plants)
    return _loaded[model_dir]


def read_table(source):
    """CSV path, DataFrame or Series -> table with a sorted datetime index."""
    if isinstance(source, (str, os.PathLike)):
        df = pd.read_csv(source, sep=None, engine='python')
    else:
        df = source.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        df = df.set_index(ts_col)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'timestamp'
    df.columns = [str(c).strip() for c in df.columns]
    return df.sort_index()


def capacity_online(plants, index):
    """MW each plant has online per hour (0 before its aligned commissioning date)."""
    t = pd.DatetimeIndex(index).to_numpy()
    cap = np.zeros((len(t), len(plants)))
    for k, p in enumerate(plants.itertuples()):
        on = np.ones(len(t), bool) if pd.isna(p.aligned_online) else t >= np.datetime64(p.aligned_online)
        cap[:, k] = on * p.capacity_mw
    return cap


def forecast(weather, past_output=None, model_dir=MODEL_DIR, radiation_convention='hour_beginning'):
    """Forecast total PV production (MW) for every hour in weather."""
    model, config, plants = load(model_dir)
    labels, names = config['feature_labels'], config['weather_plants']
    w = read_table(weather)

    # Radiation timing: training used hour-beginning radiation (mean over t to t+1)
    if radiation_convention == 'open_meteo':
        rad = [c for c in w.columns if c.rsplit('_', 1)[-1] in ('ghi', 'dni', 'dhi')]
        shifted = w[rad].copy()
        shifted.index = shifted.index - pd.Timedelta(hours=1)
        w = w.drop(columns=rad).join(shifted, how='inner')
    elif radiation_convention != 'hour_beginning':
        raise ValueError("radiation_convention must be 'hour_beginning' or 'open_meteo'")

    uses_sim = any(l.endswith('_sim_mw') for l in labels)
    lag_labels = [l for l in labels if l.startswith('pv_lag')]
    needed = {l for l in labels if l.rsplit('_', 1)[-1] in VARS}
    if uses_sim:
        needed |= {f'{n}_{v}' for n in names for v in ('ghi', 'dni', 'dhi', 'temp')}
    missing = sorted(needed - set(w.columns))
    if missing:
        raise ValueError(f"Weather columns missing ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    idx = w.index
    all_cap = capacity_online(plants, idx).sum(axis=1)
    cols = {'plant_list_online_mw': all_cap}
    for c in needed:
        cols[c] = pd.to_numeric(w[c], errors='coerce').to_numpy(float)

    physics_mw = np.full(len(idx), np.nan)
    if uses_sim:
        ph = config['physics']
        g, b, d, t = [np.nan_to_num(w[[f'{n}_{v}' for n in names]].apply(pd.to_numeric, errors='coerce')
                                    .to_numpy(float)) for v in ('ghi', 'dni', 'dhi', 'temp')]
        irr = (1 - ph['tracker_weight']) * g + ph['tracker_weight'] * (b + d)
        t_cell = t + ph['heat_k'] * irr
        cf = np.clip(ph['dc_ac'] * irr / 1000 * (1 + ph['gamma'] * (t_cell - 25)), 0, 1)
        wp = plants.set_index('name').loc[names].reset_index()
        sim = cf * capacity_online(wp, idx)
        for k, n in enumerate(names):
            cols[f'{n}_sim_mw'] = sim[:, k]
        physics_mw = sim.sum(axis=1)

    if lag_labels:
        h = config['lag_hours']
        if past_output is None:
            raise ValueError(f"This model uses Eskom's actual PV output {h} h earlier: pass past_output=...")
        po = read_table(past_output)
        po = po['total_pv_generation'] if 'total_pv_generation' in po.columns else po.iloc[:, 0]
        lag_idx = idx - pd.Timedelta(hours=h)
        lag_mw = pd.to_numeric(po, errors='coerce').reindex(lag_idx).to_numpy(float)
        lag_cap = capacity_online(plants, lag_idx).sum(axis=1)
        lag_cf = lag_mw / np.where(lag_cap > 0, lag_cap, np.nan)
        cols[f'pv_lag{h}_mw'] = lag_mw
        cols[f'pv_lag{h}_cf'] = lag_cf
        cols[f'pv_lag{h}_scaled_mw'] = lag_cf * all_cap

    X = pd.DataFrame({l: cols[l] for l in labels}, index=idx)
    complete = ~X.isna().any(axis=1).to_numpy()
    X.columns = config['feature_names']
    pred = np.full(len(X), np.nan)
    if complete.any():
        pred[complete] = np.clip(model.predict(X[complete]), 0, None)
    if not complete.all():
        why = f"missing weather or no actual output {config['lag_hours']} h earlier" if lag_labels else "missing weather"
        print(f"note: {int((~complete).sum())} of {len(X)} hours not forecast ({why})")

    return pd.DataFrame({'forecast_mw': pred, 'physics_mw': physics_mw, 'capacity_online_mw': all_cap}, index=idx)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Forecast total South African PV production from weather per plant.')
    ap.add_argument('weather')
    ap.add_argument('--past', help="Eskom actual PV output (needed if the model uses the lag)")
    ap.add_argument('--out', help='output CSV')
    ap.add_argument('--open-meteo', action='store_true', help='radiation is Open-Meteo preceding-hour mean')
    a = ap.parse_args()
    result = forecast(a.weather, past_output=a.past,
                      radiation_convention='open_meteo' if a.open_meteo else 'hour_beginning')
    if a.out:
        result.to_csv(a.out)
        print(f"Forecast for {len(result):,} hours written to {a.out}")
    else:
        print(result.to_string())
