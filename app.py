
import os
import sys
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'dashboard_config.json')) as f:
    CFG = json.load(f)


def here(path):
    """Config paths are relative to this file, so the folder runs anywhere (Colab or Streamlit Cloud)."""
    return path if not path or os.path.isabs(path) else os.path.join(HERE, path)

st.set_page_config(page_title='Wind and Solar Dashboard', layout='wide')
COLOURS = {'Wind': '#1f77b4', 'Solar PV': '#ff9f1c'}
SHOW_TZ = CFG.get('display_timezone', 'Africa/Johannesburg')
SHEAR_120_TO_100 = (100 / 120) ** 0.14   # standard wind shear, about -2.5 %

# Wind weather forecast options: (kind, Open-Meteo model, forecast days)
FORECAST_SOURCES = {
    'Ensemble: ECMWF (51 members, 15 days)': ('ensemble', 'ecmwf_ifs025', 15),
    'Deterministic (single forecast, 16 days)': ('deterministic', None, 16),
}
# PV weather: Open-Meteo variable -> short name used by the PV model
PV_OM_VARS = {'shortwave_radiation': 'ghi', 'direct_normal_irradiance': 'dni', 'diffuse_radiation': 'dhi',
              'temperature_2m': 'temp', 'cloud_cover': 'cloud'}


# =============================================================================================
# Shared Wind / Solar PV toggle (the choice carries over between pages)
# =============================================================================================
def tech_toggle(options, page):
    if st.session_state.get('tech') not in options:
        st.session_state['tech'] = options[0]
    choice = st.radio('Technology', options, index=options.index(st.session_state['tech']),
                      horizontal=True, key=f'tech_{page}')
    st.session_state['tech'] = choice
    return choice


# =============================================================================================
# Helpers and data loading (cached)
# =============================================================================================
def to_number(s):
    """Numbers saved with a decimal comma (e.g. '-30,1595' from Excel) -> float."""
    return pd.to_numeric(pd.Series(s).astype(str).str.strip().str.replace(',', '.', regex=False), errors='coerce')


def to_tz(index, from_tz, to_tz_name):
    return pd.DatetimeIndex(index).tz_localize(from_tz).tz_convert(to_tz_name).tz_localize(None)


def _series(values, times):
    return pd.Series(pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(), index=times)


@st.cache_data
def load_facilities():
    parts = []
    for source, path in CFG['facility_files'].items():
        t = pd.read_csv(here(path), sep=None, engine='python')
        t.columns = [str(c).strip() for c in t.columns]
        if 'capacity_mw' not in t.columns and 'capacity' in t.columns:
            t = t.rename(columns={'capacity': 'capacity_mw'})
        t['source'] = source
        t['name'] = t['name'].astype(str).str.strip().replace({'nan': None, '': None})
        for c in ('capacity_mw', 'latitude', 'longitude'):
            t[c] = to_number(t[c]).to_numpy() if c in t.columns else np.nan
        parts.append(t.dropna(subset=['name']))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@st.cache_data
def load_eskom(tech):
    """Eskom raw hourly data for 'wind' or 'pv': generation_mw, installed_capacity_mw, cf."""
    d = pd.read_csv(os.path.join(HERE, 'data', f'eskom_{tech}_hourly.csv'))
    d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
    d = d.dropna(subset=['timestamp']).drop_duplicates(subset='timestamp', keep='last')
    d['cf'] = d['generation_mw'] / d['installed_capacity_mw']
    return d.set_index('timestamp').sort_index()


@st.cache_data
def load_demand():
    """Eskom hourly RSA demand (MW), indexed by timestamp; empty if the setup cell found no demand column."""
    path = os.path.join(HERE, 'data', 'eskom_demand_hourly.csv')
    if not CFG.get('demand_column') or not os.path.exists(path):
        return pd.Series(dtype=float)
    d = pd.read_csv(path)
    d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
    d = d.dropna(subset=['timestamp']).drop_duplicates(subset='timestamp', keep='last')
    return d.set_index('timestamp').sort_index()['demand_mw']


@st.cache_resource
def load_wind_forecaster():
    if here(CFG['model_dir']) not in sys.path:
        sys.path.insert(0, here(CFG['model_dir']))
    import forecast_wind
    return forecast_wind


@st.cache_resource
def load_pv_forecaster():
    if here(CFG['pv_model_dir']) not in sys.path:
        sys.path.insert(0, here(CFG['pv_model_dir']))
    import forecast_pv
    return forecast_pv


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_deterministic(names, lats, lons, days):
    """Single forecast: hourly wind speed (m/s) at 120 m per site, UTC. Returns ({'forecast': df}, 120)."""
    r = requests.get('https://api.open-meteo.com/v1/forecast', timeout=60, params={
        'latitude': ','.join(f'{x:.4f}' for x in lats),
        'longitude': ','.join(f'{x:.4f}' for x in lons),
        'hourly': 'wind_speed_120m', 'wind_speed_unit': 'ms', 'timezone': 'GMT', 'forecast_days': days})
    r.raise_for_status()
    data = r.json()
    data = data if isinstance(data, list) else [data]
    df = pd.DataFrame({n: _series(loc['hourly']['wind_speed_120m'], pd.to_datetime(loc['hourly']['time']))
                       for n, loc in zip(names, data)})
    return {'forecast': df}, 120


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def fetch_ensemble(names, lats, lons, model, days, batch=10):
    """Ensemble forecast: {member: hourly wind speed (m/s) per site, UTC}, and the height used.
    Tries 100 m (the model's training height) first, then 120 m."""
    errors = []
    for height in (100, 120):
        var = f'wind_speed_{height}m'
        members, failed = {}, False
        for i in range(0, len(names), batch):
            r = requests.get('https://ensemble-api.open-meteo.com/v1/ensemble', timeout=120, params={
                'latitude': ','.join(f'{x:.4f}' for x in lats[i:i + batch]),
                'longitude': ','.join(f'{x:.4f}' for x in lons[i:i + batch]),
                'hourly': var, 'models': model, 'wind_speed_unit': 'ms', 'timezone': 'GMT',
                'forecast_days': days})
            if r.status_code != 200:
                try:
                    errors.append(f"{var}: {r.json().get('reason', r.text[:200])}")
                except Exception:
                    errors.append(f"{var}: HTTP {r.status_code}")
                failed = True
                break
            data = r.json()
            data = data if isinstance(data, list) else [data]
            for name, loc in zip(names[i:i + batch], data):
                h = loc.get('hourly', {})
                times = pd.to_datetime(h.get('time', []))
                for key, values in h.items():
                    if key == var or key.startswith(var + '_member'):
                        member = 'control' if key == var else 'm' + key.rsplit('_member', 1)[-1]
                        members.setdefault(member, {})[name] = _series(values, times)
        if failed or not members:
            continue
        frames = {m: pd.DataFrame(cols) for m, cols in members.items()}
        if np.mean([f.notna().to_numpy().mean() for f in frames.values()]) > 0.5:
            return frames, height
        errors.append(f"{var}: no values returned")
    raise RuntimeError(f"No ensemble wind speed from model '{model}'. " + ' | '.join(errors))


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pv_weather(names, lats, lons, days, batch=20):
    """Hourly PV weather per plant from Open-Meteo, in SAST (Africa/Johannesburg), as the PV model expects.
    Columns <plant>_ghi, _dni, _dhi (W/m2, mean over the preceding hour), _temp (degC), _cloud (%)."""
    cols = {}
    for i in range(0, len(names), batch):
        r = requests.get('https://api.open-meteo.com/v1/forecast', timeout=90, params={
            'latitude': ','.join(f'{x:.4f}' for x in lats[i:i + batch]),
            'longitude': ','.join(f'{x:.4f}' for x in lons[i:i + batch]),
            'hourly': ','.join(PV_OM_VARS), 'timezone': 'Africa/Johannesburg', 'forecast_days': days})
        r.raise_for_status()
        data = r.json()
        data = data if isinstance(data, list) else [data]
        for name, loc in zip(names[i:i + batch], data):
            h = loc['hourly']
            times = pd.to_datetime(h['time'])
            for om, short in PV_OM_VARS.items():
                cols[f'{name}_{short}'] = _series(h[om], times)
    return pd.DataFrame(cols)


# =============================================================================================
# Performance pages (Eskom raw hourly data)
# =============================================================================================
def page_performance(tech, title, colour, heat_scale):
    st.subheader(title)
    eskom = load_eskom(tech)
    if eskom.empty:
        st.error('No Eskom data with valid dates. Re-run the dashboard setup cell.')
        st.stop()
    start, end = eskom.index.min().date(), eskom.index.max().date()
    picked = st.date_input('Period', (start, end), min_value=start, max_value=end, key=f'period_{tech}')
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        d = eskom[(eskom.index >= pd.Timestamp(picked[0])) &
                  (eskom.index < pd.Timestamp(picked[1]) + pd.Timedelta(days=1))]
    else:
        d = eskom
    if d.empty:
        st.warning('No data in this period.')
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Energy produced', f"{d['generation_mw'].sum() / 1e6:,.2f} TWh")
    c2.metric('Average output', f"{d['generation_mw'].mean():,.0f} MW")
    c3.metric('Average capacity factor', f"{d['cf'].mean():.1%}")
    peak_t = d['generation_mw'].idxmax()
    c4.metric('Peak output', f"{d['generation_mw'].max():,.0f} MW", f'{peak_t:%d %b %Y %H:%M}', delta_color='off')
    c5.metric('Installed capacity (latest)', f"{d['installed_capacity_mw'].iloc[-1]:,.0f} MW")

    monthly = d['generation_mw'].resample('MS').sum() / 1000          # GWh
    fig = go.Figure()
    fig.add_bar(x=monthly.index, y=monthly, name='Energy', marker_color=colour,
                hovertemplate='%{x|%b %Y}: %{y:,.0f} GWh<extra></extra>')
    fig.update_layout(title='Monthly energy produced', height=380, yaxis=dict(title='GWh'),
                      showlegend=False, margin=dict(t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

    heat = d.assign(month=d.index.month, hour=d.index.hour).pivot_table(
        index='month', columns='hour', values='cf', aggfunc='mean')
    heat.index = [pd.Timestamp(2000, m, 1).strftime('%b') for m in heat.index]
    fig = px.imshow(heat, aspect='auto', color_continuous_scale=heat_scale,
                    labels=dict(x='Hour of day (SAST)', y='', color='CF'),
                    title='Average capacity factor by month and hour')
    fig.update_layout(height=420, margin=dict(t=50, b=10), coloraxis_colorbar=dict(tickformat='.0%'))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================================
# Wind forecast (ECMWF ensemble or deterministic)
# =============================================================================================
def page_wind_forecast():
    st.subheader('Wind generation forecast')
    model_tz = CFG.get('model_timezone', 'UTC')
    fw = load_wind_forecaster()
    _, model_cfg, sites = fw.load()

    top_left, top_right = st.columns([3, 1])
    source = top_left.radio('Weather forecast', list(FORECAST_SOURCES), horizontal=True)
    if top_right.button('Refresh weather forecast', key='refresh_wind'):
        fetch_ensemble.clear()
        fetch_deterministic.clear()
    kind, om_model, days = FORECAST_SOURCES[source]
    names, lats, lons = tuple(sites['name']), tuple(sites['latitude']), tuple(sites['longitude'])

    with st.spinner('Fetching wind speed forecasts for every site...'):
        try:
            if kind == 'ensemble':
                members, height = fetch_ensemble(names, lats, lons, om_model, days)
            else:
                members, height = fetch_deterministic(names, lats, lons, days)
        except Exception as e:
            st.error(f'Could not fetch the weather forecast: {e}')
            st.stop()

    scale = SHEAR_120_TO_100 if (height == 120 and CFG.get('scale_120m_to_100m', False)) else 1.0

    with st.spinner(f'Running {len(members)} forecast member(s) through the model...'):
        runs, capacity = {}, None
        for m, sp in members.items():
            sp = (sp * scale).dropna(how='any')
            if sp.empty:
                continue
            sp.index = to_tz(sp.index, 'UTC', model_tz)
            out = fw.forecast(sp, allow_missing_sites=True)
            out.index = to_tz(out.index, model_tz, SHOW_TZ)
            runs[m] = out['forecast_mw']
            if capacity is None:
                capacity = out['capacity_online_mw']
    if not runs:
        st.error('The weather forecast returned no usable hours.')
        st.stop()

    now = pd.Timestamp.now(tz=SHOW_TZ).tz_localize(None).floor('h')
    ens = pd.DataFrame(runs).sort_index()
    ens = ens[ens.index >= now].dropna(how='all')
    if ens.empty:
        st.warning('The weather forecast returned no future hours.')
        st.stop()
    n_members = ens.shape[1]
    q = ens.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    q.columns = ['p10', 'p25', 'p50', 'p75', 'p90']
    capacity = capacity.reindex(ens.index)

    total_gwh = ens.sum() / 1000
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Expected energy, P50', f"{total_gwh.median():,.0f} GWh",
              f"P10-P90: {total_gwh.quantile(0.1):,.0f}-{total_gwh.quantile(0.9):,.0f} GWh" if n_members > 1 else None,
              delta_color='off')
    c2.metric('Average output, P50', f"{q['p50'].mean():,.0f} MW")
    c3.metric('Peak (P50)', f"{q['p50'].max():,.0f} MW", f"{q['p50'].idxmax():%a %d %b %H:%M}", delta_color='off')
    c4.metric('Lowest (P50)', f"{q['p50'].min():,.0f} MW", f"{q['p50'].idxmin():%a %d %b %H:%M}", delta_color='off')
    c5.metric('Forecast members', f'{n_members}', f'wind at {height} m', delta_color='off')

    show_members = n_members > 1 and st.toggle('Show every ensemble member', value=False)
    fig = go.Figure()
    if n_members > 1:
        fig.add_scatter(x=q.index, y=q['p90'], line=dict(width=0), showlegend=False, hoverinfo='skip')
        fig.add_scatter(x=q.index, y=q['p10'], fill='tonexty', fillcolor='rgba(31,119,180,0.15)',
                        line=dict(width=0), name='P10-P90')
        fig.add_scatter(x=q.index, y=q['p75'], line=dict(width=0), showlegend=False, hoverinfo='skip')
        fig.add_scatter(x=q.index, y=q['p25'], fill='tonexty', fillcolor='rgba(31,119,180,0.30)',
                        line=dict(width=0), name='P25-P75')
    if show_members:
        for m in ens.columns:
            fig.add_scatter(x=ens.index, y=ens[m], line=dict(color='rgba(100,100,100,0.25)', width=0.7),
                            showlegend=False, hoverinfo='skip')
    fig.add_scatter(x=q.index, y=q['p50'], name='P50 (median)' if n_members > 1 else 'Forecast',
                    line=dict(color='#1f77b4', width=2.2))
    fig.add_scatter(x=capacity.index, y=capacity, name='Capacity of modelled farms', line=dict(color='grey', dash='dot'))
    fig.update_layout(title=f'Hourly forecast ({SHOW_TZ} time)', yaxis_title='MW', height=450,
                      legend=dict(orientation='h', y=1.1), margin=dict(t=60, b=10), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    daily_member = ens.resample('D').sum() / 1000
    daily = pd.DataFrame({'p10': daily_member.quantile(0.1, axis=1), 'p50': daily_member.median(axis=1),
                          'p90': daily_member.quantile(0.9, axis=1), 'hours': ens.resample('D').size()})
    st.markdown('**Daily summary**')
    table = daily.rename(columns={'p10': 'P10 (GWh)', 'p50': 'P50 (GWh)', 'p90': 'P90 (GWh)', 'hours': 'Hours'})
    table.index = table.index.strftime('%a %d %b')
    st.dataframe(table.style.format({'P10 (GWh)': '{:.1f}', 'P50 (GWh)': '{:.1f}', 'P90 (GWh)': '{:.1f}'}),
                 use_container_width=True)
    st.caption('First and last days can be partial (see Hours).')

    with st.expander(f'Forecast wind speed per site ({height} m, m/s, mean of members)'):
        mean_speed = sum(members.values()) / len(members)
        mean_speed.index = to_tz(mean_speed.index, 'UTC', SHOW_TZ)
        mean_speed = mean_speed[mean_speed.index >= now]
        fig = px.imshow(mean_speed.T, aspect='auto', color_continuous_scale='Viridis',
                        labels=dict(x='', y='', color='m/s'), height=max(400, 14 * mean_speed.shape[1]))
        st.plotly_chart(fig, use_container_width=True)

    download = q.join(capacity.rename('capacity_online_mw')).round(1)
    st.download_button('Download hourly forecast with percentiles (CSV)', download.to_csv(),
                       'wind_forecast.csv', 'text/csv')
    st.caption(f"Weather: Open-Meteo, {source}, wind at {height} m{' scaled to 100 m' if scale != 1.0 else ''}. "
               f"Each member is run through the model ({model_cfg['model']}, trained to "
               f"{model_cfg['trained_on']['end'][:10]}); P10/P50/P90 are taken across members. "
               f"Spread shows weather uncertainty only, not model error. "
               f"Generated {pd.Timestamp.now(tz=SHOW_TZ):%d %b %Y %H:%M}.")


# =============================================================================================
# PV forecast (deterministic weather forecast)
# =============================================================================================
def page_pv_forecast():
    st.subheader('Solar PV generation forecast')
    if not CFG.get('pv_model_dir') or not os.path.exists(os.path.join(here(CFG['pv_model_dir']), 'forecast_pv.py')):
        st.info('No saved PV forecast model found. Run the PV model-saving cell, then re-run the dashboard setup cell.')
        return
    fp = load_pv_forecaster()
    _, pv_cfg, plants = fp.load()
    wp = plants[plants['has_weather'].astype(str).str.lower() == 'true'].dropna(subset=['latitude', 'longitude'])
    lag = pv_cfg.get('lag_hours')

    top_left, top_right = st.columns([3, 1])
    top_left.markdown(f"Model: **{pv_cfg['model']}** (trained to {pv_cfg['trained_on']['end'][:10]}) | "
                      f"{len(wp)} plants with weather")
    if top_right.button('Refresh weather forecast', key='refresh_pv'):
        fetch_pv_weather.clear()

    past = None
    if lag:
        st.warning(f"This PV model uses Eskom's actual PV output {lag} h earlier. Without recent actual output only "
                   f"the physics-model estimate can be shown. Upload Eskom's recent hourly PV output to forecast "
                   f"the first {lag} hours after its last hour.")
        up = st.file_uploader("Recent Eskom PV output (CSV with 'timestamp' and MW columns)", type='csv')
        past = (pd.read_csv(up, sep=None, engine='python') if up is not None
                else pd.Series(dtype=float, index=pd.DatetimeIndex([], name='timestamp')))

    with st.spinner('Fetching solar weather forecasts for every PV plant...'):
        try:
            weather = fetch_pv_weather(tuple(wp['name']), tuple(wp['latitude']), tuple(wp['longitude']), 16)
        except Exception as e:
            st.error(f'Could not fetch the weather forecast: {e}')
            st.stop()

    try:
        out = fp.forecast(weather, past_output=past, radiation_convention='open_meteo')
    except Exception as e:
        st.error(f'The PV forecast program failed: {e}')
        st.stop()

    now = pd.Timestamp.now(tz=SHOW_TZ).tz_localize(None).floor('h')
    out = out[out.index >= now]
    if out.empty:
        st.warning('The weather forecast returned no future hours.')
        st.stop()

    has_model = out['forecast_mw'].notna()
    has_physics = out['physics_mw'].notna().any()
    if has_model.all():
        main, main_label = out['forecast_mw'], 'Forecast (model)'
    elif has_model.any():
        main, main_label = out['forecast_mw'].fillna(out['physics_mw']), 'Forecast (model, then physics estimate)'
        st.info(f"Model forecast for {int(has_model.sum())} hours; the rest shows the physics-model estimate.")
    elif has_physics:
        main, main_label = out['physics_mw'], 'Physics-model estimate'
    else:
        st.error('No forecast available: the model needs recent actual output and has no physics estimate.')
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Expected energy', f"{main.sum() / 1000:,.0f} GWh")
    c2.metric('Average daylight output', f"{main[main > 1].mean():,.0f} MW")
    c3.metric('Peak', f"{main.max():,.0f} MW", f"{main.idxmax():%a %d %b %H:%M}", delta_color='off')
    c4.metric('Capacity of modelled plants', f"{out['capacity_online_mw'].iloc[-1]:,.0f} MW")

    fig = go.Figure()
    fig.add_scatter(x=main.index, y=main, name=main_label, line=dict(color='#ff9f1c', width=2),
                    fill='tozeroy', fillcolor='rgba(255,159,28,0.15)')
    if has_physics and main_label.startswith('Forecast'):
        fig.add_scatter(x=out.index, y=out['physics_mw'], name='Physics model (before ML correction)',
                        line=dict(color='#2ca02c', width=1, dash='dash'))
    fig.add_scatter(x=out.index, y=out['capacity_online_mw'], name='Capacity of modelled plants',
                    line=dict(color='grey', dash='dot'))
    fig.update_layout(title=f'Hourly PV forecast ({SHOW_TZ} time)', yaxis_title='MW', height=450,
                      legend=dict(orientation='h', y=1.1), margin=dict(t=60, b=10), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    daily = pd.DataFrame({'energy_gwh': main.resample('D').sum() / 1000, 'peak_mw': main.resample('D').max(),
                          'hours': main.resample('D').size()})
    st.markdown('**Daily summary**')
    table = daily.rename(columns={'energy_gwh': 'Energy (GWh)', 'peak_mw': 'Peak (MW)', 'hours': 'Hours'})
    table.index = table.index.strftime('%a %d %b')
    st.dataframe(table.style.format({'Energy (GWh)': '{:.1f}', 'Peak (MW)': '{:,.0f}'}), use_container_width=True)
    st.caption('First and last days can be partial (see Hours).')

    with st.expander('Forecast irradiance per plant (GHI, W/m²)'):
        ghi = weather[[c for c in weather.columns if c.endswith('_ghi')]]
        ghi = ghi[ghi.index >= now]
        ghi.columns = [c.removesuffix('_ghi') for c in ghi.columns]
        fig = px.imshow(ghi.T, aspect='auto', color_continuous_scale='YlOrRd',
                        labels=dict(x='', y='', color='W/m²'), height=max(400, 14 * ghi.shape[1]))
        st.plotly_chart(fig, use_container_width=True)

    download = pd.DataFrame({'forecast_mw': main}).join(out[['physics_mw', 'capacity_online_mw']]).round(1)
    st.download_button('Download hourly PV forecast (CSV)', download.to_csv(), 'pv_forecast.csv', 'text/csv')
    st.caption(f"Weather: Open-Meteo deterministic forecast in SAST; radiation shifted 1 h to the model's hour-beginning "
               f"convention. Single weather forecast (no ensemble), so no uncertainty band is shown. "
               f"Generated {pd.Timestamp.now(tz=SHOW_TZ):%d %b %Y %H:%M}.")


# =============================================================================================
# Demand and the wind / PV share
# =============================================================================================
def page_demand():
    st.title("South Africa's electricity demand and the wind and PV share")
    demand = load_demand()
    if demand.empty:
        st.info('No demand column was found in the Eskom file, so this page has nothing to show. '
                'Check the column list printed by the dashboard setup cell.')
        return
    d = (pd.DataFrame({'demand': demand})
         .join(load_eskom('wind')['generation_mw'].rename('wind'), how='inner')
         .join(load_eskom('pv')['generation_mw'].rename('pv'), how='inner').dropna())
    d = d[d['demand'] > 0]
    if d.empty:
        st.error('No hours with demand, wind and PV data together. Re-run the dashboard setup cell.')
        st.stop()
    start, end = d.index.min().date(), d.index.max().date()
    picked = st.date_input('Period', (start, end), min_value=start, max_value=end, key='period_demand')
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        d = d[(d.index >= pd.Timestamp(picked[0])) & (d.index < pd.Timestamp(picked[1]) + pd.Timedelta(days=1))]
    if d.empty:
        st.warning('No data in this period.')
        st.stop()
    d = d.copy()
    d['share'] = (d['wind'] + d['pv']) / d['demand']
    d['residual'] = d['demand'] - d['wind'] - d['pv']          # what all other sources must supply
    total = d[['demand', 'wind', 'pv']].sum()          # MWh

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Electricity demand', f"{total['demand'] / 1e6:,.1f} TWh",
              f"average {d['demand'].mean():,.0f} MW", delta_color='off')
    c2.metric('Peak demand', f"{d['demand'].max():,.0f} MW", f"{d['demand'].idxmax():%d %b %Y %H:%M}",
              delta_color='off')
    c3.metric('Share met by wind', f"{total['wind'] / total['demand']:.1%}")
    c4.metric('Share met by PV', f"{total['pv'] / total['demand']:.1%}")
    c5.metric('Share met by wind + PV', f"{(total['wind'] + total['pv']) / total['demand']:.1%}",
              f"highest hour {d['share'].max():.0%} ({d['share'].idxmax():%d %b %Y %H:%M})", delta_color='off')

    # --- Monthly: how demand was met ----------------------------------------------------------
    m = d[['demand', 'wind', 'pv']].resample('MS').sum() / 1000          # GWh
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45],
                        subplot_titles=('Energy produced', 'Share of demand'))
    for tech, col in (('Wind', 'wind'), ('Solar PV', 'pv')):
        fig.add_bar(x=m.index, y=m[col], name=tech, marker_color=COLOURS[tech], legendgroup=tech,
                    hovertemplate='%{x|%b %Y}: %{y:,.0f} GWh<extra>' + tech + '</extra>', row=1, col=1)
        fig.add_scatter(x=m.index, y=m[col] / m['demand'], name=tech, stackgroup='share', legendgroup=tech,
                        showlegend=False, line=dict(width=0.5, color=COLOURS[tech]),
                        hovertemplate='%{x|%b %Y}: %{y:.1%}<extra>' + tech + '</extra>', row=2, col=1)
    fig.update_layout(title='Monthly wind and PV: energy and share of demand', barmode='stack', height=560,
                      legend=dict(orientation='h', y=1.1, traceorder='normal'), margin=dict(t=90, b=10),
                      hovermode='x unified')
    fig.update_yaxes(title_text='GWh', row=1, col=1)
    fig.update_yaxes(title_text='Share of demand', tickformat='.0%', rangemode='tozero', row=2, col=1)
    fig.update_xaxes(range=[m.index.min() - pd.Timedelta(days=20), m.index.max() + pd.Timedelta(days=20)])
    st.plotly_chart(fig, use_container_width=True)

    # --- Average day: demand vs residual demand -------------------------------------------------
    prof = d.groupby(d.index.hour)[['demand', 'residual']].mean()
    fig = go.Figure()
    fig.add_scatter(x=prof.index, y=prof['residual'], name='Residual demand', line=dict(color='#d62728', width=2.5))
    fig.add_scatter(x=prof.index, y=prof['demand'], name='Demand', fill='tonexty',
                    fillcolor='rgba(44,160,44,0.18)', line=dict(color='#333', width=2.5))
    fig.add_scatter(x=[None], y=[None], name='Gap = met by wind + PV', mode='markers',
                    marker=dict(size=12, symbol='square', color='rgba(44,160,44,0.35)'))
    fig.update_layout(title='Average day: demand and residual demand (demand - wind - PV)',
                      xaxis_title='Hour of day (SAST)', yaxis=dict(title='MW'), xaxis=dict(dtick=2),
                      height=400, legend=dict(orientation='h', y=1.15), margin=dict(t=70, b=10),
                      hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # --- One week -----------------------------------------------------------------------------
    last_start = (d.index.max() - pd.Timedelta(days=7)).date()
    first = d.index.min().date()
    wk_start = st.date_input('Week starting', max(last_start, first), min_value=first,
                             max_value=max(last_start, first), key='demand_week')
    wk = d[(d.index >= pd.Timestamp(wk_start)) & (d.index < pd.Timestamp(wk_start) + pd.Timedelta(days=7))]
    fig = go.Figure()
    fig.add_scatter(x=wk.index, y=wk['wind'], name='Wind', stackgroup='re', line=dict(width=0.5, color=COLOURS['Wind']))
    fig.add_scatter(x=wk.index, y=wk['pv'], name='Solar PV', stackgroup='re',
                    line=dict(width=0.5, color=COLOURS['Solar PV']))
    fig.add_scatter(x=wk.index, y=wk['demand'], name='Demand', line=dict(color='#333', width=2))
    fig.add_scatter(x=wk.index, y=wk['residual'], name='Residual demand', line=dict(color='#d62728', width=1.5,
                                                                                    dash='dash'))
    fig.update_layout(title='Hourly demand, residual demand and wind and PV output for one week', yaxis_title='MW', height=400,
                      legend=dict(orientation='h', y=1.12), margin=dict(t=70, b=10), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Demand: Eskom '{CFG.get('demand_column')}'. Wind and PV: utility-scale plants in Eskom's hourly "
               "data. Rooftop and other embedded solar are not included; they lower the demand Eskom sees "
               "rather than appearing as PV output.")


# =============================================================================================
# Sidebar: page toggle
# =============================================================================================
PAGES = ['Facilities map', 'Demand and wind/PV share', 'Performance', 'Forecast']
st.sidebar.title('Wind and Solar Dashboard')
view = st.sidebar.radio('View', PAGES)


def describe_model(path):
    """'LightGBM_SimPower_v1_2026-10-22' -> ('LightGBM SimPower v1', 'Saved 22 Oct 2026')."""
    if not path:
        return 'Not saved', ''
    parts = os.path.basename(os.path.normpath(path)).split('_')
    saved = pd.to_datetime(parts[-1], format='%Y-%m-%d', errors='coerce')
    if pd.notna(saved):
        return ' '.join(parts[:-1]), f'Saved {saved:%d %b %Y}'
    return ' '.join(parts), ''


st.sidebar.markdown('---')
st.sidebar.markdown('**Data**')
st.sidebar.caption('  \n'.join(['Eskom hourly wind and PV generation']
                               + (['Eskom hourly RSA demand'] if CFG.get('demand_column') else [])
                               + ['Wind farm and PV plant lists']))
st.sidebar.markdown('**Forecast models**')
for label, path in (('Wind', CFG.get('model_dir')), ('Solar PV', CFG.get('pv_model_dir'))):
    name, saved = describe_model(path)
    st.sidebar.caption(f'{label}: {name}' + (f'  \n{saved}' if saved else ''))

facilities = load_facilities()

if view == 'Facilities map':
    st.title('Renewable facilities in South Africa')
    options = [t for t in COLOURS if t in set(facilities['source'])]
    if not options:
        st.warning('No facility lists found.')
        st.stop()
    tech = tech_toggle(options, 'map')
    plants = facilities[facilities['source'] == tech].copy()
    mapped = plants.dropna(subset=['latitude', 'longitude']).copy()

    c1, c2, c3 = st.columns(3)
    c1.metric(f'{tech} sites', f'{len(plants)}')
    c2.metric('Installed capacity', f"{plants['capacity_mw'].sum():,.0f} MW")
    c3.metric('Shown on map', f'{len(mapped)}')

    left, right = st.columns([1.7, 1])
    with left:
        mapped['marker_size'] = mapped['capacity_mw'].fillna(20).clip(lower=5)
        hover = {c: True for c in ('capacity_mw', 'province', 'commercial_operation') if c in mapped.columns}
        hover.update({'marker_size': False, 'latitude': False, 'longitude': False})
        map_args = dict(lat='latitude', lon='longitude', size='marker_size', size_max=22,
                        hover_name='name', hover_data=hover, color_discrete_sequence=[COLOURS[tech]],
                        labels={'capacity_mw': 'Capacity (MW)', 'province': 'Province',
                                'commercial_operation': 'Commercial operation'},
                        zoom=4.4, center={'lat': -29.0, 'lon': 24.5}, height=640)
        if hasattr(px, 'scatter_map'):      # Plotly 5.24+ (scatter_mapbox was removed in newer versions)
            fig = px.scatter_map(mapped, map_style='carto-positron', **map_args)
        else:
            fig = px.scatter_mapbox(mapped, mapbox_style='carto-positron', **map_args)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Marker size shows installed capacity. Sites without coordinates are not shown on the map.')
    with right:
        table = (plants[['name', 'capacity_mw']].sort_values('capacity_mw', ascending=False)
                 .reset_index(drop=True))
        st.dataframe(table, use_container_width=True, hide_index=True, height=640,
                     column_config={'name': st.column_config.TextColumn('Name'),
                                    'capacity_mw': st.column_config.NumberColumn('Installed capacity (MW)',
                                                                                 format='%.1f')})
elif view == 'Demand and wind/PV share':
    page_demand()
elif view == 'Performance':
    st.title('Performance')
    if tech_toggle(list(COLOURS), 'performance') == 'Wind':
        page_performance('wind', 'Wind (Eskom raw hourly data)', COLOURS['Wind'], 'Blues')
    else:
        page_performance('pv', 'Solar PV (Eskom raw hourly data)', COLOURS['Solar PV'], 'YlOrBr')
else:
    st.title('Forecast')
    if tech_toggle(list(COLOURS), 'forecast') == 'Wind':
        page_wind_forecast()
    else:
        page_pv_forecast()
