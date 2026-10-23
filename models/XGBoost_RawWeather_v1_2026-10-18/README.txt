PV PRODUCTION FORECAST MODEL
============================
Created:   2026-10-18T21:33:46
Forecasts: total South African PV production (Eskom "PV", MW), hourly.
Model:     XGBoost on "Raw weather" (246 inputs).
           Physics: DC/AC 0.96, tracker weight 0.34, -0.004/degC.
           Fitted on 2022-04-03 to 2026-09-29
           (39,383 hours), 49 plants with weather, 49 in the capacity total.

HOW TO USE
  import sys
  sys.path.append('/content/drive/MyDrive/Power System Data Analytics 874/Project/PV_Production_Forecast/XGBoost_RawWeather_v1_2026-10-18')
  from forecast_pv import forecast
  result = forecast('weather.csv')

  Command line: python forecast_pv.py weather.csv --out forecast.csv

INPUT FORMAT
  input_template.csv: 'timestamp' plus <plant>_ghi, _dni, _dhi (W/m2), _temp (degC), _cloud (%) per plant.
  Times: SAST (UTC+2), hour beginning (row t covers t to t+1).
  Radiation from Open-Meteo is the mean over the PRECEDING hour: pass radiation_convention='open_meteo'
  (or --open-meteo) and the program shifts it 1 h earlier, as in training.
  example_input_weather.csv (+ example_past_output.csv) is one real week to test with.

FILES
  forecast_pv.py                 the forecast program
  model.joblib                   trained model
  config.json                    physics settings, inputs, lag, timing, test results, versions
  plants.csv                     plants, capacities, commissioning dates, coordinates
  input_template.csv             empty input layout
  example_input_weather.csv      test input (weather)
  example_past_output.csv        test input (Eskom actual PV output, for the lag)

LIMITS
  - Only plants in plants.csv are modelled. New PV plants need a retrain (new version folder).
  - The model learned from ERA5 reanalysis; forecast weather adds its own error.
  - Library versions used: sklearn 1.6.1, pandas 2.2.3, numpy 2.1.3, lightgbm 4.6.0, xgboost 3.4.1.
