WIND PRODUCTION FORECAST MODEL
==============================
Created: 2026-10-22T20:51:43
Forecasts: total South African wind production (Eskom "Wind", MW), hourly.
Input:     hourly wind speed at 100 m (m/s) per site - nothing else.
Model:     LightGBM on per-site simulated power (cubic power curve, cut-in 2.00 m/s,
           rated 8.26 m/s) x capacity online. Fitted on 2022-04-01
           to 2026-09-29 (39,432 hours), 39 sites.

HOW TO USE
  import sys
  sys.path.append('/content/drive/MyDrive/Power System Data Analytics 874/Project/Wind_Production_Forecast/LightGBM_SimPower_v1_2026-10-22')
  from forecast_wind import forecast
  result = forecast('my_wind_speeds.csv')

  Command line: python forecast_wind.py my_wind_speeds.csv forecast_output.csv

INPUT FORMAT
  See input_template.csv: a 'timestamp' column plus one column per site (names as in farms.csv).
  Use the same time convention as the training data (hour beginning, same time zone as the ERA5 file).
  Wind speeds for a real forecast come from a weather forecast model (e.g. ECMWF / GFS) at the
  coordinates in farms.csv.
  example_input_wind_speed.csv is one real week you can run to test.

FILES
  forecast_wind.py               the forecast program
  model.joblib                   trained model
  config.json                    power curve, site order, training period, test results, versions
  farms.csv                      sites, capacities, commissioning dates, coordinates
  input_template.csv             empty input layout
  example_input_wind_speed.csv   test input

LIMITS
  - Only the 39 sites in farms.csv are modelled. New wind farms need a retrain (new version folder).
  - The model learned from ERA5 reanalysis; forecast wind speeds add their own error.
  - Library versions used: lightgbm 4.6.0, scikit-learn 1.6.1, pandas 2.2.3.
