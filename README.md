# Energy Demand Forecasting (Monthly)

A complete, runnable baseline for forecasting monthly energy demand. Includes:
- Seasonal Naive baseline
- Linear Regression with time-series features
- SARIMAX with exogenous variables (if statsmodels available)
- Optional instructions for Prophet and LSTM in Colab

## Run Locally or in Colab
1) Upload `energy_forecasting_project.py` and `data/energy_monthly.csv` (this repo).
2) Install deps:
```python
!pip install statsmodels scikit-learn matplotlib pandas numpy
```
3) Execute:
```python
!python energy_forecasting_project.py
```

Outputs:
- `/mnt/data/outputs/results.csv` – MAE & MAPE by model
- `/mnt/data/outputs/forecasts.csv` – next 12 months forecast
- `/mnt/data/figures/` – PNG charts

## Optional: Prophet (Colab)
```python
!pip install prophet
from prophet import Prophet
```

## Optional: LSTM (Colab)
```python
!pip install tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
```


---
## SQL Ingestion
A SQLite database is included at `db/energy.db` with these tables:
- `meter_readings(date, energy_mwh)`
- `weather(date, temperature_c)`
- `tariff(date, price_cents_per_kwh)`
- `calendar(date, is_holiday_peak)`
- View: `v_energy_monthly` joining the above by date.

Example query:
```sql
SELECT * FROM v_energy_monthly WHERE year=2024 ORDER BY month;
```

Run the SQL-driven pipeline (includes Prophet & LSTM guards):
```python
!pip install statsmodels scikit-learn matplotlib pandas numpy prophet tensorflow
!python energy_forecasting_sql_prophet_lstm.py
```

## Tableau Migration
Use `/mnt/data/outputs/tableau_forecasts_long.csv` with fields:
- `date`
- `actual_mwh`
- `forecast_mwh`
- `model` (SeasonalNaive, LinearRegression, SARIMAX, Prophet, LSTM)
- `split` (test or future)

Suggested visuals in Tableau:
1. Lines for `forecast_mwh` by `date`, color by `model`.
2. Dual-axis with `actual_mwh` for comparison.
3. Filter: `split` to switch between test and future.
4. Tooltip: show MAE/MAPE from `/mnt/data/outputs/results_sql_prophet_lstm.csv`.
