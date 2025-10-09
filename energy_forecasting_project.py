# Energy Demand Forecasting Project
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Optional statsmodels (SARIMAX)
have_statsmodels = True
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    have_statsmodels = False

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0

DATA_PATH = "/mnt/data/data/energy_monthly.csv"
df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

os.makedirs("/mnt/data/figures", exist_ok=True)
os.makedirs("/mnt/data/outputs", exist_ok=True)

# EDA
plt.figure(figsize=(10,4))
plt.plot(df["date"], df["energy_mwh"])
plt.title("Monthly Energy Demand (MWh)")
plt.xlabel("Date")
plt.ylabel("MWh")
plt.tight_layout()
plt.savefig("/mnt/data/figures/eda_demand.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(df["date"], df["temperature_c"])
plt.title("Monthly Average Temperature (°C)")
plt.xlabel("Date")
plt.ylabel("°C")
plt.tight_layout()
plt.savefig("/mnt/data/figures/eda_temp.png")
plt.close()

def add_time_features(frame, target_col="energy_mwh", lags=(1,12), rolls=(3,6)):
    f = frame.copy()
    for L in lags:
        f[f"lag_{L}"] = f[target_col].shift(L)
    for W in rolls:
        f[f"rollmean_{W}"] = f[target_col].shift(1).rolling(W).mean()
    f["month_sin"] = np.sin(2*np.pi*f["month"]/12.0)
    f["month_cos"] = np.cos(2*np.pi*f["month"]/12.0)
    Xcols = [
        "temperature_c", "price_cents_per_kwh", "is_holiday_peak",
        "month_sin", "month_cos"
    ] + [c for c in f.columns if c.startswith("lag_") or c.startswith("rollmean_")]
    return f, Xcols

df_fe, Xcols = add_time_features(df)
df_fe = df_fe.dropna().reset_index(drop=True)

test_horizon = 12
train_df = df_fe.iloc[:-test_horizon].copy()
test_df  = df_fe.iloc[-test_horizon:].copy()

y_train = train_df["energy_mwh"].values
y_test  = test_df["energy_mwh"].values
X_train = train_df[Xcols].values
X_test  = test_df[Xcols].values

# Baseline: Seasonal Naive
seasonal_naive_pred = test_df["lag_12"].values
baseline_mae = mean_absolute_error(y_test, seasonal_naive_pred)
baseline_mape = mape(y_test, seasonal_naive_pred)

# Linear Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
linreg = LinearRegression()
linreg.fit(X_train_sc, y_train)
linreg_pred = linreg.predict(X_test_sc)
linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_mape = mape(y_test, linreg_pred)

# SARIMAX (optional)
sarimax_mae = None
sarimax_mape = None
sarimax_pred = None
if have_statsmodels:
    try:
        exog_cols = ["temperature_c", "price_cents_per_kwh", "is_holiday_peak", "month_sin", "month_cos"]
        exog_train = train_df[exog_cols]
        exog_test  = test_df[exog_cols]
        model = SARIMAX(
            train_df["energy_mwh"],
            exog=exog_train,
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)
        sarimax_pred = res.predict(start=test_df.index[0], end=test_df.index[-1], exog=exog_test)
        sarimax_mae = mean_absolute_error(y_test, sarimax_pred)
        sarimax_mape = mape(y_test, sarimax_pred)
    except Exception:
        sarimax_pred = None

# Save evaluation
rows = [
    ["SeasonalNaive", baseline_mae, baseline_mape],
    ["LinearRegression", linreg_mae, linreg_mape],
]
if sarimax_pred is not None:
    rows.append(["SARIMAX(1,1,1)(1,1,1,12)+exog", sarimax_mae, sarimax_mape])
results = pd.DataFrame(rows, columns=["Model","MAE","MAPE_percent"])
results.to_csv("/mnt/data/outputs/results.csv", index=False)

# Plot predictions on test
plt.figure(figsize=(10,4))
plt.plot(test_df["date"], y_test, label="Actual")
plt.plot(test_df["date"], seasonal_naive_pred, label="SeasonalNaive")
plt.plot(test_df["date"], linreg_pred, label="LinearReg")
if sarimax_pred is not None:
    plt.plot(test_df["date"], sarimax_pred, label="SARIMAX")
plt.title("Test Set: Actual vs Predictions")
plt.xlabel("Date")
plt.ylabel("MWh")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/figures/model_test_predictions.png")
plt.close()

# Rolling 12-month forecast using champion (SARIMAX if available, else Linear Regression)
champion = "SARIMAX" if sarimax_pred is not None else "LinearRegression"
hist = df_fe.copy()

def recompute_features(h):
    h2, cols = add_time_features(h)
    h2 = h2.dropna().reset_index(drop=True)
    return h2, cols

future_steps = 12
last_date = hist["date"].iloc[-1]
future_rows = []
for i in range(1, future_steps+1):
    next_date = last_date + pd.DateOffset(months=i)
    m = next_date.month
    month_sin = np.sin(2*np.pi*m/12.0)
    month_cos = np.cos(2*np.pi*m/12.0)
    is_holiday_peak = 1 if (m==12 or m==7) else 0
    temperature_c = hist["temperature_c"].iloc[-12+i-1] if len(hist)>=12 else hist["temperature_c"].iloc[-1]
    price_cents_per_kwh = hist["price_cents_per_kwh"].iloc[-1]
    future_rows.append({
        "date": next_date,
        "month": m,
        "temperature_c": temperature_c,
        "price_cents_per_kwh": price_cents_per_kwh,
        "is_holiday_peak": is_holiday_peak,
        "energy_mwh": np.nan,
        "month_sin": month_sin,
        "month_cos": month_cos
    })
future_df = pd.DataFrame(future_rows)

forecasts = []
if champion == "SARIMAX" and have_statsmodels:
    try:
        exog_cols = ["temperature_c", "price_cents_per_kwh", "is_holiday_peak", "month_sin", "month_cos"]
        model_full = SARIMAX(
            hist["energy_mwh"],
            exog=hist[exog_cols],
            order=(1,1,1),
            seasonal_order=(1,1,1,12),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res_full = model_full.fit(disp=False)
        for i in range(len(future_df)):
            exog_row = future_df[exog_cols].iloc[i:i+1]
            pred = res_full.predict(start=res_full.nobs, end=res_full.nobs, exog=exog_row)[0]
            forecasts.append(pred)
            new_row = future_df.iloc[i].copy()
            new_row["energy_mwh"] = pred
            hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
    except Exception:
        champion = "LinearRegression"

if champion == "LinearRegression":
    scaler = StandardScaler()
    # fit once on full hist features
    hist2, cols = recompute_features(hist)
    X = hist2[cols].values
    y = hist2["energy_mwh"].values
    Xsc = scaler.fit_transform(X)
    lr = LinearRegression().fit(Xsc, y)
    # recursive
    for i in range(len(future_df)):
        hist = pd.concat([hist, future_df.iloc[i:i+1]], ignore_index=True)
        hist2, cols = recompute_features(hist)
        X_last = hist2[cols].iloc[[-1]].values
        pred = lr.predict(scaler.transform(X_last))[0]
        forecasts.append(pred)
        hist.loc[hist.index[-1], "energy_mwh"] = pred

future_df["forecast_mwh"] = forecasts
forecast_out = future_df[["date","forecast_mwh"]].copy()
forecast_out.to_csv("/mnt/data/outputs/forecasts.csv", index=False)

plt.figure(figsize=(10,4))
plt.plot(df["date"], df["energy_mwh"], label="History")
plt.plot(future_df["date"], future_df["forecast_mwh"], label="Forecast")
plt.title(f"12-Month Forecast ({champion})")
plt.xlabel("Date")
plt.ylabel("MWh")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/figures/model_future_forecast.png")
plt.close()

print("Done. See /mnt/data/outputs and /mnt/data/figures for results.")
