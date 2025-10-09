# Energy Forecasting via SQL + Prophet + LSTM
import os, sqlite3, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

have_statsmodels = True
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    have_statsmodels = False

have_prophet = True
try:
    from prophet import Prophet
except Exception:
    have_prophet = False

have_tf = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
except Exception:
    have_tf = False

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100.0

# -----------------
# Load from SQLite
# -----------------
DB_PATH = "/mnt/data/db/energy.db"
with sqlite3.connect(DB_PATH) as con:
    df = pd.read_sql_query("SELECT * FROM v_energy_monthly", con, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# -----------------
# Feature Engineering
# -----------------
def add_time_features(frame, target_col="energy_mwh", lags=(1,12), rolls=(3,6)):
    f = frame.copy()
    for L in lags:
        f[f"lag_{L}"] = f[target_col].shift(L)
    for W in rolls:
        f[f"rollmean_{W}"] = f[target_col].shift(1).rolling(W).mean()
    f["month_sin"] = np.sin(2*np.pi*f["month"]/12.0)
    f["month_cos"] = np.cos(2*np.pi*f["month"]/12.0)
    Xcols = ["temperature_c","price_cents_per_kwh","is_holiday_peak","month_sin","month_cos"] + \
            [c for c in f.columns if c.startswith("lag_") or c.startswith("rollmean_")]
    return f, Xcols

df_fe, Xcols = add_time_features(df)
df_fe = df_fe.dropna().reset_index(drop=True)

# Split
test_horizon = 12
train_df = df_fe.iloc[:-test_horizon].copy()
test_df  = df_fe.iloc[-test_horizon:].copy()
y_train = train_df["energy_mwh"].values
y_test  = test_df["energy_mwh"].values
X_train = train_df[Xcols].values
X_test  = test_df[Xcols].values

# Baseline
seasonal_naive_pred = test_df["lag_12"].values
res_rows = [["SeasonalNaive",
             mean_absolute_error(y_test, seasonal_naive_pred),
             mape(y_test, seasonal_naive_pred)]]

# Linear Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
linreg = LinearRegression().fit(X_train_sc, y_train)
linreg_pred = linreg.predict(X_test_sc)
res_rows.append(["LinearRegression", mean_absolute_error(y_test, linreg_pred), mape(y_test, linreg_pred)])

# SARIMAX
sarimax_pred = None
if have_statsmodels:
    try:
        exog_cols = ["temperature_c","price_cents_per_kwh","is_holiday_peak","month_sin","month_cos"]
        m = SARIMAX(train_df["energy_mwh"],
                    exog=train_df[exog_cols],
                    order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
        res = m.fit(disp=False)
        sarimax_pred = res.predict(start=test_df.index[0], end=test_df.index[-1], exog=test_df[exog_cols])
        res_rows.append(["SARIMAX(1,1,1)(1,1,1,12)+exog",
                         mean_absolute_error(y_test, sarimax_pred),
                         mape(y_test, sarimax_pred)])
    except Exception:
        sarimax_pred = None

# Prophet
prophet_pred = None
if have_prophet:
    try:
        dfp = df[["date","energy_mwh"]].rename(columns={"date":"ds","energy_mwh":"y"})
        m = Prophet(yearly_seasonality=True)
        m.fit(dfp.iloc[:-test_horizon])
        future = m.make_future_dataframe(periods=test_horizon, freq="MS")
        forecast = m.predict(future)
        prophet_tail = forecast.iloc[-test_horizon:]
        prophet_pred = prophet_tail["yhat"].values
        res_rows.append(["Prophet", mean_absolute_error(y_test, prophet_pred), mape(y_test, prophet_pred)])
    except Exception:
        prophet_pred = None

# LSTM 
lstm_pred = None
if have_tf:
    try:
        
        sup_cols = [c for c in df_fe.columns if c.startswith("lag_")]
        sup = df_fe.dropna().copy()
        X_all = sup[sup_cols].values.astype(np.float32)
        y_all = sup["energy_mwh"].values.astype(np.float32)
       
        X_tr, X_te = X_all[:-test_horizon], X_all[-test_horizon:]
        y_tr, y_te = y_all[:-test_horizon], y_all[-test_horizon:]
       
        X_tr_r = X_tr.reshape((X_tr.shape[0], 1, X_tr.shape[1]))
        X_te_r = X_te.reshape((X_te.shape[0], 1, X_te.shape[1]))
        model = Sequential([
            LSTM(32, input_shape=(1, X_tr.shape[1])),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mae")
        model.fit(X_tr_r, y_tr, epochs=150, batch_size=8, verbose=0)
        lstm_pred = model.predict(X_te_r, verbose=0).flatten()
        res_rows.append(["LSTM(lag features)", mean_absolute_error(y_te, lstm_pred), mape(y_te, lstm_pred)])
    except Exception:
        lstm_pred = None

results = pd.DataFrame(res_rows, columns=["Model","MAE","MAPE_percent"])
os.makedirs("/mnt/data/outputs", exist_ok=True)
results.to_csv("/mnt/data/outputs/results_sql_prophet_lstm.csv", index=False)

# -----------------
# Tableau
# -----------------
def to_long(model_name, dates, actual, pred, split):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "actual_mwh": actual,
        "forecast_mwh": pred,
        "model": model_name,
        "split": split 
    })

long_frames = []

long_frames.append(to_long("SeasonalNaive", test_df["date"], y_test, seasonal_naive_pred, "test"))
long_frames.append(to_long("LinearRegression", test_df["date"], y_test, linreg_pred, "test"))
if sarimax_pred is not None:
    long_frames.append(to_long("SARIMAX", test_df["date"], y_test, sarimax_pred, "test"))
if prophet_pred is not None:
    long_frames.append(to_long("Prophet", test_df["date"], y_test, prophet_pred, "test"))
if lstm_pred is not None:
    long_frames.append(to_long("LSTM", test_df["date"], y_test, lstm_pred, "test"))
tableau_test = pd.concat(long_frames, ignore_index=True)

best_row = results.sort_values("MAPE_percent").iloc[0]
champion = best_row["Model"]

future_steps = 12
hist = df_fe.copy()

def recompute_features(h):
    f, cols = add_time_features(h)
    f = f.dropna().reset_index(drop=True)
    return f, cols

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scaler2 = StandardScaler()
h2, cols2 = recompute_features(hist)
X = h2[cols2].values
y = h2["energy_mwh"].values
Xsc = scaler2.fit_transform(X)
lr = LinearRegression().fit(Xsc, y)

future_rows = []
last_date = hist["date"].iloc[-1]
for i in range(1, future_steps+1):
    next_date = last_date + pd.DateOffset(months=i)
    m = next_date.month
    month_sin = np.sin(2*np.pi*m/12.0)
    month_cos = np.cos(2*np.pi*m/12.0)
    is_holiday_peak = 1 if (m==12 or m==7) else 0
    temperature_c = hist["temperature_c"].iloc[-12+i-1] if len(hist)>=12 else hist["temperature_c"].iloc[-1]
    price_cents_per_kwh = hist["price_cents_per_kwh"].iloc[-1]
    future_rows.append({"date": next_date, "year": next_date.year, "month": m,
                        "temperature_c": temperature_c, "price_cents_per_kwh": price_cents_per_kwh,
                        "is_holiday_peak": is_holiday_peak, "energy_mwh": np.nan,
                        "month_sin": month_sin, "month_cos": month_cos})
future_df = pd.DataFrame(future_rows)

future_preds = []
for i in range(len(future_df)):
    hist = pd.concat([hist, future_df.iloc[i:i+1]], ignore_index=True)
    h2, cols2 = recompute_features(hist)
    X_last = h2[cols2].iloc[[-1]].values
    pred = lr.predict(scaler2.transform(X_last))[0]
    future_preds.append(pred)
    hist.loc[hist.index[-1], "energy_mwh"] = pred

tableau_future = pd.DataFrame({
    "date": future_df["date"],
    "actual_mwh": np.nan,
    "forecast_mwh": future_preds,
    "model": "LinearRegression",
    "split": "future"
})

tableau_long = pd.concat([tableau_test, tableau_future], ignore_index=True)
tableau_long.to_csv("/mnt/data/outputs/tableau_forecasts_long.csv", index=False)

with open("/mnt/data/outputs/CHAMPION.txt", "w") as f:
    f.write(f"Champion by test MAPE: {champion}\n")
    f.write(results.sort_values('MAPE_percent').to_string(index=False))
print("Done. Metrics & Tableau CSV ready.")
