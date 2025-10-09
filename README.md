# ⚡ Energy Demand Forecasting | Data Meets Energy  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![SQL](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-TensorFlow%20%7C%20Prophet-orange?logo=tensorflow)
![Visualization](https://img.shields.io/badge/Visualization-Tableau-blueviolet?logo=tableau)
![License](https://img.shields.io/badge/License-MIT-green)

<!-- Replace username and repo-name with your GitHub handle and repository name -->
![GitHub stars](https://img.shields.io/github/stars/username/repo-name?style=social)
![GitHub forks](https://img.shields.io/github/forks/username/repo-name?style=social)
![GitHub issues](https://img.shields.io/github/issues/username/repo-name)
![GitHub last commit](https://img.shields.io/github/last-commit/username/repo-name)
![GitHub repo size](https://img.shields.io/github/repo-size/username/repo-name)
![GitHub language count](https://img.shields.io/github/languages/count/username/repo-name)


This project forecasts monthly energy demand using classical models and modern ML with a clean SQL to Python to Tableau flow

---

## 🌍 Project overview

This repository shows how forecasting models help anticipate consumption patterns and support planning cost control and sustainability

* Goal  Forecast monthly energy demand within ±5 percent accuracy  
* Data  2018 to 2024 monthly features temperature price calendar  
* Models  Seasonal Naive  Linear Regression  SARIMAX  Prophet  LSTM  
* Stack  SQL  Python  Tableau

---

## 📊 SQL integration

Data lives in SQLite at `db/energy.db` with four tables joined into the view `v_energy_monthly`

```sql
SELECT *
FROM v_energy_monthly
WHERE year = 2024
ORDER BY month;
```

---

## 🧠 Modeling approach

We compare multiple techniques to balance interpretability and accuracy

| Model | Type | Description |
|---|---|---|
| Seasonal Naive | Baseline | Same month last year reference |
| Linear Regression | Classical | Lags rolling means and seasonal encodings |
| SARIMAX | Statistical | ARIMA seasonal terms plus exogenous variables |
| Prophet | Additive ML | Yearly seasonality and holiday effects |
| LSTM | Deep learning | Sequence model on lag features |

---

## 🚀 Key results

Target  ±5 percent accuracy  
Champion  LSTM on lag features

| Model | MAE | MAPE percent |
|---|---:|---:|
| Seasonal Naive | 10.12 | 8.7 |
| Linear Regression | 6.21 | 5.3 |
| SARIMAX | 5.98 | 5.1 |
| Prophet | 5.55 | 4.9 |
| **LSTM champion** | **5.26** | **4.6** |

Insight  LSTM captures subtle lag relationships and seasonal peaks better than the linear baseline

---

## 📈 Visual results

### 1 Actual versus predicted on test set
![Test Predictions](figures/model_test_predictions.png)

### 2 Twelve month future forecast
![Future Forecast](figures/model_future_forecast.png)

---

## 📦 Outputs and direct links

Metrics by model  
* [`outputs/results_sql_prophet_lstm.csv`](outputs/results_sql_prophet_lstm.csv)

Tableau ready long format  
* [`outputs/tableau_forecasts_long.csv`](outputs/tableau_forecasts_long.csv)

Champion summary  
* [`outputs/CHAMPION.txt`](outputs/CHAMPION.txt)

LSTM specific files  
* Test predictions  [`outputs/lstm_test_predictions.csv`](outputs/lstm_test_predictions.csv)  
* Future predictions  [`outputs/lstm_future_predictions.csv`](outputs/lstm_future_predictions.csv)

---

## ▶️ Run locally or in Colab

```bash
# Python environment
pip install statsmodels scikit-learn matplotlib pandas numpy prophet tensorflow

# Baseline script
python energy_forecasting_project.py

# SQL  Prophet  LSTM script
python energy_forecasting_sql_prophet_lstm.py
```

Outputs will appear in the `outputs` folder and figures in `figures`

---

## 🔁 Switch to a real Keras LSTM in Colab

This repository ships with an MLP fallback for environments without TensorFlow  
In Google Colab you can enable a true LSTM with the steps below

```python
# In a fresh Colab notebook
!pip install tensorflow prophet statsmodels scikit-learn pandas numpy matplotlib

# Upload or clone the repo then run
!python energy_forecasting_sql_prophet_lstm.py
```

If you want a minimal inline LSTM snippet on the lag features

```python
import pandas as pd, numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/energy_monthly.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df["month"] = df["date"].dt.month

def add_feats(x, target="energy_mwh"):
    f = x.copy()
    for L in range(1,13):
        f[f"lag_{L}"] = f[target].shift(L)
    f["month_sin"] = np.sin(2*np.pi*f["month"]/12)
    f["month_cos"] = np.cos(2*np.pi*f["month"]/12)
    return f.dropna()

f = add_feats(df)
y = f["energy_mwh"].values.astype("float32")
X = f[[c for c in f.columns if c.startswith("lag_") or c in ["month_sin","month_cos"]]].values.astype("float32")

# last 12 months as test
Xtr, Xte = X[:-12], X[-12:]
ytr, yte = y[:-12], y[-12:]

sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

# reshape to [samples, timesteps, features]
Xtr = Xtr.reshape((Xtr.shape[0], 1, Xtr.shape[1]))
Xte = Xte.reshape((Xte.shape[0], 1, Xte.shape[1]))

model = Sequential([LSTM(32, input_shape=(1, Xtr.shape[2])), Dense(1)])
model.compile(optimizer="adam", loss="mae")
model.fit(Xtr, ytr, epochs=150, batch_size=8, verbose=0)

pred = model.predict(Xte, verbose=0).flatten()

pd.DataFrame({"date": f["date"].iloc[-12:], "actual_mwh": yte, "forecast_mwh": pred}).to_csv("outputs/lstm_test_predictions.csv", index=False)
```

You can then regenerate the long format for Tableau by appending these predictions to `outputs/tableau_forecasts_long.csv`

---

## 📊 Tableau dashboard

Use `outputs/tableau_forecasts_long.csv` as the data source

Suggested sheet  
* Columns  date  
* Rows  forecast_mwh and actual_mwh as dual axis  
* Color  model  
* Filter  split to toggle test versus future  
* Tooltip  join in metrics from `results_sql_prophet_lstm.csv`

---

## 🧰 Tech stack

| Layer | Tools |
|---|---|
| Data | SQL SQLite  Pandas |
| Modeling | Statsmodels  Prophet  TensorFlow |
| Visualization | Tableau  Matplotlib |
| Pipeline | Python  Scikit learn |

---

## 🌱 About me

I am passionate about clean energy  data driven decisions  and sustainable innovation  
I aim to use analytics to help build a smarter greener energy future

Connect with me  
* LinkedIn  https://www.linkedin.com/in/arametd/  
* Email  demearame@gmail.com
```



