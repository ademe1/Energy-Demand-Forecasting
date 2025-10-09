# ⚡ Energy Demand Forecasting | Data Meets Energy  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![SQL](https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-TensorFlow%20%7C%20Prophet-orange?logo=tensorflow)
![Visualization](https://img.shields.io/badge/Visualization-Tableau-blueviolet?logo=tableau)
![License](https://img.shields.io/badge/License-MIT-green)
![Built%20with%20Love](https://img.shields.io/badge/Built%20with-%E2%9D%A4%EF%B8%8F-red)

<!-- Replace `username` and `repo-name` below with your actual GitHub handle and repository name -->
![GitHub stars](https://img.shields.io/github/stars/username/repo-name?style=social)
![GitHub forks](https://img.shields.io/github/forks/username/repo-name?style=social)
![GitHub issues](https://img.shields.io/github/issues/username/repo-name)
![GitHub last commit](https://img.shields.io/github/last-commit/username/repo-name)
![GitHub repo size](https://img.shields.io/github/repo-size/username/repo-name)
![GitHub language count](https://img.shields.io/github/languages/count/username/repo-name)

This project blends my passion for **energy systems** and **data analytics** to forecast energy demand using real world methods and clean storytelling.

---

## 🌍 Project Overview

This repository demonstrates how **data science** and **forecasting models** can help us understand and anticipate **energy consumption patterns**, enabling better planning, cost savings, and sustainability.

🔹 **Goal:** Forecast monthly energy demand within ±5% accuracy  
🔹 **Data:** 2018–2024 monthly dataset (temperature, price, holiday effects)  
🔹 **Models:** Seasonal Naive, Linear Regression, SARIMAX, Prophet, LSTM  
🔹 **Stack:** SQL + Python + Tableau  

---

## 📊 SQL Integration

Data is stored and queried via **SQLite** (`db/energy.db`) using four linked tables:
- `meter_readings`: energy demand (MWh)
- `weather`: monthly average temperature
- `tariff`: price per kWh
- `calendar`: holiday/peak flags

A prebuilt view, `v_energy_monthly`, combines all sources into one analytics-ready table.

```sql
SELECT *
FROM v_energy_monthly
WHERE year = 2024
ORDER BY month;
```

---

## 🧠 Modeling Approach

We explored **multiple forecasting techniques** to balance interpretability and accuracy:

| Model | Type | Description |
|-------|------|--------------|
| Seasonal Naive | Baseline | Uses value from the same month last year |
| Linear Regression | Classical | Uses lag, rolling mean, and seasonal features |
| SARIMAX | Statistical | ARIMA with seasonality and exogenous variables |
| Prophet | Additive ML | Captures yearly seasonality and holiday effects |
| LSTM | Deep Learning | Learns temporal dependencies in sequences |

---

## 🚀 Key Results

✅ **Target:** ±5% forecasting accuracy  
✅ **Achieved:** 4.6% MAPE with LSTM (Champion Model)

| Model | MAE | MAPE (%) |
|--------|--------|-----------|
| Seasonal Naive | 10.12 | 8.7 |
| Linear Regression | 6.21 | 5.3 |
| SARIMAX | 5.98 | 5.1 |
| Prophet | 5.55 | 4.9 |
| **LSTM (Champion)** | **5.26** | **4.6** |

> 🎯 **Insight:** LSTM captured subtle lag relationships and seasonal peaks, outperforming linear and classical models.

---

## 🔍 Insights from the Data

- 🌡️ **Temperature** drives ~62% of variance — energy demand rises with cold or hot months.  
- 💰 **Price elasticity** (-0.8) shows demand drops when tariffs rise.  
- 🎄 **Seasonality:** December & July spikes due to holidays and peak usage.  
- 📈 **Trend:** Steady increase (~0.4 MWh/month), reflecting economic or population growth.  

---

## 📈 Visual Results

### 1️⃣ Actual vs Predicted (Test Set)
![Test Predictions](figures/model_test_predictions.png)

> The **Prophet** and **LSTM** models closely tracked actual demand, especially during high-variance months.

---

### 2️⃣ 12-Month Forecast (2025)
![Future Forecast](figures/model_future_forecast.png)

> Forecasts show continued seasonal oscillations with upward drift — ideal for **budgeting**, **capacity planning**, and **sustainability reporting**.

---

## 📦 Outputs

| File | Description |
|------|--------------|
| `/outputs/results_sql_prophet_lstm.csv` | Metrics by model |
| `/outputs/tableau_forecasts_long.csv` | Tableau-ready data |
| `/outputs/CHAMPION.txt` | Best model summary |

---

## 📊 Tableau Dashboard

An interactive Tableau dashboard visualizes:
- Model comparisons  
- Forecast vs actuals  
- Scenario toggles (`split = test/future`)  
- Confidence intervals  

> 🔗 *You can recreate this dashboard using `tableau_forecasts_long.csv`.*

---

## 🧰 Tech Stack

| Layer | Tools |
|-------|-------|
| Data | SQL (SQLite), Pandas |
| Modeling | Statsmodels, Prophet, TensorFlow |
| Visualization | Tableau, Matplotlib |
| Pipeline | Python, Scikit-learn |

---

## 💡 Lessons Learned

- Ensemble thinking: combining classical + ML improves stability  
- Feature engineering is key: lags, rolling stats, cyclic encodings  
- SQL + Python + Tableau = powerful, transparent analytics stack  

---

## 🌱 About Me

I’m passionate about **clean energy**, **data-driven decision-making**, and **sustainable innovation**.  
Through projects like this, I aim to use analytics to shape **a smarter, greener energy future**.

💬 Let’s connect!  
- [LinkedIn](https://www.linkedin.com/in/arametd/)  
- 📧 demearame@gmail.com  

> ⚡ *Because every watt counts — and every insight can power change.*
```


