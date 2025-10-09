# ⚡ Energy Demand Forecasting | Data Meets Energy  

👋🏾 Hi, I’m **Arame**, an energy engineer turned data explorer who believes every kilowatt tells a story.  
This project blends my passion for **energy systems** and **data analytics** to forecast energy demand using real-world methods and clean storytelling.

---

## 🌍 Project Overview

This repository demonstrates how **data science** and **forecasting models** can help us understand and anticipate **energy consumption patterns**,enabling better planning, cost savings, and sustainability.

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
