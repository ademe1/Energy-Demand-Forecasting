
# Energy Demand Forecasting Makefile

PY := python
PIP := pip

ENV_PKGS := statsmodels scikit-learn matplotlib pandas numpy prophet tensorflow

# Default target
all: setup run-advanced

setup:
	$(PIP) install $(ENV_PKGS)

run-baseline:
	$(PY) energy_forecasting_project.py

run-advanced:
	$(PY) energy_forecasting_sql_prophet_lstm.py

metrics:
	@echo "Metrics written to outputs/results_sql_prophet_lstm.csv"

tableau:
	@echo "Open outputs/tableau_forecasts_long.csv in Tableau"

clean:
	rm -f outputs/*.csv outputs/*.txt
	rm -f figures/*.png
	@echo "Cleaned outputs and figures"
