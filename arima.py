import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from InventoryDemandClassifier import InventoryDemandClassifier
import warnings
import csv

warnings.filterwarnings("ignore")

# Load existing inventory log
df = pd.read_csv("inventory_log.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

df = df.sort_values(by='timestamp')

# ARIMA forecast on demand
demand_series = df['demand']
model = ARIMA(demand_series, order=(2, 1, 2))
model_fit = model.fit()

forecast_steps = 5
forecast = model_fit.forecast(steps=forecast_steps)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(demand_series, label="Historical Demand")
plt.plot(range(len(demand_series), len(demand_series) + forecast_steps), forecast, label="Forecast", linestyle="--", marker="o")
plt.legend()
plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.title("ARIMA Demand Forecast")
plt.grid()
plt.tight_layout()
plt.show()

# Reuse last known total and current
latest_total = df['total'].iloc[-1]
latest_current = df['current'].iloc[-1]
last_timestamp = df['timestamp'].iloc[-1]

# Initialize classifier
classifier = InventoryDemandClassifier()

# Open file for appending forecasted + classified data
with open("inventory_log.csv", "a", newline='') as file:
    writer = csv.writer(file)

    for i, val in enumerate(forecast):
        timestamp = last_timestamp + timedelta(days=i+1)
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        demand = round(val, 2)

        result = classifier.classify(
            timestamp=timestamp_str,
            total=latest_total,
            current=latest_current,
            demand=demand
        )

        # Append full row
        writer.writerow([
            result["timestamp"],
            latest_total,
            latest_current,
            demand,
            result["demand_pct"],
            result["inventory_pct"],
            result["risk_ratio"],
            result["avg_demand_pct"],
            result["avg_inventory_pct"],
            result["avg_risk_ratio"],
            result["status"]
        ])

        print(f" Logged → {timestamp_str} | Demand: {demand} | Status: {result['status']}")