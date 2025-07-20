import csv
import os

CSV_FILE = "inventory_log.csv"

def init_log():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "total",
                "current",
                "demand",
                "demand_pct",
                "inventory_pct",
                "risk_ratio",
                "avg_demand_pct",
                "avg_inventory_pct",
                "avg_risk_ratio",
                "status"
            ])

def log_result(timestamp, total, current, demand, result):
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            total,
            current,
            demand,
            result["demand_pct"],
            result["inventory_pct"],
            result["risk_ratio"],
            result["avg_demand_pct"],
            result["avg_inventory_pct"],
            result["avg_risk_ratio"],
            result["status"]
        ])