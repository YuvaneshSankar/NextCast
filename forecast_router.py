from fastapi import APIRouter, HTTPException
from subprocess import run, PIPE
import os
import csv
from incremental_lstm import run_incremental_lstm

router = APIRouter()

LOG_FILE = os.path.join("models", "forecasting", "inventory_log.csv")
MODEL_DIR = os.path.join("models", "forecasting")

@router.get("/forecast")
def forecast():
    if not os.path.exists(LOG_FILE):
        raise HTTPException(status_code=404, detail="Log file not found.")

    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            log_count = sum(1 for _ in reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {str(e)}")

    if log_count >= 1000:
        try:
            forecast_df = run_incremental_lstm()
            return {
                "model_used": "lstm_forecast.py",
                "log_count": log_count,
                "forecast": forecast_df.to_dict(orient="records")
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LSTM error: {str(e)}")

    elif log_count >= 100:
        script = "arima.py"
    else:
        script = "InventoryDemandClassifier.py"

    script_path = os.path.join(MODEL_DIR, script)

    try:
        result = run(["python", script_path], stdout=PIPE, stderr=PIPE, text=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subprocess failed: {str(e)}")

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Script error: {result.stderr}")

    return {
        "model_used": script,
        "log_count": log_count,
        "output": result.stdout.strip()
    }