import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime

class LSTMDemandPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMDemandPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

LOG_PATH = "models/forecasting/inventory_log.csv"
MODEL_PATH = "models/forecasting/lstm_model.pt"
CHECKPOINT_PATH = "models/forecasting/last_trained_timestamp.txt"
FORECAST_PATH = "models/forecasting/lstm_forecast.csv"

def run_incremental_lstm(epochs=50, window_size=20, forecast_steps=10):
    df = pd.read_csv(LOG_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            last_trained_ts = pd.to_datetime(f.read().strip())
        new_data = df[df['timestamp'] > last_trained_ts]
        print(f"🕒 Found checkpoint. Retraining with {len(new_data)} new rows.")
    else:
        new_data = df
        print("No checkpoint found. Training from scratch.")

    if len(new_data) < window_size + 1:
        print("Not enough new data to train. Skipping training.")
    else:
        demand = df['demand'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        demand_scaled = scaler.fit_transform(demand)
        X, y = create_sequences(demand_scaled, window_size)
        X = X.view(-1, window_size, 1)

        model = LSTMDemandPredictor()
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH))
            print("Model loaded from disk.")
        else:
            print("No saved model found. Training fresh.")

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            output = model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        with open(CHECKPOINT_PATH, "w") as f:
            f.write(str(df['timestamp'].iloc[-1]))
        print("💾 Model + checkpoint saved.")

    model.eval()
    future_preds = []
    last_seq = scaler.transform(df['demand'].values[-window_size:].reshape(-1, 1)).tolist()

    for _ in range(forecast_steps):
        seq_tensor = torch.tensor([last_seq], dtype=torch.float32)
        with torch.no_grad():
            pred = model(seq_tensor).item()
        future_preds.append(pred)
        last_seq.append([pred])
        last_seq = last_seq