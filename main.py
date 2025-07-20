from incremental_lstm import run_incremental_lstm

if __name__ == "__main__":
    forecast_df = run_incremental_lstm()
    print("\nForecasted Demand:")
    print(forecast_df)