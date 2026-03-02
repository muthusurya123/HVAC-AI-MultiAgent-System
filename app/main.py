import pandas as pd
from agents.analyzer import AnalyzerAgent

# Load dataset
df = pd.read_csv("data/large_mall_dataset_3years_5min.csv", parse_dates=["Timestamp"])

# Initialize Analyzer Agent
analyzer = AnalyzerAgent(df)

# Generate Summary
summary = analyzer.generate_summary()

print("\n===== ANALYZER REPORT =====")
for key, value in summary.items():
    print(f"{key}: {value}")

from agents.forecasting import ForecasterAgent

print("\n===== FORECASTING =====")

forecaster = ForecasterAgent("data/large_mall_dataset_3years_5min.csv")

forecaster.load_and_prepare_data()

mae = forecaster.train_model()

print("Model MAE:", mae)

forecast = forecaster.forecast_next_24_hours()

print("Next 24 Hour Forecast:")
print(forecast)