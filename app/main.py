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

from agents.recommender import RecommenderAgent

print("\n===== RECOMMENDATIONS =====")

recommender = RecommenderAgent(summary, forecast)
recommendations = recommender.generate_recommendations()

for r in recommendations:
    print("-", r)
from agents.reporter import ReporterAgent

print("\nGenerating PDF Report...")

reporter = ReporterAgent(summary, forecast, recommendations)
file_path = reporter.generate_pdf_report()
print(f"PDF Report Generated Successfully! Saved at: {file_path}")
from agents.weather import WeatherAgent

weather = WeatherAgent("YOUR_API_KEY")

temps, humidity = weather.get_weather_forecast()

print("Weather Forecast Temps:", temps[:5])