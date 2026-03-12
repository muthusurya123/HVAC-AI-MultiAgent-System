import pandas as pd

from agents.analyzer import AnalyzerAgent
from agents.forecasting import ForecasterAgent
from agents.recommender import RecommenderAgent
from agents.reporter import ReporterAgent
from agents.weather import WeatherAgent
from agents.anomaly_detector import AnomalyAgent
from agents.impact import ImpactAgent


# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(
    "data/large_mall_dataset_3years_5min.csv",
    parse_dates=["Timestamp"]
)

# -------------------------
# Analyzer
# -------------------------

analyzer = AnalyzerAgent(df)
summary = analyzer.generate_summary()


# -------------------------
# Forecasting
# -------------------------

forecaster = ForecasterAgent(
    "data/large_mall_dataset_3years_5min.csv"
)

forecaster.load_and_prepare_data()
mae = forecaster.train_model()
forecast = forecaster.forecast_next_24_hours()


# -------------------------
# Recommendations
# -------------------------

recommender = RecommenderAgent(summary, forecast)
recommendations = recommender.generate_recommendations()


# -------------------------
# Weather
# -------------------------

weather = WeatherAgent()
temps, humidity = weather.get_weather_forecast()


# -------------------------
# Anomaly Detection
# -------------------------

anomaly_agent = AnomalyAgent(df)
anomaly_agent.train()

latest_row = df.iloc[-1]

status, explanation = anomaly_agent.detect_anomaly(latest_row)


# -------------------------
# Generate PDF Report
# -------------------------

reporter = ReporterAgent(summary, forecast, recommendations)
file_path = reporter.generate_pdf_report()


# =============================
# CLEAN DEMO OUTPUT
# =============================

print("\n")
print("="*60)
print("        AI HVAC OPTIMIZATION PLATFORM")
print("="*60)

# -------------------------
# System Analysis
# -------------------------

print("\n📊 SYSTEM ANALYSIS")
print("-"*40)

print(f"Average Efficiency (iKW/TR): {summary['Average iKW/TR']}")
print(f"Efficiency Gap: {summary['Efficiency Gap (%)']} %")
print(f"Peak Demand: {summary['Peak Demand (kW)']} kW")
print(f"Load Factor: {summary['Load Factor (%)']} %")


# -------------------------
# System Health
# -------------------------

print("\n⚠ SYSTEM HEALTH")

print(f"Degradation Status: {summary['Degradation Status']}")
print(f"Fault Severity: {summary['Fault Severity']}")
print(f"Maintenance Priority: {summary['Maintenance Priority']}")


# -------------------------
# Load Forecast
# -------------------------

print("\n📈 LOAD FORECAST (NEXT 24 HOURS)")
print("-"*40)

for i, value in enumerate(forecast):
    print(f"Hour {i+1}: {round(value,2)} kW")


# -------------------------
# Weather Forecast
# -------------------------

print("\n🌤 WEATHER FORECAST")
print("-"*40)

print("Temperature Trend:", temps)
print("Humidity Trend:", humidity)


# -------------------------
# Anomaly Detection
# -------------------------

print("\n🚨 ANOMALY DETECTION")
print("-"*40)

print("System Status:", status)
print("Explanation:", explanation)


# -------------------------
# AI Recommendations
# -------------------------

print("\n🤖 AI RECOMMENDATIONS")
print("-"*40)

for r in recommendations:
    print("•", r)


# -------------------------
# Report Generated
# -------------------------

print("\n📄 REPORT GENERATED")
print("-"*40)

print(f"Saved at: {file_path}")


# -------------------------
# Energy Impact
# -------------------------

print("\n💰 ENERGY IMPACT ANALYSIS")
print("-"*40)

impact_agent = ImpactAgent(summary, forecast)
impact = impact_agent.calculate_energy_impact()

for key, value in impact.items():
    print(f"{key}: {value}")


# -------------------------
# Completed
# -------------------------

print("\n")
print("="*60)
print("        SYSTEM EXECUTION COMPLETED")
print("="*60)