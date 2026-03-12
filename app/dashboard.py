import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

from agents.analyzer import AnalyzerAgent
from agents.forecasting import ForecasterAgent
from agents.recommender import RecommenderAgent
from agents.anomaly_detector import AnomalyAgent
from agents.weather import WeatherAgent
from agents.impact import ImpactAgent
from agents.optimizer import OptimizerAgent
from agents.digital_twin import DigitalTwinAgent
from agents.failure_predictor import FailurePredictorAgent

st.set_page_config(page_title="HVAC AI Platform", layout="wide")

# --------------------------------------------------
# LOAD DATA (CACHED)
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/large_mall_dataset_3years_5min.csv",
        parse_dates=["Timestamp"]
    )
    return df

df = load_data()

# --------------------------------------------------
# FAILURE PREDICTION MODEL
# --------------------------------------------------

@st.cache_resource
def run_failure_model(data):
    model = FailurePredictorAgent(data)
    model.train()
    return model

failure_model = run_failure_model(df)

# --------------------------------------------------
# ANALYZER
# --------------------------------------------------

@st.cache_resource
def run_analyzer(data):
    analyzer = AnalyzerAgent(data)
    return analyzer.generate_summary()

summary = run_analyzer(df)

# --------------------------------------------------
# FORECAST MODEL
# --------------------------------------------------

@st.cache_resource
def run_forecaster():
    forecaster = ForecasterAgent("data/large_mall_dataset_3years_5min.csv")
    forecaster.load_and_prepare_data()
    forecaster.train_model()
    return forecaster

forecaster = run_forecaster()
forecast = forecaster.forecast_next_24_hours()

# --------------------------------------------------
# WEATHER
# --------------------------------------------------

weather = WeatherAgent()
temps, humidity = weather.get_weather_forecast()

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

recommender = RecommenderAgent(summary, forecast)
recommendations = recommender.generate_recommendations()

# --------------------------------------------------
# ENERGY IMPACT
# --------------------------------------------------

impact_agent = ImpactAgent(summary, forecast)
impact = impact_agent.calculate_energy_impact()

# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

@st.cache_resource
def run_anomaly(data):
    anomaly = AnomalyAgent(data)
    anomaly.train()
    return anomaly

anomaly = run_anomaly(df)

latest = df.iloc[-1]
status, explanation = anomaly.detect_anomaly(latest)

# --------------------------------------------------
# AI OPTIMIZATION
# --------------------------------------------------

optimizer = OptimizerAgent(summary, forecast, temps)
decision = optimizer.optimize()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("HVAC AI Platform")

digital_twin = DigitalTwinAgent(df)
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Real-Time Monitoring",
        "Forecasting",
        "Weather",
        "Energy Impact",
        "AI Optimization",
        "Digital Twin",
        "Recommendations",
        "Architecture"
    ]
)

# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.title("AI-Powered HVAC Optimization Platform")

    st.subheader("System Health")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Efficiency (iKW/TR)", summary["Average iKW/TR"])
    col2.metric("Efficiency Gap (%)", summary["Efficiency Gap (%)"])
    col3.metric("Peak Demand (kW)", summary["Peak Demand (kW)"])
    col4.metric("Load Factor (%)", summary["Load Factor (%)"])

    st.write("Maintenance Priority:", summary["Maintenance Priority"])

    st.subheader("System Anomaly Detection")

    if status == "NORMAL":
        st.success(explanation)
    else:
        st.error(explanation)

    # Failure Risk Prediction
    st.subheader("Failure Risk Prediction")
    
    failure_prob = failure_model.predict_failure(latest)
    risk_percent = round(failure_prob * 100, 2)
    
    if risk_percent < 30:
        st.success(f"Low Failure Risk ({risk_percent}%)")
    elif risk_percent < 60:
        st.warning(f"Moderate Failure Risk ({risk_percent}%)")
    else:
        st.error(f"High Failure Risk ({risk_percent}%) - Maintenance Needed")
    
    # Feature Importance Chart
    features = [
        "AmbientTemp", "Humidity", "Occupancy", "iKW_TR",
        "PLR", "COP", "EquipmentAgeFactor", "MaintenanceScore"
    ]
    
    importance = failure_model.model.feature_importances_
    
    imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })
    
    fig = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        title="Factors Affecting HVAC Failure"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# REAL TIME MONITORING
# --------------------------------------------------
elif page == "Real-Time Monitoring":

    st.title("Real-Time HVAC Monitoring")

    placeholder = st.empty()
    live_data = df.tail(200).copy()

    for i in range(50):

        row = live_data.iloc[i]

        current_load = row["kW"] + np.random.normal(0,1)
        current_eff = row["iKW_TR"] + np.random.normal(0,0.01)
        current_temp = row["AmbientTemp"] + np.random.normal(0,0.2)

        placeholder.empty()

        with placeholder.container():

            # ----------------------
            # LIVE METRICS
            # ----------------------
            col1, col2, col3 = st.columns(3)

            col1.metric("Current Load (kW)", round(current_load,2))
            col2.metric("Current Efficiency (iKW/TR)", round(current_eff,3))
            col3.metric("Ambient Temperature (°C)", round(current_temp,2))

            # ----------------------
            # HVAC Efficiency Gauge
            # ----------------------
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=current_eff,
                title={'text': "HVAC Efficiency"},
                gauge={
                    'axis': {'range': [0,1]},
                    'steps': [
                        {'range': [0,0.6], 'color': "green"},
                        {'range': [0.6,0.75], 'color': "yellow"},
                        {'range': [0.75,1], 'color': "red"}
                    ],
                }
            ))

            st.plotly_chart(gauge, use_container_width=True)

            # ----------------------
            # GRAPH 1
            # HVAC Load Stream
            # ----------------------
            fig1 = px.line(
                live_data.iloc[:i+1],
                y="kW",
                title="Live Cooling Load (kW)"
            )

            st.plotly_chart(fig1, use_container_width=True)

            # ----------------------
            # GRAPH 2
            # Ambient Temp vs Load
            # ----------------------
            fig2 = px.scatter(
                live_data.iloc[:i+1],
                x="AmbientTemp",
                y="kW",
                title="Cooling Load vs Ambient Temperature",
                opacity=0.7
            )

            st.plotly_chart(fig2, use_container_width=True)

            # ----------------------
            # GRAPH 3
            # Efficiency Trend
            # ----------------------
            fig3 = px.line(
                live_data.iloc[:i+1],
                y="iKW_TR",
                title="Efficiency Trend (iKW/TR)"
            )

            st.plotly_chart(fig3, use_container_width=True)

        time.sleep(1)

# --------------------------------------------------
# FORECASTING
# --------------------------------------------------

elif page == "Forecasting":

    st.title("24 Hour Load Forecast")

    forecast_df = pd.DataFrame({
        "Hour": list(range(1,25)),
        "Load_kW": forecast
    })

    fig = px.line(
        forecast_df,
        x="Hour",
        y="Load_kW",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Historical Load Trend")

    fig2 = px.line(
        df.tail(500),
        x="Timestamp",
        y="kW"
    )

    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# WEATHER
# --------------------------------------------------

elif page == "Weather":

    st.title("Weather Forecast")

    weather_df = pd.DataFrame({
        "Step": list(range(1,len(temps)+1)),
        "Temperature": temps,
        "Humidity": humidity
    })

    fig1 = px.line(
        weather_df,
        x="Step",
        y="Temperature",
        title="Temperature Forecast"
    )

    fig2 = px.line(
        weather_df,
        x="Step",
        y="Humidity",
        title="Humidity Forecast"
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# ENERGY IMPACT
# --------------------------------------------------

elif page == "Energy Impact":

    st.title("Energy Impact Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Annual Energy Waste (kWh)", impact["Annual Energy Waste (kWh)"])
    col2.metric("Estimated Cost Loss (₹)", impact["Estimated Cost Loss (₹)"])
    col3.metric("Estimated Annual Savings (₹)", impact["Estimated Annual Savings (₹)"])

    st.metric("CO₂ Reduction (kg)", impact["CO2 Reduction (kg)"])

# --------------------------------------------------
# AI OPTIMIZATION
# --------------------------------------------------

elif page == "AI Optimization":

    st.title("AI HVAC Optimization Engine")

    st.subheader("AI Control Decision")

    st.metric("Recommended Action", decision["Action"])

    st.write("Reason:", decision["Reason"])

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated Daily Savings (₹)",
        decision["Estimated Daily Savings (₹)"]
    )

    col2.metric(
        "CO2 Reduction (kg)",
        decision["CO2 Reduction (kg)"]
    )

# --------------------------------------------------
# DIGITAL TWIN
# --------------------------------------------------

elif page == "Digital Twin":

    st.title("HVAC Digital Twin Simulation")

    st.write("Simulate HVAC behavior before applying real-world changes.")

    col1, col2 = st.columns(2)

    with col1:
        setpoint = st.slider(
            "Temperature Setpoint (°C)",
            18.0,
            28.0,
            23.0
        )

    with col2:
        occupancy = st.slider(
            "Occupancy Level",
            0.0,
            1.0,
            0.5
        )

    results = digital_twin.simulate(setpoint, occupancy)

    st.subheader("Simulation Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Predicted Load (kW)", results["Predicted Load (kW)"])
    c2.metric("Efficiency (iKW/TR)", results["Efficiency (iKW/TR)"])
    c3.metric("Comfort Score (%)", results["Comfort Score (%)"])

    c4, c5, c6 = st.columns(3)

    c4.metric("Daily Energy (kWh)", results["Daily Energy (kWh)"])
    c5.metric("Daily Cost (₹)", results["Daily Cost (₹)"])
    c6.metric("CO2 Emission (kg)", results["CO2 Emission (kg)"])

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

elif page == "Recommendations":

    st.title("AI Maintenance Recommendations")

    for r in recommendations:
        st.write("•", r)

    st.success("HVAC AI Optimization Analysis Completed")

# --------------------------------------------------
# ARCHITECTURE
# --------------------------------------------------

elif page == "Architecture":

    st.title("AI System Architecture")

    pipeline = pd.DataFrame({
        "Stage": [
            "Sensor Data",
            "Analyzer Agent",
            "Forecasting Model",
            "Anomaly Detection",
            "Recommendation Engine",
            "Energy Impact Analysis",
            "Dashboard"
        ],
        "Step": [1,2,3,4,5,6,7],
        "Y":[1,1,1,1,1,1,1]
    })

    fig = px.scatter(
        pipeline,
        x="Step",
        y="Y",
        text="Stage",
        size=[40]*7
    )

    fig.update_traces(
        textposition="middle center",
        marker=dict(color="#00BFFF")
    )

    fig.update_layout(
        showlegend=False,
        yaxis=dict(visible=False),
        xaxis_title="AI Processing Pipeline",
        height=250
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Models Used")

    st.markdown("""
• **XGBoost** — HVAC Load Forecasting  
• **Isolation Forest** — Anomaly Detection  
• **Weather API** — Environmental Impact Modeling  
""")

    st.subheader("Deployment Ready")

    st.markdown("""
• Real-time monitoring dashboard  
• Predictive maintenance insights  
• Energy optimization recommendations  
""")