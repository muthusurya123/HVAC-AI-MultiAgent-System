import numpy as np
import pandas as pd


class AnalyzerAgent:
    def __init__(self, df):
        self.df = df
        self.design_ikw_tr = 0.72  # Design benchmark

    # ---------------------------------------------------
    # 1️⃣ Basic KPIs
    # ---------------------------------------------------
    def average_efficiency(self):
        return round(self.df["iKW_TR"].mean(), 4)

    def calculate_efficiency_gap(self):
        actual = self.average_efficiency()
        gap = ((actual - self.design_ikw_tr) / self.design_ikw_tr) * 100
        return round(gap, 2)

    def calculate_peak_demand(self):
        return round(self.df["kW"].max(), 2)

    def calculate_load_factor(self):
        avg_load = self.df["kW"].mean()
        peak_load = self.df["kW"].max()
        return round((avg_load / peak_load) * 100, 2)

    # ---------------------------------------------------
    # 2️⃣ Monthly Efficiency Benchmark
    # ---------------------------------------------------
    def monthly_efficiency_benchmark(self):
        monthly_avg = self.df.groupby("Month")["iKW_TR"].mean()
        worst_month = monthly_avg.idxmax()
        best_month = monthly_avg.idxmin()

        return {
            "Worst Efficiency Month": int(worst_month),
            "Best Efficiency Month": int(best_month)
        }

    # ---------------------------------------------------
    # 3️⃣ Weather Sensitivity Index
    # ---------------------------------------------------
    def weather_sensitivity(self):
        correlation = self.df["kW"].corr(self.df["AmbientTemp"])
        return round(correlation, 3)

    # ---------------------------------------------------
    # 4️⃣ Degradation Trend Detection
    # ---------------------------------------------------
    def detect_degradation_trend(self):
        yearly_avg = self.df.groupby("Year")["iKW_TR"].mean()

        if len(yearly_avg) < 2:
            return "Insufficient data"

        trend = yearly_avg.iloc[-1] - yearly_avg.iloc[0]

        if trend > 0.02:
            return "Significant Degradation Detected"
        elif trend > 0.005:
            return "Mild Degradation Detected"
        else:
            return "Stable Performance"

    # ---------------------------------------------------
    # 5️⃣ Fault Severity Analysis
    # ---------------------------------------------------
    def fault_severity_analysis(self):
        fault_df = self.df[self.df["FaultFlag"] == 1]

        if len(fault_df) == 0:
            return "No faults detected"

        avg_fault_spike = fault_df["kW"].mean()
        normal_avg = self.df["kW"].mean()

        severity = ((avg_fault_spike - normal_avg) / normal_avg) * 100

        if severity > 40:
            return "High Severity Fault Pattern"
        elif severity > 20:
            return "Moderate Fault Pattern"
        else:
            return "Minor Fault Pattern"

    # ---------------------------------------------------
    # 6️⃣ Root Cause Hypothesis Engine
    # ---------------------------------------------------
    def root_cause_analysis(self):
        eff_gap = self.calculate_efficiency_gap()
        weather_corr = self.weather_sensitivity()
        degradation_status = self.detect_degradation_trend()
    
        if "Significant" in degradation_status:
            return "Performance degradation trend detected - possible fouling or aging components."
        elif eff_gap > 5 and weather_corr > 0.5:
            return "Likely condenser inefficiency due to high ambient temperature."
        elif eff_gap > 5:
            return "Possible chiller performance degradation or fouling."
        elif weather_corr > 0.6:
            return "High weather sensitivity - review load control strategy."
        else:
            return "System operating within expected parameters."

    # ---------------------------------------------------
    # 7️⃣ Maintenance Priority Scoring
    # ---------------------------------------------------
    def maintenance_priority_score(self):
        degradation_status = self.detect_degradation_trend()

        if "Significant" in degradation_status:
            return "High"
        elif "Mild" in degradation_status:
            return "Medium"
        else:
            return "Low"

    # ---------------------------------------------------
    # 8️⃣ Final Summary Report
    # ---------------------------------------------------
    def generate_summary(self):
        return {
            "Average iKW/TR": self.average_efficiency(),
            "Efficiency Gap (%)": self.calculate_efficiency_gap(),
            "Peak Demand (kW)": self.calculate_peak_demand(),
            "Load Factor (%)": self.calculate_load_factor(),
            "Weather Sensitivity Index": self.weather_sensitivity(),
            "Degradation Status": self.detect_degradation_trend(),
            "Fault Severity": self.fault_severity_analysis(),
            "Root Cause Hypothesis": self.root_cause_analysis(),
            "Maintenance Priority": self.maintenance_priority_score(),
            "Monthly Benchmark": self.monthly_efficiency_benchmark()
        }