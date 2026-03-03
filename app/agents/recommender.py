class RecommenderAgent:

    def __init__(self, analyzer_results, forecast_values):
        self.analyzer = analyzer_results
        self.forecast = forecast_values
        self.design_capacity = 300  # kW assumed design capacity

    def generate_recommendations(self):

        recommendations = []

        # 1. Peak Risk Detection
        forecast_peak = max(self.forecast)
        if forecast_peak > 0.95 * self.design_capacity:
            recommendations.append(
                f"Peak Risk Alert: Forecasted load ({forecast_peak:.2f} kW) exceeds 95% of design capacity."
            )

        # 2. Efficiency Gap Logic
        if self.analyzer["Efficiency Gap (%)"] > 3:
            recommendations.append(
                "Efficiency degradation detected. Recommend condenser inspection and heat exchanger cleaning."
            )

        # 3. Maintenance Priority
        if self.analyzer["Maintenance Priority"] == "High":
            recommendations.append(
                "High maintenance priority. Schedule preventive maintenance within 7 days."
            )

        # 4. Weather Sensitivity
        if self.analyzer["Weather Sensitivity Index"] > 0.5:
            recommendations.append(
                "System highly weather-sensitive. Consider adaptive chilled water setpoint optimization."
            )

        if not recommendations:
            recommendations.append("System operating within optimal conditions.")

        return recommendations