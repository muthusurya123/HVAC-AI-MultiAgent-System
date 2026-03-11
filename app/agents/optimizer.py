class OptimizerAgent:

    def __init__(self, summary, forecast, temps):
        self.summary = summary
        self.forecast = forecast
        self.temps = temps

    def optimize(self):

        avg_load = sum(self.forecast) / len(self.forecast)
        avg_temp = sum(self.temps) / len(self.temps)

        action = "Maintain current HVAC settings"
        reason = "System operating within optimal range"
        savings = 0
        co2 = 0

        if avg_temp > 30:
            action = "Reduce chilled water setpoint by 1°C"
            reason = "High ambient temperature increasing cooling demand"
            savings = 4200
            co2 = 12

        elif avg_load > 95:
            action = "Enable secondary chiller optimization"
            reason = "Predicted load spike detected"
            savings = 3500
            co2 = 9

        return {
            "Action": action,
            "Reason": reason,
            "Estimated Daily Savings (₹)": savings,
            "CO2 Reduction (kg)": co2
        }