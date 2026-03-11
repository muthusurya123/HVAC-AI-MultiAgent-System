class ImpactAgent:

    def __init__(self, summary, forecast):

        self.summary = summary
        self.forecast = forecast

    def calculate_energy_impact(self):

        avg_load = sum(self.forecast) / len(self.forecast)

        # Daily energy usage estimate
        daily_energy = avg_load * 24

        # Annual estimate
        annual_energy = daily_energy * 365

        efficiency_gap = self.summary["Efficiency Gap (%)"]

        # Energy waste due to inefficiency
        wasted_energy = annual_energy * (efficiency_gap / 100)

        # Electricity cost assumption
        cost_per_kwh = 8  # ₹ per kWh (commercial avg)

        annual_cost_loss = wasted_energy * cost_per_kwh

        # Optimization potential
        optimization_gain = wasted_energy * 0.6

        savings = optimization_gain * cost_per_kwh

        # CO2 emission estimate
        co2_factor = 0.82  # kg CO2 per kWh

        co2_reduction = optimization_gain * co2_factor

        return {

            "Annual Energy Waste (kWh)": round(wasted_energy, 0),
            "Estimated Cost Loss (₹)": round(annual_cost_loss, 0),
            "Optimization Potential (%)": round(self.summary["Efficiency Gap (%)"] * 0.6, 2),
            "Estimated Annual Savings (₹)": round(savings, 0),
            "CO2 Reduction (kg)": round(co2_reduction, 0)

        }