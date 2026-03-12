import numpy as np
import pandas as pd


class DigitalTwinAgent:

    def __init__(self, data):
        self.data = data

        # baseline statistics from historical data
        self.base_load = data["kW"].mean()
        self.base_temp = data["AmbientTemp"].mean()
        self.base_eff = data["iKW_TR"].mean()

    def simulate(self, setpoint_temp, occupancy):

        # temperature effect
        temp_diff = self.base_temp - setpoint_temp
        temp_effect = temp_diff * 8

        # occupancy effect
        occ_effect = occupancy * 120

        predicted_load = self.base_load + temp_effect + occ_effect

        predicted_eff = self.base_eff + (temp_diff * 0.01)

        energy_kwh = predicted_load * 24

        cost = energy_kwh * 8   # ₹8 per kWh estimate

        co2 = energy_kwh * 0.82  # India grid emission factor

        comfort = max(70, 100 - abs(setpoint_temp - 23) * 5)

        return {
            "Predicted Load (kW)": round(predicted_load,2),
            "Daily Energy (kWh)": round(energy_kwh,2),
            "Daily Cost (₹)": round(cost,2),
            "CO2 Emission (kg)": round(co2,2),
            "Efficiency (iKW/TR)": round(predicted_eff,3),
            "Comfort Score (%)": round(comfort,2)
        }