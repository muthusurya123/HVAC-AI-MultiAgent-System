import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


class ForecasterAgent:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.df = None
        self.model = None

    # -------------------------
    # Load and Feature Engineering
    # -------------------------

    def load_and_prepare_data(self):

        self.df = pd.read_csv(
            self.dataset_path,
            parse_dates=["Timestamp"]
        )

        # Time features
        self.df["Hour"] = self.df["Timestamp"].dt.hour
        self.df["DayOfWeek"] = self.df["Timestamp"].dt.dayofweek
        self.df["Month"] = self.df["Timestamp"].dt.month

        # Lag features (important for time series)
        self.df["lag_1"] = self.df["kW"].shift(1)
        self.df["lag_2"] = self.df["kW"].shift(2)
        self.df["lag_3"] = self.df["kW"].shift(3)

        # Rolling average
        self.df["rolling_mean_3"] = self.df["kW"].rolling(window=3).mean()
        self.df["rolling_mean_6"] = self.df["kW"].rolling(window=6).mean()

        # HVAC interaction features
        self.df["Temp_Occupancy"] = self.df["AmbientTemp"] * self.df["Occupancy"]
        self.df["Temp_Humidity"] = self.df["AmbientTemp"] * self.df["Humidity"]

        self.df = self.df.dropna()

        # Feature list
        self.features = [

            "Hour",
            "DayOfWeek",
            "Month",

            "AmbientTemp",
            "Humidity",
            "Occupancy",
            "TR",

            "lag_1",
            "lag_2",
            "lag_3",

            "rolling_mean_3",
            "rolling_mean_6",

            "Temp_Occupancy",
            "Temp_Humidity"
        ]

        self.target = "kW"

    # -------------------------
    # Train Forecast Model
    # -------------------------

    def train_model(self):

        X = self.df[self.features]
        y = self.df[self.target]

        split = int(len(X) * 0.8)

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        self.model = XGBRegressor(

            n_estimators=400,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        return round(mae, 2)

    # -------------------------
    # Forecast Next 24 Hours
    # -------------------------

    def forecast_next_24_hours(self):

        forecasts = []

        last_row = self.df.iloc[-1].copy()

        lag1 = last_row["kW"]
        lag2 = last_row["lag_1"]
        lag3 = last_row["lag_2"]

        rolling3 = last_row["rolling_mean_3"]
        rolling6 = last_row["rolling_mean_6"]

        temp = last_row["AmbientTemp"]
        humidity = last_row["Humidity"]
        occupancy = last_row["Occupancy"]
        tr = last_row["TR"]

        hour = int(last_row["Hour"])
        day = int(last_row["DayOfWeek"])
        month = int(last_row["Month"])

        for i in range(24):

            next_hour = (hour + i) % 24

            # Simulate realistic temperature pattern
            if 11 <= next_hour <= 16:
                temp += np.random.uniform(0.2, 0.4)
            elif 20 <= next_hour or next_hour <= 5:
                temp -= np.random.uniform(0.2, 0.3)

            # Simulate occupancy pattern
            if 9 <= next_hour <= 18:
                occupancy *= np.random.uniform(1.01, 1.03)
            else:
                occupancy *= np.random.uniform(0.96, 0.99)

            temp_occ = temp * occupancy
            temp_hum = temp * humidity

            input_df = pd.DataFrame([{

                "Hour": next_hour,
                "DayOfWeek": day,
                "Month": month,

                "AmbientTemp": temp,
                "Humidity": humidity,
                "Occupancy": occupancy,
                "TR": tr,

                "lag_1": lag1,
                "lag_2": lag2,
                "lag_3": lag3,

                "rolling_mean_3": rolling3,
                "rolling_mean_6": rolling6,

                "Temp_Occupancy": temp_occ,
                "Temp_Humidity": temp_hum
            }])

            prediction = self.model.predict(input_df)[0]

            prediction = float(prediction)

            forecasts.append(round(prediction, 2))

            # Update lag values for next prediction
            lag3 = lag2
            lag2 = lag1
            lag1 = prediction

            rolling3 = (lag1 + lag2 + lag3) / 3
            rolling6 = (rolling6 * 5 + prediction) / 6

        return forecasts