import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class ForecasterAgent:

    def __init__(self, data_path):
        self.data_path = data_path
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.hourly_df = None

    # -----------------------------------
    # Load and prepare data
    # -----------------------------------
    def load_and_prepare_data(self):
        df = pd.read_csv(self.data_path, parse_dates=["Timestamp"])
        df.set_index("Timestamp", inplace=True)
        df = df.select_dtypes(include=["number"])

        # Convert to hourly
        hourly_df = df.resample("h").mean()

        # Create lag feature
                # Create lag features
        hourly_df["kW_lag1"] = hourly_df["kW"].shift(1)
        hourly_df["kW_lag2"] = hourly_df["kW"].shift(2)
        hourly_df["kW_lag3"] = hourly_df["kW"].shift(3)
        hourly_df["kW_lag24"] = hourly_df["kW"].shift(24)

        hourly_df = hourly_df.dropna()

        hourly_df = hourly_df.dropna()

        self.hourly_df = hourly_df

    # -----------------------------------
    # Train model
    # -----------------------------------
    def train_model(self):
        X = self.hourly_df[[
         "kW_lag1",
         "kW_lag2",
         "kW_lag3",
         "kW_lag24",
         "AmbientTemp",
         "Occupancy",
         "TR",
         "Humidity"
         ]]
        y = self.hourly_df["kW"]

        split_index = int(len(self.hourly_df) * 0.8)

        X_train = X[:split_index]
        y_train = y[:split_index]

        X_test = X[split_index:]
        y_test = y[split_index:]

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)

        return round(mae, 2)

    # -----------------------------------
    # True Recursive Forecast (Next 24h)
    # -----------------------------------
    def forecast_next_24_hours(self):
        forecast_results = []

        last_rows = self.hourly_df.tail(24).copy()

        for i in range(24):

            latest = last_rows.iloc[-1]

            X_input = [[
                latest["kW_lag1"],
                latest["kW_lag2"],
                latest["kW_lag3"],
                latest["kW_lag24"],
                latest["AmbientTemp"],
                latest["Occupancy"],
                latest["TR"],
                latest["Humidity"]
            ]]

            next_kW = self.model.predict(X_input)[0]
            forecast_results.append(next_kW)

            # Create new row for next step
            new_row = latest.copy()

            new_row["kW_lag3"] = latest["kW_lag2"]
            new_row["kW_lag2"] = latest["kW_lag1"]
            new_row["kW_lag1"] = next_kW
            new_row["kW_lag24"] = last_rows.iloc[i]["kW"]

            last_rows = pd.concat([last_rows, new_row.to_frame().T])

        return forecast_results