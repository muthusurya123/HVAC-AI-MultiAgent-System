from sklearn.ensemble import IsolationForest
import pandas as pd


class AnomalyAgent:

    def __init__(self, df):
        self.df = df
        self.model = None

    def train(self):

        features = [
            "kW",
            "TR",
            "AmbientTemp",
            "Humidity",
            "Occupancy"
        ]

        X = self.df[features]

        self.model = IsolationForest(
            contamination=0.02,
            random_state=42
        )

        self.model.fit(X)

    def detect_anomaly(self, row):

        features = [
            "kW",
            "TR",
            "AmbientTemp",
            "Humidity",
            "Occupancy"
        ]

        X = pd.DataFrame([row[features]])

        prediction = self.model.predict(X)[0]

        if prediction == -1:
            return "ANOMALY", "Unusual HVAC behavior detected"
        else:
            return "NORMAL", "System operating normally"