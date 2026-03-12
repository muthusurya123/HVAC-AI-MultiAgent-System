import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class FailurePredictorAgent:

    def __init__(self, data):
        self.data = data

    def prepare_data(self):

        features = [
            "AmbientTemp",
            "Humidity",
            "Occupancy",
            "iKW_TR",
            "PLR",
            "COP",
            "EquipmentAgeFactor",
            "MaintenanceScore"
        ]

        X = self.data[features]
        y = self.data["FaultFlag"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.scaler = scaler

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train(self):

        X_train, X_test, y_train, y_test = self.prepare_data()

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        )

        model.fit(X_train, y_train)

        self.model = model

    def predict_failure(self, row):

        features = [
            "AmbientTemp",
            "Humidity",
            "Occupancy",
            "iKW_TR",
            "PLR",
            "COP",
            "EquipmentAgeFactor",
            "MaintenanceScore"
        ]

        X = pd.DataFrame([row[features]])

        X = self.scaler.transform(X)

        prob = self.model.predict_proba(X)[0][1]

        return prob