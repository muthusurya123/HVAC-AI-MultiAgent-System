import pandas as pd
import time

from agents.anomaly_detector import AnomalyAgent


df = pd.read_csv(
    "data/large_mall_dataset_3years_5min.csv",
    parse_dates=["Timestamp"]
)

anomaly = AnomalyAgent(df)

anomaly.train()

print("\n===== REAL TIME HVAC STREAM =====\n")

for index, row in df.tail(50).iterrows():

    status = anomaly.detect_anomaly(row)

    print("Time:", row["Timestamp"])
    print("Load:", round(row["kW"], 2))
    print("Status:", status)

    print("-----------------------")

    time.sleep(1)