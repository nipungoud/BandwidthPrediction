import pandas as pd
import numpy as np
import os

def generate_indian_5g_data(num_samples=10000):
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)

    timestamps = pd.date_range(start="2023-01-01", periods=num_samples, freq="s")
    mobility = np.random.choice([0, 20, 60, 100], size=num_samples)
    rssi = np.random.normal(-85, 6, size=num_samples)
    sinr = np.random.normal(15, 4, size=num_samples)
    rsrp = np.random.normal(-95, 5, size=num_samples)
    rsrq = np.random.normal(-10, 2, size=num_samples)
    lat = np.random.uniform(19.0, 28.5, num_samples)
    lon = np.random.uniform(72.8, 88.0, num_samples)

    bandwidth = (
        100 + sinr * 15 - mobility * 1.5 + np.random.normal(0, 20, num_samples)
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "latitude": lat,
        "longitude": lon,
        "mobility": mobility,
        "RSSI": rssi,
        "SINR": sinr,
        "RSRP": rsrp,
        "RSRQ": rsrq,
        "bandwidth": np.clip(bandwidth, 15, 1000)
    })

    df.to_csv("data/simulated_indian_5g.csv", index=False)
    print("✅ Simulated Indian 5G dataset saved.")

if __name__ == "__main__":
    generate_indian_5g_data()
