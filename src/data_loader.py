import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_and_prepare_data(path="data/simulated_indian_5g.csv", sequence_length=10):
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(path)
    df.drop(columns=["timestamp"], inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    joblib.dump(scaler, "models/scaler.pkl")

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i+sequence_length, :-1])
        y.append(scaled[i+sequence_length, -1])

    return torch.tensor(X).float(), torch.tensor(y).float()
