from data_loader import load_and_prepare_data
from model import LSTMModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X, y = load_and_prepare_data()
model = LSTMModel(input_size=7)
model.load_state_dict(torch.load("models/lstm_model.pth"))
model.eval()

with torch.no_grad():
    preds = model(X).squeeze().numpy()
    y_true = y.numpy()

print("RMSE:", np.sqrt(mean_squared_error(y_true, preds)))
print("MAE:", mean_absolute_error(y_true, preds))
print("R²:", r2_score(y_true, preds))

plt.figure(figsize=(10, 5))
plt.plot(y_true[:200], label="Actual")
plt.plot(preds[:200], label="Predicted")
plt.legend()
plt.grid(True)
plt.title("Bandwidth Prediction for Indian Locations")
plt.show()