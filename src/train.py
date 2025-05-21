import torch
from torch import nn, optim
from model import LSTMModel
from data_loader import load_and_prepare_data
import os

def train_model():
    X, y = load_and_prepare_data()
    model = LSTMModel(input_size=7)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/30, Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("✅ Model saved.")

if __name__ == "__main__":
    train_model()
