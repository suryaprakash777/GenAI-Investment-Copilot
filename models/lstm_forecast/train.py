# models/lstm_forecast/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.preprocessing import load_data

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model():
    X, y, scaler = load_data("data/AAPL.csv")  # example
    x_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = LSTMModel()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

    for epoch in range(30):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

    torch.save(model.state_dict(), "models/lstm_forecast/model.pt")
    return model, scaler

if __name__ == "__main__":
    train_model()
