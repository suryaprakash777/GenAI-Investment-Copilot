# models/lstm_forecast/predict.py

import torch
from models.lstm_forecast.train import LSTMModel
from utils.preprocessing import load_data
import matplotlib.pyplot as plt
import numpy as np

def predict_future(days=30):
    X, y, scaler = load_data("data/AAPL.csv")
    model = LSTMModel()
    model.load_state_dict(torch.load("models/lstm_forecast/model.pt"))
    model.eval()

    last_input = torch.Tensor(X[-1:])  # last known window
    preds = []

    for _ in range(days):
        with torch.no_grad():
            next_price = model(last_input)
        preds.append(next_price.item())
        next_input = torch.cat((last_input[:, 1:, :], next_price.unsqueeze(0).unsqueeze(2)), dim=1)
        last_input = next_input

    predicted_prices = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    #Plot
    plt.figure(figsize=(10,6))
    actual_prices = scaler.inverse_transform(y[-100:])
    plt.plot(range(len(actual_prices)), actual_prices, label="Actual")
    plt.plot(range(len(actual_prices), len(actual_prices)+days), predicted_prices, label="Predicted")
    plt.legend()
    plt.title("Stock Price Forecast")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(True)
    plt.savefig("models/lstm_forecast/forecast_plot.png")
    plt.show()

if __name__ == "__main__":
    predict_future()
