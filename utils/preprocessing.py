# utils/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath, window_size=60):
    df = pd.read_csv(filepath)
    close_prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler
