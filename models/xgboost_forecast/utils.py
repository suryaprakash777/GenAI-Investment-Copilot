# models/xgboost_forecast/utils.py

import pandas as pd

def create_lag_features(df, lags=10):
    df = df.copy()
    for i in range(1, lags+1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

def load_and_prepare(filepath, lags=10):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = create_lag_features(df, lags)
    X = df.drop(columns=["Date", "Close"])
    y = df["Close"]
    return X, y
