# models/xgboost_forecast/train.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

def create_lag_features(df, lags=10):
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    return df

def train_xgb_model():
    df = pd.read_csv("data/aapl_preprocessed.csv")
    df = create_lag_features(df, lags=10)
    df.dropna(inplace=True)

    feature_cols = ["Open", "High", "Low", "Volume"] + [f"lag_{i}" for i in range(1, 11)]
    X = df[feature_cols]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/xgb_model.pkl")
    print("✅ XGBoost model trained and saved as models/xgb_model.pkl")

if __name__ == "__main__":
    train_xgb_model()
