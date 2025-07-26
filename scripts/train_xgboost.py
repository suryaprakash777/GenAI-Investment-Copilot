# scripts/train_xgboost.py

import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def train_xgboost_model():
    # Load training data
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()

    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict on training set for quick eval
    y_pred = model.predict(X_train)

    # Metrics
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))

    print(f"✅ XGBoost model trained.")
    print(f"📈 R² Score: {r2:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("💾 Model saved to: models/xgb_model.pkl")

if __name__ == "__main__":
    train_xgboost_model()
