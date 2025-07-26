# models/xgboost_forecast/train.py

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from models.xgboost_forecast.utils import load_and_prepare
import joblib
import os

def train_model():
    X, y = load_and_prepare("data/AAPL.csv", lags=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    os.makedirs("models/xgboost_forecast", exist_ok=True)
    joblib.dump(model, "models/xgboost_forecast/model.pkl")
    print(f"Model trained. MAE: {mae:.4f}")

if __name__ == "__main__":
    train_model()
