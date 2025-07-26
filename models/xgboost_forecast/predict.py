# models/xgboost_forecast/predict.py
import sys
import os
sys.path.append(os.getcwd())

import joblib
import pandas as pd

FEATURES = ['Open', 'High', 'Low', 'Volume'] + [f'lag_{i}' for i in range(1, 11)]

def create_lag_features(df, lags=10):
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    return df

def predict_next(filepath="data/AAPL.csv", days=5):
    df = pd.read_csv(filepath)
    model = joblib.load("models/xgb_model.pkl")
    
    last_known = df[-10:].copy()
    preds = []

    for _ in range(days):
        temp = last_known.copy()
        temp = create_lag_features(temp, lags=10).iloc[-1:]
        X = temp[FEATURES]
        pred = model.predict(X)[0]
        preds.append(pred)

        new_row = pd.DataFrame({"Date": [None], "Open": [X["Open"].values[0]], "High": [X["High"].values[0]],
                                "Low": [X["Low"].values[0]], "Close": [pred], "Volume": [X["Volume"].values[0]]})
        last_known = pd.concat([last_known, new_row], ignore_index=True)

    print("📈 Next predicted values:", preds)
    return preds

def predict_phase2_model(filepath="data/X_test.csv"):
    model = joblib.load("models/xgb_model.pkl")
    df = pd.read_csv(filepath)

    if not all(col in df.columns for col in FEATURES):
        print("⚠️ Lag features missing! Creating lag features...")
        df = create_lag_features(df)
        df.dropna(inplace=True)

    try:
        X = df[FEATURES]
    except KeyError as e:
        print("🚫 Error:", e)
        print("Check if the test file has correct columns.")
        return []

    preds = model.predict(X)
    print("🔮 Phase 2 XGBoost Predictions:", preds.tolist())
    return preds.tolist()

if __name__ == "__main__":
    print("Running XGBoost prediction...")
    predict_phase2_model()
