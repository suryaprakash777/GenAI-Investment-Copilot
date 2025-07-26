# models/xgboost_forecast/predict.py

import joblib
import pandas as pd
from models.xgboost_forecast.utils import create_lag_features

def predict_next(filepath="data/AAPL.csv", days=5):
    df = pd.read_csv(filepath)
    model = joblib.load("models/xgboost_forecast/model.pkl")
    
    last_known = df[-10:].copy()
    preds = []

    for _ in range(days):
        temp = last_known.copy()
        temp = create_lag_features(temp, lags=10).iloc[-1:]
        X = temp.drop(columns=["Date", "Close"])
        pred = model.predict(X)[0]
        preds.append(pred)

        new_row = pd.DataFrame({"Date": [None], "Close": [pred]})
        last_known = pd.concat([last_known, new_row], ignore_index=True)

    print("Next predicted values:", preds)
    return preds

if __name__ == "__main__":
    predict_next()
