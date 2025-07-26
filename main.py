# main.py

import argparse
from models.xgboost_forecast.predict import predict_next
from models.lstm_forecast.predict import predict_future

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "xgb"], required=True)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--ticker", type=str, default="AAPL")

    args = parser.parse_args()
    path = f"data/{args.ticker}.csv"

    if args.model == "lstm":
        predict_future(days=args.days)
    else:
        predict_next(filepath=path, days=args.days)
