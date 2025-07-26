# frontend/forecast_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from models.xgboost_forecast.predict import predict_next
from models.lstm_forecast.predict import predict_future

st.title("📈 Stock Price Forecasting")

model = st.selectbox("Select Model", ["LSTM", "XGBoost"])
days = st.slider("Days to Predict", 1, 30, 7)
ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Predict"):
    if model == "LSTM":
        st.info("Running LSTM Forecast...")
        predict_future(days)
        st.image("models/lstm_forecast/forecast_plot.png")
    else:
        st.info("Running XGBoost Forecast...")
        preds = predict_next(f"data/{ticker}.csv", days)
        st.line_chart(preds)
