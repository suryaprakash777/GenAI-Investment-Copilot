import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from models.xgboost_forecast.predict import predict_next as xgb_predict
from models.lstm_forecast.predict import predict_future as lstm_predict

# Configure Streamlit page
st.set_page_config(page_title="📊 GenAI Investment Forecast", layout="wide")
st.title("📈 GenAI Copilot: AAPL Stock Price Forecast")
st.markdown("Forecast **Apple's stock price** using AI-powered models like **XGBoost** and **LSTM**.")

# Sidebar: Model selection
model_choice = st.selectbox("🔍 Choose a Model", ["XGBoost", "LSTM"])
forecast_days = st.slider("📆 Forecast Days", min_value=1, max_value=30, value=5)

# Forecast button
if st.button("🚀 Run Forecast"):
    st.info(f"Running {model_choice} forecast for the next {forecast_days} day(s)...")

    try:
        if model_choice == "XGBoost":
            st.subheader("🔹 XGBoost Forecast Results")
            preds = xgb_predict(filepath="data/aapl_preprocessed.csv", days=forecast_days)

            # Prepare forecast data
            start_date = datetime.today()
            forecast_dates = [(start_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(len(preds))]
            forecast_values = [round(float(p), 2) for p in preds]
            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Predicted Close Price (USD)": forecast_values
            })

            # Display table
            st.dataframe(forecast_df, use_container_width=True)

            # Plot line chart
            st.subheader("📉 XGBoost Forecast Trend")
            fig, ax = plt.subplots()
            ax.plot(forecast_df["Date"], forecast_df["Predicted Close Price (USD)"],
                    marker='o', linestyle='-', color='blue', label='Forecast')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title("AAPL Stock Forecast (XGBoost)")
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif model_choice == "LSTM":
            st.subheader("🔹 LSTM Forecast Results")
            lstm_predict(days=forecast_days)
            plot_path = "models/lstm_forecast/forecast_plot.png"

            if os.path.exists(plot_path):
                st.image(plot_path, caption=f"LSTM Forecast - Next {forecast_days} Day(s)", use_column_width=True)
            else:
                st.error("❌ Forecast plot not found. Please ensure the LSTM model has saved it.")

    except Exception as e:
        st.error(f"❌ Something went wrong during prediction: {e}")
