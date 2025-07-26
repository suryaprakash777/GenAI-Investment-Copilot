# scripts/download_data.py
import yfinance as yf
import pandas as pd

def download_aapl_data():
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="10y")
    df.reset_index(inplace=True)  # Moves 'Date' from index to column

    # Save only the required columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    df.to_csv("data/aapl.csv", index=False)
    print("✅ AAPL data saved to data/aapl.csv")

if __name__ == "__main__":
    download_aapl_data()
