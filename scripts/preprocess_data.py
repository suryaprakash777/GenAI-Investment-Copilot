# scripts/preprocess_data.py

import pandas as pd
import os

def preprocess_aapl_data():
    input_path = "data/aapl.csv"
    output_path = "data/aapl_preprocessed.csv"

    # Load the CSV
    df = pd.read_csv(input_path, parse_dates=["Date"])
    
    # Keep only essential columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Drop missing rows
    df.dropna(inplace=True)

    # Sort by date
    df.sort_values("Date", inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to: {output_path} | Rows: {len(df)}")

if __name__ == "__main__":
    preprocess_aapl_data()
