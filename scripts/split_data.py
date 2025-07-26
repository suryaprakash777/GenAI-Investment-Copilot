# scripts/split_data.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def create_features(df):
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)  # Predict next day's close
    df.dropna(inplace=True)
    return df

def split_and_save(df):
    X = df.drop(columns=["Date", "Target"])
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    os.makedirs("data", exist_ok=True)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("✅ Data split and saved.")
    print(f"🟢 X_train: {X_train.shape}, X_test: {X_test.shape}")

if __name__ == "__main__":
    df = pd.read_csv("data/aapl_preprocessed.csv")
    df = create_features(df)
    split_and_save(df)
