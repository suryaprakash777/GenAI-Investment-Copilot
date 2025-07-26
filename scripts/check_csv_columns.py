# New script: scripts/check_csv_columns.py
import pandas as pd

df = pd.read_csv("data/aapl.csv")
print("✅ Columns in aapl.csv:", df.columns.tolist())
