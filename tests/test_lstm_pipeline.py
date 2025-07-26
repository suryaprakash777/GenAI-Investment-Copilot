# tests/test_lstm_pipeline.py

from utils.preprocessing import load_data

def test_lstm_preprocessing():
    X, y, _ = load_data("data/AAPL.csv")
    assert len(X) == len(y)
    assert X.shape[1] == 60  # default window
