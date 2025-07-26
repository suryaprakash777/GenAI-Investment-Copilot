# tests/test_xgboost_pipeline.py

import pytest
from models.xgboost_forecast.utils import load_and_prepare

def test_load_and_prepare():
    X, y = load_and_prepare("data/AAPL.csv", lags=10)
    assert not X.empty
    assert not y.empty
    assert X.shape[1] == 10  # 10 lag features
