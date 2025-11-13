"""Simple feature engineering helpers for ML pipelines.

Functions to create lagged returns, moving averages and assemble X/y for weekly prediction.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List


def lagged_returns(prices: pd.Series, lags: List[int]) -> pd.DataFrame:
    """Return DataFrame of lagged pct-change returns for given lags (in days).

    Each column is named 'ret_{lag}'.
    """
    out = pd.DataFrame(index=prices.index)
    for lag in lags:
        out[f'ret_{lag}'] = prices.pct_change(lag).shift(1)
    return out


def moving_averages(prices: pd.Series, windows: List[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    for w in windows:
        out[f'sma_{w}'] = prices.rolling(window=w, min_periods=int(w * 0.8)).mean()
        out[f'ema_{w}'] = prices.ewm(span=w, adjust=False).mean()
    return out


def build_ml_dataset(prices: pd.Series, exog: pd.DataFrame = None, lags=[5,15,30,60], ma_windows=[21,63,252], forward=5):
    """Build a simple supervised dataset for predicting forward `forward`-day return.

    Returns X, y aligned with index dropped where NaN.
    """
    lagged = lagged_returns(prices, lags)
    mas = moving_averages(prices, ma_windows)

    X = pd.concat([lagged, mas], axis=1)
    if exog is not None:
        X = pd.concat([X, exog.shift(1)], axis=1)

    y = prices.pct_change(forward).shift(-forward)

    # drop rows with NaN
    df = pd.concat([X, y.rename('target')], axis=1).dropna()

    X_clean = df.drop(columns=['target'])
    y_clean = df['target']
    return X_clean, y_clean


__all__ = ["lagged_returns", "moving_averages", "build_ml_dataset"]
