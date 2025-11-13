"""Technical indicators used across strategies.

Provides: SMA, EMA, RSI, MACD, z-score
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=int(window * 0.8)).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return MACD line and signal line."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def zscore(series: pd.Series, window: int = 42) -> pd.Series:
    """Z-score relative to rolling mean/std."""
    mu = series.rolling(window=window, min_periods=int(window * 0.8)).mean()
    sigma = series.rolling(window=window, min_periods=int(window * 0.8)).std()
    return (series - mu) / sigma


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    Uses Wilder's smoothing method (as in the notebooks).
    Returns a pandas Series indexed like `series`.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Prepare series
    avg_gain = pd.Series(index=series.index, dtype=float)
    avg_loss = pd.Series(index=series.index, dtype=float)

    # First value: simple average
    if len(series) <= period:
        return pd.Series(np.nan, index=series.index)

    avg_gain.iloc[period] = gain.iloc[1 : period + 1].mean()
    avg_loss.iloc[period] = loss.iloc[1 : period + 1].mean()

    # Wilder smoothing
    for i in range(period + 1, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series


__all__ = ["sma", "ema", "macd", "zscore", "rsi"]


class TechnicalIndicators:
    """Class wrapper providing methods used by strategy classes.

    The notebooks/strategy code instantiate this class and call
    calculate_sma / calculate_ema / calculate_macd / calculate_zscore / calculate_rsi.
    """

    @staticmethod
    def calculate_sma(series: pd.Series, window: int) -> pd.Series:
        return sma(series, window)

    @staticmethod
    def calculate_ema(series: pd.Series, span: int) -> pd.Series:
        return ema(series, span)

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        return macd(series, fast, slow, signal)

    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 42) -> pd.Series:
        return zscore(series, window)

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        return rsi(series, period)

__all__.append("TechnicalIndicators")
