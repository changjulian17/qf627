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


def macd_normalized(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, norm_window: int = 42):
    """Return normalized MACD line and signal line as z-scores.
    
    Transforms MACD to be stationary by computing rolling z-score,
    making it oscillate around 0 with typical range of [-3, 3].
    """
    macd_line, signal_line = macd(series, fast, slow, signal)
    
    # Normalize MACD line to z-score
    macd_zscore = zscore(macd_line, norm_window)
    signal_zscore = zscore(signal_line, norm_window)
    
    return macd_zscore, signal_zscore


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


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3):
    """Compute Stochastic Oscillator %K and %D.
    
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA of %K over d_period
    
    Returns: %K series, %D series
    """
    lowest_low = low.rolling(window=k_period, min_periods=int(k_period * 0.8)).min()
    highest_high = high.rolling(window=k_period, min_periods=int(k_period * 0.8)).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period, min_periods=int(d_period * 0.8)).mean()
    
    return k_percent, d_percent


def rate_of_change(series: pd.Series, period: int = 12) -> pd.Series:
    """Compute Rate of Change (ROC).
    
    ROC = 100 * (Price - Price_n_periods_ago) / Price_n_periods_ago
    """
    roc = 100 * (series - series.shift(period)) / series.shift(period)
    return roc


def cumulative_volume(volume: pd.Series) -> pd.Series:
    """Compute cumulative volume.
    
    Returns the cumulative sum of volume over time.
    """
    return volume.cumsum()


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV).
    
    Approximates buying/selling pressure based on price movement:
    - If close > previous close: add volume (buyers won the day)
    - If close < previous close: subtract volume (sellers won the day)
    - If close == previous close: no change
    
    This provides a running total of "signed" volume based on daily price direction.
    """
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def obv_normalized(close: pd.Series, volume: pd.Series, norm_window: int = 42) -> pd.Series:
    """Compute normalized On-Balance Volume (OBV) as z-score.
    
    Transforms OBV to be stationary by computing rolling z-score,
    making it oscillate around 0 with typical range of [-3, 3].
    """
    obv = on_balance_volume(close, volume)
    obv_zscore = zscore(obv, norm_window)
    return obv_zscore


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    """Compute Bollinger Bands.
    
    Parameters
    ----------
    series : pd.Series
        Price series (typically closing prices)
    window : int
        Rolling window for mean and standard deviation
    num_std : float
        Number of standard deviations for upper/lower bands
    
    Returns
    -------
    tuple of (middle_band, upper_band, lower_band)
        middle_band: SMA of the price
        upper_band: SMA + (num_std * rolling std)
        lower_band: SMA - (num_std * rolling std)
    """
    middle_band = sma(series, window)
    rolling_std = series.rolling(window=window, min_periods=int(window * 0.8)).std()
    upper_band = middle_band + (num_std * rolling_std)
    lower_band = middle_band - (num_std * rolling_std)
    
    return middle_band, upper_band, lower_band


__all__ = ["sma", "ema", "macd", "macd_normalized", "zscore", "rsi", "stochastic_oscillator", "rate_of_change", "cumulative_volume", "on_balance_volume", "obv_normalized", "bollinger_bands"]


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
    def calculate_macd_normalized(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9, norm_window: int = 42):
        return macd_normalized(series, fast, slow, signal, norm_window)

    @staticmethod
    def calculate_zscore(series: pd.Series, window: int = 42) -> pd.Series:
        return zscore(series, window)

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        return rsi(series, period)
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3):
        return stochastic_oscillator(high, low, close, k_period, d_period)
    
    @staticmethod
    def calculate_roc(series: pd.Series, period: int = 12) -> pd.Series:
        return rate_of_change(series, period)
    
    @staticmethod
    def calculate_cumulative_volume(volume: pd.Series) -> pd.Series:
        return cumulative_volume(volume)
    
    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        return on_balance_volume(close, volume)
    
    @staticmethod
    def calculate_obv_normalized(close: pd.Series, volume: pd.Series, norm_window: int = 42) -> pd.Series:
        return obv_normalized(close, volume, norm_window)
    
    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
        return bollinger_bands(series, window, num_std)

__all__.append("TechnicalIndicators")
