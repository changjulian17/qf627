"""Mean reversion trading strategies."""
import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
from features.technical_indicators import TechnicalIndicators

class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy.

    This class implements a single permutation of RSI parameters (period,
    oversold, overbought). To create multiple permutations (different periods
    and thresholds), use the helper `generate_rsi_variants` provided below.
    """

    def __init__(self, data, period=14, oversold=30, overbought=70):
        # give each instance a descriptive name so backtests/comparisons are clear
        name = f"RSI_p{period}_os{oversold}_ob{overbought}"
        super().__init__(name, data)
        self.period = int(period)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
    
    def generate_signals(self):
        """Generate RSI-based signals."""
        ti = TechnicalIndicators()
        
        self.data['rsi'] = ti.calculate_rsi(
            self.data[self.data.columns[0]], self.period
        )
        self.data = self.data.dropna()
        
        # Long when crossing above oversold, short when crossing below overbought
        self.positions = pd.Series(
            np.where(
                (self.data['rsi'].shift(1) < self.oversold) & 
                (self.data['rsi'] > self.oversold), 1,
                np.where(
                    (self.data['rsi'].shift(1) > self.overbought) & 
                    (self.data['rsi'] < self.overbought), -1, 
                    np.nan
                )
            ),
            index=self.data.index
        ).ffill().fillna(0)
        
        self.trades = self.positions.diff()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class ZScoreStrategy(BaseStrategy):
    """Z-score based mean reversion strategy."""
    
    def __init__(self, data, window=42, threshold=2):
        super().__init__(f'ZScore_{window}', data)
        self.window = window
        self.threshold = threshold
    
    def generate_signals(self):
        """Generate Z-score based signals."""
        ti = TechnicalIndicators()
        
        self.data['zscore'] = ti.calculate_zscore(
            self.data[self.data.columns[0]], self.window
        )
        self.data = self.data.dropna()
        
        # Long when z < -threshold, short when z > threshold, exit at 0
        positions = np.where(
            self.data['zscore'] > self.threshold, -1,
            np.where(self.data['zscore'] < -self.threshold, 1, np.nan)
        )
        
        # Exit when crossing zero
        positions = np.where(
            self.data['zscore'] * self.data['zscore'].shift(1) < 0, 
            0, positions
        )
        
        self.positions = pd.Series(positions, index=self.data.index).ffill()
        self.trades = self.positions.diff()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band mean reversion strategy.
    
    This strategy uses Bollinger Bands to identify overbought and oversold conditions.
    It goes long when price touches the lower band and short when price touches the upper band,
    exiting when price returns to the middle band.
    """
    
    def __init__(self, data, window=20, num_std=2.0):
        name = f"BB_w{window}_std{num_std}"
        super().__init__(name, data)
        self.window = int(window)
        self.num_std = float(num_std)
    
    def generate_signals(self):
        """Generate Bollinger Band based signals."""
        ti = TechnicalIndicators()
        
        # Calculate Bollinger Bands
        middle, upper, lower = ti.calculate_bollinger_bands(
            self.data[self.data.columns[0]], 
            self.window, 
            self.num_std
        )
        
        self.data['bb_middle'] = middle
        self.data['bb_upper'] = upper
        self.data['bb_lower'] = lower
        self.data = self.data.dropna()
        
        price = self.data[self.data.columns[0]]
        
        # Signal logic:
        # Long when price crosses below lower band (oversold)
        # Short when price crosses above upper band (overbought)
        # Exit when price crosses middle band
        
        # Long signal: price touches or crosses below lower band
        long_entry = (price <= self.data['bb_lower'])
        # Short signal: price touches or crosses above upper band
        short_entry = (price >= self.data['bb_upper'])
        
        # Exit signals: price crosses middle band
        exit_long = (price >= self.data['bb_middle']) & (price.shift(1) < self.data['bb_middle'].shift(1))
        exit_short = (price <= self.data['bb_middle']) & (price.shift(1) > self.data['bb_middle'].shift(1))
        
        # Initialize positions
        positions = pd.Series(np.nan, index=self.data.index)
        
        # Set entry positions
        positions[long_entry] = 1
        positions[short_entry] = -1
        
        # Set exit positions
        positions[exit_long | exit_short] = 0
        
        # Forward fill to maintain positions
        self.positions = positions.ffill().fillna(0)
        self.trades = self.positions.diff()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


def generate_rsi_variants(data, periods=None, threshold_pairs=None):
    """Generate a list of RSIStrategy instances for all combinations of
    `periods` and `threshold_pairs`.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Price series or single-column DataFrame to pass to strategy instances.
    periods : list[int]
        List of RSI lookback periods to use. Default: [14, 21].
    threshold_pairs : list[tuple]
        List of (oversold, overbought) tuples. Default: [(30, 70), (25, 75)].

    Returns
    -------
    list[RSIStrategy]
        Strategy instances ready to be backtested.
    """
    import pandas as pd

    if periods is None:
        periods = [14, 21]
    if threshold_pairs is None:
        threshold_pairs = [(30, 70), (25, 75)]

    variants = []
    # if user passed a Series, make it a DataFrame with a single column
    if isinstance(data, pd.Series):
        price_df = data.to_frame()
    else:
        price_df = data.copy()

    for p in periods:
        for (os_val, ob_val) in threshold_pairs:
            variants.append(RSIStrategy(price_df, period=p, oversold=os_val, overbought=ob_val))

    return variants

def generate_bollinger_variants(data, windows=None, std_devs=None):
    """Generate a list of BollingerBandStrategy instances for all combinations of
    `windows` and `std_devs`.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Price series or single-column DataFrame to pass to strategy instances.
    windows : list[int]
        List of Bollinger Band lookback windows to use. Default: [20, 30].
    std_devs : list[float]
        List of standard deviation multipliers. Default: [1.5, 2.0, 2.5].

    Returns
    -------
    list[BollingerBandStrategy]
        Strategy instances ready to be backtested.
    """
    import pandas as pd

    if windows is None:
        windows = [20, 30]
    if std_devs is None:
        std_devs = [1.5, 2.0, 2.5]

    variants = []
    # if user passed a Series, make it a DataFrame with a single column
    if isinstance(data, pd.Series):
        price_df = data.to_frame()
    else:
        price_df = data.copy()

    for w in windows:
        for std in std_devs:
            variants.append(BollingerBandStrategy(price_df, window=w, num_std=std))

    return variants
