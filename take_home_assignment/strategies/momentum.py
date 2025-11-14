"""Momentum-based trading strategies."""
import numpy as np
import pandas as pd
import yfinance as yf
from .base_strategy import BaseStrategy
from features.technical_indicators import TechnicalIndicators

class SMAStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy."""
    
    def __init__(self, data, short_window=20, long_window=60):
        super().__init__(f'SMA_{short_window}_{long_window}', data)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self):
        """Generate SMA crossover signals using normalized spread."""
        ti = TechnicalIndicators()
        
        price = self.data[self.data.columns[0]]
        
        # Calculate SMAs
        sma_short = ti.calculate_sma(price, self.short_window)
        sma_long = ti.calculate_sma(price, self.long_window)
        
        # Calculate spread and normalize to z-score for stationarity
        spread = sma_short - sma_long
        spread_zscore = ti.calculate_zscore(spread, window=42)
        
        self.data[f'sma_{self.short_window}'] = sma_short
        self.data[f'sma_{self.long_window}'] = sma_long
        self.data['spread_zscore'] = spread_zscore
        
        self.data = self.data.dropna()
        
        # Use z-score of spread: long when > 0, short when < 0
        self.positions = pd.Series(
            np.where(self.data['spread_zscore'] > 0, 1, -1),
            index=self.data.index
        )
        
        self.trades = self.positions.diff()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class MACDStrategy(BaseStrategy):
    """MACD crossover strategy."""
    
    def __init__(self, data, fast=12, slow=26, signal=9):
        super().__init__(f'MACD_{fast}_{slow}_{signal}', data)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self):
        """Generate MACD crossover signals."""
        ti = TechnicalIndicators()
        
        # Use normalized MACD for stationary oscillator
        macd, signal_line = ti.calculate_macd_normalized(
            self.data[self.data.columns[0]], 
            self.fast, self.slow, self.signal,
            norm_window=42  # Z-score normalization window
        )
        
        self.data['macd'] = macd
        self.data['signal_line'] = signal_line
        self.data = self.data.dropna()
        
        self.positions = pd.Series(
            np.where(self.data['macd'] > self.data['signal_line'], 1, -1),
            index=self.data.index
        )
        
        self.trades = self.positions.diff()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator (%K, %D) strategy.
    
    Goes long when %K crosses above %D and both are below oversold threshold,
    goes short when %K crosses below %D and both are above overbought threshold.
    """
    
    def __init__(self, data, ticker='SPY', k_period=14, d_period=3, 
                 oversold=20, overbought=80):
        """
        Args:
            data: Price DataFrame
            ticker: Ticker symbol to fetch OHLC data for
            k_period: Period for %K calculation
            d_period: Period for %D (SMA of %K)
            oversold: Oversold threshold (default 20)
            overbought: Overbought threshold (default 80)
        """
        super().__init__(f'Stoch_{ticker}_k{k_period}_d{d_period}_os{oversold}_ob{overbought}', data)
        self.ticker = ticker
        self.k_period = k_period
        self.d_period = d_period
        self.oversold = oversold
        self.overbought = overbought
        self.ohlc_data = None
    
    def fetch_ohlc_data(self, start_date, end_date):
        """Fetch OHLC data for the ticker."""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            hist_data = ticker_obj.history(start=start_date, end=end_date)
            if hist_data.empty:
                print(f"Warning: No OHLC data found for {self.ticker}")
                return None
            # Remove timezone to match price data
            if hist_data.index.tz is not None:
                hist_data.index = hist_data.index.tz_localize(None)
            return hist_data[['High', 'Low', 'Close']]
        except Exception as e:
            print(f"Error fetching OHLC data for {self.ticker}: {e}")
            return None
    
    def generate_signals(self):
        """Generate Stochastic Oscillator signals."""
        # Fetch OHLC data
        start_date = self.data.index[0] - pd.Timedelta(days=self.k_period * 2)
        end_date = self.data.index[-1]
        
        self.ohlc_data = self.fetch_ohlc_data(start_date, end_date)
        
        if self.ohlc_data is None or self.ohlc_data.empty:
            print(f"Warning: Could not fetch OHLC data for {self.ticker}. Using neutral positions.")
            self.positions = pd.Series(0, index=self.data.index)
            self.data['positions'] = self.positions
            self.data['trades'] = 0
            return self.positions
        
        # Align OHLC data with price data
        self.ohlc_data = self.ohlc_data.reindex(self.data.index, method='ffill')
        
        ti = TechnicalIndicators()
        
        # Calculate %K and %D
        k_percent, d_percent = ti.calculate_stochastic(
            self.ohlc_data['High'],
            self.ohlc_data['Low'],
            self.ohlc_data['Close'],
            self.k_period,
            self.d_period
        )
        
        # Store in data
        self.data['k_percent'] = k_percent
        self.data['d_percent'] = d_percent
        
        # Generate signals
        # Long when %K crosses above %D in oversold region
        # Short when %K crosses below %D in overbought region
        k_above_d = k_percent > d_percent
        k_below_d = k_percent < d_percent
        
        in_oversold = (k_percent < self.oversold) & (d_percent < self.oversold)
        in_overbought = (k_percent > self.overbought) & (d_percent > self.overbought)
        
        self.positions = pd.Series(0, index=self.data.index)
        self.positions[k_above_d & in_oversold] = 1
        self.positions[k_below_d & in_overbought] = -1
        
        # Forward fill positions
        self.positions = self.positions.replace(0, np.nan).ffill().fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class ROCStrategy(BaseStrategy):
    """Rate of Change (ROC) momentum strategy.
    
    Goes long when ROC is positive and accelerating,
    goes short when ROC is negative and decelerating.
    """
    
    def __init__(self, data, period=12, threshold=0):
        """
        Args:
            data: Price DataFrame
            period: Lookback period for ROC calculation
            threshold: Threshold for signal generation (default 0)
        """
        super().__init__(f'ROC_p{period}_th{threshold}', data)
        self.period = period
        self.threshold = threshold
    
    def generate_signals(self):
        """Generate ROC signals."""
        ti = TechnicalIndicators()
        
        price_col = self.data.columns[0]
        
        # Calculate ROC
        roc = ti.calculate_roc(self.data[price_col], self.period)
        
        self.data['roc'] = roc
        self.data = self.data.dropna()
        
        # Generate positions: long when ROC > threshold, short when ROC < -threshold
        # Use the cleaned data index after dropna
        self.positions = pd.Series(
            np.where(self.data['roc'] > self.threshold, 1, 
                    np.where(self.data['roc'] < -self.threshold, -1, 0)),
            index=self.data.index
        )
        
        # Forward fill to avoid too many neutral positions
        self.positions = self.positions.replace(0, np.nan).ffill().fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions
