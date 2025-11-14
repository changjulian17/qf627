"""Volume-based trading strategies."""
import numpy as np
import pandas as pd
import yfinance as yf
from .base_strategy import BaseStrategy
from features.technical_indicators import TechnicalIndicators


class OBVStrategy(BaseStrategy):
    """Strategy based on On-Balance Volume (OBV).
    
    Uses price direction to estimate buying/selling pressure:
    - When close > previous close: add volume to OBV (accumulation)
    - When close < previous close: subtract volume from OBV (distribution)
    
    Goes long when OBV is trending up (above its moving average),
    short when trending down.
    """
    
    def __init__(self, data, ticker='SPY', window=20):
        """
        Args:
            data: Price DataFrame
            ticker: Ticker symbol to fetch OHLC data for
            window: Window for moving average of OBV
        """
        super().__init__(f'OBV_{ticker}_{window}', data)
        self.ticker = ticker
        self.window = window
        self.ticker_data = None
        
    def fetch_ticker_data(self, start_date, end_date):
        """Fetch OHLC data for the ticker."""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            hist_data = ticker_obj.history(start=start_date, end=end_date)
            if hist_data.empty:
                print(f"Warning: No data found for {self.ticker}")
                return None
            # Remove timezone to match price data
            if hist_data.index.tz is not None:
                hist_data.index = hist_data.index.tz_localize(None)
            return hist_data
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return None
    
    def generate_signals(self):
        """Generate signals based on OBV trend."""
        # Fetch ticker data
        start_date = self.data.index[0] - pd.Timedelta(days=self.window * 2)
        end_date = self.data.index[-1]
        
        self.ticker_data = self.fetch_ticker_data(start_date, end_date)
        
        if self.ticker_data is None or self.ticker_data.empty:
            print(f"Warning: Could not fetch data for {self.ticker}. Using neutral positions.")
            self.positions = pd.Series(0, index=self.data.index)
            self.data['positions'] = self.positions
            self.data['trades'] = 0
            return self.positions
        
        # Align ticker data with price data
        self.ticker_data = self.ticker_data.reindex(self.data.index, method='ffill')
        
        ti = TechnicalIndicators()
        
        # Extract close and volume
        close = self.ticker_data['Close']
        volume = self.ticker_data['Volume']
        
        # Calculate normalized OBV (z-score for stationarity)
        obv_norm = ti.calculate_obv_normalized(close, volume, norm_window=42)
        
        # Store in data
        self.data['volume'] = volume
        self.data['obv_zscore'] = obv_norm
        
        # Generate positions: long when OBV z-score > 0, short when below
        self.positions = pd.Series(
            np.where(obv_norm > 0, 1, -1),
            index=self.data.index
        )
        
        self.positions = self.positions.fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions

