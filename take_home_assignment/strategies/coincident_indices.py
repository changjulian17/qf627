"""Coincident indices trading strategies using correlated assets.

Strategies that use related assets (currencies, ETFs, commodities) as signals
for the primary asset.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from .base_strategy import BaseStrategy
from config import CURRENCY_INDICES, MARKET_INDICES


class CoincidentIndexStrategy(BaseStrategy):
    """Strategy based on coincident economic/market indices.
    
    Uses price movements or returns from correlated assets (e.g., DXY for USD strength,
    VIX for volatility, GLD for gold, TLT for bonds) to generate trading signals.
    """
    
    def __init__(self, data, coincident_ticker, window=20, correlation_threshold=0.0, name_suffix=""):
        """
        Args:
            data: Price DataFrame for primary asset
            coincident_ticker: Ticker symbol for coincident index (e.g., 'DX-Y.NYB', 'GLD', '^VIX')
            window: Lookback window for calculating correlations/signals
            correlation_threshold: Threshold for signal generation based on coincident index returns
            name_suffix: Optional suffix for strategy name
        """
        strategy_name = f'Coincident_{coincident_ticker}_{window}' + (f'_{name_suffix}' if name_suffix else '')
        super().__init__(strategy_name, data)
        self.coincident_ticker = coincident_ticker
        self.window = window
        self.correlation_threshold = correlation_threshold
        self.coincident_data = None
        
    def fetch_coincident_data(self, start_date, end_date):
        """Fetch data for the coincident index."""
        try:
            ticker = yf.Ticker(self.coincident_ticker)
            coincident = ticker.history(start=start_date, end=end_date)
            if coincident.empty:
                print(f"Warning: No data found for {self.coincident_ticker}")
                return None
            # Remove timezone to match primary data
            close_data = coincident['Close']
            if close_data.index.tz is not None:
                close_data.index = close_data.index.tz_localize(None)
            return close_data
        except Exception as e:
            print(f"Error fetching {self.coincident_ticker}: {e}")
            return None
    
    def generate_signals(self):
        """Generate signals based on coincident index momentum."""
        # Determine date range from primary data
        start_date = self.data.index[0] - pd.Timedelta(days=self.window * 2)
        end_date = self.data.index[-1]
        
        # Fetch coincident data
        self.coincident_data = self.fetch_coincident_data(start_date, end_date)
        
        if self.coincident_data is None or self.coincident_data.empty:
            print(f"Warning: Could not fetch {self.coincident_ticker} data. Using neutral positions.")
            self.positions = pd.Series(0, index=self.data.index)
            self.data['positions'] = self.positions
            self.data['trades'] = 0
            return self.positions
        
        # Align coincident data with primary data index
        self.coincident_data = self.coincident_data.reindex(self.data.index, method='ffill')
        
        # Calculate returns for coincident index
        coincident_returns = np.log(self.coincident_data / self.coincident_data.shift(1))
        
        # Calculate rolling mean return of coincident index
        coincident_signal = coincident_returns.rolling(window=self.window).mean()
        
        # Store in data for analysis
        self.data[f'{self.coincident_ticker}_price'] = self.coincident_data
        self.data[f'{self.coincident_ticker}_signal'] = coincident_signal
        
        # Generate positions: long when coincident index momentum is positive, short when negative
        self.positions = pd.Series(
            np.where(coincident_signal > self.correlation_threshold, 1, -1),
            index=self.data.index
        )
        
        # Fill any NaN positions with 0 (neutral)
        self.positions = self.positions.fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class MultiCoincidentStrategy(BaseStrategy):
    """Strategy combining multiple coincident indices.
    
    Aggregates signals from multiple correlated assets to generate a composite signal.
    """
    
    def __init__(self, data, coincident_tickers, window=20, aggregation='mean', name_suffix=""):
        """
        Args:
            data: Price DataFrame for primary asset
            coincident_tickers: List of ticker symbols for coincident indices
            window: Lookback window for calculating signals
            aggregation: How to combine signals ('mean', 'sum', 'majority')
            name_suffix: Optional suffix for strategy name
        """
        tickers_str = '_'.join([t.replace('^', '').replace('-', '')[:3] for t in coincident_tickers])
        strategy_name = f'MultiCoinc_{tickers_str}_{window}' + (f'_{name_suffix}' if name_suffix else '')
        super().__init__(strategy_name, data)
        self.coincident_tickers = coincident_tickers
        self.window = window
        self.aggregation = aggregation
        
    def fetch_coincident_data(self, start_date, end_date):
        """Fetch data for all coincident indices."""
        coincident_dict = {}
        for ticker in self.coincident_tickers:
            try:
                ticker_obj = yf.Ticker(ticker)
                ticker_data = ticker_obj.history(start=start_date, end=end_date)
                if not ticker_data.empty:
                    close_data = ticker_data['Close']
                    # Remove timezone to match primary data
                    if close_data.index.tz is not None:
                        close_data.index = close_data.index.tz_localize(None)
                    coincident_dict[ticker] = close_data
                else:
                    print(f"Warning: No data found for {ticker}")
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        if not coincident_dict:
            return None
        
        return pd.DataFrame(coincident_dict)
    
    def generate_signals(self):
        """Generate composite signals from multiple coincident indices."""
        # Determine date range
        start_date = self.data.index[0] - pd.Timedelta(days=self.window * 2)
        end_date = self.data.index[-1]
        
        # Fetch all coincident data
        coincident_df = self.fetch_coincident_data(start_date, end_date)
        
        if coincident_df is None or coincident_df.empty:
            print(f"Warning: Could not fetch coincident data. Using neutral positions.")
            self.positions = pd.Series(0, index=self.data.index)
            self.data['positions'] = self.positions
            self.data['trades'] = 0
            return self.positions
        
        # Align with primary data index
        coincident_df = coincident_df.reindex(self.data.index, method='ffill')
        
        # Calculate signals for each coincident index
        signals = pd.DataFrame(index=self.data.index)
        
        for ticker in coincident_df.columns:
            # Calculate returns
            returns = np.log(coincident_df[ticker] / coincident_df[ticker].shift(1))
            # Rolling mean as signal
            signal = returns.rolling(window=self.window).mean()
            # Convert to position: 1 if positive momentum, -1 if negative
            signals[ticker] = np.where(signal > 0, 1, -1)
            
            # Store raw data
            self.data[f'{ticker}_price'] = coincident_df[ticker]
            self.data[f'{ticker}_signal'] = signal
        
        # Aggregate signals
        if self.aggregation == 'mean':
            # Average of all signals (can be fractional)
            composite_signal = signals.mean(axis=1)
            self.positions = pd.Series(
                np.where(composite_signal > 0, 1, -1),
                index=self.data.index
            )
        elif self.aggregation == 'sum':
            # Sum of all signals
            composite_signal = signals.sum(axis=1)
            self.positions = pd.Series(
                np.where(composite_signal > 0, 1, -1),
                index=self.data.index
            )
        elif self.aggregation == 'majority':
            # Majority vote
            composite_signal = signals.sum(axis=1)
            self.positions = pd.Series(
                np.where(composite_signal > 0, 1, 
                        np.where(composite_signal < 0, -1, 0)),
                index=self.data.index
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        self.data['composite_signal'] = composite_signal
        self.positions = self.positions.fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions

