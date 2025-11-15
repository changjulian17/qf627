"""Coincident indices trading strategies using correlated assets.

Strategies that use related assets (currencies, ETFs, commodities) as signals
for the primary asset.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from .base_strategy import BaseStrategy
from features.technical_indicators import TechnicalIndicators
from config import CURRENCY_INDICES, MARKET_INDICES
from utils.yfinance_cache import get_cached_ticker_data


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
        self._cached_raw_data = None  # Cache for Yahoo Finance data to ensure consistency
        
    def fetch_coincident_data(self, start_date, end_date):
        """Fetch data for the coincident index."""
        # Return in-memory cached data if available (for multiple calls within same run)
        if self._cached_raw_data is not None:
            return self._cached_raw_data
        
        # Use persistent disk cache to ensure consistency across runs
        def _fetch():
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
        
        try:
            close_data = get_cached_ticker_data(
                self.coincident_ticker, start_date, end_date, _fetch
            )
            # Cache in memory for subsequent calls within same run
            self._cached_raw_data = close_data
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
        
        # FIX DATA LEAKAGE: Shift by 1 to use only PAST data
        # Signal at time t should only use data up to time t-1 (not including t)
        # This ensures we don't use today's close price (which isn't known until after market close)
        coincident_signal = coincident_returns.shift(1).rolling(window=self.window).mean()
        
        # Store in data for analysis
        self.data[f'{self.coincident_ticker}_price'] = self.coincident_data
        self.data[f'{self.coincident_ticker}_signal'] = coincident_signal
        
        # Add z-score of price for ML features (stationary version)
        # Also shift to avoid leakage
        ti = TechnicalIndicators()
        self.data[f'{self.coincident_ticker}_price_zscore'] = ti.calculate_zscore(
            self.coincident_data.shift(1), window=self.window
        )
        
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
    Can also optionally exclude individual ticker features to avoid redundancy.
    """
    
    def __init__(self, data, coincident_tickers, window=20, aggregation='mean', 
                 include_individual_features=True, correlation_window=60, name_suffix=""):
        """
        Args:
            data: Price DataFrame for primary asset
            coincident_tickers: List of ticker symbols for coincident indices
            window: Lookback window for calculating signals
            aggregation: How to combine signals ('mean', 'sum', 'majority', 'correlation_weighted')
            include_individual_features: If False, only composite_signal is kept for ML features
            correlation_window: Window for calculating correlations (used for 'correlation_weighted')
            name_suffix: Optional suffix for strategy name
        """
        tickers_str = '_'.join([t.replace('^', '').replace('-', '')[:3] for t in coincident_tickers])
        strategy_name = f'MultiCoinc_{tickers_str}_{window}' + (f'_{name_suffix}' if name_suffix else '')
        super().__init__(strategy_name, data)
        self.coincident_tickers = coincident_tickers
        self.window = window
        self.aggregation = aggregation
        self.include_individual_features = include_individual_features
        self.correlation_window = correlation_window
        self._cached_raw_data = None  # Cache for Yahoo Finance data to ensure consistency
        
    def fetch_coincident_data(self, start_date, end_date):
        """Fetch data for all coincident indices."""
        # Return in-memory cached data if available (for multiple calls within same run)
        if self._cached_raw_data is not None:
            return self._cached_raw_data
        
        coincident_dict = {}
        for ticker in self.coincident_tickers:
            # Use persistent disk cache for each ticker
            def _fetch():
                ticker_obj = yf.Ticker(ticker)
                ticker_data = ticker_obj.history(start=start_date, end=end_date)
                if not ticker_data.empty:
                    close_data = ticker_data['Close']
                    # Remove timezone to match primary data
                    if close_data.index.tz is not None:
                        close_data.index = close_data.index.tz_localize(None)
                    return close_data
                else:
                    print(f"Warning: No data found for {ticker}")
                    return None
            
            try:
                data = get_cached_ticker_data(ticker, start_date, end_date, _fetch)
                if data is not None:
                    coincident_dict[ticker] = data
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        if not coincident_dict:
            return None
        
        result = pd.DataFrame(coincident_dict)
        # Cache in memory for subsequent calls within same run
        self._cached_raw_data = result
        return result
    
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
        
        # Calculate SPY returns for correlation weighting
        price_col = self.data.columns[0]  # First column is price
        spy_returns = np.log(self.data[price_col] / self.data[price_col].shift(1))
        
        # Calculate signals for each coincident index
        signals = pd.DataFrame(index=self.data.index)
        correlations = pd.DataFrame(index=self.data.index)
        
        for ticker in coincident_df.columns:
            # Calculate returns
            returns = np.log(coincident_df[ticker] / coincident_df[ticker].shift(1))
            # FIX DATA LEAKAGE: Shift by 1 to use only PAST data
            # Signal at time t should only use data up to time t-1
            signal = returns.shift(1).rolling(window=self.window).mean()
            # Convert to position: 1 if positive momentum, -1 if negative
            signals[ticker] = np.where(signal > 0, 1, -1)
            
            # Calculate rolling correlation for weighting (also shifted to avoid leakage)
            if self.aggregation == 'correlation_weighted':
                # Use shifted returns to avoid leakage
                corr = returns.shift(1).rolling(window=self.correlation_window).corr(spy_returns.shift(1))
                # Use absolute correlation for weighting (both positive and negative correlations are useful)
                correlations[ticker] = corr.abs()
            
            # Only store individual features if requested (avoid redundancy with CoincidentIndexStrategy)
            if self.include_individual_features:
                # Store raw data
                self.data[f'{ticker}_price'] = coincident_df[ticker]
                self.data[f'{ticker}_signal'] = signal
                
                # Add z-score of price for ML features (stationary version)
                # Also shift to avoid leakage
                from features.technical_indicators import TechnicalIndicators
                ti = TechnicalIndicators()
                self.data[f'{ticker}_price_zscore'] = ti.calculate_zscore(
                    coincident_df[ticker].shift(1), window=self.window
                )
        
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
        elif self.aggregation == 'correlation_weighted':
            # Weighted by historical correlation to SPY returns
            # Normalize correlations to sum to 1 for each row (avoiding division by zero)
            weights = correlations.div(correlations.sum(axis=1), axis=0)
            weights = weights.fillna(1.0 / len(self.coincident_tickers))  # Equal weight if no correlation data
            
            # Weight the signals by correlation
            weighted_signals = signals * weights
            composite_signal = weighted_signals.sum(axis=1)
            
            self.positions = pd.Series(
                np.where(composite_signal > 0, 1, -1),
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

