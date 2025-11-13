"""Multi-window returns strategies.

Strategies that compute returns over multiple lookback windows and use them
as features or signals for trading.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from .base_strategy import BaseStrategy


class MultiWindowReturnsStrategy(BaseStrategy):
    """Strategy based on returns calculated over multiple time windows.
    
    Computes returns over various lookback periods and generates signals based on
    momentum patterns across different time horizons.
    """
    
    def __init__(self, data, windows=[5, 10, 20, 60], signal_method='weighted_average', name_suffix=""):
        """
        Args:
            data: Price DataFrame for the asset
            windows: List of lookback windows (in days) for computing returns
            signal_method: Method to generate signals from multi-window returns
                          ('weighted_average', 'majority_vote', 'all_positive', 'trend_alignment')
            name_suffix: Optional suffix for strategy name
        """
        windows_str = '_'.join(map(str, windows))
        strategy_name = f'MultiWindow_{signal_method}_{windows_str}' + (f'_{name_suffix}' if name_suffix else '')
        super().__init__(strategy_name, data)
        self.windows = sorted(windows)
        self.signal_method = signal_method
        
    def generate_signals(self):
        """Generate signals based on multi-window returns."""
        price_col = self.data.columns[0]
        
        # Calculate returns for each window
        returns_dict = {}
        for window in self.windows:
            # Log returns over the lookback window
            ret = np.log(self.data[price_col] / self.data[price_col].shift(window))
            returns_dict[f'ret_{window}d'] = ret
            self.data[f'ret_{window}d'] = ret
        
        returns_df = pd.DataFrame(returns_dict, index=self.data.index)
        
        # Generate signals based on chosen method
        if self.signal_method == 'weighted_average':
            # Weight longer-term returns more heavily
            weights = np.array([1/w for w in self.windows])
            weights = weights / weights.sum()  # Normalize
            
            weighted_signal = sum(returns_df[f'ret_{w}d'] * weight 
                                for w, weight in zip(self.windows, weights))
            
            self.positions = pd.Series(
                np.where(weighted_signal > 0, 1, -1),
                index=self.data.index
            )
            self.data['weighted_signal'] = weighted_signal
            
        elif self.signal_method == 'majority_vote':
            # Each window votes: +1 for positive return, -1 for negative
            votes = pd.DataFrame(index=self.data.index)
            for window in self.windows:
                votes[f'vote_{window}'] = np.where(returns_df[f'ret_{window}d'] > 0, 1, -1)
            
            majority_signal = votes.sum(axis=1)
            self.positions = pd.Series(
                np.where(majority_signal > 0, 1, -1),
                index=self.data.index
            )
            self.data['majority_signal'] = majority_signal
            
        elif self.signal_method == 'all_positive':
            # Go long only if ALL windows show positive returns, short if ALL negative
            all_positive = returns_df.gt(0).all(axis=1)
            all_negative = returns_df.lt(0).all(axis=1)
            
            self.positions = pd.Series(
                np.where(all_positive, 1, 
                        np.where(all_negative, -1, 0)),
                index=self.data.index
            )
            self.data['all_positive'] = all_positive
            self.data['all_negative'] = all_negative
            
        elif self.signal_method == 'trend_alignment':
            # Check if windows show aligned trend (short < medium < long term returns)
            # This indicates accelerating momentum
            if len(self.windows) >= 3:
                # Check if returns are monotonically increasing across windows
                aligned_up = True
                aligned_down = True
                
                for i in range(len(self.windows) - 1):
                    curr_col = f'ret_{self.windows[i]}d'
                    next_col = f'ret_{self.windows[i+1]}d'
                    
                    if i == 0:
                        aligned_up = returns_df[curr_col] < returns_df[next_col]
                        aligned_down = returns_df[curr_col] > returns_df[next_col]
                    else:
                        aligned_up = aligned_up & (returns_df[curr_col] < returns_df[next_col])
                        aligned_down = aligned_down & (returns_df[curr_col] > returns_df[next_col])
                
                self.positions = pd.Series(
                    np.where(aligned_up, 1, 
                            np.where(aligned_down, -1, 0)),
                    index=self.data.index
                )
                self.data['aligned_up'] = aligned_up
                self.data['aligned_down'] = aligned_down
            else:
                # Fall back to simple average if not enough windows
                avg_signal = returns_df.mean(axis=1)
                self.positions = pd.Series(
                    np.where(avg_signal > 0, 1, -1),
                    index=self.data.index
                )
                self.data['avg_signal'] = avg_signal
        else:
            raise ValueError(f"Unknown signal_method: {self.signal_method}")
        
        # Fill NaN positions with 0 (neutral)
        self.positions = self.positions.fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class CrossAssetWindowReturnsStrategy(BaseStrategy):
    """Strategy using multi-window returns from a different asset as signals.
    
    Similar to MultiWindowReturnsStrategy but uses returns from a coincident asset
    (like SPY, QQQ, or sector ETFs) to generate signals for the primary asset.
    """
    
    def __init__(self, data, reference_ticker='SPY', windows=[5, 10, 20, 60], 
                 signal_method='weighted_average', name_suffix=""):
        """
        Args:
            data: Price DataFrame for primary asset
            reference_ticker: Ticker symbol for reference asset (e.g., 'SPY', 'QQQ')
            windows: List of lookback windows (in days) for computing returns
            signal_method: Method to generate signals from multi-window returns
            name_suffix: Optional suffix for strategy name
        """
        windows_str = '_'.join(map(str, windows))
        strategy_name = f'CrossAsset_{reference_ticker}_{signal_method}_{windows_str}' + (f'_{name_suffix}' if name_suffix else '')
        super().__init__(strategy_name, data)
        self.reference_ticker = reference_ticker
        self.windows = sorted(windows)
        self.signal_method = signal_method
        self.reference_data = None
        
    def fetch_reference_data(self, start_date, end_date):
        """Fetch data for the reference asset."""
        try:
            ticker = yf.Ticker(self.reference_ticker)
            ref_data = ticker.history(start=start_date, end=end_date)
            if ref_data.empty:
                print(f"Warning: No data found for {self.reference_ticker}")
                return None
            # Remove timezone to match primary data
            close_data = ref_data['Close']
            if close_data.index.tz is not None:
                close_data.index = close_data.index.tz_localize(None)
            return close_data
        except Exception as e:
            print(f"Error fetching {self.reference_ticker}: {e}")
            return None
    
    def generate_signals(self):
        """Generate signals based on reference asset's multi-window returns."""
        # Determine date range (add buffer for longest window)
        max_window = max(self.windows)
        start_date = self.data.index[0] - pd.Timedelta(days=max_window * 2)
        end_date = self.data.index[-1]
        
        # Fetch reference data
        self.reference_data = self.fetch_reference_data(start_date, end_date)
        
        if self.reference_data is None or self.reference_data.empty:
            print(f"Warning: Could not fetch {self.reference_ticker} data. Using neutral positions.")
            self.positions = pd.Series(0, index=self.data.index)
            self.data['positions'] = self.positions
            self.data['trades'] = 0
            return self.positions
        
        # Align reference data with primary data index
        self.reference_data = self.reference_data.reindex(self.data.index, method='ffill')
        self.data[f'{self.reference_ticker}_price'] = self.reference_data
        
        # Calculate returns for each window on reference asset
        returns_dict = {}
        for window in self.windows:
            ret = np.log(self.reference_data / self.reference_data.shift(window))
            returns_dict[f'{self.reference_ticker}_ret_{window}d'] = ret
            self.data[f'{self.reference_ticker}_ret_{window}d'] = ret
        
        returns_df = pd.DataFrame(returns_dict, index=self.data.index)
        
        # Generate signals using same logic as MultiWindowReturnsStrategy
        if self.signal_method == 'weighted_average':
            weights = np.array([1/w for w in self.windows])
            weights = weights / weights.sum()
            
            weighted_signal = sum(returns_df.iloc[:, i] * weight 
                                for i, weight in enumerate(weights))
            
            self.positions = pd.Series(
                np.where(weighted_signal > 0, 1, -1),
                index=self.data.index
            )
            self.data['weighted_signal'] = weighted_signal
            
        elif self.signal_method == 'majority_vote':
            votes = (returns_df > 0).astype(int) * 2 - 1  # Convert True/False to 1/-1
            majority_signal = votes.sum(axis=1)
            
            self.positions = pd.Series(
                np.where(majority_signal > 0, 1, -1),
                index=self.data.index
            )
            self.data['majority_signal'] = majority_signal
            
        elif self.signal_method == 'all_positive':
            all_positive = returns_df.gt(0).all(axis=1)
            all_negative = returns_df.lt(0).all(axis=1)
            
            self.positions = pd.Series(
                np.where(all_positive, 1, 
                        np.where(all_negative, -1, 0)),
                index=self.data.index
            )
            
        else:  # Default to simple average
            avg_signal = returns_df.mean(axis=1)
            self.positions = pd.Series(
                np.where(avg_signal > 0, 1, -1),
                index=self.data.index
            )
            self.data['avg_signal'] = avg_signal
        
        self.positions = self.positions.fillna(0)
        
        self.trades = self.positions.diff().abs()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions
