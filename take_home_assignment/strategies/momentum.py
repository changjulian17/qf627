"""Momentum-based trading strategies."""
import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
from features.technical_indicators import TechnicalIndicators

class SMAStrategy(BaseStrategy):
    """Simple Moving Average crossover strategy."""
    
    def __init__(self, data, short_window=20, long_window=60):
        super().__init__(f'SMA_{short_window}_{long_window}', data)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self):
        """Generate SMA crossover signals."""
        ti = TechnicalIndicators()
        
        self.data[f'sma_{self.short_window}'] = ti.calculate_sma(
            self.data[self.data.columns[0]], self.short_window
        )
        self.data[f'sma_{self.long_window}'] = ti.calculate_sma(
            self.data[self.data.columns[0]], self.long_window
        )
        
        self.data = self.data.dropna()
        
        self.positions = pd.Series(
            np.where(
                self.data[f'sma_{self.short_window}'] > 
                self.data[f'sma_{self.long_window}'], 
                1, -1
            ),
            index=self.data.index
        )
        
        self.trades = self.positions.diff()
        self.data['positions'] = self.positions
        self.data['trades'] = self.trades
        
        return self.positions


class MACDStrategy(BaseStrategy):
    """MACD crossover strategy."""
    
    def __init__(self, data, fast=12, slow=26, signal=9):
        super().__init__('MACD', data)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def generate_signals(self):
        """Generate MACD crossover signals."""
        ti = TechnicalIndicators()
        
        macd, signal_line = ti.calculate_macd(
            self.data[self.data.columns[0]], 
            self.fast, self.slow, self.signal
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