"""Base strategy class for all trading strategies."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name, data):
        self.name = name
        self.data = data.copy()
        self.positions = None
        self.trades = None
        self.returns = None
    
    @abstractmethod
    def generate_signals(self):
        """Generate trading signals. Must be implemented by subclasses."""
        pass
    
    def calculate_returns(self):
        """Calculate strategy returns."""
        # Always regenerate signals to ensure consistency with current price data
        # This is important when strategies are reused across multiple runs or feature building
        self.generate_signals()
        
        # Passive returns
        self.data['passive_returns'] = np.log(
            self.data[self.data.columns[0]] / 
            self.data[self.data.columns[0]].shift(1)
        ).fillna(0)
        
        # Strategy returns
        self.data['strategy_returns'] = (
            self.data['passive_returns'] * 
            self.positions.shift(1).fillna(0)
        )
        
        # Cumulative returns
        self.data['cum_passive_returns'] = (
            self.data['passive_returns'].cumsum().apply(np.exp)
        )
        self.data['cum_strategy_returns'] = (
            self.data['strategy_returns'].cumsum().apply(np.exp)
        )
        
        return self.data
    
    def get_positions(self):
        """Return positions DataFrame."""
        if self.positions is None:
            self.generate_signals()
        return self.positions