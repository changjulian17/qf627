"""Backtesting engine with commission handling."""
import numpy as np
import pandas as pd

class BacktestEngine:
    """Execute backtests with realistic commission costs."""
    
    def __init__(self, initial_capital=100_000, commission=5):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run_backtest(self, strategy_data):
        """Run backtest with commission costs."""
        capital = self.initial_capital
        portfolio_values = []
        
        for idx, row in strategy_data.iterrows():
            # Apply returns
            capital *= np.exp(row['strategy_returns'])
            
            # Deduct commission on trades
            if row.get('trades', 0) != 0:
                capital -= self.commission
            
            portfolio_values.append(capital)
        
        strategy_data['portfolio_value'] = portfolio_values
        strategy_data['portfolio_returns'] = (
            pd.Series(portfolio_values, index=strategy_data.index).pct_change()
        )
        
        final_value = portfolio_values[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        return strategy_data, final_value, total_return