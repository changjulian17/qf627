"""Performance metric calculations."""
import numpy as np
import pandas as pd

class PerformanceMetrics:
    """Calculate various performance metrics for strategies."""
    
    @staticmethod
    def calculate_sharpe_ratio(returns, periods_per_year=252):
        """Calculate annualized Sharpe ratio."""
        return np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1)
    
    @staticmethod
    def calculate_cagr(cum_returns):
        """Calculate Compound Annual Growth Rate."""
        cum_returns = cum_returns.dropna()
        n_days = (cum_returns.index[-1] - cum_returns.index[0]).days
        cagr = (cum_returns.iloc[-1] / cum_returns.iloc[0]) ** (365.25/n_days) - 1
        return cagr
    
    @staticmethod
    def calculate_max_drawdown(cum_returns):
        """Calculate maximum drawdown."""
        drawdown = cum_returns / cum_returns.cummax() - 1
        return drawdown.min()
    
    @staticmethod
    def calculate_longest_drawdown_duration(cum_returns):
        """Calculate longest drawdown period in days."""
        drawdown = cum_returns / cum_returns.cummax() - 1
        periods = np.diff(
            np.append(drawdown[drawdown == 0].index, drawdown.index[-1:])
        )
        return periods.max() / np.timedelta64(1, "D")
    
    @staticmethod
    def get_drawdown_periods(cum_returns):
        """Get detailed drawdown period statistics."""
        drawdown = cum_returns / cum_returns.cummax() - 1
        dd_reset = pd.DataFrame({'Date': cum_returns.index, 'dd': drawdown.values})
        dd_reset['period'] = (dd_reset['dd'] == 0).cumsum()
        
        dd_nonzero = dd_reset[dd_reset['dd'] != 0]
        period_stats = dd_nonzero.groupby('period').agg(
            start_date=('Date', 'min'),
            end_date=('Date', 'max'),
            max_dd=('dd', 'min'),
            duration=('dd', 'count')
        ).sort_values('max_dd')
        
        return period_stats
    
    @staticmethod
    def calculate_all_metrics(strategy_returns, cum_returns, train_mse=None, test_mse=None):
        """Calculate all performance metrics."""
        pm = PerformanceMetrics()
        
        metrics = {
            'Sharpe Ratio': pm.calculate_sharpe_ratio(strategy_returns),
            'CAGR': pm.calculate_cagr(cum_returns),
            'Max Drawdown': pm.calculate_max_drawdown(cum_returns),
            'Longest DD Duration': pm.calculate_longest_drawdown_duration(cum_returns),
            'Total Return': cum_returns.iloc[-1] - 1,
            'Volatility': strategy_returns.std() * np.sqrt(252),
        }
        
        # Add MSE metrics if provided (for ML strategies)
        if train_mse is not None:
            metrics['Train MSE'] = train_mse
        if test_mse is not None:
            metrics['Test MSE'] = test_mse
        
        return metrics