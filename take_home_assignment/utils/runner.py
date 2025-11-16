"""Runner module: modular orchestration for data load, strategies, backtests and benchmark.

This keeps `main.py` small and testable. Exposes a single `run_all()` function that
returns (results, comparison_df) mirroring the previous `main()` behavior.
"""
import pandas as pd
import numpy as np
from config import *
from data.data_loader import DataLoader
from utils.strategy_factory import init_strategies
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics
from backtesting.results_manager import compute_spy_benchmark, save_and_print_results, save_detailed_results


def load_prices_and_test_start(ticker: str = TICKER):
    """Load price data for ticker and compute time-based test start date.

    Returns: prices (DataFrame), test_start_date (Timestamp)
    """
    print(f"Loading {ticker} data from {START_DATE} to {END_DATE}...")
    loader = DataLoader(ticker, START_DATE, END_DATE)
    prices = loader.get_prices()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    train_pct = float(ML_PARAMS.get('train_test_split', 0.75))
    if not (0.0 < train_pct < 1.0):
        train_pct = 0.75
    split_idx = int(len(prices) * train_pct)
    split_idx = max(1, min(len(prices) - 1, split_idx))
    test_start_date = prices.index[split_idx]
    return prices, test_start_date


def run_backtests_on_test_period(strategies, test_start_date):
    """Run backtests for each strategy restricted to the test period. Returns results dict."""
    results = {}
    backtest_engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION_PER_TRADE)
    
    # Print test period information
    print("\n" + "="*70)
    print("TEST PERIOD INFORMATION")
    print("="*70)
    print(f"Test Start Date: {test_start_date.date()}")
    print(f"Test End Date:   {END_DATE.date()}")
    print(f"Duration:        {(END_DATE - test_start_date).days} days")
    print("="*70)

    for strategy in strategies:
        # print(f"\nBacktesting {strategy.name}...")
        strategy.calculate_returns()

        test_mask = strategy.data.index >= test_start_date
        test_data = strategy.data.loc[test_mask].copy()

        if test_data.empty:
            print(f"Warning: no test-period rows for strategy {strategy.name}; skipping.")
            continue

        # Enforce day-1 alignment: zero-out returns on the first test-period row
        first_idx = test_data.index[0]
        if 'passive_returns' in test_data.columns:
            test_data.loc[first_idx, 'passive_returns'] = 0.0
        if 'strategy_returns' in test_data.columns:
            test_data.loc[first_idx, 'strategy_returns'] = 0.0

        # Recalculate cumulative returns for test period only (starting from 1.0)
        test_data['cum_passive_returns'] = test_data['passive_returns'].cumsum().apply(np.exp)
        test_data['cum_strategy_returns'] = test_data['strategy_returns'].cumsum().apply(np.exp)

        bt_data, final_value, total_return = backtest_engine.run_backtest(test_data)

        # Check if this is an ML strategy with MSE metrics
        train_mse = getattr(strategy, 'train_mse', None)
        test_mse = getattr(strategy, 'test_mse', None)

        metrics = PerformanceMetrics.calculate_all_metrics(
            bt_data['strategy_returns'],
            bt_data['cum_strategy_returns'],
            train_mse=train_mse,
            test_mse=test_mse,
            final_value=final_value
        )

        results[strategy.name] = {
            'data': bt_data,
            'final_value': final_value,
            'total_return': total_return,
            'metrics': metrics
        }

    return results


def run_backtests_on_train_period(strategies, test_start_date):
    """Run backtests for each strategy restricted to the train period. Returns results dict."""
    results = {}
    backtest_engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION_PER_TRADE)

    # Print train period information
    print("\n" + "="*70)
    print("TRAIN PERIOD INFORMATION")
    print("="*70)
    print(f"Train Start Date: {START_DATE.date()}")
    print(f"Train End Date:   {test_start_date.date()}")
    print(f"Duration:        {(test_start_date - START_DATE).days} days")
    print("="*70)

    for strategy in strategies:
        # print(f"\nBacktesting (Train) {strategy.name}...")
        strategy.calculate_returns()

        train_mask = strategy.data.index < test_start_date
        train_data = strategy.data.loc[train_mask].copy()

        if train_data.empty:
            print(f"Warning: no train-period rows for strategy {strategy.name}; skipping.")
            continue

        # Enforce day-1 alignment: zero-out returns on the first train-period row
        first_idx = train_data.index[0]
        if 'passive_returns' in train_data.columns:
            train_data.loc[first_idx, 'passive_returns'] = 0.0
        if 'strategy_returns' in train_data.columns:
            train_data.loc[first_idx, 'strategy_returns'] = 0.0

        # Recalculate cumulative returns for train period only (starting from 1.0)
        train_data['cum_passive_returns'] = train_data['passive_returns'].cumsum().apply(np.exp)
        train_data['cum_strategy_returns'] = train_data['strategy_returns'].cumsum().apply(np.exp)

        bt_data, final_value, total_return = backtest_engine.run_backtest(train_data)

        # Check if this is an ML strategy with MSE metrics
        train_mse = getattr(strategy, 'train_mse', None)
        test_mse = getattr(strategy, 'test_mse', None)

        metrics = PerformanceMetrics.calculate_all_metrics(
            bt_data['strategy_returns'],
            bt_data['cum_strategy_returns'],
            train_mse=train_mse,
            test_mse=test_mse,
            final_value=final_value
        )

        results[strategy.name] = {
            'data': bt_data,
            'final_value': final_value,
            'total_return': total_return,
            'metrics': metrics
        }

    return results


