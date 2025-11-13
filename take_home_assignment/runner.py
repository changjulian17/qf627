"""Runner module: modular orchestration for data load, strategies, backtests and benchmark.

This keeps `main.py` small and testable. Exposes a single `run_all()` function that
returns (results, comparison_df) mirroring the previous `main()` behavior.
"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from config import *
from data.data_loader import DataLoader
from strategies.mean_reversion import RSIStrategy, ZScoreStrategy, generate_rsi_variants
from strategies.momentum import SMAStrategy, MACDStrategy
from strategies.coincident_indices import CoincidentIndexStrategy, MultiCoincidentStrategy
from strategies.multi_window_returns import MultiWindowReturnsStrategy, CrossAssetWindowReturnsStrategy
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics
from strategies.ml_strategies import create_ml_models, MLStrategy


def create_ml_strategies(prices: pd.DataFrame, strategies: list, test_start_date, 
                        tune_hyperparameters=True, checkpoint_dir='checkpoints'):
    """Create multiple ML strategy instances using different models.
    
    Args:
        prices: Price DataFrame
        strategies: List of rule-based strategies to extract features from
        test_start_date: Date to split train/test
        tune_hyperparameters: If True, use GridSearchCV to tune each model's hyperparameters
        checkpoint_dir: Directory to save/load checkpoints
    
    Returns a list of MLStrategy instances.
    """
    ml_strategies = []
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    # Load checkpoint if exists
    checkpoint_file = checkpoint_path / 'ml_strategies_checkpoint.pkl'
    completed_models = set()
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                ml_strategies = checkpoint_data.get('strategies', [])
                completed_models = checkpoint_data.get('completed_models', set())
            print(f"\n✓ Loaded checkpoint: {len(ml_strategies)} strategies already completed")
            print(f"  Completed models: {', '.join(sorted(completed_models))}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            ml_strategies = []
            completed_models = set()
    
    ml_models = create_ml_models(tune_hyperparameters=tune_hyperparameters)
    
    tuning_msg = " with hyperparameter tuning" if tune_hyperparameters else ""
    print(f"\nCreating ML strategies{tuning_msg}...")
    total_models = len(ml_models)
    remaining_models = {k: v for k, v in ml_models.items() if k not in completed_models}
    
    if remaining_models:
        print(f"Progress: {len(completed_models)}/{total_models} models completed, {len(remaining_models)} remaining")
    
    for idx, (model_name, (model_instance, short_name)) in enumerate(remaining_models.items(), 1):
        try:
            print(f"\n[{len(completed_models) + idx}/{total_models}] Training {model_name}...")
            
            ml_strat = MLStrategy(
                prices=prices,
                feature_strategies=strategies,
                test_start_date=test_start_date,
                model=model_instance,
                model_name=short_name
            )
            # Pre-compute to validate
            ml_strat.calculate_returns()
            if ml_strat.data is not None and not ml_strat.data.empty:
                ml_strategies.append(ml_strat)
                completed_models.add(model_name)
                
                # Print best params if tuned
                if tune_hyperparameters and hasattr(ml_strat.model, 'best_params_'):
                    print(f"✓ {ml_strat.name}: best_params={ml_strat.model.best_params_}")
                else:
                    print(f"✓ {ml_strat.name}")
                
                # Save checkpoint after each successful model
                try:
                    checkpoint_data = {
                        'strategies': ml_strategies,
                        'completed_models': completed_models,
                        'timestamp': pd.Timestamp.now()
                    }
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    print(f"  → Checkpoint saved ({len(completed_models)}/{total_models} complete)")
                except Exception as e:
                    print(f"  Warning: Could not save checkpoint: {e}")
            else:
                print(f"✗ {model_name} produced empty data; skipping.")
        except Exception as e:
            print(f"✗ Could not create {model_name} strategy: {e}")
            print(f"  → Progress saved. You can resume by running again.")
    
    # Remove checkpoint file when all done
    if len(completed_models) == total_models and checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print(f"\n✓ All models complete! Checkpoint file removed.")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}")
    
    return ml_strategies


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


def init_strategies(prices: pd.DataFrame, test_start_date=None):
    """Create strategy instances for the given price DataFrame.
    
    If test_start_date is provided, also builds and adds an ML strategy trained on features
    from the rule-based strategies.
    """
    strategies = []
    for short, long in MOMENTUM_PARAMS['sma_short_long_pairs']:
        strategies.append(SMAStrategy(prices, short, long))

    strategies.append(MACDStrategy(prices, *MOMENTUM_PARAMS['macd_params']))

    strategies.append(ZScoreStrategy(prices,
                                     MEAN_REVERSION_PARAMS['zscore_window'],
                                     MEAN_REVERSION_PARAMS['zscore_threshold']))

    rsi_variants = generate_rsi_variants(
        prices,
        periods=MEAN_REVERSION_PARAMS.get('rsi_period'),
        threshold_pairs=MEAN_REVERSION_PARAMS.get('rsi_threshhold_pairs')
    )
    strategies.extend(rsi_variants)
    
    # Add coincident indices strategies
    if COINCIDENT_INDICES_PARAMS.get('enabled', False):
        # Single coincident index strategies
        for ticker, window in COINCIDENT_INDICES_PARAMS.get('single_indices', []):
            try:
                coinc_strat = CoincidentIndexStrategy(
                    prices, 
                    coincident_ticker=ticker,
                    window=window,
                    correlation_threshold=COINCIDENT_INDICES_PARAMS.get('correlation_threshold', 0.0)
                )
                strategies.append(coinc_strat)
            except Exception as e:
                print(f"Warning: Could not create CoincidentIndexStrategy for {ticker}: {e}")
        
        # Multi-coincident index strategies
        for tickers, window, aggregation in COINCIDENT_INDICES_PARAMS.get('multi_indices', []):
            try:
                multi_coinc_strat = MultiCoincidentStrategy(
                    prices,
                    coincident_tickers=tickers,
                    window=window,
                    aggregation=aggregation
                )
                strategies.append(multi_coinc_strat)
            except Exception as e:
                print(f"Warning: Could not create MultiCoincidentStrategy: {e}")
    
    # Add multi-window returns strategies
    if MULTI_WINDOW_PARAMS.get('enabled', False):
        # Same-asset multi-window strategies
        for windows, signal_method in MULTI_WINDOW_PARAMS.get('same_asset_strategies', []):
            try:
                mw_strat = MultiWindowReturnsStrategy(
                    prices,
                    windows=windows,
                    signal_method=signal_method
                )
                strategies.append(mw_strat)
            except Exception as e:
                print(f"Warning: Could not create MultiWindowReturnsStrategy: {e}")
        
        # Cross-asset multi-window strategies
        for ref_ticker, windows, signal_method in MULTI_WINDOW_PARAMS.get('cross_asset_strategies', []):
            try:
                cross_strat = CrossAssetWindowReturnsStrategy(
                    prices,
                    reference_ticker=ref_ticker,
                    windows=windows,
                    signal_method=signal_method
                )
                strategies.append(cross_strat)
            except Exception as e:
                print(f"Warning: Could not create CrossAssetWindowReturnsStrategy for {ref_ticker}: {e}")
    
    # Add ML strategies if test_start_date provided
    if test_start_date is not None:
        tune_hyperparams = ML_PARAMS.get('tune_hyperparameters', True)
        ml_strats = create_ml_strategies(prices, strategies, test_start_date, tune_hyperparams)
        strategies.extend(ml_strats)
    
    return strategies


def run_backtests_on_test_period(strategies, test_start_date):
    """Run backtests for each strategy restricted to the test period. Returns results dict."""
    results = {}
    backtest_engine = BacktestEngine(INITIAL_CAPITAL, COMMISSION_PER_TRADE)

    for strategy in strategies:
        print(f"\nBacktesting {strategy.name}...")
        strategy.calculate_returns()

        test_mask = strategy.data.index >= test_start_date
        test_data = strategy.data.loc[test_mask].copy()

        if test_data.empty:
            print(f"Warning: no test-period rows for strategy {strategy.name}; skipping.")
            continue

        # Recalculate cumulative returns for test period only (starting from 1.0)
        test_data['cum_passive_returns'] = test_data['passive_returns'].cumsum().apply(np.exp)
        test_data['cum_strategy_returns'] = test_data['strategy_returns'].cumsum().apply(np.exp)

        bt_data, final_value, total_return = backtest_engine.run_backtest(test_data)

        metrics = PerformanceMetrics.calculate_all_metrics(
            bt_data['strategy_returns'],
            bt_data['cum_strategy_returns']
        )

        results[strategy.name] = {
            'data': bt_data,
            'final_value': final_value,
            'total_return': total_return,
            'metrics': metrics
        }

    return results


def compute_spy_benchmark(test_start_date):
    """Fetch SPY and compute passive buy-and-hold metrics for the test period.

    Returns a dict {'data': spy_test_df, 'metrics': metrics} or None on failure.
    """
    try:
        spy_loader = DataLoader('SPY', START_DATE, END_DATE)
        spy_prices = spy_loader.get_prices()
        if isinstance(spy_prices, pd.Series):
            spy_prices = spy_prices.to_frame()
    except Exception as e:
        print(f"Warning: could not load SPY benchmark ({e})")
        return None

    try:
        spy_test = spy_prices.loc[spy_prices.index >= test_start_date].copy()
        if spy_test.empty:
            print("Warning: SPY has no data in test period; benchmark not added.")
            return None

        price_col = spy_test.columns[0]
        spy_test['passive_returns'] = (spy_test[price_col] / spy_test[price_col].shift(1)).apply(
            lambda x: np.log(x) if x > 0 else 0
        )
        spy_test['passive_returns'] = spy_test['passive_returns'].fillna(0)
        spy_test['cum_passive_returns'] = spy_test['passive_returns'].cumsum().apply(np.exp)

        spy_metrics = PerformanceMetrics.calculate_all_metrics(
            spy_test['passive_returns'],
            spy_test['cum_passive_returns']
        )

        return {
            'data': spy_test,
            'metrics': spy_metrics
        }
    except Exception as e:
        print(f"Warning computing SPY benchmark metrics: {e}")
        return None


def save_and_print_results(results: dict):
    """Build comparison DataFrame from results, print and save to CSV.

    Returns the comparison DataFrame.
    """
    comparison_df = pd.DataFrame({name: res['metrics'] for name, res in results.items()}).T

    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(comparison_df.to_string())

    comparison_df.to_csv('strategy_comparison.csv')
    print("\nResults saved to strategy_comparison.csv")
    return comparison_df


def run_all():
    prices, test_start_date = load_prices_and_test_start()
    strategies = init_strategies(prices, test_start_date)
    results = run_backtests_on_test_period(strategies, test_start_date)

    spy_res = compute_spy_benchmark(test_start_date)
    if spy_res is not None:
        results['SPY'] = {
            'data': spy_res['data'],
            'final_value': None,
            'total_return': None,
            'metrics': spy_res['metrics']
        }

    comparison_df = save_and_print_results(results)
    return results, comparison_df

