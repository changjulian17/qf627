"""Strategy factory for creating trading strategy instances.

Centralizes strategy instantiation logic to keep orchestration code clean.
"""
from __future__ import annotations
import pandas as pd
from typing import List
from config import (
    MOMENTUM_PARAMS, MEAN_REVERSION_PARAMS, VOLUME_OSCILLATOR_PARAMS,
    COINCIDENT_INDICES_PARAMS, MULTI_WINDOW_PARAMS, ML_PARAMS
)
from strategies.mean_reversion import ZScoreStrategy, generate_rsi_variants, generate_bollinger_variants
from strategies.momentum import SMAStrategy, MACDStrategy, StochasticStrategy, ROCStrategy
from strategies.volume_based import OBVStrategy
from strategies.coincident_indices import CoincidentIndexStrategy, MultiCoincidentStrategy
from strategies.multi_window_returns import MultiWindowReturnsStrategy, CrossAssetWindowReturnsStrategy
from strategies.ml_strategies import MLStrategy
from utils.model_factory import create_ml_models
from utils.checkpoint_manager import CheckpointManager
from features.feature_engineering import build_ml_features_from_strategies


def create_rule_based_strategies(prices: pd.DataFrame) -> List:
    """Create all rule-based (non-ML) trading strategies.
    
    Args:
        prices: Price DataFrame
    
    Returns:
        List of strategy instances
    """
    strategies = []
    
    # Momentum strategies
    for short, long in MOMENTUM_PARAMS['sma_short_long_pairs']:
        strategies.append(SMAStrategy(prices, short, long))

    for fast, slow, signal in MOMENTUM_PARAMS['macd_params']:
        strategies.append(MACDStrategy(prices, fast, slow, signal))

    # Mean reversion strategies
    strategies.append(ZScoreStrategy(
        prices,
        MEAN_REVERSION_PARAMS['zscore_window'],
        MEAN_REVERSION_PARAMS['zscore_threshold']
    ))

    rsi_variants = generate_rsi_variants(
        prices,
        periods=MEAN_REVERSION_PARAMS.get('rsi_period'),
        threshold_pairs=MEAN_REVERSION_PARAMS.get('rsi_threshhold_pairs')
    )
    strategies.extend(rsi_variants)
    
    bollinger_variants = generate_bollinger_variants(
        prices,
        windows=MEAN_REVERSION_PARAMS.get('bollinger_windows'),
        std_devs=MEAN_REVERSION_PARAMS.get('bollinger_std_devs')
    )
    strategies.extend(bollinger_variants)
    
    # Volume and oscillator strategies
    if VOLUME_OSCILLATOR_PARAMS.get('enabled', False):
        for ticker, window in VOLUME_OSCILLATOR_PARAMS.get('obv', []):
            try:
                strategies.append(OBVStrategy(prices, ticker, window))
            except Exception as e:
                print(f"Warning: Could not create OBVStrategy for {ticker}: {e}")
        
        for ticker, k_period, d_period, oversold, overbought in VOLUME_OSCILLATOR_PARAMS.get('stochastic', []):
            try:
                strategies.append(StochasticStrategy(
                    prices, ticker, k_period, d_period, oversold, overbought
                ))
            except Exception as e:
                print(f"Warning: Could not create StochasticStrategy for {ticker}: {e}")
        
        for period, threshold in VOLUME_OSCILLATOR_PARAMS.get('roc', []):
            try:
                strategies.append(ROCStrategy(prices, period, threshold))
            except Exception as e:
                print(f"Warning: Could not create ROCStrategy: {e}")
    
    # Coincident indices strategies
    if COINCIDENT_INDICES_PARAMS.get('enabled', False):
        for ticker, window in COINCIDENT_INDICES_PARAMS.get('single_indices', []):
            try:
                strategies.append(CoincidentIndexStrategy(
                    prices, 
                    coincident_ticker=ticker,
                    window=window,
                    correlation_threshold=COINCIDENT_INDICES_PARAMS.get('correlation_threshold', 0.0)
                ))
            except Exception as e:
                print(f"Warning: Could not create CoincidentIndexStrategy for {ticker}: {e}")
        
        for tickers, window, aggregation in COINCIDENT_INDICES_PARAMS.get('multi_indices', []):
            try:
                strategies.append(MultiCoincidentStrategy(
                    prices,
                    coincident_tickers=tickers,
                    window=window,
                    aggregation=aggregation,
                    include_individual_features=False,
                    correlation_window=MULTI_WINDOW_PARAMS.get('correlation_window', 60)
                ))
            except Exception as e:
                print(f"Warning: Could not create MultiCoincidentStrategy: {e}")
    
    # Multi-window returns strategies
    if MULTI_WINDOW_PARAMS.get('enabled', False):
        for windows, signal_method in MULTI_WINDOW_PARAMS.get('same_asset_strategies', []):
            try:
                strategies.append(MultiWindowReturnsStrategy(
                    prices,
                    windows=windows,
                    signal_method=signal_method
                ))
            except Exception as e:
                print(f"Warning: Could not create MultiWindowReturnsStrategy: {e}")
        
        for ref_ticker, windows, signal_method in MULTI_WINDOW_PARAMS.get('cross_asset_strategies', []):
            try:
                strategies.append(CrossAssetWindowReturnsStrategy(
                    prices,
                    reference_ticker=ref_ticker,
                    windows=windows,
                    signal_method=signal_method
                ))
            except Exception as e:
                print(f"Warning: Could not create CrossAssetWindowReturnsStrategy for {ref_ticker}: {e}")
    
    return strategies


def create_ml_strategies(prices: pd.DataFrame, rule_based_strategies: List, 
                        test_start_date, tune_hyperparameters: bool = True,
                        checkpoint_dir: str = 'checkpoints') -> List:
    """Create ML-based trading strategies using rule-based features.
    
    NOTE: Always trains new models from scratch. Checkpoints are saved with 
    datetime stamps for historical reference but never loaded.
    
    Args:
        prices: Price DataFrame
        rule_based_strategies: List of rule-based strategies to extract features from
        test_start_date: Date to split train/test
        tune_hyperparameters: If True, use GridSearchCV to tune hyperparameters
        checkpoint_dir: Directory for saving checkpoints (with datetime stamps)
    
    Returns:
        List of MLStrategy instances
    """
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    # Never load saved params - always train fresh
    saved_best_params = {}
    
    # Create ML models
    ml_models = create_ml_models(
        tune_hyperparameters=tune_hyperparameters,
        saved_best_params=saved_best_params
    )
    
    # Load checkpoint (will always be empty - just for consistency)
    ml_strategies, completed_models = checkpoint_mgr.load_checkpoint(ml_models)
    
    # Build features once for all ML strategies (optimization)
    print("\nBuilding ML features from rule-based strategies...")
    X, y = build_ml_features_from_strategies(prices, rule_based_strategies, test_start_date)
    if X.empty or y.empty:
        print("Warning: No ML features could be built; skipping all ML strategies.")
        return []
    print(f"✓ Built {X.shape[1]} features from {len(rule_based_strategies)} strategies")
    print(f"  Feature shape: {X.shape}, Target shape: {y.shape}")
    
    # Train all models (fresh each run)
    tuning_msg = " with hyperparameter tuning" if tune_hyperparameters else ""
    print(f"\nCreating ML strategies{tuning_msg}...")
    total_models = len(ml_models)
    
    # Track starting count to calculate correct progress numbers
    initial_completed = len(completed_models)
    
    for idx, (model_name, (model_instance, short_name)) in enumerate(ml_models.items(), 1):
        try:
            current_progress = initial_completed + idx
            print(f"\n[{current_progress}/{total_models}] Training {model_name}...")
            
            ml_strat = MLStrategy(
                prices=prices,
                feature_strategies=rule_based_strategies,
                test_start_date=test_start_date,
                model=model_instance,
                model_name=short_name,
                prebuilt_features=(X, y)
            )
            
            ml_strat.calculate_returns()
            
            if ml_strat.data is not None and not ml_strat.data.empty:
                ml_strategies.append(ml_strat)
                completed_models.add(model_name)
                
                # Extract and save best params if tuned
                if hasattr(ml_strat.model, 'best_params_'):
                    saved_best_params[model_name] = ml_strat.model.best_params_
                    print(f"✓ {ml_strat.name}: best_params={ml_strat.model.best_params_}")
                else:
                    print(f"✓ {ml_strat.name}")
                
                # Save checkpoint after each successful model (with datetime stamp)
                try:
                    checkpoint_mgr.save_checkpoint(ml_strategies, completed_models)
                    if saved_best_params:
                        checkpoint_mgr.save_best_params(saved_best_params)
                    print(f"  → Checkpoint saved with timestamp ({len(completed_models)}/{total_models} complete)")
                except Exception as e:
                    print(f"  Warning: Could not save checkpoint: {e}")
            else:
                print(f"✗ {model_name} produced empty data; skipping.")
        except Exception as e:
            print(f"✗ Could not create {model_name} strategy: {e}")
            print(f"  → Progress saved. You can resume by running again.")
    
    # Don't cleanup checkpoint - keep it with timestamp for historical reference
    if len(completed_models) == total_models:
        print(f"\n✓ All models complete!")
        print(f"✓ Best hyperparameters saved to: {checkpoint_mgr.best_params_file}")
        print(f"✓ Checkpoint saved to: {checkpoint_mgr.checkpoint_file}")
    
    return ml_strategies


def init_strategies(prices: pd.DataFrame, test_start_date=None) -> List:
    """Create all trading strategy instances.
    
    Args:
        prices: Price DataFrame
        test_start_date: Optional date to split train/test for ML strategies
    
    Returns:
        List of all strategy instances (rule-based + ML if test_start_date provided)
    """
    # Create rule-based strategies
    strategies = create_rule_based_strategies(prices)
    
    # Add ML strategies if test_start_date provided
    if test_start_date is not None:
        tune_hyperparams = ML_PARAMS.get('tune_hyperparameters', True)
        ml_strats = create_ml_strategies(prices, strategies, test_start_date, tune_hyperparams)
        strategies.extend(ml_strats)
    
    return strategies


__all__ = ["create_rule_based_strategies", "create_ml_strategies", "init_strategies"]
