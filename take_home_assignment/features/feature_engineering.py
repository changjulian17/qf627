"""Feature engineering for ML trading strategies.

Functions to create lagged returns, moving averages, build features from strategies,
and perform feature selection using Information Coefficient (IC).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from .technical_indicators import zscore
from backtesting.metrics import PerformanceMetrics
from config import ML_PARAMS, ZSCORE_WINDOW


def lagged_returns(prices: pd.Series, lags: List[int]) -> pd.DataFrame:
    """Return DataFrame of lagged pct-change returns for given lags (in days).

    Each column is named 'ret_{lag}'.
    """
    out = pd.DataFrame(index=prices.index)
    for lag in lags:
        out[f'ret_{lag}'] = prices.pct_change(lag).shift(1)
    return out


def moving_averages(prices: pd.Series, windows: List[int]) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    for w in windows:
        out[f'sma_{w}'] = prices.rolling(window=w, min_periods=int(w * 0.8)).mean()
        out[f'ema_{w}'] = prices.ewm(span=w, adjust=False).mean()
    return out


def build_ml_dataset(prices: pd.Series, exog: pd.DataFrame = None, lags=[5,15,30,60], ma_windows=[21,63,252], forward=5):
    """Build a simple supervised dataset for predicting forward `forward`-day return.

    Returns X, y aligned with index dropped where NaN.
    """
    lagged = lagged_returns(prices, lags)
    mas = moving_averages(prices, ma_windows)

    X = pd.concat([lagged, mas], axis=1)
    if exog is not None:
        X = pd.concat([X, exog.shift(1)], axis=1)

    y = prices.pct_change(forward).shift(-forward)

    # drop rows with NaN
    df = pd.concat([X, y.rename('target')], axis=1).dropna()

    X_clean = df.drop(columns=['target'])
    y_clean = df['target']
    return X_clean, y_clean


def filter_features_by_ic(X: pd.DataFrame, y: pd.Series, test_start_date, 
                          ic_percentile: float, correlation_threshold: float = 0.8) -> Tuple[pd.DataFrame, pd.Series]:
    """Filter features by Sharpe ratio using two-stage filtering (train-only selection).

    Stage 1: Rank features by the absolute Sharpe ratio of a strategy that uses
             positions based on feature sign: positions = np.where(feature > 0, 1, -1),
             then calculates strategy_returns = y_train * positions.shift(1).
             This matches the actual strategy logic (see BaseStrategy.calculate_returns).
             Keep the top ``ic_percentile`` percent by Sharpe.

    Stage 2: Remove highly correlated feature pairs (based on training data
             correlations), keeping the one with higher Stage 1 score.

    Prevents look-ahead bias by calculating Sharpe and correlations only on
    training data, then applying the same feature selection to the full dataset.

    Args:
        X: Feature DataFrame with datetime index
        y: Target Series with datetime index (e.g., next-period log returns)
        test_start_date: Date that splits train/test periods
        ic_percentile: Top percentile of features to keep (0-100)
        correlation_threshold: Threshold for removing correlated features (default 0.8)

    Returns:
        Tuple of (X_filtered, y_filtered) with aligned indices after dropping NaN rows
    """
    if test_start_date is None:
        print("Warning: Sharpe filtering requires test_start_date to prevent look-ahead bias. Skipping filtering.")
        return X, y
    
    # Calculate metrics using ONLY training data
    train_mask = X.index < test_start_date
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    if X_train.empty or y_train.empty:
        print("Warning: No training data available for Sharpe filtering. Skipping filtering.")
        return X, y
    
    # Remove columns with more than threshold% NaN values before filtering
    nan_threshold = ML_PARAMS.get('nan_threshold', 0.3)
    nan_percentages = X.isna().mean()
    cols_to_drop = nan_percentages[nan_percentages > nan_threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Pre-filtering: Removing {len(cols_to_drop)} columns with >{nan_threshold*100}% NaN values")
        X = X.drop(columns=cols_to_drop)
        X_train = X[train_mask]
    
    # STAGE 1: Calculate Sharpe ratio for each feature using actual strategy logic
    # Match the train period calculation: passive_returns * positions.shift(1)
    # where positions = np.where(feature > 0, 1, -1) to replicate Coincident strategy logic
    # 
    # IMPORTANT: Features like `TLT_signal` are ALREADY shifted in strategy generation,
    # so we should NOT shift positions again. The feature value at time t uses data up to t-1.
    # Strategy logic: positions[t] = where(signal[t] > 0, 1, -1)
    #                 strategy_returns[t] = passive_returns[t] * positions[t-1]
    # Since signal is already lagged, positions[t] uses info up to t-1
    # Then positions.shift(1) gives us positions[t-1] for time t
    # 
    # For features that are pre-shifted signals: positions = where(feature > 0, 1, -1)
    # Then: strategy_returns = y_train * positions.shift(1)
    sharpe_values = {}
    sharpe_values_signed = {}
    for col in X_train.columns:
        # Generate positions from feature: +1 when feature > 0, -1 otherwise
        positions = pd.Series(np.where(X_train[col] > 0, 1, -1), index=X_train.index)
        # Calculate strategy returns using same logic as BaseStrategy.calculate_returns()
        # strategy_returns = passive_returns * positions.shift(1)
        strat_ret = (y_train.shift(1) * positions.shift(1)).fillna(0)
        
        if strat_ret.std(ddof=1) == 0 or strat_ret.empty:
            sharpe = 0.0
        else:
            sharpe = float(PerformanceMetrics.calculate_sharpe_ratio(strat_ret, periods_per_year=252))
            
        # sharpe_values_signed[col] = sharpe
        sharpe_values[col] = abs(sharpe)

    # Calculate the threshold for top percentile
    sharpe_series = pd.Series(sharpe_values)
    
    # Print all Sharpe values (disable truncation)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("\nAll Feature Sharpe Ratios (sorted descending):")
        print(sharpe_series.sort_values(ascending=False))
    
    threshold = sharpe_series.quantile((100 - ic_percentile) / 100)

    # Select features above the threshold
    stage1_sorted = sharpe_series[sharpe_series >= threshold].sort_values(ascending=False, kind='mergesort')
    stage1_features = stage1_sorted.index.tolist()
    
    if len(stage1_features) == 0:
        print(f"Warning: IC filtering with percentile={ic_percentile} removed all features. Using all features.")
        return X, y

    
    # STAGE 2: Remove highly correlated features, keeping the one with higher IC
    # Calculate correlation matrix using ONLY training data
    X_train_stage1 = X_train[stage1_features]
    corr_matrix = X_train_stage1.corr().abs()
    
    # Track features to keep
    features_to_keep = []
    features_removed = []
    
    # Iterate through features sorted by Sharpe (highest first)
    for feature in stage1_features:
        # Check if this feature is already removed due to correlation with a higher-IC feature
        if feature in features_removed:
            continue
        
        # Keep this feature
        features_to_keep.append(feature)
        
        # Find and remove features highly correlated with this one
        for other_feature in stage1_features:
            if other_feature == feature or other_feature in features_removed:
                continue
            
            # If highly correlated, remove the one with lower Sharpe (which is the current other_feature)
            corr_val = corr_matrix.loc[feature, other_feature]
            if pd.notna(corr_val) and corr_val >= correlation_threshold:
                features_removed.append(other_feature)
    
    if len(features_to_keep) == 0:
        print(f"Warning: Correlation filtering removed all features. Using Stage 1 features.")
        features_to_keep = stage1_features

    # Apply same feature selection to entire dataset (train + test)
    X_filtered = X[features_to_keep]
    
    # Drop rows with any remaining NaN values from the entire dataset
    rows_before_total = len(X_filtered)
    X_filtered = X_filtered.dropna()
    rows_dropped_total = rows_before_total - len(X_filtered)
    if rows_dropped_total > 0:
        print(f"Final cleaning: Dropped {rows_dropped_total} rows with NaN values from full dataset ({rows_dropped_total/rows_before_total*100:.1f}%)")
    
    print(f"Stage 2 - Correlation filtering: removed {len(features_removed)} highly correlated features (threshold={correlation_threshold})")
    print(f"  Final feature count: {len(features_to_keep)}/{len(stage1_features)}")
    if features_removed:
        print(f"  Removed features: {features_removed[:10]}{'...' if len(features_removed) > 10 else ''}")
    
    # Align y with the cleaned X_filtered (same rows)
    y_filtered = y.loc[X_filtered.index]
    
    return X_filtered, y_filtered


def build_ml_features_from_strategies(prices: pd.DataFrame, strategies: list, 
                                      test_start_date=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Aggregate numeric indicator columns from each strategy into a feature DataFrame.

    Args:
        prices: Price DataFrame
        strategies: List of rule-based strategies to extract features from
        test_start_date: Optional date to split train/test for IC filtering (prevents look-ahead bias)

    Returns:
        Tuple of (X, y) where X is feature DataFrame and y is Series of forward log returns.
    """
    feature_frames = []

    # Ensure strategies have indicator columns populated
    for strat in strategies:
        # compute indicators/returns if not yet present
        try:
            strat.calculate_returns()
        except Exception:
            # ignore if already computed or not applicable
            pass

        df = strat.data.copy()
        # Exclude: returns, positions, trades, Close (redundant with price)
        # Also exclude raw SMAs/EMAs (they're trending - only keep normalized spreads)
        # Also exclude price column from each strategy (trending, use price_zscore instead)
        # Also exclude individual window returns (redundant across strategies - keep only signals)
        exclude = {
            "strategy_returns", "cum_strategy_returns", "passive_returns", 
            "cum_passive_returns", "position", "trades", "Close", "positions"
        }
        
        # Get the price column name (first column in prices DataFrame)
        price_col = prices.columns[0] if len(prices.columns) > 0 else 'price'
        
        indicator_cols = []
        for c in df.select_dtypes(include=["number"]).columns:
            if c in exclude:
                continue
            # Exclude price column (trending, we'll use price_zscore instead)
            if c == price_col:
                continue
            # Exclude correlation/weight diagnostic columns from multi-coincident strategies
            if c.endswith('_correlation') or c.endswith('_weight'):
                continue
            # Exclude raw SMA/EMA columns (keep only normalized spreads/z-scores)
            if c.startswith('sma_') or c.startswith('ema_'):
                continue
            # Exclude raw coincident index prices (they're trending)
            # Keep _price_zscore columns (already normalized in strategy)
            if c.endswith('_price'):
                continue
            # Exclude individual window returns (redundant - keep only aggregated signals)
            # These returns appear multiple times across different multi-window strategies
            if c.startswith('ret_') or c.endswith('_ret_5d') or c.endswith('_ret_10d') or \
               c.endswith('_ret_20d') or c.endswith('_ret_60d'):
                continue
            indicator_cols.append(c)
        
        if indicator_cols:
            tmp = df[indicator_cols].copy()
            
            # prefix columns with strategy name to avoid collisions
            tmp.columns = [f"{strat.name}__{c}" for c in tmp.columns]
            feature_frames.append(tmp)

    if not feature_frames:
        return pd.DataFrame(index=prices.index), pd.Series(dtype=float)

    # Add normalized price (z-score) instead of raw price
    price_series = prices[prices.columns[0]]
    price_zscore = zscore(price_series, window=ZSCORE_WINDOW)
    price_df = pd.DataFrame({'price_zscore': price_zscore}, index=prices.index)
    
    X = pd.concat(feature_frames + [price_df], axis=1).sort_index()
    print(f"Built ML feature set with {X.shape[1]} features from {len(strategies)} strategies.")
    print(f"Feature names: {X.columns.tolist()}")

    # compute forward return as target (using original price for returns calculation)
    log_ret = np.log(price_series / price_series.shift(1))
    y = log_ret.shift(-1).rename('fwd_log_ret')

    # drop rows with NaNs in features or target
    combined = pd.concat([X, y], axis=1)
    X_clean = combined.drop(columns=['fwd_log_ret'])
    y_clean = combined['fwd_log_ret']
    
    # Filter features by Information Coefficient (IC) if percentile is specified
    # IMPORTANT: IC filtering uses ONLY training data to prevent look-ahead bias
    ic_percentile = ML_PARAMS.get('ic_percentile')
    correlation_threshold = ML_PARAMS.get('correlation_threshold', 0.8)
    if ic_percentile is not None and 0 < ic_percentile <= 100:
        X_clean, y_clean = filter_features_by_ic(X_clean, y_clean, test_start_date, ic_percentile, correlation_threshold)
    
    return X_clean, y_clean


__all__ = ["lagged_returns", "moving_averages", "build_ml_dataset", 
           "filter_features_by_ic", "build_ml_features_from_strategies"]
