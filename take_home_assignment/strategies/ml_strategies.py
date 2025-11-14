"""ML-based strategy wrappers for multiple model types.

Contains strategy classes that train regressors to predict forward returns 
and convert predictions into long/short signals.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from typing import Optional, Any    
from features.technical_indicators import zscore
from config import ML_HYPERPARAMETER_GRIDS, ML_PARAMS, ZSCORE_WINDOW


class MLStrategy:
    """ML-based trading strategy that works with any sklearn-compatible model.
    
    Builds features from rule-based strategies, trains a regressor to predict forward returns,
    and creates long/short positions based on predictions.
    """

    def __init__(self, prices: pd.DataFrame, feature_strategies: list, test_start_date, 
                 model: Any, model_name: str = "ML", threshold: float = 0.0):
        """
        Args:
            prices: Price DataFrame
            feature_strategies: List of rule-based strategies to extract features from
            test_start_date: Date to split train/test
            model: sklearn-compatible model instance
            model_name: Short name for the model (e.g., 'RF', 'LinReg')
            threshold: Prediction threshold for long/short signals (default 0.0)
        """
        self.prices = prices
        self.feature_strategies = feature_strategies
        self.test_start_date = test_start_date
        self.model = model
        self.model_name = model_name
        self.name = model_name
        self.threshold = threshold
        self.data = None
        self._trained = False
        self.train_mse = None
        self.test_mse = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the underlying model."""
        self.model.fit(X, y)
        self._trained = True

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict forward returns using the trained model."""
        if not self._trained:
            raise RuntimeError("Model not fitted. Call calculate_returns() first.")
        return pd.Series(self.model.predict(X), index=X.index)

    def signals_from_preds(self, preds: pd.Series) -> pd.Series:
        """Convert predictions to long/short signals: long if pred > threshold else short."""
        # signal = pd.Series(np.where(preds > self.threshold, 1, 
        #                           np.where(preds > -.01, -1, np.nan)), index=preds.index).ffill()
        signal = pd.Series(np.where(preds > self.threshold, 1, 
                                  np.where(preds > -.01, 0, -1)), index=preds.index)
        return signal
        
    def calculate_returns(self):
        """Build features, train on train period, generate signals/returns for full period."""
        if self.data is not None:
            return  # already computed
            
        X, y = build_ml_features_from_strategies(self.prices, self.feature_strategies)
        if X.empty or y.empty:
            print(f"Warning: No ML features available for {self.name}; creating empty data.")
            self.data = pd.DataFrame(index=self.prices.index)
            return
        
        # Update name with feature count
        self.name = f"{self.model_name}_{len(X.columns)}f"
        
        # split by time index
        X_train = X.loc[X.index < self.test_start_date]
        y_train = y.loc[y.index < self.test_start_date]
        
        if X_train.empty:
            print(f"Warning: Insufficient training data for {self.name}; creating empty data.")
            self.data = pd.DataFrame(index=self.prices.index)
            return
        
        # Train on training period
        self.fit(X_train, y_train)
        
        # Predict on full period where features exist
        preds = self.predict(X)
        
        # Calculate MSE for train and test sets
        X_test = X.loc[X.index >= self.test_start_date]
        y_test = y.loc[y.index >= self.test_start_date]
        
        train_preds = self.predict(X_train)
        test_preds = self.predict(X_test)
        
        self.train_mse = ((y_train - train_preds) ** 2).mean()
        self.test_mse = ((y_test - test_preds) ** 2).mean()
        
        signals = self.signals_from_preds(preds)
        
        # Build data DataFrame matching other strategies
        price_col = self.prices.columns[0]
        self.data = pd.DataFrame({price_col: self.prices[price_col].reindex(X.index)})
        self.data['passive_returns'] = np.log(self.data[price_col] / self.data[price_col].shift(1)).fillna(0)
        # Standardize on 'positions' (plural) across all strategies
        self.data['positions'] = signals.reindex(self.data.index).fillna(0)
        self.data['strategy_returns'] = self.data['positions'].shift(1).fillna(0) * self.data['passive_returns']
        # Cumulative returns: cumulative product of exp(log_returns)
        self.data['cum_passive_returns'] = self.data['passive_returns'].cumsum().apply(np.exp)
        self.data['cum_strategy_returns'] = self.data['strategy_returns'].cumsum().apply(np.exp)
        self.data['trades'] = self.data['positions'].diff().abs().fillna(0)


def create_ml_models(tune_hyperparameters=False, saved_best_params=None):
    """Create a dictionary of ML models for strategy testing.
    
    Args:
        tune_hyperparameters: If True, returns GridSearchCV wrapped models for tuning.
                             If False, returns models with default parameters.
        saved_best_params: Dict mapping model names to their best parameters from previous runs.
                          If provided, these params will be used instead of tuning.
    
    Returns a dict with model_name -> (model_instance, short_name) pairs.
    """
    if saved_best_params is None:
        saved_best_params = {}
    
    random_seed = ML_PARAMS.get('random_seed', 42)
    
    base_models = {
        'Linear Regression': (LinearRegression(), 'LinReg'),
        'Elastic Net': (ElasticNet(random_state=random_seed), 'ElasticNet'),
        'LASSO': (Lasso(random_state=random_seed), 'LASSO'),
        'XGBoost' : (XGBRegressor(random_state=random_seed, n_jobs=ML_PARAMS.get('n_jobs', -1)), 'XGB'),
        # 'Support Vector Machine': (SVR(), 'SVM'),
        'K-Nearest Neighbor': (KNeighborsRegressor(), 'KNN'),
        'Decision Tree': (DecisionTreeRegressor(random_state=random_seed), 'DTree'),
        'Extra Trees': (ExtraTreesRegressor(random_state=random_seed), 'ExtraTrees'),
        'Random Forest': (RandomForestRegressor(random_state=random_seed), 'RF'),
        'Gradient Boosting': (GradientBoostingRegressor(random_state=random_seed), 'GBT'),
        'Adaptive Boosting': (AdaBoostRegressor(random_state=random_seed), 'AdaBoost'),
    }
    
    tuned_models = {}
    
    for model_name, (model, short_name) in base_models.items():
        # Check if we have saved best params for this model
        if model_name in saved_best_params:
            print(f"  → Using saved best params for {model_name}: {saved_best_params[model_name]}")
            model.set_params(**saved_best_params[model_name])
            tuned_models[model_name] = (model, short_name)
        elif tune_hyperparameters:
            # Only tune if we don't have saved params
            param_grid = ML_HYPERPARAMETER_GRIDS.get(model_name, {})
            
            if param_grid:
                # Use GridSearchCV with 3-fold CV, optimize for negative MSE
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=ML_PARAMS.get('n_jobs', -1),
                    verbose=0
                )
                tuned_models[model_name] = (grid_search, short_name)
            else:
                # No hyperparameters to tune (e.g., LinearRegression)
                tuned_models[model_name] = (model, short_name)
        else:
            # No tuning, just use default params
            tuned_models[model_name] = (model, short_name)
    
    return tuned_models


def build_ml_features_from_strategies(prices: pd.DataFrame, strategies: list):
    """Aggregate numeric indicator columns from each strategy into a feature DataFrame.

    Returns X (DataFrame) and y (Series of forward log returns).
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
    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined.drop(columns=['fwd_log_ret'])
    y_clean = combined['fwd_log_ret']
    return X_clean, y_clean


__all__ = ["MLStrategy", "create_ml_models", "build_ml_features_from_strategies"]