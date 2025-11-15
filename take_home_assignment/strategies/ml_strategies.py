"""ML-based strategy wrappers for multiple model types.

Contains strategy classes that train regressors to predict forward returns 
and convert predictions into long/short signals.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, Any
from features.feature_engineering import build_ml_features_from_strategies


class MLStrategy:
    """ML-based trading strategy that works with any sklearn-compatible model.
    
    Builds features from rule-based strategies, trains a regressor to predict forward returns,
    and creates long/short positions based on predictions.
    """

    def __init__(self, prices: pd.DataFrame, feature_strategies: list, test_start_date, 
                 model: Any, model_name: str = "ML", threshold: float = 0.0,
                 prebuilt_features: tuple = None):
        """
        Args:
            prices: Price DataFrame
            feature_strategies: List of rule-based strategies to extract features from
            test_start_date: Date to split train/test
            model: sklearn-compatible model instance
            model_name: Short name for the model (e.g., 'RF', 'LinReg')
            threshold: Prediction threshold for long/short signals (default 0.0)
            prebuilt_features: Optional tuple of (X, y) pre-computed features to avoid rebuilding
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
        self.prebuilt_features = prebuilt_features
        
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
            
        # Use prebuilt features if available, otherwise build from strategies
        if self.prebuilt_features is not None:
            X, y = self.prebuilt_features
        else:
            X, y = build_ml_features_from_strategies(self.prices, self.feature_strategies, self.test_start_date)
        
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


__all__ = ["MLStrategy"]