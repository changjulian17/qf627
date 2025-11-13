"""Configuration parameters for the trading strategy project."""
import datetime as dt

# Data parameters
START_DATE = dt.datetime(2006, 11, 1)
END_DATE = dt.datetime(2025, 11, 12)
TICKER = 'SPY'  # S&P 500

# Portfolio parameters
INITIAL_CAPITAL = 100_000
COMMISSION_PER_TRADE = 0

# Strategy parameters
MOMENTUM_PARAMS = {
    'sma_short_long_pairs': [(20, 60), (50, 200), (24, 58)],
    'macd_params': (12, 26, 9),
}

MEAN_REVERSION_PARAMS = {
    'rsi_period': [14, 21],
    'rsi_threshhold_pairs': [(30, 70), (25, 75)],
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'zscore_window': 42,
    'zscore_threshold': 2,
}

ML_PARAMS = {
    'train_test_split': 0.75,
    'random_seed': 627,
    'models': ['RandomForest', 'GradientBoosting', 'LASSO'],
    'tune_hyperparameters': True,  # Use GridSearchCV to tune each model
    'n_jobs': 1,  # Number to 1 to manage resource usage
}

# Coincident indices strategy parameters
COINCIDENT_INDICES_PARAMS = {
    'enabled': True,
    'single_indices': [
        ('GLD', 20),    # Gold ETF with 20-day window
        ('^VIX', 10),   # Volatility Index with 10-day window
    ],
    'multi_indices': [
        (['GLD', 'TLT', 'UUP'], 20, 'majority'),  # Tickers, window, aggregation method
    ],
    'correlation_threshold': 0.0,
}

# Multi-window returns strategy parameters
MULTI_WINDOW_PARAMS = {
    'enabled': True,
    'same_asset_strategies': [
        ([5, 10, 20, 60], 'weighted_average'),
        ([5, 10, 20], 'majority_vote'),
    ],
    'cross_asset_strategies': [
        ('SPY', [5, 10, 20, 60], 'weighted_average'),
        ('QQQ', [10, 20, 60], 'majority_vote'),
    ],
}

# Hyperparameter grids for ML model tuning
ML_HYPERPARAMETER_GRIDS = {
    'Linear Regression': {},  # No hyperparameters to tune
    
    'Elastic Net': {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    
    'LASSO': {
        'alpha': [0.001, 0.01, 0.1, 1.0]
    },
    
    'Support Vector Machine': {
        'C': [0.1, 1.0, 10.0],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf', 'linear']
    },
    
    'K-Nearest Neighbor': {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    },
    
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    },
    
    'Extra Trees': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    },
    
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    },
    
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    },
    
    'Adaptive Boosting': {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]
    },
    
    'XGBoost': {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
}

# Common coincident indices for strategy use
CURRENCY_INDICES = {
    'DXY': 'DX-Y.NYB',  # US Dollar Index
    'EUR': 'EURUSD=X',   # Euro/USD
    'JPY': 'JPY=X',      # Japanese Yen
    'GBP': 'GBPUSD=X',   # British Pound
}

MARKET_INDICES = {
    'VIX': '^VIX',       # Volatility Index
    'TNX': '^TNX',       # 10-Year Treasury Yield
    'GLD': 'GLD',        # Gold ETF
    'TLT': 'TLT',        # 20+ Year Treasury Bond ETF
    'USO': 'USO',        # Oil ETF
    'UUP': 'UUP',        # US Dollar Bullish ETF
    # SPY Industry/Sector ETFs
    'XLK': 'XLK',        # Technology Select Sector SPDR
    'XLF': 'XLF',        # Financial Select Sector SPDR
    'XLV': 'XLV',        # Health Care Select Sector SPDR
    'XLE': 'XLE',        # Energy Select Sector SPDR
    'XLI': 'XLI',        # Industrial Select Sector SPDR
    'XLP': 'XLP',        # Consumer Staples Select Sector SPDR
    'XLY': 'XLY',        # Consumer Discretionary Select Sector SPDR
    'XLU': 'XLU',        # Utilities Select Sector SPDR
    'XLRE': 'XLRE',      # Real Estate Select Sector SPDR
    'XLB': 'XLB',        # Materials Select Sector SPDR
    'XLC': 'XLC',        # Communication Services Select Sector SPDR
}

# Performance metrics
TRADING_DAYS_PER_YEAR = 252