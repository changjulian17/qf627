"""Configuration parameters for the trading strategy project."""
import datetime as dt

# Data parameters
START_DATE = dt.datetime(2006, 11, 1)
END_DATE = dt.datetime(2025, 11, 12)
TICKER = 'SPY'  # S&P 500

# Portfolio parameters
INITIAL_CAPITAL = 100_000
COMMISSION_PER_TRADE = 0

# Feature normalization parameters
ZSCORE_WINDOW = 200  # Window for z-score normalization of features (price, coincident signals, etc.)

# Coincident index windows for multi-timeframe analysis
COINCIDENT_WINDOWS = [5, 20, 252]  # Short, medium, and long-term momentum windows
MULTI_COINCIDENT_WINDOW = 20  # Window for multi-coincident index strategies

# Strategy parameters
MOMENTUM_PARAMS = {
    'sma_short_long_pairs': [(20, 60), (50, 200), (24, 58)],
    'macd_params': [
        (12, 26, 9),   # Standard MACD
        (5, 35, 5),    # Faster MACD
        (19, 39, 9),   # Slower MACD
    ],
}

MEAN_REVERSION_PARAMS = {
    'rsi_period': [14, 21],
    'rsi_threshhold_pairs': [(30, 70), (25, 75)],
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'zscore_window': 42,
    'zscore_threshold': 2,
    'bollinger_windows': [20, 30, 50],
    'bollinger_std_devs': [1.5, 2.0, 2.5],
}

# Volume and oscillator strategy parameters
VOLUME_OSCILLATOR_PARAMS = {
    'enabled': True,
    'obv': [
        ('SPY', 20),   # ticker, window
        ('SPY', 50),
    ],
    'stochastic': [
        ('SPY', 14, 3, 20, 80),  # ticker, k_period, d_period, oversold, overbought
        ('SPY', 5, 3, 20, 80),   # Faster stochastic
    ],
    'roc': [
        (12, 0),   # period, threshold
        (20, 0),
        (5, 0),    # Faster ROC
    ],
}

ML_PARAMS = {
    'train_test_split': 0.75,
    'random_seed': 627,
    'models': ['RandomForest', 'GradientBoosting', 'LASSO'],
    'tune_hyperparameters': True,  # Use GridSearchCV to tune each model
    'n_jobs': 1,  # Number to 1 to manage resource usage
    'ic_percentile': 5,  # Top percentile of features by Information Coefficient (0-100). None = no filtering
    'correlation_threshold': 1,  # Threshold for removing highly correlated features in Stage 2 filtering (0-1)
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

# Coincident indices strategy parameters
COINCIDENT_INDICES_PARAMS = {
    'enabled': True,
    'single_indices': [
        # SPY itself (rolling mean of returns as momentum signal) - multiple windows
        *[('SPY', window) for window in COINCIDENT_WINDOWS],
        # All currency indices with multiple windows
        *[(ticker, window) for ticker in CURRENCY_INDICES.values() for window in COINCIDENT_WINDOWS],
        # All market indices with multiple windows
        *[(ticker, window) for ticker in MARKET_INDICES.values() for window in COINCIDENT_WINDOWS],
    ],
    'multi_indices': [
        # All currency indices combined with mean aggregation
        (list(CURRENCY_INDICES.values()), MULTI_COINCIDENT_WINDOW, 'mean'),
        # All market indices combined with majority vote
        (list(MARKET_INDICES.values()), MULTI_COINCIDENT_WINDOW, 'majority'),
        # All currency + market indices with correlation-weighted aggregation
        (list(CURRENCY_INDICES.values()) + list(MARKET_INDICES.values()), MULTI_COINCIDENT_WINDOW, 'correlation_weighted'),
    ],
    'correlation_threshold': 0.0,
}

# Multi-window returns strategy parameters
MULTI_WINDOW_PARAMS = {
    'enabled': True,
    'correlation_window': 60,  # Window for rolling correlation calculation (used in correlation_weighted aggregation)
    'same_asset_strategies': [
        (COINCIDENT_WINDOWS, 'weighted_average'),  # Short, medium, long-term weighted average
        (COINCIDENT_WINDOWS, 'majority_vote'),  # Short, medium, long-term majority vote
    ],
    'cross_asset_strategies': [
        ('SPY', COINCIDENT_WINDOWS, 'weighted_average'),
        ('QQQ', COINCIDENT_WINDOWS, 'majority_vote'),  # Custom windows for QQQ
    ],
}

# Hyperparameter grids for ML model tuning
ML_HYPERPARAMETER_GRIDS = {
    'Linear Regression': {},  # No hyperparameters to tune
    
    'Elastic Net': {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.5, 0.9]
    },
    
    'LASSO': {
        'alpha': [0.01, 0.1, 1.0]
    },
    
    'Support Vector Machine': {
        'C': [1.0, 10.0],
        'gamma': ['scale'],
        'kernel': ['rbf']
    },
    
    'K-Nearest Neighbor': {
        'n_neighbors': [5, 10],
        'weights': ['uniform', 'distance']
    },
    
    'Decision Tree': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
    },
    
    'Extra Trees': {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
    },
    
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, None],
        'min_samples_split': [2, 10],
        'min_samples_leaf': [1, 4],
    },
    
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 10],
    },
    
    'Adaptive Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 1.0]
    },
    
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
    }
}

# Performance metrics
TRADING_DAYS_PER_YEAR = 252