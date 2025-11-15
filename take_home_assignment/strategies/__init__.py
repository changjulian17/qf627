"""Strategies package for take_home_assignment."""

__all__ = ["base_strategy", "momentum", "mean_reversion", "ml_strategies"]

# Import commonly used strategy classes for easier access
from .mean_reversion import (
    RSIStrategy, 
    ZScoreStrategy, 
    BollingerBandStrategy,
    generate_rsi_variants,
    generate_bollinger_variants
)
