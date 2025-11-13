"""Test checkpoint functionality with a small subset of models."""
import sys
sys.path.insert(0, '/Users/ju/Projects/qf627/take_home_assignment')

from runner import load_prices_and_test_start, init_strategies, create_ml_strategies
from utils.checkpoint_manager import show_checkpoint_status, clear_checkpoint

# First, clear any existing checkpoint
print("Clearing any existing checkpoint...")
clear_checkpoint()
print()

# Load data
print("Loading data...")
prices, test_start = load_prices_and_test_start()
print()

# Create base strategies
print("Creating base strategies...")
base_strategies = init_strategies(prices, test_start)
# Only use first 5 for quick test
base_strategies = base_strategies[:5]
print(f"Using {len(base_strategies)} base strategies for feature generation")
print()

# Temporarily modify config to only use 2 models for testing
import config
original_grids = config.ML_HYPERPARAMETER_GRIDS.copy()

# Use only 2 models with smaller grids for quick test
config.ML_HYPERPARAMETER_GRIDS = {
    'Linear Regression': {},
    'LASSO': {'alpha': [0.01, 0.1]},  # Small grid for quick test
}

try:
    print("=" * 70)
    print("CHECKPOINT TEST: Creating first model...")
    print("=" * 70)
    
    # This will create and checkpoint the first model
    ml_strats = create_ml_strategies(prices, base_strategies, test_start, 
                                    tune_hyperparameters=True)
    
    print()
    print("=" * 70)
    print("Checkpoint status after first run:")
    print("=" * 70)
    show_checkpoint_status()
    
finally:
    # Restore original config
    config.ML_HYPERPARAMETER_GRIDS = original_grids

print()
print("✓ Checkpoint test complete!")
print("Try running this script again - it should resume from the checkpoint!")
