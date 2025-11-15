"""Checkpoint management utilities for ML grid search."""
from __future__ import annotations
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, Tuple, List, Any, Optional
import pandas as pd


def get_checkpoint_info(checkpoint_dir='checkpoints'):
    """Get information about the current checkpoint.
    
    Returns:
        dict with checkpoint info or None if no checkpoint exists
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_file = checkpoint_path / 'ml_strategies_checkpoint.pkl'
    
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        return {
            'num_strategies': len(checkpoint_data.get('strategies', [])),
            'completed_models': list(checkpoint_data.get('completed_models', [])),
            'timestamp': checkpoint_data.get('timestamp'),
            'file_size': checkpoint_file.stat().st_size / 1024,  # KB
        }
    except Exception as e:
        return {'error': str(e)}


def clear_checkpoint(checkpoint_dir='checkpoints'):
    """Clear the checkpoint file to start fresh.
    
    Returns:
        bool: True if checkpoint was cleared, False if no checkpoint existed
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_file = checkpoint_path / 'ml_strategies_checkpoint.pkl'
    
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print(f"✓ Checkpoint cleared: {checkpoint_file}")
            return True
        except Exception as e:
            print(f"✗ Error clearing checkpoint: {e}")
            return False
    else:
        print("No checkpoint file found.")
        return False


def show_checkpoint_status(checkpoint_dir='checkpoints'):
    """Display current checkpoint status."""
    info = get_checkpoint_info(checkpoint_dir)
    
    if info is None:
        print("=" * 60)
        print("CHECKPOINT STATUS")
        print("=" * 60)
        print("No checkpoint found - starting fresh")
        print("=" * 60)
        return
    
    if 'error' in info:
        print("=" * 60)
        print("CHECKPOINT STATUS")
        print("=" * 60)
        print(f"Error reading checkpoint: {info['error']}")
        print("=" * 60)
        return
    
    print("=" * 60)
    print("CHECKPOINT STATUS")
    print("=" * 60)
    print(f"Checkpoint found: {info['num_strategies']} strategies completed")
    print(f"File size: {info['file_size']:.2f} KB")
    if info['timestamp']:
        print(f"Last updated: {info['timestamp']}")
    print()
    print("Completed models:")
    for model in sorted(info['completed_models']):
        print(f"  ✓ {model}")
    print("=" * 60)
    print("You can:")
    print("  - Run main.py to resume from this checkpoint")
    print("  - Call clear_checkpoint() to start fresh")
    print("=" * 60)


if __name__ == '__main__':
    # Show status when run as script
    show_checkpoint_status()


class CheckpointManager:
    """Manages checkpoint files for ML strategy training."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory path for storing checkpoint files
        """
        self.checkpoint_path = Path(checkpoint_dir)
        self.checkpoint_path.mkdir(exist_ok=True)
        # Use datetime stamp for new checkpoint files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.checkpoint_file = self.checkpoint_path / f'ml_strategies_checkpoint_{timestamp}.pkl'
        self.best_params_file = self.checkpoint_path / f'ml_best_params_{timestamp}.pkl'
    
    def load_checkpoint(self, ml_models: Dict[str, Tuple[Any, str]]) -> Tuple[List, Set[str]]:
        """Load checkpoint with completed strategies and model names.
        
        NOTE: Always returns empty results to force training new models each run.
        Old checkpoints are preserved with timestamps.
        
        Args:
            ml_models: Dictionary of model_name -> (model_instance, short_name) pairs
                      Used to validate checkpoint against current configuration
        
        Returns:
            Tuple of (empty ml_strategies list, empty completed_models set)
        """
        ml_strategies = []
        completed_models = set()
        
        # Always start fresh - never load old checkpoints
        print("\n✓ Starting fresh training run (checkpoints disabled)")
        
        return ml_strategies, completed_models
    
    def save_checkpoint(self, ml_strategies: List, completed_models: Set[str]) -> None:
        """Save checkpoint with current progress.
        
        Args:
            ml_strategies: List of completed MLStrategy instances
            completed_models: Set of model names that have been completed
        """
        try:
            checkpoint_data = {
                'strategies': ml_strategies,
                'completed_models': completed_models,
                'timestamp': pd.Timestamp.now()
            }
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception as e:
            raise RuntimeError(f"Could not save checkpoint: {e}")
    
    def remove_checkpoint(self) -> None:
        """Remove checkpoint file after all models complete."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                print(f"\n✓ All models complete! Checkpoint file removed.")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint file: {e}")
    
    def load_best_params(self) -> Dict[str, Dict]:
        """Load saved best hyperparameters from previous runs.
        
        NOTE: Always returns empty dict to force hyperparameter tuning each run.
        Old parameters are preserved with timestamps.
        
        Returns:
            Empty dictionary (always trains with fresh hyperparameter search)
        """
        saved_best_params = {}
        
        # Always start fresh - never load old parameters
        print("✓ Hyperparameter tuning will run for all models (saved params disabled)")
        
        return saved_best_params
    
    def save_best_params(self, best_params: Dict[str, Dict]) -> None:
        """Save best hyperparameters for future runs.
        
        Args:
            best_params: Dictionary mapping model_name -> best_params dict
        """
        try:
            with open(self.best_params_file, 'wb') as f:
                pickle.dump(best_params, f)
        except Exception as e:
            raise RuntimeError(f"Could not save best parameters: {e}")
    
    def checkpoint_exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_file.exists()
    
    def best_params_exists(self) -> bool:
        """Check if best params file exists."""
        return self.best_params_file.exists()


__all__ = ["CheckpointManager", "get_checkpoint_info", "clear_checkpoint", "show_checkpoint_status"]
