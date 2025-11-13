"""Checkpoint management utilities for ML grid search."""
import pickle
from pathlib import Path
from datetime import datetime


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
