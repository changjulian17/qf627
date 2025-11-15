"""Results management for backtesting output.

Handles saving, formatting, and benchmarking of strategy performance results.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional
from config import START_DATE, END_DATE
from data.data_loader import DataLoader
from backtesting.metrics import PerformanceMetrics


def compute_spy_benchmark(test_start_date) -> Optional[Dict]:
    """Fetch SPY and compute passive buy-and-hold metrics for the test period.
    
    Args:
        test_start_date: Start date for test period
    
    Returns:
        Dict with 'data' and 'metrics' keys, or None on failure
    """
    try:
        spy_loader = DataLoader('SPY', START_DATE, END_DATE)
        spy_prices = spy_loader.get_prices()
        if isinstance(spy_prices, pd.Series):
            spy_prices = spy_prices.to_frame()
    except Exception as e:
        print(f"Warning: could not load SPY benchmark ({e})")
        return None

    try:
        spy_test = spy_prices.loc[spy_prices.index >= test_start_date].copy()
        if spy_test.empty:
            print("Warning: SPY has no data in test period; benchmark not added.")
            return None

        price_col = spy_test.columns[0]
        spy_test['passive_returns'] = (spy_test[price_col] / spy_test[price_col].shift(1)).apply(
            lambda x: np.log(x) if x > 0 else 0
        )
        spy_test['passive_returns'] = spy_test['passive_returns'].fillna(0)
        spy_test['cum_passive_returns'] = spy_test['passive_returns'].cumsum().apply(np.exp)

        spy_metrics = PerformanceMetrics.calculate_all_metrics(
            spy_test['passive_returns'],
            spy_test['cum_passive_returns']
        )

        return {
            'data': spy_test,
            'metrics': spy_metrics
        }
    except Exception as e:
        print(f"Warning computing SPY benchmark metrics: {e}")
        return None


def save_and_print_results(results: Dict) -> pd.DataFrame:
    """Build comparison DataFrame from results, print and save to CSV.
    
    Args:
        results: Dictionary mapping strategy_name -> result dict with 'metrics' key
    
    Returns:
        Comparison DataFrame sorted by Total Return (descending)
    """
    comparison_df = pd.DataFrame({name: res['metrics'] for name, res in results.items()}).T
    
    # Sort by Total Return in descending order
    comparison_df = comparison_df.sort_values('Total Return', ascending=False)

    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(comparison_df.to_string())

    comparison_df.to_csv('strategy_comparison.csv')
    print("\nResults saved to strategy_comparison.csv")
    return comparison_df


def save_detailed_results(results: Dict, out_dir: str = "results") -> None:
    """Save per-strategy backtest DataFrames and a summary CSV for diagnostics.
    
    Args:
        results: Dictionary mapping strategy_name -> result dict
        out_dir: Output directory for results
    
    Writes:
        - results/strategy_data/<strategy>.csv for each strategy
        - results/summary.csv with aggregated metrics
    """
    out_path = Path(out_dir)
    data_dir = out_path / "strategy_data"
    out_path.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    def safe_name(name: str) -> str:
        """Convert strategy name to safe filename."""
        return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)[:200]

    summary_rows = []
    for name, res in results.items():
        bt_data = res.get('data')
        final_value = res.get('final_value')
        total_return = res.get('total_return')
        metrics = res.get('metrics', {}) or {}

        # Save per-strategy data if available
        if isinstance(bt_data, pd.DataFrame) and not bt_data.empty:
            file_path = data_dir / f"{safe_name(name)}.csv"
            try:
                bt_data.to_csv(file_path, index_label='date')
            except Exception as e:
                print(f"Warning: Failed to save data for {name}: {e}")

        # Build summary row
        row = {
            'strategy': name,
            'final_value': final_value,
            'total_return_pct': total_return,
        }
        # Merge metrics (ensure flat dict)
        if isinstance(metrics, dict):
            row.update(metrics)
        summary_rows.append(row)

    # Save summary CSV
    try:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_path / 'summary.csv', index=False)
        print(f"Detailed results saved to: {out_path.resolve()} (data + summary.csv)")
    except Exception as e:
        print(f"Warning: Failed to save summary CSV: {e}")


__all__ = ["compute_spy_benchmark", "save_and_print_results", "save_detailed_results"]
