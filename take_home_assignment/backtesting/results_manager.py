"""Results management for backtesting output.

Handles saving, formatting, and benchmarking of strategy performance results.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Optional, Iterable
from config import START_DATE, END_DATE
from data.data_loader import DataLoader
from backtesting.metrics import PerformanceMetrics
from pathlib import Path


def compute_spy_benchmark(test_start_date, align_indices: Optional[Iterable[pd.Index]] = None, period: str = 'test') -> Optional[Dict]:
    """Fetch SPY and compute passive buy-and-hold metrics for the requested period.

    Args:
        test_start_date: Timestamp splitting train/test
        align_indices: Optional iterable of indices to align SPY to the common
            intersection across provided indices for apples-to-apples metrics.
        period: 'test' (>= test_start_date) or 'train' (< test_start_date)

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
        if period == 'train':
            spy_window = spy_prices.loc[spy_prices.index < test_start_date].copy()
            period_name = 'train'
        else:
            spy_window = spy_prices.loc[spy_prices.index >= test_start_date].copy()
            period_name = 'test'
        if spy_window.empty:
            print(f"Warning: SPY has no data in {period_name} period; benchmark not added.")
            return None

        price_col = spy_window.columns[0]
        # Compute log returns (same convention as strategies)
        spy_window['passive_returns'] = (spy_window[price_col] / spy_window[price_col].shift(1)).apply(
            lambda x: np.log(x) if x > 0 else 0
        )
        spy_window['passive_returns'] = spy_window['passive_returns'].fillna(0)
        
        # IMPORTANT: Align with strategy convention where positions are shifted by 1 day
        # The first test-period strategy return is 0 due to the 1-day position shift.
        # To compare apples-to-apples, set SPY's first test-period return to 0 as well.
        if not spy_window['passive_returns'].empty:
            spy_window.iloc[0, spy_window.columns.get_loc('passive_returns')] = 0.0

        # Optional alignment to strategy indices (common intersection)
        if align_indices:
            try:
                iter_indices = list(align_indices)
                if len(iter_indices) > 0:
                    common_idx = iter_indices[0]
                    for idx in iter_indices[1:]:
                        common_idx = common_idx.intersection(idx)
                    # Slice SPY to the common index intersection
                    aligned = spy_window.loc[spy_window.index.intersection(common_idx)].copy()
                    if not aligned.empty:
                        # Reset first aligned day's passive return to 0 and recompute cumsum base
                        first_idx = aligned.index[0]
                        aligned.iloc[0, aligned.columns.get_loc('passive_returns')] = 0.0
                        spy_window = aligned
            except Exception as e:
                print(f"Warning: Could not align SPY to provided indices: {e}")

        spy_window['cum_passive_returns'] = spy_window['passive_returns'].cumsum().apply(np.exp)

        spy_metrics = PerformanceMetrics.calculate_all_metrics(
            spy_window['passive_returns'],
            spy_window['cum_passive_returns']
        )

        return {
            'data': spy_window,
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
        summary_csv_path = out_path / 'summary.csv'
        summary_df.sort_values('total_return_pct', ascending=False).to_csv(summary_csv_path, index=False)
        print(f"Detailed results saved to: {out_path.resolve()} (data + summary.csv)")

        # Optional: Create Sharpe alignment debug CSV if Stage 1 diagnostics exist
        try:
            stage1_csv = Path(__file__).resolve().parents[1] / 'results' / 'debug_feature_engineering_sharpes.csv'
            if stage1_csv.exists():
                sharpe_df = pd.read_csv(stage1_csv)
                # Choose best feature per strategy by abs Sharpe
                idx_best = sharpe_df.groupby('strategy')['sharpe_stage1_abs'].idxmax()
                best_view = sharpe_df.loc[idx_best].reset_index(drop=True).rename(
                    columns={'feature': 'best_feature'}
                )

                # Merge with summary Sharpe Ratio as fallback if train/test comparison isn't available
                res_df = summary_df.copy()
                if 'strategy' not in res_df.columns:
                    # Attempt to find strategy column
                    if res_df.columns[0].lower() in ('strategy', 'name', 'strategy_name'):
                        res_df = res_df.rename(columns={res_df.columns[0]: 'strategy'})
                keep_cols = ['strategy'] + [c for c in res_df.columns if 'sharpe' in c.lower()]
                res_sel = res_df[keep_cols]
                merged = best_view.merge(res_sel, on='strategy', how='left')
                out_path_align = out_path / 'debug_sharpe_alignment.csv'
                merged.to_csv(out_path_align, index=False)
                print(f"Sharpe alignment (with summary.csv) saved to: {out_path_align}")
            else:
                print("Sharpe alignment skipped: Stage 1 diagnostics not found.")
        except Exception as e:
            print(f"Warning: Failed to produce Sharpe alignment CSV: {e}")
    except Exception as e:
        print(f"Warning: Failed to save summary CSV: {e}")


__all__ = ["compute_spy_benchmark", "save_and_print_results", "save_detailed_results"]
