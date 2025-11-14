"""Main execution script for the trading strategy project.

This file now shows the high-level logical flow by calling the
small functions implemented in `utils/runner.py`. That way you can
review the orchestration steps in `main.py` while the heavy
lifting remains in `utils/runner.py`.
"""
from pathlib import Path
from lets_plot import LetsPlot, ggsave

from utils.runner import (
    load_prices_and_test_start,
    init_strategies,
    run_backtests_on_test_period,
    compute_spy_benchmark,
    save_and_print_results,
    save_detailed_results,
)
from visualisation.plotting import (
    plot_top_strategy_vs_benchmark,
    plot_top_n_strategies_vs_benchmark,
)


def save_plot(plot_obj, filename, script_dir):
    """Save a plot to HTML file without printing HTML content.
    
    Args:
        plot_obj: The plot object to save
        filename: Name of the output HTML file
        script_dir: Directory where the script is located
        
    Returns:
        Path to the saved file
    """
    if plot_obj is None:
        return None
    
    plot_dir = script_dir / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / filename
    
    # Save without printing HTML
    ggsave(plot_obj, str(plot_path.resolve()))
    
    return plot_path.resolve()


def main():
    """High-level orchestration visible in main for review.

    Steps:
    1. Load prices and compute test start date
    2. Initialize strategies
    3. Run backtests restricted to the test period
    4. Compute SPY benchmark metrics for the test period
    5. Save and print the comparison table
    6. Save detailed results to CSV files
    7. Generate and save plots
    """
    # Initialize lets-plot (suppress HTML output in console)
    LetsPlot.setup_html()
    
    script_dir = Path(__file__).parent
    
    # ============================================================
    # STEP 1-3: Load Data, Initialize Strategies, Run Backtests
    # ============================================================
    prices, test_start_date = load_prices_and_test_start()
    strategies = init_strategies(prices, test_start_date)
    results = run_backtests_on_test_period(strategies, test_start_date)

    # ============================================================
    # STEP 4: Add SPY Benchmark
    # ============================================================
    spy_res = compute_spy_benchmark(test_start_date)
    if spy_res is not None:
        results['SPY'] = {
            'data': spy_res['data'],
            'final_value': None,
            'total_return': None,
            'metrics': spy_res['metrics']
        }

    # ============================================================
    # STEP 5: Save Results
    # ============================================================
    comparison_df = save_and_print_results(results)
    save_detailed_results(results)
    
    # ============================================================
    # STEP 6: Generate Plots
    # ============================================================
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Plot 1: Top strategy vs SPY
    print("\nGenerating: Top Strategy vs SPY Benchmark...")
    plot1 = plot_top_strategy_vs_benchmark(results, comparison_df)
    plot1_path = save_plot(plot1, "top_strategy_vs_spy.html", script_dir)
    if plot1_path:
        print(f"✓ Saved to: {plot1_path}")
    
    # Plot 2: Top 5 strategies vs SPY
    print("\nGenerating: Top 5 Strategies vs SPY Benchmark...")
    plot2 = plot_top_n_strategies_vs_benchmark(results, comparison_df, top_n=5)
    plot2_path = save_plot(plot2, "top_5_strategies_vs_spy.html", script_dir)
    if plot2_path:
        print(f"✓ Saved to: {plot2_path}")
    
    print("\n" + "="*70)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*70 + "\n")
    
    return results, comparison_df


if __name__ == "__main__":
    main()
