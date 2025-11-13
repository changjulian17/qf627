"""Main execution script for the trading strategy project.

This file now shows the high-level logical flow by calling the
small functions implemented in `runner.py`. That way you can
review the orchestration steps in `main.py` while the heavy
lifting remains in `runner.py`.
"""
from runner import (
    load_prices_and_test_start,
    init_strategies,
    run_backtests_on_test_period,
    compute_spy_benchmark,
    save_and_print_results,
)


def main():
    """High-level orchestration visible in main for review.

    Steps:
    1. Load prices and compute test start date
    2. Initialize strategies
    3. Run backtests restricted to the test period
    4. Compute SPY benchmark metrics for the test period
    5. Save and print the comparison table
    """
    # 1. Load data and compute test split point
    prices, test_start_date = load_prices_and_test_start()

    # 2. Build strategy instances (including ML strategy trained on rule-based features)
    strategies = init_strategies(prices, test_start_date)

    # 3. Run backtests on the test period
    results = run_backtests_on_test_period(strategies, test_start_date)

    # 4. Add SPY benchmark metrics (passive buy-and-hold)
    spy_res = compute_spy_benchmark(test_start_date)
    if spy_res is not None:
        results['SPY'] = {
            'data': spy_res['data'],
            'final_value': None,
            'total_return': None,
            'metrics': spy_res['metrics']
        }

    # 5. Save and print
    comparison_df = save_and_print_results(results)
    return results, comparison_df


if __name__ == "__main__":
    results, comparison = main()