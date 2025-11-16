"""Plotting helpers for strategy visualization."""
from __future__ import annotations
import pandas as pd
import numpy as np
from lets_plot import *

LetsPlot.setup_html()

# Don't set a global size - let plots be responsive


def plot_price_with_signals(df: pd.DataFrame, price_col: str = None, sig_col: str = 'trades'):
    """Plot price and mark buy/sell signals (where `sig_col` != 0)."""
    if price_col is None:
        price_col = df.columns[0]

    # Prepare data
    plot_df = df[[price_col, sig_col]].copy()
    plot_df['date'] = plot_df.index
    
    trades = df[df[sig_col] != 0]
    buys = trades[trades[sig_col] > 0]
    sells = trades[trades[sig_col] < 0]
    
    buy_df = pd.DataFrame({
        'date': buys.index,
        'price': df.loc[buys.index, price_col],
        'type': 'Buy'
    })
    
    sell_df = pd.DataFrame({
        'date': sells.index,
        'price': df.loc[sells.index, price_col],
        'type': 'Sell'
    })
    
    signals_df = pd.concat([buy_df, sell_df])
    
    # Create plot with responsive size
    p = (ggplot() + 
        geom_line(aes(x='date', y=price_col), data=plot_df, color='gray', size=1) + 
        geom_point(aes(x='date', y='price', color='type', shape='type'), 
                   data=signals_df, size=3) + 
        scale_color_manual(values={'Buy': 'green', 'Sell': 'red'}) + 
        scale_shape_manual(values={'Buy': 24, 'Sell': 25}) + 
        labs(title='Price with Buy/Sell Signals',
             x='Date',
             y='Price') + 
        theme_light())
    
    return p


def plot_cumulative_returns(df: pd.DataFrame, cum_col: str = 'cum_strategy_returns'):
    """Plot cumulative returns."""
    plot_df = df[[cum_col]].copy()
    plot_df['date'] = plot_df.index
    
    p = (ggplot(plot_df, aes(x='date', y=cum_col)) + 
        geom_line(size=1.5, color='blue') + 
        labs(title='Cumulative Strategy Returns',
             x='Date',
             y='Growth Factor') + 
        theme_light())
    
    return p


def plot_top_strategy_vs_benchmark(results, comparison_df, initial_capital=100000):
    """Plot wealth curve of top strategy vs SPY with buy/sell signals.
    
    Args:
        results: Dict of strategy results from run_backtests_on_test_period
        comparison_df: DataFrame with strategy metrics sorted by Total Return
        initial_capital: Starting capital for wealth calculations
    
    Returns:
        lets_plot GGBunch with two plots
    """
    # Get top strategy name (first row, excluding SPY if present)
    top_strategies = comparison_df.index.tolist()
    top_strategy_name = None
    
    for strategy_name in top_strategies:
        if strategy_name != 'SPY':
            top_strategy_name = strategy_name
            break
    
    if top_strategy_name is None:
        print("Warning: No strategy found to plot.")
        return None
    
    # Get data for top strategy and SPY
    if top_strategy_name not in results or 'SPY' not in results:
        print(f"Warning: Missing data for {top_strategy_name} or SPY")
        return None
    
    top_data = results[top_strategy_name]['data']
    spy_data = results['SPY']['data']
    
    # Calculate wealth curves (normalized to start at initial_capital)
    top_wealth = top_data['cum_strategy_returns'] * initial_capital
    spy_wealth = spy_data['cum_passive_returns'] * initial_capital
    
    # Align the data on the same index (use inner join to ensure same length)
    wealth_df = pd.DataFrame({
        'strategy_wealth': top_wealth,
        'spy_wealth': spy_wealth
    }).dropna()  # Remove any NaN values
    
    # Reset index to get date as a column (robustly name the first column 'date')
    wealth_df = wealth_df.reset_index()
    if wealth_df.columns[0] != 'date':
        wealth_df = wealth_df.rename(columns={wealth_df.columns[0]: 'date'})
    
    # Reshape for lets-plot
    wealth_long = pd.concat([
        pd.DataFrame({
            'date': wealth_df['date'],
            'wealth': wealth_df['strategy_wealth'],
            'type': top_strategy_name
        }),
        pd.DataFrame({
            'date': wealth_df['date'],
            'wealth': wealth_df['spy_wealth'],
            'type': 'SPY (Buy & Hold)'
        })
    ])
    
    positions = top_data['positions']
    wealth_long["date"] = pd.to_datetime(wealth_long["date"])
    
    # Plot 1: Wealth curves with buy/sell signals
    p1 = (ggplot(wealth_long, aes(x='date', y='wealth', color='type')) + 
        geom_line(size=1.5, tooltips=layer_tooltips().format('@date', '%Y-%m-%d')) + 
        scale_color_manual(values={top_strategy_name: 'blue', 'SPY (Buy & Hold)': 'gray'}) + 
        labs(title=f'Wealth Curve: {top_strategy_name} vs SPY Benchmark',
             x='Date',
             y='Portfolio Value ($)',
             color='Strategy') + 
        theme_light() + 
        theme(legend_position="top", 
              panel_grid_minor='blank',
              panel_grid_major_x='blank') +
        scale_x_datetime(format="%b %Y") +
        ggsize(2000, 1200))
    
    # Add buy/sell signals if positions available
    if positions is not None:
        # Get SPY price from the data
        spy_price_col = spy_data.columns[0]
        spy_price = spy_data[spy_price_col]
        spy_price_normalized = (spy_price / spy_price.iloc[0]) * initial_capital

        # Find buy/sell signals (position state changes)
        buy_signals = (positions == 1) & (positions.shift(1) != 1)
        sell_signals = (positions != 1) & (positions.shift(1) == 1)

        signals_list = []
        if buy_signals.any():
            buy_dates = buy_signals[buy_signals].index
            buy_dates = spy_price_normalized.index.intersection(buy_dates)
            buy_prices = spy_price_normalized.loc[buy_dates]
            for date, price in zip(buy_dates, buy_prices):
                signals_list.append({'date': date, 'price': price, 'signal': 'Buy'})

        if sell_signals.any():
            sell_dates = sell_signals[sell_signals].index
            sell_dates = spy_price_normalized.index.intersection(sell_dates)
            sell_prices = spy_price_normalized.loc[sell_dates]
            for date, price in zip(sell_dates, sell_prices):
                signals_list.append({'date': date, 'price': price, 'signal': 'Sell'})

        if signals_list:
            signals_df = pd.DataFrame(signals_list)
            p1 = p1 + geom_point(aes(x='date', y='price', shape='signal', fill='signal'),
                                 data=signals_df, size=3, color='black') + \
                scale_shape_manual(values={'Buy': 24, 'Sell': 25}) + \
                scale_fill_manual(values={'Buy': 'green', 'Sell': 'red'})
    
    # Get metrics for annotation
    top_return = comparison_df.loc[top_strategy_name, 'Total Return']
    top_sharpe = comparison_df.loc[top_strategy_name, 'Sharpe Ratio']
    spy_return = comparison_df.loc['SPY', 'Total Return']
    spy_sharpe = comparison_df.loc['SPY', 'Sharpe Ratio']
    
    metrics_text = f'{top_strategy_name}: Total Return={top_return:.2%}, Sharpe={top_sharpe:.4f}\n'
    metrics_text += f'SPY: Total Return={spy_return:.2%}, Sharpe={spy_sharpe:.4f}\n'
    metrics_text += f'Alpha: {top_return - spy_return:.2%}'
    
    # # Plot 2: Position indicator
    # if positions is not None:
    #     position_df = pd.DataFrame({
    #         'date': positions.index,
    #         'positions': positions.values
    #     })
    #     
    #     # Map positions to labels
    #     position_df['position_label'] = position_df['positions'].map({
    #         1: 'Long', -1: 'Short', 0: 'Neutral'
    #     })
    #     
    #     # Calculate position statistics
    #     long_days = (positions == 1).sum()
    #     short_days = (positions == -1).sum()
    #     neutral_days = (positions == 0).sum()
    #     total_days = len(positions)
    #     
    #     position_text = f'Long: {long_days} ({long_days/total_days*100:.1f}%), '
    #     position_text += f'Short: {short_days} ({short_days/total_days*100:.1f}%), '
    #     position_text += f'Neutral: {neutral_days} ({neutral_days/total_days*100:.1f}%)'
    #     
    #     p2 = (
    #         ggplot(position_df, aes(x='date', y='positions', fill='position_label')) + 
    #         geom_bar(stat='identity', alpha=0.6) + 
    #         geom_point(
    #             aes(x='date', y='positions', shape='positions', fill='position_label'),
    #             data=position_df,
    #             size=6,
    #             color='black',
    #             stroke=1.5
    #         ) + 
    #         scale_fill_manual(values={'Long': 'green', 'Short': 'red', 'Neutral': 'gray'}) + 
    #         labs(title=f'Trading Positions - {position_text}',
    #              x='Date',
    #              y='Position',
    #              fill='Position Type') + 
    #         scale_y_continuous(breaks=[-1, 0, 1], labels=['Short', 'Neutral', 'Long']) + 
    #         theme_light() + 
    #         theme(legend_position='top')
    #         )
    # else:
    #     # Empty plot with message
    #     p2 = (ggplot() + 
    #         labs(title='Position data not available') + 
    #         theme_void())
    # 
    # # Combine plots vertically
    # from lets_plot import gggrid
    # grid = gggrid([p1, p2], ncol=1) + ggsize(2400, 1800)
    
    print(f"\n{metrics_text}")
    
    return p1
    
    # # Combine plots vertically
    # from lets_plot import gggrid
    # grid = gggrid([p1, p2], ncol=1) + ggsize(2400, 1800)
    # grid = gggrid([p1, p2], ncol=1) + ggsize(2400, 1800)
    
    # print(f"\n{metrics_text}")
    
    # return grid


def plot_top_n_strategies_vs_benchmark(results, comparison_df, top_n=5, initial_capital=100000):
    """Plot wealth curves of top N strategies vs SPY on the same axis.
    
    Args:
        results: Dict of strategy results from run_backtests_on_test_period
        comparison_df: DataFrame with strategy metrics sorted by Total Return
        top_n: Number of top strategies to plot
        initial_capital: Starting capital for wealth calculations
    
    Returns:
        lets_plot figure
    """
    # Get top N strategy names (excluding SPY)
    top_strategies = [name for name in comparison_df.index.tolist() 
                      if name != 'SPY'][:top_n]
    
    if not top_strategies:
        print("Warning: No strategies found to plot.")
        return None
    
    # Check if SPY exists
    if 'SPY' not in results:
        print("Warning: SPY benchmark not found in results.")
        return None
    
    # Prepare data for all strategies
    wealth_data_list = []
    
    # Add SPY benchmark
    spy_data = results['SPY']['data']
    spy_wealth = spy_data['cum_passive_returns'] * initial_capital
    
    for date, wealth in zip(spy_wealth.index, spy_wealth.values):
        wealth_data_list.append({
            'date': date,
            'wealth': wealth,
            'strategy': 'SPY (Benchmark)',
            'type': 'Benchmark'
        })
    
    # Add top N strategies
    for strategy_name in top_strategies:
        if strategy_name not in results:
            continue
        
        strategy_data = results[strategy_name]['data']
        strategy_wealth = strategy_data['cum_strategy_returns'] * initial_capital
        
        for date, wealth in zip(strategy_wealth.index, strategy_wealth.values):
            wealth_data_list.append({
                'date': date,
                'wealth': wealth,
                'strategy': strategy_name,
                'type': 'Strategy'
            })
    
    wealth_df = pd.DataFrame(wealth_data_list)
    
    # Create color palette
    n_strategies = len(top_strategies)
    strategy_colors = {}
    strategy_colors['SPY (Benchmark)'] = 'gray'
    
    # Use a color palette for strategies
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, strategy_name in enumerate(top_strategies):
        strategy_colors[strategy_name] = color_palette[i % len(color_palette)]
    
    # Create plot
    p = (ggplot(wealth_df, aes(x='date', y='wealth', color='strategy')) + 
        geom_line(size=1.5, alpha=0.8) + 
        scale_color_manual(values=strategy_colors) + 
        labs(title=f'Top {len(top_strategies)} Strategies vs SPY Benchmark',
             x='Date',
             y='Portfolio Value ($)',
             color='Strategy') + 
        theme_light() + 
        theme(legend_position='right') +
        ggsize(2000, 1200))
    
    # Add performance summary
    print("\n" + "="*80)
    print(f"TOP {len(top_strategies)} STRATEGIES PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"\n{'Rank':<6} {'Strategy':<40} {'Total Return':<12} {'Sharpe':<10} {'Max DD':<10}")
    print("-" * 80)
    
    spy_return = comparison_df.loc['SPY', 'Total Return']
    spy_sharpe = comparison_df.loc['SPY', 'Sharpe Ratio']
    spy_dd = comparison_df.loc['SPY', 'Max Drawdown']
    
    print(f"{'   -':<6} {'SPY (Benchmark)':<40} {spy_return:>10.2%} {spy_sharpe:>9.4f} {-spy_dd:>9.2%}")
    print("-" * 80)
    
    for i, strategy_name in enumerate(top_strategies, 1):
        ret = comparison_df.loc[strategy_name, 'Total Return']
        sharpe = comparison_df.loc[strategy_name, 'Sharpe Ratio']
        dd = comparison_df.loc[strategy_name, 'Max Drawdown']
        alpha = ret - spy_return
        
        print(f"{i:<6} {strategy_name:<40} {ret:>10.2%} {sharpe:>9.4f} {-dd:>9.2%}")
        print(f"{'':6} {'  → Alpha vs SPY:':<40} {alpha:>10.2%}")
    
    print("="*80 + "\n")
    
    return p


__all__ = [
    "plot_price_with_signals", 
    "plot_cumulative_returns",
    "plot_top_strategy_vs_benchmark",
    "plot_top_n_strategies_vs_benchmark"
]