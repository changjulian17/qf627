"""Plotting helpers for strategy visualization."""
from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd


def plot_price_with_signals(df: pd.DataFrame, price_col: str = None, sig_col: str = 'trades'):
    """Plot price and mark buy/sell signals (where `sig_col` != 0)."""
    if price_col is None:
        price_col = df.columns[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[price_col], color='grey', label=price_col)
    trades = df[df[sig_col] != 0]
    buys = trades[trades[sig_col] > 0]
    sells = trades[trades[sig_col] < 0]
    ax.scatter(buys.index, df.loc[buys.index, price_col], marker='^', color='g', s=50, label='Buy')
    ax.scatter(sells.index, df.loc[sells.index, price_col], marker='v', color='r', s=50, label='Sell')
    ax.set_title('Price with Buy/Sell Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    plt.grid(True, alpha=0.3)
    return fig, ax


def plot_cumulative_returns(df: pd.DataFrame, cum_col: str = 'cum_strategy_returns'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[cum_col], lw=2)
    ax.set_title('Cumulative Strategy Returns')
    ax.set_ylabel('Growth Factor')
    ax.grid(True, alpha=0.3)
    return fig, ax


__all__ = ["plot_price_with_signals", "plot_cumulative_returns"]
