# QF627 Take-Home Assignment

Quantitative trading strategy backtesting system with multiple algorithmic approaches and machine learning integration.

## 📁 Project Structure

```
take_home_assignment/
├── main.py                 # Main execution script
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
│
├── data/                  # Data loading and management
├── features/              # Technical indicators & feature engineering
├── strategies/            # Trading strategy implementations
│   ├── momentum/         # Momentum-based strategies
│   ├── mean_reversion/   # Mean reversion strategies
│   └── ml/              # Machine learning strategies
├── backtesting/          # Backtesting engine and metrics
├── visualisation/        # Plotting and visualization tools
├── utils/                # Utility functions
│   ├── runner.py        # Orchestration logic
│   ├── helpers.py       # Helper functions
│   └── checkpoint_manager.py  # Model checkpoint management
│
├── checkpoints/          # Saved model checkpoints
├── results/             # Output files
│   ├── plots/          # Generated visualizations
│   └── strategy_data/  # Per-strategy CSV results
└── scripts/            # Utility scripts
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
python -m pip install -r requirements.txt
```

### Running the System

```bash
# Run full backtest pipeline (generates results and plots)
python main.py

# Generate plots from existing results
python generate_plots.py
```

## 📊 Output

The system generates:

1. **Strategy Comparison CSV** (`strategy_comparison.csv`) - Performance metrics for all strategies
2. **Detailed Results** (`results/strategy_data/*.csv`) - Per-strategy time series data
3. **Visualizations** (`results/plots/*.html`) - Interactive plots:
   - Top strategy vs SPY benchmark with buy/sell signals
   - Top 5 strategies comparison

## 🎯 Features

- **Multiple Strategy Types**: Momentum, mean reversion, coincident, and ML-based
- **Comprehensive Metrics**: Returns, Sharpe ratio, max drawdown, win rate, etc.
- **Interactive Visualizations**: HTML plots with tooltips showing full dates and values
- **Benchmark Comparison**: All strategies compared against SPY buy-and-hold
- **ML Integration**: Machine learning strategies trained on rule-based features

## 📝 Documentation

- `ML_FEATURES_DATA_DICTIONARY.md` - Documentation of ML features used in strategies

## 🔧 Configuration

Edit `config.py` to modify:
- Train/test split dates
- Strategy parameters
- Initial capital
- Data sources
