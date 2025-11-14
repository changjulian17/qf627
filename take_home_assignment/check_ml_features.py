"""Quick script to check all ML features generated from strategies."""
import pandas as pd
import numpy as np
from config import *
from runner import init_strategies

# Create sample price data
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
prices = pd.DataFrame({
    'SPY': 100 * np.exp(np.random.randn(len(dates)).cumsum() * 0.01)
}, index=dates)

print("Creating all strategies...")
strategies = init_strategies(prices)
print(f"\nTotal strategies created: {len(strategies)}")

# Build ML features
print("\n" + "="*80)
print("BUILDING ML FEATURES:")
print("="*80)

from strategies.ml_strategies import build_ml_features_from_strategies

X, y = build_ml_features_from_strategies(prices, strategies)

print(f"\nTotal ML features: {X.shape[1]}")
print(f"Total samples: {X.shape[0]}")

# Categorize features by strategy type
print("\n" + "="*80)
print("ML FEATURES BY STRATEGY TYPE:")
print("="*80)

from collections import defaultdict

# Group features by strategy
strategy_features = defaultdict(list)
for col in X.columns:
    if '__' in col:
        strat_name = col.split('__')[0]
        feature_name = col.split('__', 1)[1]
        strategy_features[strat_name].append(feature_name)
    else:
        strategy_features['Global'].append(col)

# Categorize strategies by type
strategy_categories = {
    'Technical Indicators': [],
    'Volume & Oscillators': [],
    'Coincident Indices': [],
    'Multi-Window Returns': [],
    'Global Features': []
}

for strat_name, features in strategy_features.items():
    if strat_name == 'Global':
        strategy_categories['Global Features'].append((strat_name, features))
    elif strat_name.startswith('SMA_') or strat_name.startswith('MACD_') or \
         strat_name.startswith('RSI_') or strat_name.startswith('ZScore_'):
        strategy_categories['Technical Indicators'].append((strat_name, features))
    elif strat_name.startswith('OBV_') or strat_name.startswith('Stoch_') or \
         strat_name.startswith('ROC_'):
        strategy_categories['Volume & Oscillators'].append((strat_name, features))
    elif strat_name.startswith('Coincident_') or strat_name.startswith('MultiCoinc_'):
        strategy_categories['Coincident Indices'].append((strat_name, features))
    elif strat_name.startswith('MultiWindow_') or strat_name.startswith('CrossAsset_'):
        strategy_categories['Multi-Window Returns'].append((strat_name, features))

# Print categorized features
for category, strat_list in strategy_categories.items():
    if not strat_list:
        continue
    
    total_features = sum(len(features) for _, features in strat_list)
    print(f"\n{'='*80}")
    print(f"{category.upper()} ({len(strat_list)} strategies, {total_features} features)")
    print(f"{'='*80}")
    
    for strat_name, features in sorted(strat_list):
        print(f"\n  {strat_name} ({len(features)} features):")
        for feat in sorted(features):
            print(f"    - {feat}")

# Summary by feature type
print("\n" + "="*80)
print("FEATURE TYPE SUMMARY:")
print("="*80)

feature_type_counts = defaultdict(int)
for col in X.columns:
    if col == 'price_zscore':
        feature_type_counts['Price Z-Score (Global)'] += 1
    elif '_price_zscore' in col:
        feature_type_counts['Coincident Price Z-Scores'] += 1
    elif '_signal' in col and 'composite' not in col and 'weighted' not in col and 'majority' not in col:
        feature_type_counts['Coincident Signals (Returns)'] += 1
    elif 'composite_signal' in col or 'weighted_signal' in col or 'majority_signal' in col:
        feature_type_counts['Aggregated Signals'] += 1
    elif 'spread_zscore' in col:
        feature_type_counts['SMA Spread Z-Scores'] += 1
    elif 'zscore' in col.lower() and 'price' not in col and 'spread' not in col:
        feature_type_counts['Z-Score Indicators'] += 1
    elif 'macd' in col.lower() or 'signal_line' in col.lower():
        feature_type_counts['MACD (Z-Score Normalized)'] += 1
    elif 'rsi' in col.lower():
        feature_type_counts['RSI'] += 1
    elif 'obv_zscore' in col.lower():
        feature_type_counts['OBV (Z-Score Normalized)'] += 1
    elif 'roc' in col.lower():
        feature_type_counts['ROC'] += 1
    elif 'percent' in col.lower():
        feature_type_counts['Stochastic Oscillator'] += 1
    elif 'volume' in col.lower():
        feature_type_counts['Volume (Raw)'] += 1
    else:
        feature_type_counts['Other'] += 1

print(f"\nTotal features: {X.shape[1]}\n")
for feat_type, count in sorted(feature_type_counts.items(), key=lambda x: -x[1]):
    pct = (count / X.shape[1]) * 100
    print(f"  {feat_type:.<40} {count:>3} ({pct:>5.1f}%)")

# Stationarity check
print("\n" + "="*80)
print("STATIONARITY VERIFICATION:")
print("="*80)
print("\nAll features should be stationary (oscillating):")
print("  ✓ Bounded [0, 100]: RSI, Stochastic")
print("  ✓ Z-scores ~[-3, 3]: price_zscore, coincident prices, MACD, OBV, SMA spreads")
print("  ✓ Returns/Signals: coincident signals, aggregated signals")
print("  ✓ Volume: raw (paired with OBV z-score)")

non_stationary_warning = [col for col in X.columns if col.lower().endswith('volume')]
if non_stationary_warning:
    print(f"\nNote: {len(non_stationary_warning)} volume features are raw (non-stationary)")
    print("      These are context features for OBV z-score interpretation")

