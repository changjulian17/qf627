# ML Features Data Dictionary

**Project:** QF627 Take-Home Assignment - Trading Strategy Backtesting System

**Total Features:** 194 stationary features for machine learning models

**Total Strategies:** 100 rule-based strategies generating features

**Last Updated:** November 16, 2025

---

## Overview

This document describes all machine learning features used in the ML-based trading strategies. All features are engineered to be **stationary** (mean-reverting, no unit root) to improve model performance and prevent overfitting to trending data.

The features are derived from **100 rule-based strategies** across multiple categories:
- **Technical indicators** (11 strategies: MACD, RSI, SMA, Z-Score)
- **Volume & oscillators** (7 strategies: OBV, Stochastic, ROC)
- **Coincident indices** (69 strategies: cross-asset correlations)
- **Multi-window returns** (4 strategies: aggregated signals across timeframes)

All non-stationary features (e.g., raw prices) are transformed using:
- **Z-score normalization** - Centers data with zero mean, unit variance
- **Percentage returns** - Daily/periodic price changes
- **Spread ratios** - Normalized price differences

### Feature Breakdown by Type

| Feature Type | Notes |
|-------------|-------|
| Coincident Signals (Returns) | Expanded coverage at 5, 20, 252-day windows across currencies, sectors, and market indices. |
| Coincident Price Z-Scores | Added 5 and 252-day variants alongside 20-day context where applicable. |
| Aggregated/Composite Signals | Multi-asset composite and multi-window weighted/majority signals updated to 5/20/252 schemas. |
| Technical Indicators | RSI, MACD, SMA spreads, Z-Score maintained. |
| Oscillators & Volume | Stochastic, ROC, OBV (z-scored) retained. |
| Global Price Z-Score | Core normalized SPY price feature retained. |
| **Total** | **194** features generated this run. |

---

## Feature Categories

### 1. Price Features (1 feature)

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `price_zscore` | Z-score normalized SPY price | Z-score | ~[-3, 3] | Stationary |

**Purpose:** Normalized representation of the primary asset (SPY) price, making it stationary and comparable across different time periods.

---

### 2. Technical Indicator Features (25 features)

#### 2.1 RSI (Relative Strength Index) - 4 features

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `RSI_p14_os30_ob70__rsi` | RSI with 14-day period, oversold=30, overbought=70 | Oscillator | [0, 100] | Stationary |
| `RSI_p14_os25_ob75__rsi` | RSI with 14-day period, oversold=25, overbought=75 | Oscillator | [0, 100] | Stationary |
| `RSI_p21_os30_ob70__rsi` | RSI with 21-day period, oversold=30, overbought=70 | Oscillator | [0, 100] | Stationary |
| `RSI_p21_os25_ob75__rsi` | RSI with 21-day period, oversold=25, overbought=75 | Oscillator | [0, 100] | Stationary |

**Purpose:** Momentum oscillators indicating overbought/oversold conditions. Values >70 suggest overbought, <30 suggest oversold.

#### 2.2 MACD (Moving Average Convergence Divergence) - 6 features

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `MACD_12_26_9__macd` | MACD line (12,26,9) z-score normalized | Z-score | ~[-3, 3] | Stationary |
| `MACD_12_26_9__signal_line` | MACD signal line (12,26,9) z-score normalized | Z-score | ~[-3, 3] | Stationary |
| `MACD_5_35_5__macd` | Fast MACD line (5,35,5) z-score normalized | Z-score | ~[-3, 3] | Stationary |
| `MACD_5_35_5__signal_line` | Fast MACD signal line (5,35,5) z-score normalized | Z-score | ~[-3, 3] | Stationary |
| `MACD_19_39_9__macd` | Slow MACD line (19,39,9) z-score normalized | Z-score | ~[-3, 3] | Stationary |
| `MACD_19_39_9__signal_line` | Slow MACD signal line (19,39,9) z-score normalized | Z-score | ~[-3, 3] | Stationary |

**Purpose:** Trend-following momentum indicators. Positive values suggest bullish momentum, negative suggest bearish. Signal line crossovers indicate potential trend changes.

#### 2.3 SMA Spread Z-Score - 3 features

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `SMA_20_60__spread_zscore` | Z-score of (SMA_20 - SMA_60) spread | Z-score | ~[-3, 3] | Stationary |
| `SMA_50_200__spread_zscore` | Z-score of (SMA_50 - SMA_200) spread (Golden/Death Cross) | Z-score | ~[-3, 3] | Stationary |
| `SMA_24_58__spread_zscore` | Z-score of (SMA_24 - SMA_58) spread | Z-score | ~[-3, 3] | Stationary |

**Purpose:** Normalized difference between fast and slow moving averages. Positive values suggest uptrend, negative suggest downtrend.

#### 2.4 Z-Score Strategy - 1 feature

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `ZScore_42__zscore` | 42-period z-score of price for mean reversion | Z-score | ~[-3, 3] | Stationary |

**Purpose:** Mean reversion indicator. Extreme values (>2 or <-2) suggest potential reversals.

#### 2.5 OBV (On-Balance Volume) - 4 features

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `OBV_SPY_20__volume` | Trading volume (20-period context) | Volume | [0, ∞) | Non-stationary |
| `OBV_SPY_20__obv_zscore` | Z-score normalized OBV (20-period) | Z-score | ~[-3, 3] | Stationary |
| `OBV_SPY_50__volume` | Trading volume (50-period context) | Volume | [0, ∞) | Non-stationary |
| `OBV_SPY_50__obv_zscore` | Z-score normalized OBV (50-period) | Z-score | ~[-3, 3] | Stationary |

**Purpose:** Volume-based momentum indicator. Positive OBV z-score suggests buying pressure, negative suggests selling pressure.

#### 2.6 Stochastic Oscillator - 4 features

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Stoch_SPY_k14_d3_os20_ob80__k_percent` | %K line of stochastic (14,3) | Oscillator | [0, 100] | Stationary |
| `Stoch_SPY_k14_d3_os20_ob80__d_percent` | %D line (signal) of stochastic (14,3) | Oscillator | [0, 100] | Stationary |
| `Stoch_SPY_k5_d3_os20_ob80__k_percent` | Fast %K line of stochastic (5,3) | Oscillator | [0, 100] | Stationary |
| `Stoch_SPY_k5_d3_os20_ob80__d_percent` | Fast %D line (signal) of stochastic (5,3) | Oscillator | [0, 100] | Stationary |

**Purpose:** Momentum oscillators comparing closing price to price range. Values >80 suggest overbought, <20 suggest oversold.

#### 2.7 ROC (Rate of Change) - 3 features

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `ROC_p12_th0__roc` | 12-period rate of change | Percentage | (-∞, ∞) | Stationary |
| `ROC_p20_th0__roc` | 20-period rate of change | Percentage | (-∞, ∞) | Stationary |
| `ROC_p5_th0__roc` | Fast 5-period rate of change | Percentage | (-∞, ∞) | Stationary |

**Purpose:** Momentum indicators measuring percentage price change. Positive values suggest upward momentum, negative suggest downward.

---

### 3. Multi-Window Return Features (updated to 5/20/252 windows)

#### 3.1 Same-Asset Multi-Window Returns

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `MultiWindow_weighted_average_5_20_252__ret_5d` | 5-day log return of SPY | Return | (-∞, ∞) | Stationary |
| `MultiWindow_weighted_average_5_20_252__ret_20d` | 20-day log return of SPY | Return | (-∞, ∞) | Stationary |
| `MultiWindow_weighted_average_5_20_252__ret_252d` | 252-day log return of SPY | Return | (-∞, ∞) | Stationary |
| `MultiWindow_weighted_average_5_20_252__weighted_signal` | Weighted average signal from 5/20/252 windows | Signal | [-1, 1] | Stationary |
| `MultiWindow_majority_vote_5_20_252__ret_5d` | 5-day log return of SPY | Return | (-∞, ∞) | Stationary |
| `MultiWindow_majority_vote_5_20_252__ret_20d` | 20-day log return of SPY | Return | (-∞, ∞) | Stationary |
| `MultiWindow_majority_vote_5_20_252__ret_252d` | 252-day log return of SPY | Return | (-∞, ∞) | Stationary |
| `MultiWindow_majority_vote_5_20_252__majority_signal` | Majority vote signal from 5/20/252 windows | Signal | {-1, 0, 1} | Stationary |

**Purpose:** Multi-timeframe momentum indicators capturing short, medium, and long-term trends simultaneously.

#### 3.2 Cross-Asset Returns (SPY/QQQ, 5/20/252 windows)

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `CrossAsset_SPY_weighted_average_5_20_252__SPY_ret_5d` | SPY 5-day log return | Return | (-∞, ∞) | Stationary |
| `CrossAsset_SPY_weighted_average_5_20_252__SPY_ret_20d` | SPY 20-day log return | Return | (-∞, ∞) | Stationary |
| `CrossAsset_SPY_weighted_average_5_20_252__SPY_ret_252d` | SPY 252-day log return | Return | (-∞, ∞) | Stationary |
| `CrossAsset_SPY_weighted_average_5_20_252__weighted_signal` | Weighted signal from SPY returns | Signal | [-1, 1] | Stationary |
| `CrossAsset_QQQ_majority_vote_5_20_252__QQQ_ret_5d` | QQQ 5-day log return | Return | (-∞, ∞) | Stationary |
| `CrossAsset_QQQ_majority_vote_5_20_252__QQQ_ret_20d` | QQQ 20-day log return | Return | (-∞, ∞) | Stationary |
| `CrossAsset_QQQ_majority_vote_5_20_252__QQQ_ret_252d` | QQQ 252-day log return | Return | (-∞, ∞) | Stationary |
| `CrossAsset_QQQ_majority_vote_5_20_252__majority_signal` | Majority vote signal from QQQ returns | Signal | {-1, 0, 1} | Stationary |

**Purpose:** Cross-asset momentum indicators using correlated tech-heavy QQQ to predict SPY movements.

---

### 4. Coincident Index Features (expanded, 5/20/252 windows)

These features track currency pairs, market indices, and sector ETFs that have economic relationships with SPY.

#### 4.1 Currency Index Features (16 features)

**DX-Y.NYB (US Dollar Index):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_DX-Y.NYB_5/20/252__DX-Y.NYB_signal` | Rolling mean return of DXY | Return | (-∞, ∞) | Stationary |
| `Coincident_DX-Y.NYB_5/20/252__DX-Y.NYB_price_zscore` | Z-score normalized DXY price | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__DX-Y.NYB_signal` | DXY signal in multi-currency strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__DX-Y.NYB_price_zscore` | DXY price z-score in multi-currency strategy | Z-score | ~[-3, 3] | Stationary |

**EURUSD=X (Euro/USD Exchange Rate):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_EURUSD=X_5/20/252__EURUSD=X_signal` | Rolling mean return of EUR/USD | Return | (-∞, ∞) | Stationary |
| `Coincident_EURUSD=X_5/20/252__EURUSD=X_price_zscore` | Z-score normalized EUR/USD rate | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__EURUSD=X_signal` | EUR/USD signal in multi-currency strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__EURUSD=X_price_zscore` | EUR/USD z-score in multi-currency strategy | Z-score | ~[-3, 3] | Stationary |

**JPY=X (Japanese Yen/USD Exchange Rate):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_JPY=X_5/20/252__JPY=X_signal` | Rolling mean return of JPY/USD | Return | (-∞, ∞) | Stationary |
| `Coincident_JPY=X_5/20/252__JPY=X_price_zscore` | Z-score normalized JPY/USD rate | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__JPY=X_signal` | JPY/USD signal in multi-currency strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__JPY=X_price_zscore` | JPY/USD z-score in multi-currency strategy | Z-score | ~[-3, 3] | Stationary |

**GBPUSD=X (British Pound/USD Exchange Rate):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_GBPUSD=X_5/20/252__GBPUSD=X_signal` | Rolling mean return of GBP/USD | Return | (-∞, ∞) | Stationary |
| `Coincident_GBPUSD=X_5/20/252__GBPUSD=X_price_zscore` | Z-score normalized GBP/USD rate | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__GBPUSD=X_signal` | GBP/USD signal in multi-currency strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_DXY_EUR_JPY_GBP_20__GBPUSD=X_price_zscore` | GBP/USD z-score in multi-currency strategy | Z-score | ~[-3, 3] | Stationary |

**Purpose:** Currency strength indicators. Dollar strength (DXY up) often correlates with equity weakness; foreign currency strength may indicate risk-on sentiment.

#### 4.2 Market Index Features (72 features)

**^VIX (CBOE Volatility Index):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_^VIX_5/20/252__^VIX_signal` | Rolling mean return of VIX | Return | (-∞, ∞) | Stationary |
| `Coincident_^VIX_5/20/252__^VIX_price_zscore` | Z-score normalized VIX level | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__^VIX_signal` | VIX signal in multi-market strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__^VIX_price_zscore` | VIX z-score in multi-market strategy | Z-score | ~[-3, 3] | Stationary |

**^TNX (10-Year Treasury Yield):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_^TNX_5/20/252__^TNX_signal` | Rolling mean return of 10Y yield | Return | (-∞, ∞) | Stationary |
| `Coincident_^TNX_5/20/252__^TNX_price_zscore` | Z-score normalized 10Y yield | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__^TNX_signal` | TNX signal in multi-market strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__^TNX_price_zscore` | TNX z-score in multi-market strategy | Z-score | ~[-3, 3] | Stationary |

**GLD (Gold ETF):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_GLD_5/20/252__GLD_signal` | Rolling mean return of gold | Return | (-∞, ∞) | Stationary |
| `Coincident_GLD_5/20/252__GLD_price_zscore` | Z-score normalized gold price | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__GLD_signal` | GLD signal in multi-market strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__GLD_price_zscore` | GLD z-score in multi-market strategy | Z-score | ~[-3, 3] | Stationary |

**TLT (20+ Year Treasury Bond ETF):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_TLT_5/20/252__TLT_signal` | Rolling mean return of long bonds | Return | (-∞, ∞) | Stationary |
| `Coincident_TLT_5/20/252__TLT_price_zscore` | Z-score normalized TLT price | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__TLT_signal` | TLT signal in multi-market strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__TLT_price_zscore` | TLT z-score in multi-market strategy | Z-score | ~[-3, 3] | Stationary |

**USO (Oil ETF):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_USO_5/20/252__USO_signal` | Rolling mean return of oil | Return | (-∞, ∞) | Stationary |
| `Coincident_USO_5/20/252__USO_price_zscore` | Z-score normalized oil price | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__USO_signal` | USO signal in multi-market strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__USO_price_zscore` | USO z-score in multi-market strategy | Z-score | ~[-3, 3] | Stationary |

**UUP (US Dollar Bullish ETF):** (windows: 5, 20, 252)
| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `Coincident_UUP_5/20/252__UUP_signal` | Rolling mean return of USD ETF | Return | (-∞, ∞) | Stationary |
| `Coincident_UUP_5/20/252__UUP_price_zscore` | Z-score normalized UUP price | Z-score | ~[-3, 3] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__UUP_signal` | UUP signal in multi-market strategy | Return | (-∞, ∞) | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__UUP_price_zscore` | UUP z-score in multi-market strategy | Z-score | ~[-3, 3] | Stationary |

#### 4.3 S&P 500 Sector ETF Features (44 features)

Each sector follows the same pattern with `_signal` and `_price_zscore` variants, now generated at 5, 20 and 252-day windows, in both single and multi-coincident strategies:

**XLK (Technology Sector):**
- `Coincident_XLK_20__XLK_signal` / `Coincident_XLK_20__XLK_price_zscore`
- `MultiCoinc_...__XLK_signal` / `MultiCoinc_...__XLK_price_zscore`

**XLF (Financial Sector):**
- `Coincident_XLF_20__XLF_signal` / `Coincident_XLF_20__XLF_price_zscore`
- `MultiCoinc_...__XLF_signal` / `MultiCoinc_...__XLF_price_zscore`

**XLV (Healthcare Sector):**
- `Coincident_XLV_20__XLV_signal` / `Coincident_XLV_20__XLV_price_zscore`
- `MultiCoinc_...__XLV_signal` / `MultiCoinc_...__XLV_price_zscore`

**XLE (Energy Sector):**
- `Coincident_XLE_20__XLE_signal` / `Coincident_XLE_20__XLE_price_zscore`
- `MultiCoinc_...__XLE_signal` / `MultiCoinc_...__XLE_price_zscore`

**XLI (Industrial Sector):**
- `Coincident_XLI_20__XLI_signal` / `Coincident_XLI_20__XLI_price_zscore`
- `MultiCoinc_...__XLI_signal` / `MultiCoinc_...__XLI_price_zscore`

**XLP (Consumer Staples Sector):**
- `Coincident_XLP_20__XLP_signal` / `Coincident_XLP_20__XLP_price_zscore`
- `MultiCoinc_...__XLP_signal` / `MultiCoinc_...__XLP_price_zscore`

**XLY (Consumer Discretionary Sector):**
- `Coincident_XLY_20__XLY_signal` / `Coincident_XLY_20__XLY_price_zscore`
- `MultiCoinc_...__XLY_signal` / `MultiCoinc_...__XLY_price_zscore`

**XLU (Utilities Sector):**
- `Coincident_XLU_20__XLU_signal` / `Coincident_XLU_20__XLU_price_zscore`
- `MultiCoinc_...__XLU_signal` / `MultiCoinc_...__XLU_price_zscore`

**XLRE (Real Estate Sector):**
- `Coincident_XLRE_20__XLRE_signal` / `Coincident_XLRE_20__XLRE_price_zscore`
- `MultiCoinc_...__XLRE_signal` / `MultiCoinc_...__XLRE_price_zscore`

**XLB (Materials Sector):**
- `Coincident_XLB_20__XLB_signal` / `Coincident_XLB_20__XLB_price_zscore`
- `MultiCoinc_...__XLB_signal` / `MultiCoinc_...__XLB_price_zscore`

**XLC (Communication Services Sector):**
- `Coincident_XLC_20__XLC_signal` / `Coincident_XLC_20__XLC_price_zscore`
- `MultiCoinc_...__XLC_signal` / `MultiCoinc_...__XLC_price_zscore`

**Purpose:** Sector rotation indicators. Strong/weak sector performance can predict overall market movements.

#### 4.4 Composite Signals (2 features)

| Feature Name | Description | Type | Range | Stationarity |
|-------------|-------------|------|-------|--------------|
| `MultiCoinc_DXY_EUR_JPY_GBP_20__composite_signal` | Mean aggregation of 4 currency signals | Signal | [-1, 1] | Stationary |
| `MultiCoinc_VIX_TNX_GLD_TLT_USO_UUP_XLK_XLF_XLV_XLE_XLI_XLP_XLY_XLU_XLR_XLB_XLC_20__composite_signal` | Majority vote of 17 market index signals | Signal | {-1, 0, 1} | Stationary |

**Purpose:** Aggregated sentiment from multiple correlated assets, providing ensemble-based predictions.

---

## Feature Engineering Principles

### Stationarity Transformation
All features are made stationary through one of these methods:
1. **Bounded Oscillators**: RSI, Stochastic (naturally bounded [0,100])
2. **Z-Score Normalization**: Prices, MACD, OBV, SMA spreads (transformed to ~[-3,3])
3. **Returns/Differences**: Multi-window returns, coincident signals (stationary by construction)

### Window Sizes
- **Short-term**: 5-20 days (capturing immediate momentum)
- **Medium-term**: 20-60 days (capturing trend persistence)
- **Long-term**: 200 days (for z-score normalization, ZSCORE_WINDOW=200)

### No Leakage
- All features use only historical data available at time t
- Forward returns are excluded from features (used only as target variable)
- No future information is incorporated

---

## Target Variable (Not Included in Features)

| Variable Name | Description | Type | Range |
|--------------|-------------|------|-------|
| `fwd_log_ret` | Forward 1-day log return of SPY | Return | (-∞, ∞) |

**Calculation:** `log(price[t+1] / price[t])`

**Purpose:** Predict next-day return for long/short position generation

---

## Data Quality Notes

1. **Missing Data Handling**: NaN values are dropped after feature/target concatenation
2. **Timezone Alignment**: All external data (yfinance) is timezone-localized to match primary data
3. **Volume Features**: Raw volume included only as context for OBV z-score interpretation
4. **Redundancy Elimination**: 
   - Raw prices excluded (use z-scores instead)
   - Raw SMAs/EMAs excluded (use spreads instead)
   - Duplicate price columns from strategies excluded
   - Close price excluded (redundant with price feature)

---

## Usage Recommendations

### For Regression Models
- Use all 167 features
- Target: `fwd_log_ret` (continuous)
- Convert predictions to positions: long if pred > 0, short if pred < 0

### For Classification Models
- Bin target into classes: {-1: negative return, 0: ~zero return, 1: positive return}
- Use all 167 features
- Directly predict position {-1, 0, 1}

### Feature Selection
**High-Information Features:**
- `price_zscore`: Core momentum indicator
- Coincident price z-scores (66 features): Cross-asset regime detection
- Coincident signals (69 features): Cross-asset momentum
- Multi-window returns: Multi-timeframe momentum aggregation
- RSI/Stochastic: Overbought/oversold conditions

**Correlation Groups** (consider dimensionality reduction):
- Multiple RSI configurations (4 features)
- Multiple MACD configurations (6 features)  
- Multiple Stochastic configurations (4 features)
- Sector ETFs (11 sectors × 3 windows = 66 features) - high correlation within sectors
- Currency pairs (4 currencies × 3 windows = 24 features)

### Stationarity Verification

All features are designed to be stationary:
- ✅ **Bounded [0, 100]**: RSI (4), Stochastic (4)
- ✅ **Z-scores ~[-3, 3]**: Price z-scores (67), MACD (6), OBV (2), SMA spreads (3)
- ✅ **Returns/Signals**: Coincident signals (69), aggregated signals (7), returns (2)
- ⚠️ **Volume (2)**: Raw volume features are non-stationary but provide context for OBV z-scores

---

## Current Feature Summary (As of November 14, 2025)

### By Strategy Category

| Category | Strategies | Features | Description |
|----------|-----------|----------|-------------|
| **Technical Indicators** | 11 | 14 | MACD, RSI, SMA spreads, Z-Score |
| **Volume & Oscillators** | 7 | 11 | OBV, Stochastic, ROC |
| **Coincident Indices** | 69 | 135 | Cross-asset signals and price z-scores |
| **Multi-Window Returns** | 4 | 6 | Aggregated multi-timeframe signals |
| **Global Features** | 1 | 1 | SPY price z-score |
| **Total** | **91** | **167** | All features |

### Asset Coverage

**Currencies (4):**
- DX-Y.NYB (US Dollar Index)
- EURUSD=X (Euro)
- JPY=X (Japanese Yen)
- GBPUSD=X (British Pound)

**Market Indicators (2):**
- ^VIX (Volatility Index)
- ^TNX (10-Year Treasury Yield)

**Commodities (3):**
- GLD (Gold)
- TLT (Long-Term Bonds)
- USO (Oil)
- UUP (US Dollar ETF)

**Sector ETFs (11):**
- XLK (Technology)
- XLF (Financials)
- XLV (Healthcare)
- XLE (Energy)
- XLI (Industrials)
- XLP (Consumer Staples)
- XLY (Consumer Discretionary)
- XLU (Utilities)
- XLRE (Real Estate)
- XLB (Materials)
- XLC (Communication Services)

**Primary Asset:**
- SPY (S&P 500 ETF)
- QQQ (Nasdaq ETF) - in cross-asset features

### Window Periods Used

- **Short-term**: 5 days
- **Medium-term**: 20 days (monthly)
- **Long-term**: 252 days (yearly)

---

## Changelog

**v2.1 - November 16, 2025**
- Updated to 194 features (from 167) across 100 strategies.
- Standardized multi-window and cross-asset features to 5/20/252 schemas.
- Expanded coincident features to include 5 and 252-day windows across assets.
- Updated documentation sections to reflect new windows and feature names.

**v2.0 - November 14, 2025**
- Updated to 167 features (from 130)
- Added 91 rule-based strategies generating features
- Comprehensive asset coverage: 4 currencies, 11 sectors, 3 commodities, 2 market indicators
- All features verified for stationarity
- Detailed feature type breakdown added

**v1.0 - November 14, 2025**
- Initial data dictionary
- 130 stationary features (outdated)
- Basic feature engineering documented
