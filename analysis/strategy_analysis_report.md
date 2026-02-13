# Strategy Analysis Report

## Executive Summary

This report provides a detailed technical analysis of three trading strategies:
1. **High Beta Growth Strategy** - Aggressive growth stock selection
2. **Bear Beta Strategy** - Defensive stock selection for downturns
3. **Regime Adaptive Strategy** - Dynamic blending based on market regime

---

## 1. High Beta Growth Strategy

### Overview
The High Beta Growth Strategy identifies stocks with characteristics historically associated with market outperformance. Based on analysis of S&P 500 stocks from 2014-2026, it targets high-beta, high-growth, high-profitability stocks.

### Historical Findings (from analysis)
- Outperformers had average Beta of **1.26** vs **0.83** for underperformers
- **92%** of top performers had Beta > 1.0
- **Technology sector** represented 64% of outperformers
- High profitability (ROE > 25%) combined with high growth (>15%)

### Data Sources

| Data Type | Source | Lookback Period | Update Frequency |
|-----------|--------|-----------------|------------------|
| Price History | yfinance | 504 days (~2 years) | On rebalance |
| Fundamentals | yfinance API | Current snapshot | 7-day cache |
| Beta | Calculated | 504 days | 30-day cache |
| Sector Data | yfinance API | Current | 7-day cache |

### Position Selection Process

#### Step 1: Candidate Universe (Lines 476-488)
```
1. Load tickers from fundamentals database with >= 3 years of data
2. Filter out invalid tickers (warrants, units, rights)
3. Prioritize PRIORITY_TICKERS (top S&P 500 + historical outperformers)
4. Evaluate up to 500 candidates total
```

**Priority Tickers Include:**
- Top S&P 500 by weight: AAPL, MSFT, NVDA, GOOGL, AMZN, META, etc.
- Historical outperformers: LRCX, ANET, KLAC, CDNS, FTNT, MU, PANW, etc.

#### Step 2: Beta Calculation (Lines 188-222)
```python
Beta = Covariance(stock_returns, SPY_returns) / Variance(SPY_returns)
```
- Uses 504 trading days (~2 years) of price data
- Aligns stock and SPY returns by date
- Requires minimum 200 common data points
- Cached for 30 days to reduce API calls

#### Step 3: Fundamentals Fetching (Lines 224-251)
From yfinance API:
- **Profitability**: ROE, ROA, Operating Margin, Gross Margin, Profit Margin
- **Growth**: Revenue Growth, Earnings Growth
- **Balance Sheet**: Debt/Equity, Current Ratio, Free Cash Flow
- **Size**: Market Cap
- **Classification**: Sector

#### Step 4: Scoring System (Lines 354-455)

**Total Score: 0-100 points**

| Component | Max Points | Criteria |
|-----------|------------|----------|
| **Profitability** | 30 | ROE > 15% (+10), ROE > 25% (+5), OpMargin > 15% (+10), GrossMargin > 40% (+5) |
| **Growth** | 25 | RevGrowth > 10% (+10), RevGrowth > 20% (+5), EarnGrowth > 15% (+10) |
| **Balance Sheet** | 15 | D/E < 100 (+5), D/E < 50 (+5), FCF > 0 (+5) |
| **Beta** | 15 | Beta > 1.0 (+10), Beta > 1.2 (+5) |
| **Sector** | 15 | Technology (+15), Healthcare (+10), Consumer Disc. (+10), Financials (+5) |

**Beta Exception Rule:**
- Stocks with Beta 0.7-1.0 allowed if quality score (Profitability + Growth) >= 45

#### Step 5: Portfolio Construction (Lines 510-548)

```
1. Sort candidates by score (highest first)
2. Apply sector weight limit (max 50% per sector)
3. Calculate position weight: base_weight + score_bonus
   - base_weight = 90% / max_positions
   - score_bonus = up to 5% for high scores
4. Cap individual positions at 15%
5. Select top candidates until max_positions reached
```

### Rebalancing
- **Frequency**: Every 90 days (configurable)
- **Trigger**: Date check on each execution

---

## 2. Bear Beta Strategy

### Overview
The Bear Beta Strategy identifies defensive stocks that perform well (or lose less) during market downturns. It uses a novel "bear beta" metric that measures stock behavior specifically during market down days.

### Core Concept: Bear Beta

```
Beta_bear = Cov(stock, market | market < -1%) / Var(market | market < -1%)
```

**Interpretation:**
| Bear Beta | Meaning | Example |
|-----------|---------|---------|
| < 0 | Moves UP when market crashes | Gold miners (NEM, GOLD) |
| ~ 0 | Unaffected by crashes | Utilities (NEE, DUK) |
| 0.5 | Loses half of market decline | Consumer Staples (PG, KO) |
| > 1 | Amplifies losses | High-beta tech |

### Data Sources

| Data Type | Source | Lookback Period | Update Frequency |
|-----------|--------|-----------------|------------------|
| Price History | yfinance | 756 days (~3 years) | On rebalance |
| SPY Returns | yfinance | 756 days | On rebalance |
| Fundamentals | yfinance API | Current snapshot | Per session |

### Position Selection Process

#### Step 1: Candidate Universe (Lines 553-565)
```
1. ALWAYS include PRIORITY_DEFENSIVE stocks (known defensive names)
2. Add additional tickers from database up to 200 total
```

**Priority Defensive Stocks Include:**
- **Consumer Staples**: PG, KO, PEP, WMT, COST, PM, MO, CL, etc.
- **Healthcare**: JNJ, UNH, PFE, MRK, ABBV, LLY, TMO, ABT, etc.
- **Utilities**: NEE, DUK, SO, D, AEP, EXC, SRE, XEL, etc.
- **Gold Miners**: NEM, GOLD, AEM, FNV, WPM, RGLD

#### Step 2: Bear Beta Calculation (Lines 168-274)

```python
# Identify down days (market drops > 1%)
down_days = spy_returns < -0.01

# Calculate bear beta using only down days
stock_down_returns = stock_returns[down_days]
spy_down_returns = spy_returns[down_days]
bear_beta = Cov(stock_down, spy_down) / Var(spy_down)
```

**Additional Metrics Calculated:**
- **Bull Beta**: Same calculation for up days (market > +1%)
- **Down Capture**: avg(stock_down) / avg(spy_down)
- **Up Capture**: avg(stock_up) / avg(spy_up)
- **Total Return**: Annualized return over period
- **Volatility**: Annualized standard deviation

#### Step 3: Scoring System (Lines 381-531)

**Total Score: 0-115 points**

| Component | Max Points | Criteria |
|-----------|------------|----------|
| **Bear Beta** | 35 | < 0 (+35), < 0.3 (+30), < 0.5 (+25), < 0.7 (+15), else (+5) |
| **Down Capture** | 20 | < 50% (+20), < 70% (+15), < 90% (+10), else (+5) |
| **Asymmetry** | 15 | up_capture/down_capture > 1.5 (+15), > 1.2 (+10), > 1.0 (+5) |
| **Return** | 10 | > 15% (+10), > 10% (+8), > 5% (+5), > 0% (+3) |
| **Sector** | 15 | Consumer Defensive (+15), Healthcare (+12), Utilities (+12), etc. |
| **Quality** | 10 | Dividend yield, profit margin, current ratio bonuses |
| **Market Cap** | 10 | > $100B (+10), > $50B (+8), > $10B (+5), > $1B (+2) |
| **Priority Stock** | 10 | Bonus for known defensive stocks |

#### Step 4: Filters
- **max_bear_beta**: 0.8 (stocks with bear_beta > 0.8 rejected)
- **min_total_return**: -10% (avoid persistent losers)
- **min_score**: 40

#### Step 5: Portfolio Construction (Lines 596-622)
```
1. Sort by score (highest = most defensive)
2. Apply sector weight limit (max 40% per sector)
3. Equal weight positions: 90% / max_positions
4. Cap individual positions at 10%
```

### Rebalancing
- **Frequency**: Every 90 days (configurable)

---

## 3. Regime Adaptive Strategy

### Overview
The Regime Adaptive Strategy dynamically blends the High Beta and Bear Beta strategies based on a "bear score" that predicts market corrections. It uses the **MacroMom** (Macro Momentum) indicator system.

### MacroMom Performance (from backtesting)
- **80% detection rate** for major corrections (>15% drawdown)
- **173 days average lead time** before market peaks
- Best score at market peaks: 53.8 average

### Key Insight
> "Rate of CHANGE in indicators is more predictive than levels."

### Data Sources for Regime Detection

| Indicator | Source | Data Used | Weight |
|-----------|--------|-----------|--------|
| Yield Curve Momentum | ^TNX, ^IRX | 10Y-3M spread change | 30% |
| Breadth Momentum | SPY | % above 200 DMA change | 25% |
| VIX Momentum | ^VIX | VIX level, change, acceleration | 25% |
| Price Momentum | SPY | Multi-timeframe momentum | 20% |

### Regime Signal Components

#### 1. Yield Curve Momentum Score (Lines 36-58 in regime.py)

```python
def yield_curve_momentum_score(y10, y3m):
    spread = y10 - y3m  # Current spread
    spread_now = spread[-1]
    spread_13w = spread[-13]  # 13 weeks ago
    spread_change = spread_now - spread_13w

    # Fast flattening = bearish signal
    momentum_score = clamp(0.5 - spread_change * 0.25)

    # Bonus for actual inversion
    if spread_now < 0:
        momentum_score += 0.15

    return momentum_score
```

**Why it works:** Yield curve **flattening speed** precedes corrections by 3-6 months. The rate of change catches the transition before the actual inversion.

#### 2. Breadth Momentum Score (Lines 61-88)

```python
def breadth_momentum_score(pct_above_200dma):
    # Current level (30% weight)
    level_score = clamp((60 - pct_now) / 60)

    # 13-week change (50% weight) - KEY SIGNAL
    pct_change = pct_now - pct_13w
    momentum_score = clamp(0.5 - pct_change * 0.02)

    # Trend (20% weight)
    trend = linear_regression_slope(last_13_weeks)
    trend_score = clamp(0.5 - trend * 0.1)

    return level_score * 0.3 + momentum_score * 0.5 + trend_score * 0.2
```

**Why it works:** Declining breadth (fewer stocks above 200 DMA) often precedes price weakness. The momentum component catches deterioration early.

#### 3. VIX Momentum Score (Lines 91-132)

```python
def vix_momentum_score(vix, vix_3m):
    # VIX level percentile (25% weight)
    level_score = percentile(vix_current, last_252_days)

    # VIX 13-week change (35% weight) - KEY SIGNAL
    vix_change = vix_current / vix_13w - 1
    momentum_score = clamp(0.5 + vix_change * 2)

    # VIX acceleration (20% weight)
    change_now = vix[-1] / vix[-13] - 1
    change_prev = vix[-13] / vix[-26] - 1
    accel = change_now - change_prev
    accel_score = clamp(0.5 + accel * 3)

    # Term structure (20% weight)
    # VIX > VIX3M = backwardation = stress
    term_ratio = vix_current / vix_3m_current
    term_score = clamp((term_ratio - 0.9) * 2)

    return weighted_average(scores)
```

**Why it works:** Rising VIX momentum and inverted term structure (VIX > VIX3M) signal increasing stress before it fully manifests.

#### 4. Price Momentum Score (Lines 135-174)

```python
def price_momentum_score(sp500):
    # 1-month momentum (20%)
    mom_21d = sp500[-1] / sp500[-21] - 1

    # 3-month momentum (30%)
    mom_63d = sp500[-1] / sp500[-63] - 1

    # Momentum DECELERATION (30%) - KEY SIGNAL
    mom_now = sp500[-1] / sp500[-21] - 1
    mom_prev = sp500[-21] / sp500[-42] - 1
    decel = mom_now - mom_prev

    # Distance from 200 MA (20%)
    dist_from_ma = sp500[-1] / ma_200 - 1

    # Negative momentum/deceleration = bearish
    return weighted_score
```

**Why it works:** Momentum deceleration (slowing gains) often precedes trend reversals.

### Allocation Thresholds

#### Optimized Thresholds (Default)
| Bear Score | High Beta | Defensive | Risk Level |
|------------|-----------|-----------|------------|
| <= 55 | 100% | 0% | LOW |
| <= 65 | 85% | 15% | ELEVATED |
| <= 75 | 60% | 40% | HIGH |
| > 75 | 30% | 70% | EXTREME |

#### Conservative Thresholds (Original)
| Bear Score | High Beta | Defensive | Risk Level |
|------------|-----------|-----------|------------|
| <= 40 | 100% | 0% | LOW |
| <= 50 | 80% | 20% | WATCH |
| <= 55 | 60% | 40% | CAUTION |
| <= 65 | 40% | 60% | ELEVATED |
| <= 75 | 20% | 80% | HIGH |
| > 75 | 10% | 90% | EXTREME |

### Regime Switching Logic (Optimized)

#### Asymmetric Switching (Lines 230-255 in regime_adaptive_strategy.py)

```python
# Going DEFENSIVE: Require confirmation
if new_allocation[0] < old_allocation[0]:  # More defensive
    # Require: momentum confirmation AND persistence
    if momentum_confirmed AND consecutive_high_scores >= 2:
        allocation = new_allocation
    else:
        allocation = old_allocation  # Stay aggressive

# Going AGGRESSIVE: Allow immediately
else:
    allocation = new_allocation
```

**Key Safeguards:**
1. **Momentum Confirmation**: Only go defensive if SPY < 50-day MA
2. **Persistence Requirement**: Need 2+ consecutive high-risk readings
3. **Asymmetric Switching**: Fast to aggressive, slow to defensive

This prevents **false positives** that hurt returns in bull markets.

### Historical Regime Signal Examples

| Date | Bear Score | Level | What Happened |
|------|------------|-------|---------------|
| Mar 2020 | 78 | HIGH RISK | COVID crash (-34%) |
| Q4 2018 | ~65 | ELEVATED | -20% correction |
| Jan 2022 | ~60 | CAUTION | Start of bear market |
| Apr 2025 | 75.4 | HIGH RISK | (in backtest) |

---

## 4. Performance Comparison

Based on 10-year backtest (2015-2025):

| Strategy | Total Return | CAGR | Max Drawdown | Sharpe |
|----------|--------------|------|--------------|--------|
| Regime-Adaptive (Optimized) | 4563% | 42.2% | 44.2% | High |
| High Beta Only | 1465% | 28.7% | 45.4% | 4.95 |
| Regime-Adaptive (Conservative) | 481% | 17.5% | 45.7% | 3.49 |
| Bear Beta Only | 198% | 10.5% | 15.3% | 4.07 |
| SPY Benchmark | 298% | 13.5% | - | - |

### Key Takeaways

1. **High Beta** captures upside but has high drawdowns (45%)
2. **Bear Beta** has lowest drawdown (15%) but lowest returns (10.5% CAGR)
3. **Conservative Regime** switches too often, missing rallies
4. **Optimized Regime** stays aggressive longer, uses confirmation to avoid false positives

---

## 5. File Locations

| File | Purpose |
|------|---------|
| `strategies/high_beta_strategy.py` | High Beta Growth Strategy |
| `strategies/bear_beta_strategy.py` | Bear Beta Defensive Strategy |
| `strategies/regime_adaptive_strategy.py` | Regime Adaptive Strategy |
| `data/regime.py` | MacroMom indicator calculations |
| `backtest/run_optimized_regime_backtest.py` | Comparison backtest script |

---

## 6. Usage

```bash
# Run the full comparison backtest
python backtest/run_optimized_regime_backtest.py

# With interactive chart (hover for positions)
python backtest/run_optimized_regime_backtest.py --interactive

# Quick run with less output
python backtest/run_optimized_regime_backtest.py --quiet
```

---

*Report generated: 2026-02-01*
