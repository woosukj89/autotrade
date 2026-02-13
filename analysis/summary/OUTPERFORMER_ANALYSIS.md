# S&P 500 Outperformer Analysis (2014-2026)

## Executive Summary

Analyzed 162 large-cap S&P 500 stocks over the past 12 years to identify characteristics of outperformers.

| Metric | S&P 500 (SPY) | Average Stock | Top Outperformers |
|--------|---------------|---------------|-------------------|
| Total Return | 386% | 964% | 6,022% (Tech avg) |
| CAGR | 14.1% | - | 30.4% |

**Key Finding**: Only 25 out of 162 stocks (15%) beat both the S&P 500 and the average stock return.

---

## Top 15 Outperformers (2014-2026)

| Rank | Ticker | Sector | Total Return | CAGR | Beta | Sharpe |
|------|--------|--------|--------------|------|------|--------|
| 1 | NVDA | Technology | 52,575% | 68.8% | 1.75 | 1.42 |
| 2 | AVGO | Technology | 8,270% | 44.7% | 1.45 | 1.11 |
| 3 | AMD | Technology | 7,114% | 43.0% | 1.69 | 0.71 |
| 4 | LRCX | Technology | 5,712% | 40.4% | 1.66 | 0.93 |
| 5 | ANET | Technology | 4,100% | 37.9% | 1.33 | 0.80 |
| 6 | KLAC | Technology | 3,772% | 35.7% | 1.55 | 0.88 |
| 7 | TSLA | Consumer Disc | 3,599% | 35.2% | 1.60 | 0.58 |
| 8 | LLY | Health Care | 2,387% | 30.8% | 0.69 | 1.01 |
| 9 | CDNS | Technology | 2,059% | 29.3% | 1.24 | 0.86 |
| 10 | FTNT | Technology | 1,891% | 28.4% | 1.23 | 0.67 |
| 11 | MU | Technology | 1,834% | 28.1% | 1.66 | 0.55 |
| 12 | PANW | Technology | 1,743% | 27.6% | 1.09 | 0.66 |
| 13 | MSCI | Financials | 1,578% | 26.6% | 1.14 | 0.83 |
| 14 | AAPL | Technology | 1,539% | 26.3% | 1.21 | 0.86 |
| 15 | CTAS | Industrials | 1,459% | 25.8% | 1.02 | 0.94 |

---

## Quantifiable Factors

### 1. Market Exposure (Beta)

| Metric | Outperformers | Underperformers | Signal |
|--------|---------------|-----------------|--------|
| Average Beta | **1.26** | 0.83 | Higher = Better |
| Volatility | 35.8% | 26.1% | Higher accepted |
| Max Drawdown | -50% | -49% | Similar risk |

**Rule**: Beta > 1.0 (92% of outperformers had beta > 1.0)

### 2. Profitability Metrics

| Metric | Outperformers | Underperformers | Signal |
|--------|---------------|-----------------|--------|
| ROE | **34.4%** | 21.4% | Higher = Better |
| Operating Margin | **31.6%** | 22.0% | Higher = Better |
| Gross Margin | **61.7%** | 55.4% | Higher = Better |
| Profit Margin | **28.4%** | 14.9% | Higher = Better |
| ROA | **15.3%** | 6.1% | Higher = Better |

**Rules**:
- ROE > 15% (minimum)
- ROE > 25% (preferred)
- Operating Margin > 15%
- Gross Margin > 40%

### 3. Growth Metrics

| Metric | Outperformers | Underperformers | Signal |
|--------|---------------|-----------------|--------|
| Revenue Growth (5Y) | **15.9%** | 2.7% | 6x higher |
| Earnings Growth (5Y) | **20.8%** | 4.7% | 4x higher |

**Rules**:
- Revenue Growth > 10% YoY
- Earnings Growth > 15% YoY

### 4. Balance Sheet

| Metric | Outperformers | Underperformers | Signal |
|--------|---------------|-----------------|--------|
| Debt/Equity | **44.2** | 68.7 | Lower = Better |

**Rules**:
- Debt/Equity < 100 (prefer < 50)
- Positive Free Cash Flow

### 5. Valuation (Accepted, Not Required)

| Metric | Outperformers | Underperformers | Signal |
|--------|---------------|-----------------|--------|
| P/E Ratio | 42.6 | 24.4 | Higher accepted |
| P/S Ratio | 12.7 | 3.7 | Higher accepted |
| EV/EBITDA | 27.8 | 14.4 | Higher accepted |

**Key Insight**: Outperformers traded at premium valuations. Value investing would have missed them.

### 6. Sector Concentration

| Sector | % of Outperformers | Avg Return |
|--------|-------------------|------------|
| Technology | **64%** | 6,021% |
| Consumer Discretionary | 12% | 1,968% |
| Health Care | 12% | 1,516% |
| Financials | 4% | 1,578% |
| Industrials | 4% | 1,459% |

---

## Quantifiable Scoring System (for coding)

```python
def score_outperformance_potential(stock):
    """
    Score a stock's potential to outperform (0-100).
    Based on 12-year analysis of S&P 500 outperformers.
    """
    score = 0

    # PROFITABILITY (max 30 points)
    if stock['ROE'] > 0.15: score += 10
    if stock['ROE'] > 0.25: score += 5
    if stock['operating_margin'] > 0.15: score += 10
    if stock['gross_margin'] > 0.40: score += 5

    # GROWTH (max 25 points)
    if stock['revenue_growth'] > 0.10: score += 10
    if stock['revenue_growth'] > 0.20: score += 5
    if stock['earnings_growth'] > 0.15: score += 10

    # BALANCE SHEET (max 15 points)
    if stock['debt_to_equity'] < 100: score += 5
    if stock['debt_to_equity'] < 50: score += 5
    if stock['free_cash_flow'] > 0: score += 5

    # MARKET EXPOSURE (max 15 points)
    if stock['beta'] > 1.0: score += 10
    if stock['beta'] > 1.2: score += 5

    # SECTOR (max 15 points)
    if stock['sector'] == 'Technology': score += 15
    elif stock['sector'] in ['Health Care', 'Consumer Discretionary']: score += 10
    elif stock['sector'] in ['Financials', 'Industrials']: score += 5

    return score  # Buy threshold: score >= 60
```

---

## Key Takeaways

### Why These Companies Outperformed

1. **Secular Growth Themes**: AI, cloud computing, semiconductors, biotech
2. **High Profitability + High Growth**: Not either/or, but BOTH
3. **Market Leaders**: Dominant positions in growing markets
4. **Higher Beta Accepted**: Willingness to accept volatility for returns
5. **Premium Valuations Justified**: Growth justified high P/E ratios

### What Traditional Value Investing Missed

- Excluded high P/E stocks (NVDA at 40+ P/E)
- Underweighted technology (would have missed 64% of winners)
- Required margin of safety that quality growth never offered
- Focused on cheap stocks instead of great businesses

### Recommended Strategy Adjustments

1. **Raise ROE/ROA thresholds** (ROE > 15%, prefer > 25%)
2. **Add growth requirements** (Revenue growth > 10%)
3. **Accept premium valuations** for quality growth
4. **Overweight technology** and secular growth sectors
5. **Prefer higher beta** (> 1.0) for market upside capture
6. **Focus on quality + growth**, not just cheap valuation

---

## Data Files

- `sp500_outperformers.csv` - 25 outperforming stocks with metrics
- `sp500_all_analysis.csv` - All 162 analyzed stocks
