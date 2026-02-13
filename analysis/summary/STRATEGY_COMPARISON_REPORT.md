# Strategy Comparison Report: Value Strategies vs High Beta Strategy

## Executive Summary

The Value Strategies and High Beta Strategy find fundamentally different portfolios because they optimize for **different objectives** using **different data sources** and **different selection criteria**.

| Aspect | Value Strategies | High Beta Strategy |
|--------|------------------|-------------------|
| **Philosophy** | Buy undervalued quality companies | Buy high-growth market leaders |
| **Data Source** | Historical SEC filings (fundamentals.sqlite) | Real-time yfinance data |
| **Key Metric** | Intrinsic value discount | Score (0-100) based on beta + growth |
| **Valuation** | DCF-based intrinsic value | No intrinsic value - just quality scores |
| **Typical Stocks** | Value stocks: BEN, CMI, INTC | Growth stocks: NVDA, META, AMZN |

---

## Portfolio Comparison

### Value Strategies Selected (2014-2016):
```
Buy and Hold:     BEN, CMI, LMT, QVCGB, ACCS
Compounders:      CMI, ACCS, BEN, ORCL, APEI, HAL, INTC, KMB, AMGN, HRB
True Buffett:     ALK, ORCL, PRAA, CE, IT, CMI
Earnings Growth:  CMI, BEN, INTC, ADI, HAL
```

**Common picks:** CMI (Cummins), BEN (Franklin Resources), INTC (Intel), ORCL (Oracle)

### High Beta Strategy Selected:
```
LRCX (Lam Research)     - Score 100, Beta 1.31, Technology
NVDA (NVIDIA)           - Score 100, Beta 1.30, Technology
MU (Micron)             - Score 95,  Beta 1.87, Technology
MSFT (Microsoft)        - Score 90,  Beta 1.09, Technology
ADBE (Adobe)            - Score 90,  Beta 1.24, Technology
META (Meta)             - Score 80,  Beta 1.52, Communication
GOOG (Google)           - Score 80,  Beta 1.15, Communication
NFLX (Netflix)          - Score 80,  Beta 1.31, Communication
AMZN (Amazon)           - Score 75,  Beta 1.29, Consumer Cyclical
MA (Mastercard)         - Score 75,  Beta 1.43, Financial Services
```

**Zero overlap** between the two approaches.

---

## Why Value Strategies Missed High Beta Stocks

### 1. Intrinsic Value Requirement

Value strategies require stocks to trade **below intrinsic value** (margin of safety):

```python
# From improved_strategies.py
mos = (intrinsic - price) / intrinsic  # Margin of safety
if mos < self.min_mos:  # Typically 10-20%
    continue  # Skip this stock
```

**Problem:** High-growth stocks like NVDA, META, AMZN almost never trade at a discount to DCF-based intrinsic value. They trade at **premium valuations** because the market prices in future growth.

| Stock | Typical P/E | Value Strategy View |
|-------|-------------|---------------------|
| NVDA  | 40-80x | "Overvalued" - skip |
| META  | 25-35x | "Overvalued" - skip |
| BEN   | 8-12x  | "Undervalued" - buy |
| CMI   | 10-15x | "Undervalued" - buy |

### 2. DCF Penalizes High-Growth Companies

The DCF model uses conservative growth assumptions:

```python
# From improved_strategies.py
growth_allowance = min(base_growth, 0.06)  # Cap at 6% for most stocks
discount_rate = 0.08 to 0.10
```

**Problem:** A 6% growth cap massively undervalues companies growing at 20-50%:
- NVDA revenue grew ~50% annually (2019-2024)
- DCF with 6% growth produces intrinsic value far below market price
- Value strategy sees this as "overvalued" and skips it

### 3. ROIC Calculation Favors Mature Businesses

```python
# Value strategy ROIC
roic = (operating_income * 0.75) / invested_capital
if roic < self.min_roic:  # 12-15%
    continue
```

**Problem:** Fast-growing tech companies often reinvest heavily, depressing short-term ROIC:
- Amazon had low ROIC for years (reinvesting in AWS)
- Tesla had negative ROIC during growth phase
- Value strategy would skip these

### 4. Historical Data Lag

Value strategies use **SEC filings** which have significant lag:
- Annual reports filed 60-90 days after year-end
- Quarterly reports filed 45 days after quarter-end
- Data reflects past performance, not current trajectory

**High Beta Strategy** uses **real-time yfinance data**:
- Current quarter growth rates
- Live market cap and sector data
- Forward-looking analyst estimates

---

## Why High Beta Strategy Misses Value Stocks

### 1. Beta Requirement

```python
# High Beta Strategy
if beta < self.min_beta:  # 1.0
    return None
```

**Excludes:**
- Utility companies (beta 0.3-0.6)
- Consumer staples (beta 0.5-0.8)
- Healthcare (beta 0.6-0.9)

BEN, CMI, LMT all have lower betas than the 1.0 threshold.

### 2. Sector Scoring Bias

```python
SECTOR_SCORES = {
    'Technology': 15,
    'Health Care': 10,
    'Consumer Discretionary': 10,
    'Financials': 5,
    'Industrials': 5,
}
```

**Problem:** Traditional value stocks are often in low-scoring sectors:
- CMI (Industrials): +5 points
- BEN (Financials): +5 points
- LMT (Industrials): +5 points
- vs NVDA (Technology): +15 points

### 3. No Valuation Check

High Beta Strategy has **no price/value comparison**:

```python
# Only checks fundamentals quality, not price
if score >= self.min_score:  # 50
    buy(ticker)
```

**Result:** May buy expensive stocks, may miss cheap stocks that don't have high growth metrics.

---

## Fundamental Difference: What Drives Returns?

### Value Strategy Belief:
> "Price will converge to intrinsic value. Buy below value, wait for reversion."

**Works when:** Market temporarily misprices quality companies

**Fails when:** Market correctly prices growth (the "value trap")

### High Beta Strategy Belief:
> "High-beta, high-quality, high-growth companies outperform over long periods."

**Works when:** Bull markets, growth stocks leading

**Fails when:** Market corrections (high beta = high drawdowns)

---

## Historical Performance Context

From the Outperformer Analysis (2014-2024):

| Metric | Top Outperformers | Value Stocks |
|--------|-------------------|--------------|
| Avg Beta | 1.26 | 0.83 |
| Avg ROE | 34.4% | 21.4% |
| Revenue Growth | 15.9% | 2.7% |
| Sector | 64% Tech | Mixed |
| Avg Return | 6,642% | ~300% |

**Key insight:** The outperformers were NOT cheap stocks that reverted to value. They were expensive stocks that grew into even higher valuations.

---

## Specific Stock Analysis

### Why CMI (Cummins) Was Selected by Value, Not High Beta:

| Metric | CMI Value | High Beta Threshold |
|--------|-----------|---------------------|
| Beta | 0.95 | < 1.0 = FAIL |
| P/E | 12x | N/A |
| ROE | 18% | > 15% = Pass |
| Revenue Growth | 3% | < 10% = Low score |
| Sector | Industrials | +5 points only |

CMI fails the beta filter and gets low scores for growth and sector.

### Why NVDA Was Selected by High Beta, Not Value:

| Metric | NVDA Value | Value Strategy Check |
|--------|------------|---------------------|
| P/E | 60x | Overvalued = FAIL |
| Beta | 1.75 | N/A in value |
| ROE | 45% | Excellent |
| Revenue Growth | 50%+ | Capped at 6% in DCF |
| DCF Intrinsic | $50 (hypothetical) | Price $120 = "50% overvalued" |

NVDA fails the margin of safety requirement because DCF underestimates its growth.

---

## Recommendations

### For Value Strategies:
1. **Raise growth caps** for high-ROIC companies (allow 15% vs 6%)
2. **Accept fair-value purchases** for exceptional quality (no discount required)
3. **Add sector consideration** (don't penalize tech)

### For High Beta Strategy:
1. **Add valuation sanity check** (avoid extreme P/E stocks)
2. **Consider lower-beta quality** (like LLY with 0.69 beta but 2,387% return)
3. **Track historical fundamentals** from SEC filings for consistency

### Hybrid Approach:
Combine both:
- Use High Beta's growth/sector scoring
- Add Value's ROIC consistency check
- Remove strict margin of safety requirement
- Keep sector diversification limits

---

## Conclusion

The strategies find different portfolios because:

1. **Value strategies** are backwards-looking (historical fundamentals) and require price < value
2. **High Beta strategy** is forward-looking (current fundamentals) and requires high growth + high beta

Neither is "wrong" - they represent different investment philosophies:
- Value: "Pay less than something is worth"
- Growth: "Own the best companies regardless of price"

The historical evidence (2014-2024) suggests the growth approach outperformed, but this may not hold in all market regimes. Value strategies excel during market corrections when high-beta stocks crash.
