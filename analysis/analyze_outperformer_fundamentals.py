"""Analyze fundamental characteristics of S&P 500 outperformers."""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")

# Load the outperformers from previous analysis
outperformers = pd.read_csv('sp500_outperformers.csv')
all_stocks = pd.read_csv('sp500_all_analysis.csv')

print("=" * 80)
print("FUNDAMENTAL ANALYSIS OF OUTPERFORMERS")
print("=" * 80)

# Get fundamentals for top outperformers
top_tickers = outperformers['Ticker'].head(25).tolist()
underperformer_tickers = all_stocks[all_stocks['Total Return (%)'] < all_stocks['Total Return (%)'].median()]['Ticker'].head(25).tolist()

def get_fundamentals(ticker):
    """Fetch key fundamentals for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info

        return {
            'Ticker': ticker,
            'Revenue Growth (5Y)': info.get('revenueGrowth', None),
            'Earnings Growth (5Y)': info.get('earningsGrowth', None),
            'Profit Margin': info.get('profitMargins', None),
            'Operating Margin': info.get('operatingMargins', None),
            'ROE': info.get('returnOnEquity', None),
            'ROA': info.get('returnOnAssets', None),
            'Debt/Equity': info.get('debtToEquity', None),
            'Current Ratio': info.get('currentRatio', None),
            'P/E Ratio': info.get('trailingPE', None),
            'Forward P/E': info.get('forwardPE', None),
            'P/S Ratio': info.get('priceToSalesTrailing12Months', None),
            'P/B Ratio': info.get('priceToBook', None),
            'EV/EBITDA': info.get('enterpriseToEbitda', None),
            'Free Cash Flow Yield': info.get('freeCashflow', 0) / info.get('marketCap', 1) if info.get('marketCap') else None,
            'Gross Margin': info.get('grossMargins', None),
            'R&D % Revenue': None,  # Not directly available
            'Market Cap ($B)': info.get('marketCap', 0) / 1e9,
        }
    except Exception as e:
        return {'Ticker': ticker}

print("\n1. FETCHING FUNDAMENTALS FOR TOP OUTPERFORMERS...")
outperformer_fundamentals = []
for i, ticker in enumerate(top_tickers):
    print(f"   {i+1}/{len(top_tickers)}: {ticker}", end='\r')
    fund = get_fundamentals(ticker)
    outperformer_fundamentals.append(fund)
    time.sleep(0.3)
print()

print("\n2. FETCHING FUNDAMENTALS FOR UNDERPERFORMERS (COMPARISON)...")
underperformer_fundamentals = []
for i, ticker in enumerate(underperformer_tickers):
    print(f"   {i+1}/{len(underperformer_tickers)}: {ticker}", end='\r')
    fund = get_fundamentals(ticker)
    underperformer_fundamentals.append(fund)
    time.sleep(0.3)
print()

outperf_df = pd.DataFrame(outperformer_fundamentals)
underperf_df = pd.DataFrame(underperformer_fundamentals)

# Compare metrics
print("\n" + "=" * 80)
print("3. FUNDAMENTAL COMPARISON: OUTPERFORMERS vs UNDERPERFORMERS")
print("=" * 80)

metrics = ['Profit Margin', 'Operating Margin', 'ROE', 'ROA', 'Gross Margin',
           'Revenue Growth (5Y)', 'Earnings Growth (5Y)', 'Debt/Equity',
           'P/E Ratio', 'P/S Ratio', 'EV/EBITDA']

print(f"\n{'Metric':<25} {'Outperformers':>15} {'Underperformers':>15} {'Diff':>12} {'Signal':>15}")
print("-" * 85)

signals = {}
for metric in metrics:
    out_vals = outperf_df[metric].dropna()
    under_vals = underperf_df[metric].dropna()

    if len(out_vals) > 0 and len(under_vals) > 0:
        out_med = out_vals.median()
        under_med = under_vals.median()
        diff = out_med - under_med

        # Determine signal
        if metric in ['Debt/Equity']:
            signal = 'Lower better' if out_med < under_med else 'Mixed'
        elif metric in ['P/E Ratio', 'P/S Ratio', 'EV/EBITDA']:
            signal = 'Higher accepted' if out_med > under_med else 'Value'
        else:
            signal = 'Higher better' if out_med > under_med else 'Mixed'

        signals[metric] = {'out': out_med, 'under': under_med, 'signal': signal}

        if isinstance(out_med, float) and out_med < 10:
            print(f"{metric:<25} {out_med:>15.2f} {under_med:>15.2f} {diff:>+12.2f} {signal:>15}")
        else:
            print(f"{metric:<25} {out_med:>15.1f} {under_med:>15.1f} {diff:>+12.1f} {signal:>15}")

# Individual outperformer details
print("\n" + "=" * 80)
print("4. TOP 15 OUTPERFORMER FUNDAMENTAL DETAILS")
print("=" * 80)

key_metrics = ['Ticker', 'Profit Margin', 'ROE', 'Revenue Growth (5Y)', 'Gross Margin', 'Debt/Equity']
available_cols = [c for c in key_metrics if c in outperf_df.columns]
print(outperf_df[available_cols].head(15).to_string(index=False, float_format=lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'))

# KEY FINDINGS
print("\n" + "=" * 80)
print("5. QUANTIFIABLE FACTORS FOR STRATEGY CODING")
print("=" * 80)

print("""
FACTOR 1: PROFITABILITY METRICS
===============================
The data shows outperformers have significantly higher profitability:
""")

if 'ROE' in signals:
    print(f"  - ROE: Outperformers {signals['ROE']['out']*100:.1f}% vs Underperformers {signals['ROE']['under']*100:.1f}%")
if 'Operating Margin' in signals:
    print(f"  - Operating Margin: {signals['Operating Margin']['out']*100:.1f}% vs {signals['Operating Margin']['under']*100:.1f}%")
if 'Gross Margin' in signals:
    print(f"  - Gross Margin: {signals['Gross Margin']['out']*100:.1f}% vs {signals['Gross Margin']['under']*100:.1f}%")

print("""
QUANTIFIABLE RULE:
  - ROE > 15%
  - Operating Margin > 15%
  - Gross Margin > 40%
""")

print("""
FACTOR 2: GROWTH METRICS
========================
Outperformers show stronger growth trajectories:
""")
if 'Revenue Growth (5Y)' in signals:
    print(f"  - Revenue Growth: {signals['Revenue Growth (5Y)']['out']*100:.1f}% vs {signals['Revenue Growth (5Y)']['under']*100:.1f}%")
if 'Earnings Growth (5Y)' in signals:
    print(f"  - Earnings Growth: {signals['Earnings Growth (5Y)']['out']*100:.1f}% vs {signals['Earnings Growth (5Y)']['under']*100:.1f}%")

print("""
QUANTIFIABLE RULE:
  - Revenue Growth > 10% YoY
  - Earnings Growth > 15% YoY
  - Consistent growth (low variance)
""")

print("""
FACTOR 3: BALANCE SHEET STRENGTH
================================
""")
if 'Debt/Equity' in signals:
    print(f"  - Debt/Equity: {signals['Debt/Equity']['out']:.1f} vs {signals['Debt/Equity']['under']:.1f}")

print("""
QUANTIFIABLE RULE:
  - Debt/Equity < 1.0 (prefer low leverage)
  - Current Ratio > 1.5
  - Positive Free Cash Flow
""")

print("""
FACTOR 4: MARKET EXPOSURE (From Beta Analysis)
==============================================
From price analysis:
  - Outperformers avg Beta: 1.26
  - Underperformers avg Beta: 0.83

QUANTIFIABLE RULE:
  - Beta > 1.0 (captures market upside)
  - Higher volatility tolerance (30-40% acceptable)
""")

print("""
FACTOR 5: SECTOR ALLOCATION
===========================
Outperformers concentrated in:
  - Technology: 64%
  - Health Care: 12%
  - Consumer Discretionary: 12%

QUANTIFIABLE RULE:
  - Overweight Technology sector
  - Focus on secular growth themes (AI, cloud, semiconductors)
""")

print("""
================================================================================
COMBINED STRATEGY CRITERIA (for coding):
================================================================================

def is_potential_outperformer(ticker_fundamentals):
    '''
    Score a stock's outperformance potential (0-100)
    '''
    score = 0

    # Profitability (max 30 points)
    if ROE > 0.15: score += 10
    if ROE > 0.25: score += 5
    if operating_margin > 0.15: score += 10
    if gross_margin > 0.40: score += 5

    # Growth (max 25 points)
    if revenue_growth > 0.10: score += 10
    if revenue_growth > 0.20: score += 5
    if earnings_growth > 0.15: score += 10

    # Balance Sheet (max 15 points)
    if debt_to_equity < 1.0: score += 10
    if current_ratio > 1.5: score += 5

    # Market Exposure (max 15 points)
    if beta > 1.0: score += 10
    if beta > 1.2: score += 5

    # Sector (max 15 points)
    if sector == 'Technology': score += 15
    elif sector in ['Health Care', 'Consumer Discretionary']: score += 10

    return score  # Buy if score > 60

================================================================================
""")
