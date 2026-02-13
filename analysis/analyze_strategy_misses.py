"""Analyze why the High Beta strategy missed top outperformers."""

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

import sqlite3
import pandas as pd
import yfinance as yf

# Top 15 outperformers from analysis
TOP_OUTPERFORMERS = [
    ('NVDA', 'Technology', 52575, 1.75),
    ('AVGO', 'Technology', 8270, 1.45),
    ('AMD', 'Technology', 7114, 1.69),
    ('LRCX', 'Technology', 5712, 1.66),
    ('ANET', 'Technology', 4100, 1.33),
    ('KLAC', 'Technology', 3772, 1.55),
    ('TSLA', 'Consumer Cyclical', 3599, 1.60),
    ('LLY', 'Healthcare', 2387, 0.69),
    ('CDNS', 'Technology', 2059, 1.24),
    ('FTNT', 'Technology', 1891, 1.23),
    ('MU', 'Technology', 1834, 1.66),
    ('PANW', 'Technology', 1743, 1.09),
    ('MSCI', 'Financials', 1578, 1.14),
    ('AAPL', 'Technology', 1539, 1.21),
    ('CTAS', 'Industrials', 1459, 1.02),
]

# Strategy's selected stocks
STRATEGY_PICKS = ['MU', 'MSFT', 'CRDO', 'APP', 'HOOD', 'RCL', 'AMAT', 'GWRE',
                  'RELY', 'NUTX', 'SOFI', 'ACAD', 'LNG', 'GGG', 'COLL']

db_path = "fundamentals.sqlite"

print("=" * 90)
print("ANALYSIS: Why Strategy Missed Top Outperformers")
print("=" * 90)

# 1. Check which outperformers are in the database
print("\n1. DATABASE PRESENCE CHECK")
print("-" * 90)

conn = sqlite3.connect(db_path)

# Check fundamentals table
df_funds = pd.read_sql_query(
    """
    SELECT ticker, COUNT(DISTINCT fy) as years
    FROM fundamentals
    WHERE statement_type = 'income'
    GROUP BY ticker
    """,
    conn
)
tickers_in_db = set(df_funds['ticker'])

print(f"\n{'Ticker':<8} {'In DB?':<8} {'Years Data':<12} {'Historical Return':<20} {'Historical Beta'}")
print("-" * 70)

for ticker, sector, hist_return, hist_beta in TOP_OUTPERFORMERS:
    in_db = ticker in tickers_in_db
    years = df_funds[df_funds['ticker'] == ticker]['years'].values[0] if in_db else 0
    status = "YES" if in_db else "NO"
    picked = " [PICKED]" if ticker in STRATEGY_PICKS else ""
    print(f"{ticker:<8} {status:<8} {years:<12} {hist_return:>15,}% {hist_beta:>15.2f}{picked}")

# 2. Fetch current fundamentals for outperformers
print("\n\n2. CURRENT FUNDAMENTALS (from yfinance)")
print("-" * 90)

def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
        return {
            'ticker': ticker,
            'roe': info.get('returnOnEquity'),
            'operating_margin': info.get('operatingMargins'),
            'gross_margin': info.get('grossMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'debt_to_equity': info.get('debtToEquity'),
            'free_cash_flow': info.get('freeCashflow', 0),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
        }
    except:
        return {'ticker': ticker}

print("\nFetching fundamentals for top outperformers...")
outperformer_funds = []
for ticker, _, _, _ in TOP_OUTPERFORMERS:
    fund = get_fundamentals(ticker)
    outperformer_funds.append(fund)
    print(f"  {ticker}...", end=" ")

print("\n")

# 3. Score each outperformer using strategy's scoring system
print("\n3. SCORING ANALYSIS (using strategy criteria)")
print("-" * 90)

SECTOR_SCORES = {
    'Technology': 15,
    'Health Care': 10,
    'Healthcare': 10,
    'Consumer Discretionary': 10,
    'Consumer Cyclical': 10,
    'Financials': 5,
    'Financial Services': 5,
    'Industrials': 5,
}

def score_stock(fund, hist_beta):
    """Score using same logic as HighBetaGrowthStrategy."""
    score = 0
    breakdown = {}

    # PROFITABILITY (max 30 points)
    roe = fund.get('roe') or 0
    op_margin = fund.get('operating_margin') or 0
    gross_margin = fund.get('gross_margin') or 0

    prof_score = 0
    if roe > 0.15: prof_score += 10
    if roe > 0.25: prof_score += 5
    if op_margin > 0.15: prof_score += 10
    if gross_margin > 0.40: prof_score += 5
    score += prof_score
    breakdown['profitability'] = prof_score

    # GROWTH (max 25 points)
    rev_growth = fund.get('revenue_growth') or 0
    earn_growth = fund.get('earnings_growth') or 0

    growth_score = 0
    if rev_growth > 0.10: growth_score += 10
    if rev_growth > 0.20: growth_score += 5
    if earn_growth > 0.15: growth_score += 10
    score += growth_score
    breakdown['growth'] = growth_score

    # BALANCE SHEET (max 15 points)
    debt_equity = fund.get('debt_to_equity') or 0
    fcf = fund.get('free_cash_flow') or 0

    balance_score = 0
    if debt_equity < 100: balance_score += 5
    if debt_equity < 50: balance_score += 5
    if fcf > 0: balance_score += 5
    score += balance_score
    breakdown['balance_sheet'] = balance_score

    # MARKET EXPOSURE (max 15 points)
    beta_score = 0
    if hist_beta > 1.0: beta_score += 10
    if hist_beta > 1.2: beta_score += 5
    score += beta_score
    breakdown['beta'] = beta_score

    # SECTOR (max 15 points)
    sector = fund.get('sector', 'Unknown')
    sector_score = SECTOR_SCORES.get(sector, 0)
    score += sector_score
    breakdown['sector'] = sector_score

    return score, breakdown

print(f"\n{'Ticker':<8} {'Score':>6} {'Prof':>6} {'Growth':>7} {'Bal':>5} {'Beta':>6} {'Sect':>6} {'Meets Min?':<12} {'Issue'}")
print("-" * 90)

issues_found = []
for i, (ticker, sector, hist_return, hist_beta) in enumerate(TOP_OUTPERFORMERS):
    fund = outperformer_funds[i]
    score, breakdown = score_stock(fund, hist_beta)

    meets_min = score >= 50
    meets_beta = hist_beta >= 1.0
    in_db = ticker in tickers_in_db
    picked = ticker in STRATEGY_PICKS

    # Determine issue
    issue = ""
    if picked:
        issue = "SELECTED OK"
    elif not in_db:
        issue = "NOT IN DB"
    elif not meets_beta:
        issue = f"BETA < 1.0 ({hist_beta:.2f})"
    elif not meets_min:
        issue = f"SCORE < 50"
    else:
        issue = "NOT IN TOP 500 CANDIDATES"

    issues_found.append((ticker, issue, score, hist_return))

    status = "YES" if meets_min else "NO"
    print(f"{ticker:<8} {score:>6} {breakdown['profitability']:>6} {breakdown['growth']:>7} "
          f"{breakdown['balance_sheet']:>5} {breakdown['beta']:>6} {breakdown['sector']:>6} "
          f"{status:<12} {issue}")

# 4. Compare historical returns
print("\n\n4. RETURN COMPARISON: Outperformers vs Strategy Picks")
print("-" * 90)

# Get historical returns for strategy picks
print("\nFetching historical data for strategy picks...")
strategy_returns = []
for ticker in STRATEGY_PICKS:
    try:
        data = yf.download(ticker, start='2014-01-01', end='2024-12-31', progress=False)
        if len(data) > 100:
            start_price = data['Close'].iloc[0]
            end_price = data['Close'].iloc[-1]
            if hasattr(start_price, 'item'):
                start_price = start_price.item()
                end_price = end_price.item()
            total_return = (end_price / start_price - 1) * 100
            strategy_returns.append((ticker, total_return))
    except:
        pass

print(f"\n{'OUTPERFORMERS (Historical)':<40} | {'STRATEGY PICKS (Historical)':<40}")
print("-" * 85)

# Sort both by return
outperf_sorted = sorted([(t, r) for t, _, r, _ in TOP_OUTPERFORMERS], key=lambda x: -x[1])
strategy_sorted = sorted(strategy_returns, key=lambda x: -x[1])

max_rows = max(len(outperf_sorted), len(strategy_sorted))
for i in range(max_rows):
    left = f"{outperf_sorted[i][0]}: {outperf_sorted[i][1]:,.0f}%" if i < len(outperf_sorted) else ""
    right = f"{strategy_sorted[i][0]}: {strategy_sorted[i][1]:,.0f}%" if i < len(strategy_sorted) else ""
    print(f"{left:<40} | {right:<40}")

# Calculate averages
outperf_avg = sum(r for _, r in outperf_sorted) / len(outperf_sorted)
strategy_avg = sum(r for _, r in strategy_sorted) / len(strategy_sorted) if strategy_returns else 0

print("-" * 85)
print(f"{'Average: ' + f'{outperf_avg:,.0f}%':<40} | {'Average: ' + f'{strategy_avg:,.0f}%':<40}")

# 5. Root cause analysis
print("\n\n5. ROOT CAUSE ANALYSIS")
print("=" * 90)

print("\nWhy the strategy missed top outperformers:\n")

missed_due_to_beta = [t for t, issue, _, _ in issues_found if "BETA" in issue]
missed_due_to_db = [t for t, issue, _, _ in issues_found if "NOT IN DB" in issue]
missed_due_to_score = [t for t, issue, _, _ in issues_found if "SCORE" in issue]
missed_due_to_limit = [t for t, issue, _, _ in issues_found if "TOP 500" in issue]

print(f"  a) Beta < 1.0 requirement excluded: {missed_due_to_beta}")
print(f"  b) Not in fundamentals database:   {missed_due_to_db}")
print(f"  c) Score below 50 threshold:       {missed_due_to_score}")
print(f"  d) Not in first 500 candidates:    {missed_due_to_limit}")

print("\n\n6. RECOMMENDED STRATEGY ADJUSTMENTS")
print("=" * 90)

print("""
ISSUE 1: Candidate Selection Limited to First 500
-------------------------------------------------
Current: candidates = list(self._eligible_tickers)[:500]
Problem: Uses arbitrary ordering, may exclude major stocks like NVDA, AAPL

FIX: Sort by market cap or add explicit "must include" list of S&P 500 stocks

ISSUE 2: Beta Threshold Excludes LLY (Beta 0.69)
------------------------------------------------
Current: min_beta = 1.0
Problem: LLY (Eli Lilly) returned 2,387% but has beta 0.69

FIX: Lower beta threshold to 0.7 OR create separate low-beta quality bucket

ISSUE 3: Growth Metrics May Be Stale/Missing
---------------------------------------------
Current: Uses current yfinance fundamentals
Problem: Some stocks may have missing or outdated growth data, scoring 0

FIX: Use TTM (trailing twelve months) data OR calculate growth from historical financials

ISSUE 4: No Prioritization of Established Large Caps
----------------------------------------------------
Current: All candidates treated equally
Problem: NVDA, AAPL, MSFT should get priority but may be excluded by random ordering

FIX: Add market cap weighting OR create "core holdings" list from S&P 500
""")

conn.close()
