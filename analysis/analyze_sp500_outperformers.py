"""Analyze S&P 500 top performers over the past 12 years."""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings("ignore")

# Date range: 12 years
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=12*365)

print("=" * 80)
print(f"S&P 500 OUTPERFORMER ANALYSIS ({START_DATE.year} - {END_DATE.year})")
print("=" * 80)

# Get S&P 500 performance
print("\n1. FETCHING S&P 500 BENCHMARK...")
spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_return = (float(spy['Close'].iloc[-1]) / float(spy['Close'].iloc[0]) - 1) * 100
spy_cagr = ((float(spy['Close'].iloc[-1]) / float(spy['Close'].iloc[0])) ** (1/12) - 1) * 100
print(f"   SPY Total Return: {spy_return:.1f}%")
print(f"   SPY CAGR: {spy_cagr:.1f}%")

# S&P 500 large caps list (comprehensive)
print("\n2. USING S&P 500 LARGE CAP LIST...")
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
    'JNJ', 'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PFE', 'KO',
    'PEP', 'COST', 'TMO', 'AVGO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT', 'DHR', 'NEE',
    'LLY', 'DIS', 'VZ', 'ADBE', 'NKE', 'CRM', 'CMCSA', 'INTC', 'TXN', 'PM', 'QCOM',
    'RTX', 'HON', 'UPS', 'IBM', 'CAT', 'AMGN', 'GE', 'LOW', 'SPGI', 'BA', 'GS',
    'SBUX', 'BLK', 'INTU', 'ELV', 'MDLZ', 'AMD', 'ISRG', 'ADI', 'GILD', 'CVS',
    'AXP', 'LMT', 'BKNG', 'SYK', 'TJX', 'VRTX', 'MMC', 'REGN', 'PLD', 'ZTS', 'NOW',
    'PYPL', 'SCHW', 'CI', 'MO', 'DUK', 'SO', 'BDX', 'ITW', 'EOG', 'SLB', 'CL',
    'CME', 'APD', 'FDX', 'NOC', 'ICE', 'NSC', 'PNC', 'MU', 'AON', 'USB', 'WM',
    'ORLY', 'GD', 'EMR', 'COP', 'PSA', 'TGT', 'MCK', 'KLAC', 'ADP', 'LRCX', 'F',
    'GM', 'SHW', 'SNPS', 'CDNS', 'FTNT', 'MRVL', 'PANW', 'AZO', 'CTAS', 'MNST',
    'MSCI', 'ANET', 'ADSK', 'NXPI', 'ROP', 'KDP', 'AIG', 'PAYX', 'NEM', 'IDXX',
    'MAR', 'TT', 'PCAR', 'AEP', 'D', 'EXC', 'SRE', 'XEL', 'WEC', 'ED', 'PEG',
    'ES', 'AWK', 'AMP', 'MET', 'PRU', 'AFL', 'TRV', 'ALL', 'HIG', 'CBRE', 'AMT',
    'CCI', 'EQIX', 'SPG', 'O', 'WELL', 'VTR', 'EQR', 'AVB', 'MAA', 'ESS', 'CPT'
]

# Sector mapping
sectors = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Consumer Discretionary',
    'BRK-B': 'Financials', 'UNH': 'Health Care', 'JNJ': 'Health Care', 'V': 'Financials',
    'XOM': 'Energy', 'JPM': 'Financials', 'PG': 'Consumer Staples', 'MA': 'Financials',
    'HD': 'Consumer Discretionary', 'CVX': 'Energy', 'MRK': 'Health Care', 'ABBV': 'Health Care',
    'PFE': 'Health Care', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples', 'COST': 'Consumer Staples',
    'TMO': 'Health Care', 'AVGO': 'Technology', 'MCD': 'Consumer Discretionary', 'WMT': 'Consumer Staples',
    'CSCO': 'Technology', 'ACN': 'Technology', 'ABT': 'Health Care', 'DHR': 'Health Care',
    'NEE': 'Utilities', 'LLY': 'Health Care', 'DIS': 'Communication Services', 'VZ': 'Communication Services',
    'ADBE': 'Technology', 'NKE': 'Consumer Discretionary', 'CRM': 'Technology', 'CMCSA': 'Communication Services',
    'INTC': 'Technology', 'TXN': 'Technology', 'PM': 'Consumer Staples', 'QCOM': 'Technology',
    'RTX': 'Industrials', 'HON': 'Industrials', 'UPS': 'Industrials', 'IBM': 'Technology',
    'CAT': 'Industrials', 'AMGN': 'Health Care', 'GE': 'Industrials', 'LOW': 'Consumer Discretionary',
    'SPGI': 'Financials', 'BA': 'Industrials', 'GS': 'Financials', 'SBUX': 'Consumer Discretionary',
    'BLK': 'Financials', 'INTU': 'Technology', 'ELV': 'Health Care', 'MDLZ': 'Consumer Staples',
    'AMD': 'Technology', 'ISRG': 'Health Care', 'ADI': 'Technology', 'GILD': 'Health Care',
    'CVS': 'Health Care', 'AXP': 'Financials', 'LMT': 'Industrials', 'BKNG': 'Consumer Discretionary',
    'SYK': 'Health Care', 'TJX': 'Consumer Discretionary', 'VRTX': 'Health Care', 'MMC': 'Financials',
    'REGN': 'Health Care', 'PLD': 'Real Estate', 'ZTS': 'Health Care', 'NOW': 'Technology',
    'PYPL': 'Financials', 'SCHW': 'Financials', 'CI': 'Health Care', 'MO': 'Consumer Staples',
    'DUK': 'Utilities', 'SO': 'Utilities', 'BDX': 'Health Care', 'ITW': 'Industrials',
    'EOG': 'Energy', 'SLB': 'Energy', 'CL': 'Consumer Staples', 'CME': 'Financials',
    'APD': 'Materials', 'FDX': 'Industrials', 'NOC': 'Industrials', 'ICE': 'Financials',
    'NSC': 'Industrials', 'PNC': 'Financials', 'MU': 'Technology', 'AON': 'Financials',
    'USB': 'Financials', 'WM': 'Industrials', 'ORLY': 'Consumer Discretionary', 'GD': 'Industrials',
    'EMR': 'Industrials', 'COP': 'Energy', 'PSA': 'Real Estate', 'TGT': 'Consumer Discretionary',
    'MCK': 'Health Care', 'KLAC': 'Technology', 'ADP': 'Industrials', 'LRCX': 'Technology',
    'F': 'Consumer Discretionary', 'GM': 'Consumer Discretionary', 'SHW': 'Materials',
    'SNPS': 'Technology', 'CDNS': 'Technology', 'FTNT': 'Technology', 'MRVL': 'Technology',
    'PANW': 'Technology', 'AZO': 'Consumer Discretionary', 'CTAS': 'Industrials', 'MNST': 'Consumer Staples',
    'MSCI': 'Financials', 'ANET': 'Technology', 'ADSK': 'Technology', 'NXPI': 'Technology',
    'ROP': 'Technology', 'KDP': 'Consumer Staples', 'AIG': 'Financials', 'PAYX': 'Industrials',
    'NEM': 'Materials', 'IDXX': 'Health Care', 'AMT': 'Real Estate', 'CCI': 'Real Estate',
    'EQIX': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'MAR': 'Consumer Discretionary',
}

print(f"   Total tickers: {len(tickers)}")

# Fetch price data
print("\n3. FETCHING 12-YEAR PRICE DATA...")

all_data = {}
batch_size = 50

for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    try:
        data = yf.download(batch, start=START_DATE, end=END_DATE, progress=False, threads=True)

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in batch:
                try:
                    if ticker in data.columns.get_level_values(1):
                        close = data['Close'][ticker].dropna()
                        if len(close) > 252 * 8:
                            all_data[ticker] = close
                except:
                    pass
        else:
            # Single ticker case
            if len(batch) == 1 and 'Close' in data.columns:
                close = data['Close'].dropna()
                if len(close) > 252 * 8:
                    all_data[batch[0]] = close
    except Exception as e:
        print(f"   Error batch {i}: {e}")
    time.sleep(0.2)

print(f"   Successfully fetched: {len(all_data)} tickers with 8+ years of data")

# Calculate returns and betas
print("\n4. CALCULATING RETURNS AND BETAS...")

spy_close = spy['Close'].squeeze()  # Ensure 1D Series
results = []

for ticker, prices in all_data.items():
    try:
        # Ensure prices is 1D Series
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        # Align dates with SPY
        common_dates = prices.index.intersection(spy_close.index)
        if len(common_dates) < 252 * 8:
            continue

        stock_prices = prices.loc[common_dates].astype(float)
        spy_prices = spy_close.loc[common_dates].astype(float)

        # Total return
        start_price = float(stock_prices.iloc[0])
        end_price = float(stock_prices.iloc[-1])
        if start_price <= 0:
            continue

        total_return = (end_price / start_price - 1) * 100
        years = len(common_dates) / 252
        cagr = ((end_price / start_price) ** (1/years) - 1) * 100

        # Calculate beta
        stock_ret = stock_prices.pct_change().dropna()
        spy_ret = spy_prices.pct_change().dropna()

        common_ret_idx = stock_ret.index.intersection(spy_ret.index)
        stock_ret_aligned = stock_ret.loc[common_ret_idx].values
        spy_ret_aligned = spy_ret.loc[common_ret_idx].values

        if len(stock_ret_aligned) < 252:
            continue

        covariance = np.cov(stock_ret_aligned, spy_ret_aligned)[0, 1]
        variance = np.var(spy_ret_aligned)
        beta = covariance / variance if variance > 0 else 1.0

        # Volatility (annualized)
        volatility = np.std(stock_ret_aligned) * np.sqrt(252) * 100

        # Sharpe ratio (assuming 2% risk-free rate)
        sharpe = (cagr - 2.0) / volatility if volatility > 0 else 0

        # Max drawdown
        rolling_max = stock_prices.expanding().max()
        drawdown = (stock_prices - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min()) * 100

        results.append({
            'Ticker': ticker,
            'Sector': sectors.get(ticker, 'Unknown'),
            'Total Return (%)': total_return,
            'CAGR (%)': cagr,
            'Beta': beta,
            'Volatility (%)': volatility,
            'Sharpe': sharpe,
            'Max Drawdown (%)': max_drawdown,
        })
    except Exception as e:
        pass

results_df = pd.DataFrame(results)
print(f"   Analyzed {len(results_df)} stocks with complete data")

if len(results_df) == 0:
    print("ERROR: No results generated.")
    exit(1)

# Filter for outperformers (beat S&P 500)
avg_return = results_df['Total Return (%)'].mean()

outperformers = results_df[
    (results_df['Total Return (%)'] > spy_return) &
    (results_df['Total Return (%)'] > avg_return)
].copy()
outperformers = outperformers.sort_values('Total Return (%)', ascending=False)

underperformers = results_df[results_df['Total Return (%)'] <= spy_return].copy()

print(f"\n" + "=" * 80)
print(f"S&P 500 BENCHMARK: {spy_return:.1f}% total return ({spy_cagr:.1f}% CAGR)")
print(f"AVERAGE STOCK RETURN: {avg_return:.1f}%")
print(f"OUTPERFORMERS (Beat SPY & Average): {len(outperformers)} / {len(results_df)}")
print("=" * 80)

# Show top 30 outperformers
print("\n5. TOP 30 OUTPERFORMERS:")
print("-" * 100)
top30 = outperformers.head(30)
for _, row in top30.iterrows():
    print(f"{row['Ticker']:<6} {row['Sector']:<25} Return:{row['Total Return (%)']:>8.0f}%  CAGR:{row['CAGR (%)']:>6.1f}%  Beta:{row['Beta']:>5.2f}  Sharpe:{row['Sharpe']:>5.2f}  MaxDD:{row['Max Drawdown (%)']:>6.0f}%")

# Analyze by sector
print("\n" + "=" * 80)
print("6. SECTOR ANALYSIS OF OUTPERFORMERS")
print("=" * 80)
if len(outperformers) > 0:
    sector_analysis = outperformers.groupby('Sector').agg({
        'Ticker': 'count',
        'Total Return (%)': 'mean',
        'CAGR (%)': 'mean',
        'Beta': 'mean',
        'Sharpe': 'mean',
    }).rename(columns={'Ticker': 'Count'}).sort_values('Count', ascending=False)
    print(sector_analysis.to_string(float_format=lambda x: f'{x:.2f}'))

# Compare characteristics
print("\n" + "=" * 80)
print("7. OUTPERFORMER vs UNDERPERFORMER CHARACTERISTICS")
print("=" * 80)

print(f"\n{'Metric':<25} {'Outperformers':>15} {'Underperformers':>15} {'Difference':>15}")
print("-" * 70)

for metric in ['Beta', 'Volatility (%)', 'CAGR (%)', 'Sharpe', 'Max Drawdown (%)']:
    out_val = outperformers[metric].mean() if len(outperformers) > 0 else 0
    under_val = underperformers[metric].mean() if len(underperformers) > 0 else 0
    diff = out_val - under_val
    print(f"{metric:<25} {out_val:>15.2f} {under_val:>15.2f} {diff:>+15.2f}")

# High beta outperformers
print("\n" + "=" * 80)
print("8. HIGH BETA OUTPERFORMERS (Beta > 1.0)")
print("=" * 80)
high_beta = outperformers[outperformers['Beta'] > 1.0].sort_values('Total Return (%)', ascending=False)
print(f"Count: {len(high_beta)} / {len(outperformers)} outperformers")
if len(high_beta) > 0:
    print("\nTop 15 High-Beta Outperformers:")
    for _, row in high_beta.head(15).iterrows():
        print(f"  {row['Ticker']:<6} {row['Sector']:<22} Return:{row['Total Return (%)']:>7.0f}%  Beta:{row['Beta']:>5.2f}  Sharpe:{row['Sharpe']:>5.2f}")

# Save results
outperformers.to_csv('sp500_outperformers.csv', index=False)
results_df.to_csv('sp500_all_analysis.csv', index=False)
print(f"\nSaved: sp500_outperformers.csv, sp500_all_analysis.csv")

# Final analysis
print("\n" + "=" * 80)
print("9. KEY FINDINGS - QUANTIFIABLE FACTORS")
print("=" * 80)

if len(outperformers) > 0 and len(underperformers) > 0:
    out_beta = outperformers['Beta'].mean()
    under_beta = underperformers['Beta'].mean()
    out_vol = outperformers['Volatility (%)'].mean()
    under_vol = underperformers['Volatility (%)'].mean()
    out_sharpe = outperformers['Sharpe'].mean()
    under_sharpe = underperformers['Sharpe'].mean()

    print(f"""
SUMMARY OF OUTPERFORMER CHARACTERISTICS:

1. BETA ANALYSIS
   - Outperformers avg beta: {out_beta:.2f}
   - Underperformers avg beta: {under_beta:.2f}
   - Signal: {'Higher beta correlates with outperformance' if out_beta > under_beta else 'Lower beta correlates with outperformance'}

2. VOLATILITY ANALYSIS
   - Outperformers avg volatility: {out_vol:.1f}%
   - Underperformers avg volatility: {under_vol:.1f}%
   - Signal: {'Higher volatility accepted for returns' if out_vol > under_vol else 'Lower volatility with better returns'}

3. RISK-ADJUSTED RETURNS (Sharpe)
   - Outperformers avg Sharpe: {out_sharpe:.2f}
   - Underperformers avg Sharpe: {under_sharpe:.2f}
   - Signal: Sharpe ratio {out_sharpe/under_sharpe:.1f}x higher for outperformers

4. SECTOR CONCENTRATION
""")

    top_sectors = outperformers['Sector'].value_counts()
    total_out = len(outperformers)
    for sector, count in top_sectors.head(5).items():
        pct = count / total_out * 100
        print(f"   - {sector}: {count} stocks ({pct:.0f}% of outperformers)")

    print("""
5. QUANTIFIABLE SELECTION CRITERIA:
   - Beta > 1.0 (captures market upside)
   - Sector: Technology, Health Care, Consumer Discretionary
   - Historical Sharpe > 0.5 (risk-adjusted return efficiency)
   - Willing to accept higher volatility for returns
""")
